# 06 · 扩散 Action Head 原理（如何"想"出动作）

> 目标：讲清 action head 为什么用"流匹配/扩散"来生成动作，以及推理时那一层 `for t in range(num_steps)`
> 到底在做什么。这是全课程**最核心的原理章**。
> 对应代码：`Isaac-GR00T/gr00t/model/action_head/flow_matching_action_head.py`、`cross_attention_dit.py`

## 1. 为什么不能用"一次前向"直接出动作？

- 机器人动作是**多模态的**：同一场景下"往左够"可能是对的，"往右够"也可能是对的（同样合理）。
- 回归模型（直接预测动作）会"取平均"，得到**模糊/不合理**的中间动作；且难以表达多解。
- 生成模型（扩散/流匹配）能**采样**：每次都从噪声还原出**一条完整合理的动作轨迹**，天然多解。


### 1.1 对照表：两种建模哲学

| | 回归头 | 生成式头 |
|---|---|---|
| 学的对象 | 条件均值 E[a\|obs] | 条件分布 p(a\|obs) |
| 多解场景的输出 | 均值（哪个解都不是） | 每次采样命中**某一个**解 |
| 确定性 | 同输入同输出 | 同输入可出不同轨迹（受噪声种子控制） |

> 一行钉死：MSE 的最优解 **=条件均值**（对固定 obs，使 `E‖f(obs)−a‖²` 最小的 f 就是取期望）。
> "左右两可"场景输出均值不是训练不充分，而是**目标函数层面判了死刑**——数据再多、模型再大都救不回来。

### 1.2 插叙：开环评估 vs 闭环控制——"采样随机"是坏事吗？

术语来自控制论，"环"指反馈环：

```
开环(离线回放): obs(数据集帧) ──► 模型 ──► â ──► 与演示动作 a 算 MSE，完。
              预测错了也不改变下一步看到什么——没有反馈环。
闭环(真机执行): obs ──► 模型 ──► a 被执行 ──► 机器人到新状态 ──► 新 obs ──► ...
              预测误差会改变后续输入——环闭上了。
```

- **开环 MSE 对多模态策略是"错配的考官"**：策略采样出同样合理的"右侧绕行"，演示录的是"左够"——MSE 判它错，真机上它没错。生成式头在小数据集上的开环 MSE **天然偏高且不公允**，这是评估口径问题而非模型问题（第 12/13 章反复叮嘱"换动作空间/换采样种子的 MSE 不许横比"，根源在此）。
- **闭环才是终审**：采样换模式在闭环里常是优点——这次采样卡住了，下一窗口可能采到另一模式，等于免费重试（第 08 章重试例子的机理）。
- 记法：**开环问"你猜得像不像演示"，闭环问"你把活干成了没有"。**

## 2. 流匹配（Flow Matching）直觉

![流匹配训练与推理](images/common/flow_matching.png)
*图：自绘——左：训练时在高斯噪声 x₀ 与真实动作 x₁ 之间取直线插值，红箭头即模型要学的常速度场 u=x₁−x₀（右上的两个点簇=动作的多峰分布，这正是"回归求均值会坍缩"要解决的问题）；右：推理时从噪声出发，用 DiT 预测的速度场做 K 步 Euler 积分，逐步落到某个动作模式上*


想象在"纯噪声 ε ~ N(0,1)"和"真实动作 x"之间，画一条**直线插值**：

- 训练时我们知道"数据点 x 和随机噪声 ε"，定义：
  - 噪声轨迹：`noisy = (1-t)·ε + t·x`（t 从 0→1，从噪声"走"到数据）
  - 目标速度场：`velocity = x - ε`（数据相对噪声的位移）
- 模型学的是：给定 `noisy` 和时刻 `t`，预测 `velocity`。即模型是一个**速度预测器**。
- **推理（采样）时**：从纯噪声开始，沿着模型预测的速度场用 **Euler 法逐步积分**，`t: 0→1`，
  噪声就被"一步一步搬"成了动作。

> 类比：想象让一团噪声"被风吹着"走一条路，模型负责告诉每一时刻"该朝哪个方向、以多快速度走"，
> 走到底就是一张清晰的图（这里是一段动作轨迹）。

### 训练里的关键代码（forward）
```python
noise = torch.randn(actions.shape, ...)
t = self.sample_time(B, ...)          # 从 Beta 分布采时间(偏向两端/中段)
t = t[:, None, None]                  # (B,1,1) 广播
noisy_trajectory = (1 - t) * noise + t * actions   # 直线插值
velocity = actions - noise                            # 速度场目标
...
loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
```
- `sample_time` 用 Beta 分布调度 t，使得对中段/端点分配不同权重（`noise_s` 控制），学得更稳。
- 时间被离散化成 `num_timestep_buckets` 个桶，喂给正弦位置编码（`SinusoidalPositionalEncoding`）。


### 2.5 训练-推理对偶：学的是"全域指路牌"，不是一条轨迹

易混点：训练时每条样本只被问**一次**（"在时刻 t、半糊位置 noisy，速度多少？"），凭什么支撑推理 K 步？

- `t` 是网络输入的一部分（离散桶 + 正弦编码），训练覆盖**整个 (位置, 时刻) 空间**——网络学的是"位置+时刻 → 速度"的**全域指路牌**，不是背下一条条轨迹；
- 推理是把 K 次**单点提问串成链**：第 k 步问完、走一步、**到新位置重新问**。每次重新提问就是重新纠错——上一步折线折歪了，下一步拿到的是针对歪点的新速度，误差不会盲目放大；
- 防误区：这不是"半衰期式越迭代越接近答案"。K 增大减小的是欧拉法**离散化误差**（~O(1/K)）——弯路径用 K 段小直线折逼，仅此而已。

> 一句话：训练=在河道里立好指路牌；推理=放一条船，看一块牌子漂一段。牌子立得好的河
> 不会有"平均方向"的船——河水分叉，每条水分子只走一条。多模态问题就这样被几何化地解决了：
> 粒子落进哪个汇水区(basin)，就出哪个动作。

## 3. 推理采样：get_action 里的去噪循环

```python
@torch.no_grad()
def get_action(self, backbone_output, action_input):
    backbone_output = self.process_backbone_output(backbone_output)  # layernorm + vl self-attn
    vl_embs        = backbone_output.backbone_features
    embodiment_id  = action_input.embodiment_id

    # 状态条件编码
    state_features = self.state_encoder(action_input.state, embodiment_id)

    # 从纯噪声出发
    actions = torch.randn((B, self.config.action_horizon, self.config.action_dim), ...)

    num_steps = self.num_inference_timesteps          # 去噪步数(默认可调)
    dt = 1.0 / num_steps

    for t in range(num_steps):                        # 逐步 Euler 积分
        t_cont = t / num_steps
        t_discretized = int(t_cont * self.num_timestep_buckets)

        # 把"当前动作轨迹+时间"编码
        action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
        if self.config.add_pos_embed:
            action_features = action_features + self.position_embedding(pos_ids)

        # 拼接 状态 + future tokens + 动作 作为自注意力序列
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(B,-1,-1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

        # DiT 前向：以 vl_embs(视觉语言) 为 cross-attention 条件
        model_output = self.model(hidden_states=sa_embs,
                                  encoder_hidden_states=vl_embs,
                                  timestep=timesteps_tensor)
        pred = self.action_decoder(model_output, embodiment_id)
        pred_velocity = pred[:, -self.action_horizon:]   # 只取动作部分

        actions = actions + dt * pred_velocity           # Euler 一步更新

    return BatchFeature(data={"action_pred": actions})
```
核心就是那一行：
```python
actions = actions + dt * pred_velocity     # 欧拉法求解 ODE: dx/dt = velocity
```
- **步数越多越精细**：`num_inference_timesteps`（即 policy 里的 `denoising_steps`）是"质量 ↔ 延迟"
  的旋钮。GR00T-N1.5 常用 4 步左右已经能出不错结果。
- `@torch.no_grad()`：推理不求梯度，省显存、更快。


### 3.5 思想实验：完美"1 步采样"恰好落在均值上（mode averaging 幽灵回归）

符号：`actions_{k+1} = actions_k + dt · net(actions_k, k/K, 条件)`。取 `K=1`（`t_cont=0, dt=1`），整条推理退化为一步：`actions_1 = ε + net(ε, 0, 条件)`。

"完美网络"永远输出**条件期望速度**（MSE 最优解=条件期望，§1.1 的老朋友）。关键观察：**t=0 时刻网络手里只有独立抽样的纯噪声 ε，关于"该去哪个模式"零信息**，此刻它对 `x−ε` 的最优猜测只能对所有可能的 x 求平均：

```
net(ε, 0) = E[x|obs] − ε          （对所有合理模式求均值！）
actions_1 = ε + (E[x|obs] − ε) = E[x|obs]    ← 正是回归头的 mode-averaging 输出
```

所以 **K 步的意义不是"更精细地去噪"，而是给轨迹时间早一点"选边"**：走第一步后 `actions_{1/K}` 已微偏一侧，下一次查询的条件期望不再全模式平均，而是偏向那侧的更尖锐速度。**多步采样 = 边积分边"揭示信息"，从平均逐步坍缩到某一个模式。** 两面：

- **单模式**（强条件下动作 chunk 常接近单峰）：平均速度=真速度，直线一条，1 步真够用——GR00T 敢用 4 步不是巧合；
- **多模式**：1 步必糊，糊法恰等于回归头输出；多步才选边。

![流匹配欧拉积分：选边与均值坍缩（玩具实证）](images/ch06/fm_euler_paths.png)
*图：自绘玩具实证（可复跑：`python3 tools/mk_fig_ch06_denoise_paths.py`）——2D 双簇数据（两个"合理动作模式"）+ 解析可算的**完美速度场** `u(x_t,t)=E[x|x_t]−ε`，对同一批噪声起点真跑欧拉积分。
① K=16：轨迹分别入两个 basin（橙=选边段，蓝=精修段，灰箭头=t=0.25 时刻的"指路牌"）；
② K=1：12 条轨迹全部精准落到全局均值（红 X）——上面推导的"完美 1 步=均值"当场复现；
③ 选边完成时刻 t_commit（簇后验首次越 95%）全部落在 t≈0.31~0.56 的前中段，此后再无人改主意。*

### 3.6 "选边"发生在哪一段：分岔口全在前段

- **t 小**：`x_t` 几乎全是噪声，`E[x|x_t]` 横跨所有模式，速度场微小差异就把轨迹推向不同 basin——**分岔口全在前段**；
- **t 大**：`x_t ≈ 0.9x + 0.1ε`，点已深居某 basin 内，条件期望几乎无方差，网络只是"扶着你滑到终点"，想改主意在数值上已不可能；
- 实践含义：**后段步数对"去哪个模式"几乎没有投票权，它们买的是落点精度**——K=4 时前两步的选边质量决定成败。

这也解释了训练侧 `sample_time` Beta 调度"两端多练"的现实理由：

- **t≈0 端最难**：条件期望最"平均"、目标最暧昧，诸模式命运未定——这里学歪 1°，分岔口就进错 basin，误差后果最不可逆；
- **t≈1 端最精**：条件期望已尖锐，输出的是落点细节修正，最后那段误差≈直接的动作误差；
- **中段最易**：直线路径的速度场接近线性、信噪比高，稀疏采样就够；
- **部署耦合**：推理仅 4 步时，模型实际被查询的时刻只有 `t ∈ {0, .25, .5, .75}` 附近——训练时刻密度与推理查询时刻对齐，正是 `noise_s` 存在的现实理由。

## 4. action head 的组成部件（各自干嘛）

| 部件 | 作用 |
|---|---|
| `state_encoder`(CategorySpecificMLP) | 把机器人当前 state 编码成条件特征 |
| `action_encoder`(MultiEmbodimentActionEncoder) | 把"当前(噪声)动作轨迹 + 时间 t"编码成 token |
| `action_decoder`(CategorySpecificMLP) | 把 DiT 输出解回动作空间 `(B, H, action_dim)` |
| `DiT`(cross_attention_dit) | 扩散 transformer 主体，做自注意力+以 vl 为条件的交叉注意力 |
| `future_tokens`(nn.Embedding) | 一组可学习的"未来目标" token，辅助生成 |
| `vl_self_attention` + `vlln`(LayerNorm) | 对 backbone 特征先做层归一化+自身注意力提炼 |
| `position_embedding` | 给动作 token 加位置信息（顺序/时间） |

## 5. Multi-embodiment 怎么实现（CategorySpecific*）

```python
class CategorySpecificLinear(nn.Module):
    self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
    self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))
    def forward(self, x, cat_ids):
        return torch.bmm(x, self.W[cat_ids]) + self.b[cat_ids].unsqueeze(1)
```
- 对每一种 embodiment（`cat_ids`）保存**一套独立的线性权重** → 同一个模型可服务多种机器人，只换投影头。
- `bmm` 按 batch 里每条样本的 `cat_ids` 取对应权重矩阵做批量矩阵乘。
- 这正是第 03 章 `EMBODIMENT_TAG_MAPPING`（gr1:24, oxe_droid:17, ...）落地的位置：
  `embodiment_id` 从字符串映射成一个下标，用来选 CategorySpecific 的权重。

## 6. 串联全链路（本节课的"成品图"）

```
backbone_features (看懂世界) ─┐
                              ├─► process_backbone_output(LN+自注意力) → vl_embs → DiT 的交叉注意力条件
state(当前状态)                ┘
动作轨迹(从噪声开始) ──► action_encoder(+时间) ──► [state | future | action] 序列 ──► DiT ──► action_decoder ──► velocity
                                                                                              ▲
                     for t in 0..num_steps: actions += dt * velocity    (欧拉去噪)
```
- backbone 提供"语义条件"，DiT 在这个条件下"逐步把噪声变成动作"。这就是双脑分工的原理落点。


## 6.5 事故复盘（课堂）：把"循环内编码"挪出循环——伪装成加速的算法事故

某加速实验论证"encoder 输入形状又没变，何必每步重算"，把 `action_encoder(actions, timesteps_tensor, ...)` 从循环内挪到循环外，循环里只剩 `actions += dt·pred_velocity`。

**坍缩代数**：每步 DiT 输入完全相同（陈旧 action_features + 恒为 0 的时刻嵌入）→ 每步吐出**同一个 v₁** → K 步线性合并：

```
actions = ε + K·dt·v₁ = ε + 1·v₁        (dt = 1/K)
```

一步没浪费——4 小步因速度恒定，**合并成一个横跨 t=0→1 的大步**，恰是 §3.5 的"完美 1 步采样"。症状精确预言：输出不是乱码不是 NaN，而是**平滑、合理-looking、但哪个模式都不是的平均轨迹**（两可场景直冲障碍物）。

三条教训：

1. `action_encoder` 是循环内"**每步重新提问**"的入口，`(x_k, t_k)` 是其左右手；挪出去等于撕掉路牌，只剩 t=0 那一块。
2. 这类 bug **伪装成质量问题**：MSE 只表现为"偏高"。归因要靠**多次采样方差探针**——同 obs 采样多次，正常头在多模态场景样本间轨迹有差异、归一化后一致；冻结头方差为零（对应第 09 章"可归因"纪律）。
3. 措辞校准：cross-attention 的 `vl_embs` 作为 key/value **恒定是设计使然**（世界不随你的草稿变）；坏事的冻结在 **query 侧**——动作 token 不再更新。"题目不变是应该的，学生不再答题才是 bug。"

## 7. 本章小结

- action head 用**流匹配/扩散**：学一个"速度场"，从噪声还原动作 → 支持多模态动作。
- 训练：`noisy=(1-t)ε+t·x`，目标 `velocity=x-ε`，MSE 损失。
- 推理：从随机噪声出发，`num_steps` 步欧拉积分去噪，得到 `action_pred`。
- 全程无梯度（`@torch.no_grad()`）；`denoising_steps` 是质量/延迟旋钮。
- category-specific 权重实现"一个模型多种机器人本体"。


### 迁移总结（案例 → 普遍原理：换 pi0/ACT/任何 VLA 仍成立）

| 本章机制（gr00t 案例） | 普遍原理 |
|---|---|
| 禁用 MSE 回归头、改流匹配 | 多模态动作分布上，逐点损失的最优解=均值=非法动作；要分布不要均值 |
| `noisy=(1−t)ε+tx`，`v=x−ε` | 训练学"全域指路牌"（位置+时刻→速度），单样本单次前向即有精确监督；K 步推理=沿牌欧拉积分 |
| `for t: actions += dt·v` | 采样=逐步揭示信息：前段选边、后段精修；步数是质量↔延迟旋钮（`denoising_steps` ≠ `action_horizon`） |
| vl_embs 走 cross-attn、state/action 走 self-attn | 条件与草稿分通道；条件每步每层被重新查询 |
| CategorySpecific 权重动物园 + `bmm` 索引 | 共享躯干+两端私有翻译=多本体经济学（新本体边际成本 MB 级）；下标传错=安静灾难，需服务层校验/golden 闸门兜底 |

## 自己动手

1. 手推一遍：如果 `num_steps=1`（`dt=1`），`get_action` 变成 `actions = noise + 1·velocity ≈ x`，
   说明即使 1 步也能近似还原——验证欧拉积分离散。
2. 打开 `cross_attention_dit.py`，找到 `DiT.forward`，确认 `encoder_hidden_states=vl_embs` 用的
   cross-attention，以及 `timestep` 是怎么进入各层的（加分项）。
3. 复跑课堂配图：`python3 tools/mk_fig_ch06_denoise_paths.py`。改两个旋钮观察：
   把簇心距离从 ±2.2 拉近到 ±0.8（模式模糊化），t_commit 分布怎么移动？把 K=16 改成
   K=2，看多少轨迹在 2 步内来不及选边、落回均值附近（橙段变长、蓝段变短的失败形态）。

## 疑问与批注

### 2026-09-24 · 授课问答存档（首轮）

> 详细论述已并入正文（§1.1/§1.2/§2.5/§3.5/§3.6/§6.5 与配图脚本），此处只存问答判卷记录。

- **Q1(a) MSE 为何在两可场景必出畸形轨迹**：✔ "统计加权均值"即条件均值；补刀：目标函数层面的死刑，数据与模型容量救不回（§1.1）。
- **Q1(b) 采样随机对开环/闭环的意义**：学生先前半句判对（"贴近高概率模式但偏离演示 → 逐点 MSE 不最小"），开环/闭环术语当时不明——由 §1.2 插叙补齐。
- **Q2(a) 一次前向怎么撑 K 步**：✔ "每步依据当前状态重新问网络、走向下一个半糊"抓中要害；修正"半衰期"类比——K 减小的是 ODE 离散化误差 O(1/K)，非指数逼近（§2.5）。另答学生质疑"t=0.5 算不算真半糊"：流匹配直线插值**字面各占一半**；其怀疑属于 VP 扩散的方差保持路径（那里"糊度"与 t 是弯曲关系）——正是流匹配的卖点之一。
- **Q2(b) 完美 1 步思想实验**：学生要求展开公式重问，当堂推导验证（§3.5）；并配玩具实证图。
- **Q3(a) 选边在轨迹哪段**：✔ "前半段、晚了就来不及"；曾把"两边"误解为"旧速度 vs 新速度"，已修正为**动作模式之间**选（§3.6）。
- **Q3(b) Beta 调度为何两端多练**：✔ "前端快走、后端末修"方向对；补两端难/贵精确原因与 4 步查询时刻耦合（§3.6）。
- **Q4 挪出循环事故**：✔ 两破坏点（x_k 陈旧 + t_k 冻结）与"4 步退化成 1 步"全对；奖励坍缩代数——速度恒定使 K 小步**线性合并**为 t=0→1 大步，症状=平均轨迹而非乱码（§6.5）。措辞校准：cross-attn 的 K/V 恒定是设计，坏事在 query 侧。
- **Q5(a) 新增本体成本**：✔ 64×1536≈1e5 参数 ≈190KB(bf16)，对 3B 是 0.003%；结论："**共享 3B 的理解，私有 0.1% 的翻译**"。
- **Q5(b) 换本体变动带**：85 分——归一化统计、backbone、process_backbone_output、state_encoder、DiT 裁决全对；**漏掉 action_encoder 与 action_decoder**（恰是本节点名的 MultiEmbodiment/CategorySpecific 件）。判据：**碰"本体物理坐标"的层换，活在 1536 语义空间的层不换**。

#### 三个超越标准答案的收获

1. K 步速度恒定 → **线性坍缩成 1 步** → 完美 1 步 = 条件均值 = 回归头死法（"选边发生在前段"的数学根源，§3.5 + 配图实证）；
2. "循环内编码挪出循环"是**伪装成性能问题的算法事故**，MSE 只见"偏高"，归因需多次采样方差探针（§6.5）；
3. 换本体分界线：**物理↔语义的两次翻译（入头、出动作）本体私有，中间的思考共享**——与第 05 章 `eagle_linear` adapter 思路同构。

