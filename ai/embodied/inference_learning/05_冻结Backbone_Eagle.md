# 05 · 冻结 Backbone（Eagle VLM 如何"看懂世界"）

> 目标：理解 backbone 在推理里扮演的角色 —— 把视频帧与语言指令编码成 action head 能用的特征，
> 以及它为什么是"冻结"的、如何省算力。
> 对应代码：`Isaac-GR00T/gr00t/model/backbone/eagle_backbone.py`

## 1. 为什么需要"看懂世界"的 backbone

![Eagle grounding 示例](images/ch12/grounding_example1.png)
*图：N1.5 的 Eagle VLM 具备"指称表达→像素区域"的 grounding 能力（GR-1 场景 IoU 40.4，优于同级 Qwen2.5-VL-3B）——这正是"冻结它、借它看懂世界"的底气。原文：<https://research.nvidia.com/labs/gear/gr00t-n1_5/>*


动作不能凭空生成——机器人必须**先理解**"看到了什么、被要求做什么"。这部分"理解"能力
由一个大 VLM 提供：`Eagle 2.5`（视觉-语言模型）。
- 它把多张视频帧 + 一段语言指令，变成一串**语义特征** `backbone_features`。
- 这些特征作为"条件"（condition）喂给 action head，告诉它"在当前的场景里该朝哪动"。

## 2. backbone 的构成（`EagleBackbone`）

```python
class EagleBackbone(nn.Module):
    def __init__(self, tune_llm=False, tune_visual=False,
                 select_layer=-1, eagle_path=None, project_to_dim=1536, ...):
        self.eagle_model = AutoModel.from_config(config, trust_remote_code=True)
        # VLM 输出 hidden 维度 2048 → 投影到 project_to_dim(默认1536)
        self.eagle_linear = nn.Linear(2048, project_to_dim)
        # 只保留前 select_layer 层，省算力（后续层不用就被裁掉）
        while len(self.eagle_model.language_model.model.layers) > select_layer:
            self.eagle_model.language_model.model.layers.pop(-1)
        self.select_layer = select_layer
        self.eagle_linear = ...  # 若 project_to_dim 为 None → Identity
```
要点：
- 模型权重从**本地** `eagle2_hg_model/`（仓库自带）加载，不依赖网络。
- `select_layer` 决定取**哪一层的 hidden state** 作为特征，并据此**裁剪**语言模型尾部层——少算。
- `eagle_linear` 把 VLM 的高维特征（2048）投影到 action head 期望的维度（1536）。

## 3. "冻结"是什么意思（`set_trainable_parameters`）

```python
def set_trainable_parameters(self, tune_llm, tune_visual):
    for p in self.parameters(): p.requires_grad = True
    if not tune_llm:    self.eagle_model.language_model.requires_grad_(False)  # 冻结 LLM
    if not tune_visual: self.eagle_model.vision_model.requires_grad_(False)
                        self.eagle_model.mlp1.requires_grad_(False)            # 冻结视觉
```
- `requires_grad_(False)`：这些参数的梯度不再计算、权重不更新 → "冻结"。
- 推理场景本来就不反向传播；**冻结**的真正意义是省显存/显式表达"不动它"。
- 有个训练相关细节 `set_frozen_modules_to_eval_mode()`：HF 每次训练会 `model.train()`，
  但冻结模块要保持在 `eval()`（关掉 dropout/BatchNorm 的随机行为）——推理时整体就是 eval，无影响。

## 4. 前向：把输入变成 backbone_features（`forward`）

```python
def forward(self, vl_input):
    self.set_frozen_modules_to_eval_mode()
    eagle_embeds, eagle_mask = self.forward_eagle(vl_input)
    ...
    return BatchFeature(data={
        "backbone_features": eagle_embeds,        # [B, T, hidden]
        "backbone_attention_mask": eagle_mask,    # 用于注意力掩码
    })

def forward_eagle(self, vl_input):
    eagle_input = { k.removeprefix("eagle_") : v
                    for k, v in vl_input.items() if k.startswith("eagle_") }
    del eagle_input["image_sizes"]
    eagle_output = self.eagle_model(**eagle_input,
                                    output_hidden_states=True, return_dict=True)
    eagle_features = eagle_output.hidden_states[self.select_layer]  # 取指定层
    eagle_features = self.eagle_linear(eagle_features)               # 投影
    return eagle_features, eagle_input["attention_mask"]
```
- `prepare_input` 这里只是把 batch 原样包成 `BatchFeature`（真正拆字段发生在模型级 prepare_input 之前）。
- 输入约定：前缀 `eagle_` 的字段会被剥掉前缀后交给 VLM；`image_sizes` 被剔除。
- 输出键固定：`backbone_features`（特征）+ `backbone_attention_mask`（掩码），
  与第 04 章 `validate_data` 要求一致。

## 5. backbone 在整条链路里的"上下游"

```
观测(video+annotation)
   ▼ eagle_backbone
backbone_features (B,T,hidden) + mask
   ▼ 作为条件送入
FlowmatchingActionHead.get_action(backbone_features, action_inputs)
   ▼
action_pred (B, horizon, dim)
```
backbone 的输出**不直接是动作**，而是"世界的理解"，作为条件引导动作生成。这是理解
"双脑"分工的关键：一个脑负责"看/想语义"，另一个负责"生成动作"。

## 6. 本章小结

- Backbone = 冻结的 Eagle VLM + 一层线性投影（2048→1536）+ 层裁剪省算力。
- 作用：把视频+语言变成 `backbone_features`（语义条件）。
- 冻结即 `requires_grad_(False)`；推理下无梯度，天然轻量。
- 输出 `backbone_features` + `backbone_attention_mask`，作为 action head 的条件输入。

## 自己动手

1. 打开 `eagle_backbone.py`，数一下 `forward` 输出了哪些键，和 `gr00t_n1.py::validate_data`
   需要的键对不对得上。
2. 思考：如果 `project_to_dim != action_head 期望维度`，会发生什么？在哪个层报维度错误？

## 疑问与批注

### 2026-09-23 · 授课问答存档（首轮）+ 三个超越标准答案的收获

> 本章以"老师提问 → 学生作答 → 判卷修正"的问答形式走完全章。以下按授课顺序存档，
> 含两处易错点修正与一张"泛化能不能兜住"的判据表（复习时优先看这张表）。

#### Q&A 实录（要点式）

- **Q1 backbone_features 的角色**：✔ 答对——它是条件（condition），"指导"动作生成；
  具体动作由扩散/流匹配从噪声迭代恢复。补充精确化：它不是"开头喂一次"，
  而是在去噪的**每一步**都被 action head 通过 cross-attention 反复查询（第 06 章细讲）。
- **Q2 backbone 输出的两个键**：当时遗忘，答案为 `backbone_features` + `backbone_attention_mask`
  （与第 04 章 `validate_data` 的契约一致）。
- **Q3 推理时 `requires_grad_(False)` 还剩什么用**：学生答"对部署微调设限"——对了一半（契约/防误调）。
  补齐另一半：① **防呆保险**：若有人绕过 Policy 的 `no_grad` 裸调 backbone 且输入带梯度，
  整张计算图会被建起、激活全扣住，`requires_grad_(False)` 保证图建不起来；
  ② "冻结省显存"的真正红利在**训练态**：不存梯度、不进优化器状态（Adam 约 2× 参数量）。
- **Q4 删掉 `set_frozen_modules_to_eval_mode()` 埋什么雷**：学生答"训练时 dropout 是正则、
  推理关闭很正常，为何影响推理？"——由此引出本节最有价值的一轮展开（见下方"泛化判据"）。
  途中纠正一个概念偏差：**dropout 丢的是激活值，不是参数**；train/eval 切换改的是
  "前向计算方式"，与 requires_grad（梯度与否）正交。
- **Q5a `project_to_dim=512` 与 head 期望 1536 不符，谁报错**：✔ 答对——action head 内部
  第一个消费 1536 的权重矩阵处爆（DiT cross-attention 的 K/V 投影），报通用
  `RuntimeError: shapes cannot be multiplied`。两个补充：① backbone 与 head 靠 **dict 键名
  约定**松耦合（非 `nn.Sequential` 串联），故报错时机是**运行时**而非加载时，
  `validate_data` 只查键名不查形状也拦不住；② 排查技巧：看报错栈里权重矩阵属于哪个模块，
  那个模块就是"按旧维度建的消费者"，往上游找生产者。
- **Q5b `select_layer` 与 `requires_grad_(False)` 各省什么**：大方向对，三处精度修正：
  ① 裁剪发生在**加载之后**（`from_config` 先全量建模再 `pop`），省的是驻留显存 + 每步前向
  计算/激活，**不省加载 I/O 与内存峰值**；② `select_layer` 还有一半语义是"**从第 16 层取特征**"——
  先确认中间层理解特征够用，才敢砍它后面（理解 ≠ 生成）；
  ③ 训练期冻结省三样：反向计算、**为反向保存的激活**（显存大头）、梯度+优化器状态。
- **追问：泛化不该兜住 train/eval 错位吗？**（学生主动质疑，见判据表）
- **追问：换整个 backbone 复用旧 head 属于哪类？**：✔ 学生裁决"急性事故"，正确，且排序
  （比 BN 错位更严重）也对——BN 错位是同分布的仿射搬移，可小校正救；换骨干是特征**语义坐标系**
  整个换掉，head 的 K/V"读数刻度"逐维失效。救治路径：冻结新骨干，**重训 `eagle_linear` +
  action head**——这正是投影层作为骨干与任务头之间**可重训缓冲区（adapter）**的深层意义。
- **追问（学生主动提出）**："噪声加在输入图像侧比加在 backbone 输出特征侧更利于泛化"——
  具身领域的主流正确答案，两个深化理由：
  ① 在流形上 vs 出流形：图像增广经冻结 VLM 后产生的特征变化对应"真实世界拍得到的东西"；
  dropout 抠通道则**没有任何真实图像能让 VLM 吐出那种残缺特征**，买到的鲁棒性部署时兑不了现。
  ② 具身特有约束：**增广必须对动作标签等变**——分类里翻图像标签不变；机器人任务里
  翻转图像 = 最优动作的左右分量要跟着翻转，否则注入的是"图文不符"的假样本，比不加还糟。

#### 核心判据表：train/eval 错位时，"泛化"兜不兜得住？

> 判据一句话：**测试时的偏移是否属于训练时按设计注入过的那个随机过程。**

| 错位类型 | 偏移性质 | 泛化能否兜住 | 临床表现 |
|---|---|---|---|
| 冻结塔 dropout 没关→推理关闭 | 期望对齐、只差方差（inverted dropout 保均值，clean 特征落在噪声族期望处） | **大体兜得住**，但上限被削：head 花容量平均噪声，条件信噪比被人为压低 | 慢性损耗：动作变毛躁，离线指标未必可见 |
| BN train 统计量→eval running 统计量 | **系统性仿射搬移**（通道均值/尺度搬家），训练噪声族之外 | **兜不住** | 急性事故：性能可崩得很直接 |
| 换整个 backbone 复用旧 head | 语义坐标系整体更换，比仿射搬移更远 | **兜不住**，但有标准救治路径（重训 eagle_linear+head） | 换城市级错位 |
| #168 预处理黑边（第 02 章，同族案例） | 训练从未出现过的几何分布 | **兜不住** | 真机翻车、离线可能好看 |

配套结论（设计过的随机性 vs 泄漏的随机性）：

| | 设计过（CFG 条件 dropout、流匹配加噪） | 泄漏（忘关 eval） |
|---|---|---|
| 噪声形式谁决定 | 显式写进训练配方，可调 | 藏在冻结塔配置里的 dropout=0.1 |
| 推理侧对应策略 | 有，按设计执行 | 没有，纯错位 |
| 可消融/回归 | 可（改超参 A/B） | 难（破坏"可归因"纪律） |

以及具身特有的加重因素：机器人是**闭环**，条件带噪 → 本步动作偏 → 到达没见过的状态 →
下一步观测本身出分布 → **误差复利**（covariate shift / DAgger 治的病）。分类任务的
单样本容错逻辑在这里不成立。

#### 三个超越标准答案的收获（复习优先）

1. **泛化覆盖判据**（上表）：能指望泛化的前提是"训练时按设计注入过同族噪声"；
   一致性优先于"希望它顺便泛化一下"。随机性应加在**部署时真实会扰动的地方（观测侧）**，
   且两侧口径写进同一份契约。
2. **裁剪时机**：`select_layer` 是"加载后裁剪"（先全量 `from_config` 再 pop），省驻留显存
   与每步前向，不省加载 I/O——与"按需加载"工程含义不同。
3. **`eagle_linear` = 可重训缓冲区（adapter）**：冻结骨干与任务头之间唯一的薄翻译层，
   骨干一换只需重训它 + head，是"一个模型适配多本体/多骨干"生态的接口设计。

#### 本章迁移收束表（案例 → 普遍原理）

| 本章机制（gr00t 案例） | 普遍原理（换任何 VLA 仍成立） |
|---|---|
| Eagle + `select_layer=16` 裁剪 | 理解 ≠ 生成：条件提取不必跑完为 next-token 调优的深层 LLM（pi0/RT-2 取中间层同款） |
| `eagle_` 前缀剥壳喂原生 VLM | 用命名空间约定实现模块松耦合，第三方权重零改动复用 |
| `set_frozen_modules_to_eval_mode()` | 训练/推理一致性：冻结供应商的交货口径必须前后一致（与 #168 同族） |
| `requires_grad_(False)` + 可训练 `eagle_linear` | 冻结主干 + 只训"翻译层"与任务头：省资源且防灾难性遗忘 |
| 输出 `[B,T,1536]` 作条件 | VLA 的接缝在"语义特征"层，两边只靠形状+键名契约握手 |

#### 遗留动手题

打开真实 `eagle_backbone.py` 数一遍 `forward` 输出的键，与 `gr00t_n1.py::validate_data`
交叉验证（本章笔记口径为两键：`backbone_features` / `backbone_attention_mask`，以真实代码为准）。

