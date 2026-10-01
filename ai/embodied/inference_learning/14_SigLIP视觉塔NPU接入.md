# 14 · SigLIP 视觉塔：从调用链到 NPU 融合算子

> 本章把"视觉塔那一半"专项收口：SigLIP 的实现在哪、被谁在何时调用、
> `transformers_npu` 如何把它逐个结构换到 LPU kernel（当前主链）、`groot_ops`
> 又备好的"整层一次调用"融合入口为何还没上主链。
> **基准**：`Isaac-GR00T@n1.5-release` + `transformers 4.51.3` + `groot_ops@v0.2(7df9a04)`；
> trace 口径 = 09 章 §10.6 的 CUR 热轮（`trace_current_clean/n15_sd3_chrome_trace.json`，单轮 infer）。
> 前置：05 章（视觉塔结构与图 5-1/5-2）、09 章（T0/T1/T2 三级入口、静默失效框架）。

## 1. 为什么单列一章

> **分工定位**：SigLIP 的原生结构与参数全表在 05 章 §2.5（图 5-1/5-2）；本章的主语是
> **NPU 接入**——调用链（§3）只讲到"能归因算子"为止，随后就交给 kernel/trace。
> 所以它是 09 章方法论（T0/T1/T2、版本契约、计数纪律）在视觉塔上的**专项落实**，
> 排在 09/13 之后、作为专题存在，而不是 05 章的续篇。"三件套（ViT+LLM+DiT）"这个
> 心智模型本身的校准，在 04 章 §1.1。

1. **它在瓶颈上**。热轮一轮 infer 的 prologue（backbone 段）≈55 ms，占整轮 36%（09 章 §10.6），
   而 prologue 里视觉塔 27 层是最大的逐层发射源：`unified_mha_run_die_batch_emb` 单轮恰好 ×27，
   与层数一一对应——这就是"每层都在单独发 attention kernel"的铁证。
2. **它每轮只跑一遍**（DiT 去噪 4 步复用 backbone 特征，06 章），所以视觉塔的融合收益
   不会被"步数摊销"稀释，也不会被误记到去噪头上——归因干净。
3. **代码形态特殊**：本地 `gr00t/transformers_npu/npu/siglip.py` **源码被删只剩 pyc**。
   本章顺带演示一套"marshal 考古"方法（§4.1），从字节码符号表重建接入清单——
   这与 09 章"版本契约/静默失效"是同一件事的两面：先要能*看见*接了什么，才谈得上验证没失效。

## 2. 实现归属：SigLIP 的"身体"在 transformers 里，不在 gr00t 里

在 gr00t 仓库里 `grep -r "class Siglip"` 结果是**零**。所有 `SiglipEncoderLayer / SiglipVisionModel`
的类定义都在 `site-packages/transformers/models/siglip/modeling_siglip.py`（conda env `gr00t`，4.51.3）。
gr00t 里能 grep 到的只有三处**胶水**（`gr00t/model/backbone/eagle2_hg_model/modeling_eagle2_5_vl.py`，
vendored remote-code）：

| 位置 | 内容 | 性质 |
| --- | --- | --- |
| `:22` | `from transformers.models.siglip.modeling_siglip import SiglipVisionModel` | import |
| `:110-112` | `model_type=="siglip_vision_model"` → 强制 `_attn_implementation="flash_attention_2"` → 实例化 | 分派实例化 |
| `:62` | `_no_split_modules` 列表里的字符串 `"SiglipEncoderLayer"` | **只是字符串**（device_map 切层用），不是调用 |

所以"SigLIP 怎么算"（LN→MHA→残差→LN→MLP→残差的 pre-norm 结构、`gelu_pytorch_tanh`）
由 transformers 决定；而"NPU 换掉它"（09 章 T0）也只能 patch **transformers 的类符号**。
这正是版本契约的由来：**transformers 一旦升版、类名/前向签名一变，补丁静默失效**。

塔本体参数（`eagle2_hg_model/config.json`，结构详见 05 章 §2.5 与图 5-1）：
27 层 / hidden 1152 / 16 头（head_dim 72）/ patch 14 / 输入 224 / MLP inter 4304 /
**无 CLS**（没有 pooling head 那条路，见 §4.2）/ 可学习 PE `[256,1152]` / 每图 `(224/14)² = 256` token。

## 2.5 从像素到 256 个 token：14×14 patch 与一发 Conv2d

先校准一个高频口误：**"14×14"说的是每个小块的空间尺寸（14×14 像素），不是块数**。
一张 224×224 按 14 一格切开，切出的是 **16×16 = 256 块**。

整个入口只有**一个 Conv2d 加一次加法**（`modeling_siglip.py:245` `SiglipVisionEmbeddings`，
transformers 4.51.3，行号实测）：

```python
self.patch_embedding = nn.Conv2d(3, 1152, kernel_size=14, stride=14, padding="valid")  # :253
...
patch_embeds = self.patch_embedding(pixel_values)              # :307  [B,1152,16,16]
embeddings   = patch_embeds.flatten(2).transpose(1, 2)         # :308  [B,256,1152]
embeddings   = embeddings + self.position_embedding(self.position_ids)  # 可学习 PE [256,1152]
```

**kernel == stride == 14 是全部机关**：卷积窗以 14 像素为步长滑过整图，
**不重叠、不留缝**（224 = 14×16 恰好整除），每个像素恰好被一个窗口看到一次。
输出特征图 16×16，每个空间位置 = 一块 patch 的嵌入。

**Conv2d 是化装的 Linear**：普通卷积窗口重叠、权重跨窗口共享；这里窗口恰好铺满
图像，于是这发 conv 严格等价于——

> 把每块 patch 摊平成 `14×14×3 = 588` 维向量（3 通道 × 14 行 × 14 列），
> 过**同一个** `Linear(588→1152)`。权重 `[1152,3,14,14]` reshape 成 `[1152,588]` 即用。

卷积语法买到的只有两件事：588 个数的 gather 与矩阵乘在**一个 kernel** 里完成；
访存模式保持图像式的规整。于是每个输出通道是一个**可学习的 14×14 彩色滤波器**：
权重的 588 个数与该块像素值做内积、当场求和塌缩成一个标量再加 bias，1152 通道
= 1152 种"视觉词"的打分。

> **直觉核查（真实权重实测，图 14-4 中部）**：民俗说法"CNN 第一层自组织成 Gabor
> 条纹"常被照抄给 ViT 的 patch embedding——**这副 SigLIP 权重不是**。对全部 1152
> 行做 2D 频谱，"单峰主导"（正弦条纹的指纹）占比 >0.3 的只有 **3 行**（逐通道口径
> 3456 个里仅 1 个），权重行目视像高维噪声。原因不神秘：这层对 patch 内部零局部性
> 假设，588→1152 又是升维的随机超平面族投影（信息近乎无损），"抽边缘"的活儿整层
> 根本不需要干——27 层双向 attention 有的是容量。条纹检测器是 CNN conv1 的民俗
> 图像，不是所有 patch embedding 的必经形态（教训形态学：查图不查 folklore）。

**账本（每图 224×224）**：

| 项 | 值 |
|---|---|
| patch 数 | (224/14)² = **256**；无 CLS、无额外 token（SigLIP 特色，§4.2） |
| 每 patch 参数 | 588×1152 + 1152(bias) ≈ **678k**（即 conv 权重 reshape [1152,588]） |
| 每 patch 计算 | 588×1152 ≈ 0.68 MMAC |
| 每图计算 | 256 × 0.68 ≈ **173 MMAC**（~0.35 GFLOP） |
| PE 参数 | 256×1152 ≈ 295k，**可学习**（不用 RoPE——patch 无"序列相对位置"可言，位置即格子） |

PE 必须在塔头加，因为 attention 对 token 顺序是置换不变的；256 个位置向量把每个
query 的打分分布"钉"在图像的确定方位上。源码里的 `interpolate_pos_encoding`
（PE 网格双三次重采样）是为多分辨率输入的分支——**gr00t 输入恒 224，永远走不到**。

**三相机**：同一塔同一权重，三张图拼 batch 一趟过（trace 里 `conv2d_patch_embed ×1`
每轮一发，是整塔唯一锚点，§5），出塔共 `256×n_img` 个 token，过 `mlp1 1152→2048`
后按 `151669` 槽位原位回写 LLM 序列（图 14-1）。

**NPU 落点**：这是全塔唯一一处输入是**像素而不是激活**的地方，不属于"通用矩阵乘"
的形状，所以 groot_ops 给它单独一核 `conv2d_patch_embed_run_die`——patch 的
gather + matmul 融合成一发射，输出直接是后续核认得的布局货币。

![SigLIP 的 Conv2d：分块 → 摊平 → 共享 Linear（含真实权重实测与 PE 结构）](images/ch14/siglip_patch_embed.svg)

*图 14-4：自绘（`tools/mk_fig_ch14_patch_embed.py`，直接读 GR00T-N1.5-3B safetensors）——上：224×224 → 16×16 patch 平铺 → flatten 588 维 → [1152,588] GEMM → 1152 维打分；中：均匀抽样 12 行权重 reshape 回 3×14×14 的**真实长相**（噪声状，非条纹；附 1152 行全谱统计）；下：PE 的 PCA-RGB 色场（连续平滑 ⇒ 学出来的坐标系）与邻居余弦统计。*

**Q&A · patchify 四连问（课堂问答存档）**

> **Q1 图像切成 14×14，卷积核也 14×14——是巧合吗？**
> 不是巧合，是恒等式："一块 patch = 一次卷积计算 = 一个 token"要求窗口==步长。
> kernel<stride 漏缝丢像素；kernel>stride 窗口重叠、相邻 token 的嵌入互相纠缠，
> 而且"conv ≡ 分块∘Linear"的等价也随之失效。唯一自由的设计量是 patch 尺寸本身——
> 它决定 token 数 (224/p)²，attention 是二次方付费：SigLIP 14（256 token）、原 ViT 16（196 token）。

> **Q2 窗口对齐、逐位乘再求和——这算传统卷积运算吗？**
> 算术上是（逐位乘求和 + bias，权重跨 256 格共享 ⇒ 仅剩的平移等变性也在）；角色上
> 不是滤波而是**打包机**：卷积家族的威力来自重叠滑窗 + 输出仍是图，这里窗口平铺、
> 分辨率除以 14，算子严格分解为"硬分块（无参数）∘ 共享 Linear(588→1152)"。
> 谱系上是 AlexNet 11×11/s4 这类大步长下采样卷积的极端情形；在传统图像处理里它
> 真正对应的操作是 **JPEG 先切 8×8 块再做 DCT**——把固定 DCT 基换成学出来的基而已。

> **Q3 14×14 点乘，再加上 14×14 个 bias？**
> 两处修正：①点乘之后**当场求和**，588 项塌缩成一个标量——"14×14"只活在求和之前；
> ②bias 是**每输出通道一个**（共 1152 个），加在各自通道的求和之后。一块 patch
> 出口是 1152 个标量 = 1152 次内积 + 1152 个 bias，没有任何 196 这个数字的 bias。

> **Q4 那到底算卷积还是线性变换？**
> 作为**算子**它是卷积的角例（stride=kernel 的互相关）；作为**函数族**它是位置共享
> 的仿射映射，且矩阵没有任何二维结构——不像 DCT/FFT 有固定、可分离、正交的基，
> 它是学出来的稠密非方阵 [1152,588]，reshape 即展开，二维性只剩"行优先取 588 个
> 数"的 gather 约定。图 14-4 中部进一步说明：连"自组织出二维花纹"都不是必然——
> 这副权重的行像噪声，图像先验全在塔里的 27 层。**Conv2d 是载体，分块线性投影是
> 本体，"视觉词表"是产物。**

> **Q5 1152 这个维度是为了迁就 Qwen 吗？**
> 不是。1152 进不了 Qwen3（hidden 2048），也进不了 DiT（1536）。它是 SigLIP 塔自己的
> 带宽：16 头 × 72 head_dim = 1152（SigLIP-so400m 家族签名；NPU 融合核的硬约束
> `HEADS×D==N` 同样是这个等式，见 §6）。与语言塔的接缝只有一处、在塔外：
> `mlp1 Linear(1152→2048)`（05 章 §2.6）——换 LLM 只需重搭桥，塔不动。三塔宽度从不
> 直接对接：ViT→LLM 靠 mlp1，LLM→DiT 靠 `eagle_linear`/vl 投影（2048→1536）。


## 3. 调用链：一次 get_action 里 SigLIP 出现在哪（图 14-1）

![SigLIP 在 gr00t 中的调用链与 NPU 替换点](images/ch14/siglip_call_flow.svg)

*图 14-1：左列八步 = 一次 `get_action` 的调用顺序（file:line 实测）；右侧红虚线框 = `transformers_npu` 的替换点，绿框 = groot_ops 已导出但主链未启用的整层融合。再生成：`tools/mk_fig_ch14_siglip_flow.py`。*

### 3.1 逐级走读

1. `Gr00tN1Model.get_action`（`gr00t_n1.py:172`）收到 obs（3 相机 224×224 + state + 语言指令）。
2. collate（`gr00t/model/transforms.py:55-83`）：`Eagle2_5_VLProcessor` 出 `pixel_values / input_ids`，
   且**所有 key 加 `eagle_` 前缀**——这是 backbone 与 state/language 处理器共用一个 batch 的分隔手段。
3. `EagleBackbone.forward_eagle`（`eagle_backbone.py:100-111`）：剥 `eagle_` 前缀 →
   `self.eagle_model(**input, output_hidden_states=True)` → 取
   `hidden_states[select_layer]` → 过 `eagle_linear` 出 `backbone_features`。
4. Eagle 前向内 `Eagle2_5_VLModel.forward` 调 `extract_feature(pixel_values)`
   （调用点 `:237` / `:358`，定义 `:311-338`）。
5. **`extract_feature` 是 SigLIP 全仓库唯一调用点**：`select_layer==-1` 时直接取
   `vision_model(...).last_hidden_state`（`:312-322`）→ `mlp1`（1152→2048，`:334-338`）。
6. 回写：`input_ids == image_token_index(151669)` 的位置**原位替换**为视觉嵌入（`:247`）。
7. Qwen3 前 12 层消费图文混合序列，出 `backbone_features`（LM 侧 NPU：
   `qwen3_attention_run_die×12 + mlp_swiglu×12`，那是 09 章的戏份）。
8. Action Head：DiT 去噪 **4 步全部复用第 7 步的输出，不再回视觉塔**——
   所以 trace 里 `conv2d_patch_embed_run_die` 单轮只出现 ×1。

### 3.2 一次还是多次：用 trace 验证"心智模型"

"DiT 复用 hidden、SigLIP 每轮只跑一遍"不是从注释里读来的，是 trace 里数出来的：
`conv2d_patch_embed_run_die` ×1、`unified_mha_run_die_batch_emb` ×27（=27 层×1 遍）。
若哪天重构后这个 27 变成 27×K，说明有人把编码挪进了去噪循环——正是 06 章 §6.5
"循环内编码坍缩事故"的监测器。**计数即断言**。

### 3.3 陷阱备忘：两个同名 `select_layer`

| 出处 | 语义 | 典型值 |
| --- | --- | --- |
| eagle config（`modeling_eagle2_5_vl.py:99`） | **视觉塔**取层：-1=last_hidden_state，否则取 `hidden_states[k]` | -1 |
| backbone 配置（`eagle_backbone.py:56-58`） | **LM**：加载后 `layers.pop` 裁到前 k 层 + 特征 tap 层 | 12 |

两个 `select_layer` 同名不同物、作用于塔的不同部位；改任何一个都不会报错，只会悄悄改特征来源。
05 章批注已记过一次，这里再钉一遍：**读 backbone 代码先问"哪个 select_layer"**。

## 4. T0 注入点：NPU 补丁换掉了哪些符号

### 4.1 pyc 考古：源码只剩字节码时怎么读

`gr00t/transformers_npu/npu/siglip.py` 本地只剩 `__pycache__/siglip.cpython-311.pyc`。
用 `marshal` 直读字节码常量表即可重建结构（python3.11 环境下）：

```python
import marshal
code = marshal.loads(open('gr00t/transformers_npu/npu/__pycache__/siglip.cpython-311.pyc','rb').read()[16:])
def walk(c, d=0):
    print(' '*d + c.co_name)
    for k in c.co_consts:
        if hasattr(k, 'co_names'): walk(k, d+1)
walk(code)
```

得到的骨架（实测输出）：

```
NPU_SiglipVisionEmbeddings   __init__ / _load_lpu / forward
NPU_SiglipAttention          __init__ / _load_lpu / _buf / _tables / forward
_attl_unified_siglip
NPU_SiglipMLP                __init__ / _load_lpu / forward
siglip_encoder_layer_forward_wrapper → new_forward
siglip_encoder_forward_wrapper      → new_forward
vision_transformer_forward_wrapper  → new_forward
_dump / _sav / _ln
```

`_load_lpu`（T1 权重期懒上设备）、`_buf/_tables`（T2 前向期预分配/常量表）正是
09 章三级入口里 T1/T2 的形态；`_dump/_sav` + 字符串 `'/tmp/npu_%s.npy'`、
`NPU_DUMP_SIGLIP` 是探针（§7）。

### 4.2 替换点清单（从 `npu_patch.pyc` 提取）

| transformers 符号 | 换成 | 注入时机 |
| --- | --- | --- |
| `siglip.modeling_siglip.SiglipEncoderLayer` | `siglip_encoder_layer_forward_wrapper` 的 `new_forward`（内部逐结构走 NPU_* 类） | `install_backbone_whole` |
| `siglip.modeling_siglip.SiglipMultiheadAttentionPoolingHead` | `_npu_siglip_pooling_forward` | `install()` |
| （塔级）`vision_transformer_forward_wrapper`、`siglip_encoder_forward_wrapper` | 包 `SiglipVisionTransformer/SiglipEncoder.forward` | 同上 |

两条纪律（09 章 §3.3 的老结论，此处对号入座）：

- **必须早于 `AutoModel.from_config`**（`eagle_backbone.py:50-51`）：类符号替换发生在
  实例化之前才生效；晚一步就是静默失效，症状是"逐 op 现算链复活，~30 ms/轮 host"。
- 换的是 **transformers 模块上的类符号**，不是 gr00t 的胶水——所以版本契约锁的是
  `transformers==4.51.3`，与 gr00t 自身改不改无关。

## 5. Per-op 接入清单：当前主链每一层发什么核（图 14-2）

![一个 SiglipEncoderLayer 的三栏映射](images/ch14/siglip_layer_ops_map.svg)

*图 14-2：同一个 SiglipEncoderLayer（×27）的三栏映射——原生结构 | 当前主链的 per-op kernel | groot_ops 已导出的整层融合。再生成：`tools/mk_fig_ch14_siglip_flow.py`。*

`groot_ops@v0.2` 中 SigLIP 相关算子（`ops/torch_ops/` 三件套 + `*_ffi.cc`）与主链实测计数：

| 层内结构 | ffi 名（torch_evo.*） | kernel 符号 | 单轮 trace 计数 | 归因备注 |
| --- | --- | --- | --- | --- |
| patch embed conv | `conv2d_patch_embed` | `conv2d_patch_embed_run_die(/v2)` | **×1** | 唯一识别视觉塔 |
| LN1 / LN2 / 塔尾 post-LN | `layernorm` | `layernorm_run_die(/v2)` | ×68 | **共享核**（LM/头也用），不能整除 27 |
| q/k/v/out 投影 | `gemm_bias`（导出名见 `gemm_bias_msplit`） | `gemm_bias_run_die(/v2/_m_split)` | ×129 | 共享核；视觉名义 4×27=108 |
| MHA 本体 | `unified_mha`、`unified_mha_qv_emb` | `unified_mha_run_die_batch_emb` | **×27** | 与层数一一对应 |
| MLP（fc1+gelu+fc2） | `mlp_gelu(+_msplit/relu/silu)` | `mlp_gelu_run_die(/_m_split/_v2)` | ×31 | ffi 带 `residual_enabled`（残差可折入） |
| 残差加 | `residual_add` | `residual_add_run_die` | ×25 | **< 名义 2×27**：部分残差已折进邻居核 |
| （备而未用）QKV 三合一 | `linear_qkv` | `linear_qkv_run_fused` | ×4 | 计数显示**不走视觉塔**（视觉 qkv 仍是 4 发 gemm_bias） |

pyc 侧的调用点符号与之对得上：`conv2d_patch_embed, gemm_bias, layernorm_buf, mlp_gelu,
residual_add_buf, unified_mha_forward / unified_mha_forward_qv_T, clone_split, post_layernorm`。

**计数归因三纪律**（本章反复用到的方法论）：

1. **名 + @op 归组、trace 为准**：先按 kernel 名归组，再与结构对数，不吻合处以 trace 为准。
2. **区分"唯一识别核"与"共享核"**：`conv2d_patch_embed(×1)`、`unified_mha_batch_emb(×27)`
   唯一属于视觉塔，是锚点；`layernorm/gemm_bias/mlp_gelu` 全模型共用，计数不能按
   "27 层教科书结构"硬拆（否则一定得出"数目对不上"的假 bug）。
3. **导出 ≠ 启用**：`linear_qkv_run_fused` 有导出、有 ×4 计数，但那 4 次不在视觉塔；
   验证"某个融合是否真在跑"，永远看它的 kernel 在 trace 里的次数。

图 14-2 是"结构→核"的三栏账本；图 14-3 换成与 01 章 llama_decoder.png 同款的
**逐运算画法**：主干竖线 = hidden 逐级下行、左侧绕行 = 残差、中间 qkv 三叉 →
⊗ 打分 → softmax → ⊗V → out proj 折回，MLP 展开为 fc1→gelu_tanh→fc2。
**右侧红标签 = 当前主链真正发射的 LPU kernel 及其 trace 计数**，一眼看清
"每一格运算落在哪个核上"。ViT 与 Llama 骨架的五处不同（无 causal mask、无 RoPE、
无 CLS/尾头、无 KV cache、非 SwiGLU）在图底部集中列出。

![SigLIP 27 层逐运算展开 + NPU kernel 落点](images/ch14/siglip_vit_stack.svg)
*图 14-3：自绘（`tools/mk_fig_vit_llm_dit_stacks.py`）——pre-norm 双残差与 llama_decoder.png 逐格同构；红标签计数纪律见 §5 末三条。与 05 图 5-3、06 图 6-1 同一套读图语法。*

## 5.5 attention 的“注意”住在哪一格：α 路由图（图 14-5）

§2.5 讲的是**入口**——588 → 1152 的一次线性变换，与图像内容无关、块与块之间零交换。
本节讲**核心**：27 层里真正在搬信息的那一步，也就是图 14-3 中间那三格
`⊗ → softmax → ⊗V`。三行算式（本头 `head_dim=72`，72 = 1152/16）：

```
s_j = (q · k_j) / √72      j = 1..256          # 打分：我和 256 个格子各有多相关
α   = softmax(s)           ⇒ 256 个非负数、和恰为 1
o   = Σ_j α_j · v_j                            # 加权搬运：新 1152 维向量送回残差流
```

**Q/K/V 是三种角色，不是三堆数**：q「我要找什么」、k「我这儿有什么、快来找我」、
v「选中我就把我带走」。K 是**索引**、V 是**内容**——同一个 patch，在别人的注意力里
被查的是它的 K、被搬走的是它的 V。这也解释了为什么 k_proj 与 v_proj 必须是两个矩阵。

**“和为 1”是全部机关**：α 是一笔**竞争性预算**——给 (6,8) 多分一分，给 (11,0) 就少一分。
所以“注意”的实体不是“算得多”，而是**同一份预算下按内容重新分配**。归一化之前那只是
打分，归一化之后才成为分配；这是 softmax 在这里不可替换的原因（换成 sigmoid 就退化成
256 个独立的门，预算不再互斥，“注意”当场消失）。

**静态查表 vs 动态路由**——本节要立的分野，也正是「入口 §2.5」与「核心 §5.5」的分界线：

| | 静态查表（卷积 / 查表） | 动态路由（attention） |
|---|---|---|
| 权重与输入的关系 | 无关：学一次用到底 | 由当前 q、k **现算** |
| 一张表管多少 | 全图所有位置共享同一组系数 | 每层、每头、每 query 各一张新表 |
| 学到的东西 | 一组固定滤波器 | **规则**（W_Q/W_K 怎么打分）；**实例**（本帧本格的 α）当场算 |
| 类比 | 词典：一个词一个释义 | 语境：同一个词在句子里现取释义（词义消歧） |

### 真实权重实测（图 14-5，不是示意图）

图 14-5 的四张小热力图与右侧连线图**是同一行 α**：真实 demo 帧（`demo_data/cube_to_bowl_5`
前相机第 0 帧，ffmpeg 抽帧后按管线 resize 224×224）过真实 GR00T-N1.5-3B SigLIP 权重
（剥前缀 `backbone.eagle_model.vision_model.` 直接 load 进 `SiglipVisionModel`，
missing=0），`eager` attention + `output_attentions=True` 抓到的 27 层 × 16 头 softmax
之后的值；被追踪的 query 是第 120 格（第 7 行第 8 列，立方体右下缘旁的一块桌面）。
四张表给同一块桌面格发了四种完全不同的指令：

| 面板 | (层,头) | top-1 | 熵（均匀=ln256=5.55） | 这一头在干什么 |
|---|---|---|---|---|
| A | L0 h4 | 0.012（≈1/256） | **5.51** | 均匀表 ⇒ 等价“取全图平均”，一个免费的全局上下文池化 |
| B | L1 h2 | 0.963 → 紧邻格 (6,8) | 0.21 | 局部滤波；全图 256 个 query 的目标两两不同（239 个不同目标）⇒ **不是位置查表** |
| C | L12 h3 | 0.998 → (11,0) | 0.02 | 左缘一块桌面把**全部 256 个 query** 的 top-1 全吸走 ⇒ 注意力汇（sink） |
| D | L1 h13 | 0.095 | 4.35 | 质量摊到立方体/绿碗/机械臂/橙球四个物体上（右图连线） |

顺手量的另外两个事实（同一份 α，全 256 个 query 统计，不挑好看的）：

- **局部性是 U 型不是单调**：top-1 目标到 query 的曼哈顿距离中位数 L0=12 → L9=3 →
  L21=12，“≤2 格”占比 0.12 → 0.43 → 0.06 ⇒ 浅层补局部、中层最局部、深层几乎全远程。
  “ViT 浅层看局部、深层看全局”这句口头禅，前半段成立，后半段要打个折（见下条）。
- **视觉塔里也有注意力汇**：L12/L18/L20/L21/L22/L26 至少 9 个头把全部 256 个 query 的
  top-1 投到同一格 (11,0)（左缘桌面）。这不是 bug：softmax 需要一个“谁都不像”的格子
  倾倒多余质量，与 LLM 里首个 token 吸走注意力的 sink 同源。**读论文看到 attention
  map 全图发亮时，先查汇，再谈“模型在看什么”。**

### α 数学上存在，硬件上从不物化

一层 16 张 256×256 表、一塔 27 层 = **432 张/图**，合计 **28,311,552** 个 α，fp16 存下
**54 MiB/图**（三相机 batch 一趟 = 85 M 个、162 MiB）。flash/online-softmax 的做法是分块
滚动归一，**整张 α 从不落进任何内存**；在 NPU 上更是连“块”都不给 host 看——§5 表里
`unified_mha_run_die_batch_emb` 一发吃掉整格运算（核内 `partial_softmax<B=128>`），
host 侧只见 `q^T / k / v^T+1` 和输出 `O`。**α 存在的意义是“解释”，不是“驻留”。**

![同一个 patch 的 256 个权重：真实权重 + 真实帧的 α 路由图](images/ch14/siglip_alpha_routing.svg)

*图 14-5：自绘——数据由 `tools/cap_ch14_siglip_alpha.py` 从真实权重与真实 demo 帧抓出
（`/tmp/siglip_alpha.npz`），再由 `tools/mk_fig_ch14_alpha_routing.py` 画图；与图 14-4
正好配成一对：**14-4 是入口（与内容无关的线性变换），14-5 是核心（与内容纠缠的分配）**。
物体名是按格子坐标目视标注的读图辅助，不是模型输出。*

**Q&A · “注意”到底在哪（课堂问答存档）**

> **Q1 入口只是线性变换，那“看到整张图”这件事发生在哪一步？**
> 发生在 α 那一行。conv 出口处 256 个 token 各带各的 1152 维、彼此零交换；第一次发生
> “块与块之间按内容交换信息”的地方就是 `o = Σα·v`。所以“SigLIP 把图看成整体”不是
> 那发 conv 的功劳，是 27 轮 α 路由的功劳。

> **Q2 16 个头、27 层各买到什么？**
> 头 = 并行的关系通道：同一层 16 笔互不相干的预算，实测有的取全图平均（A）、有的只看
> 邻格（B）、有的绑背景锚点（C）、有的同时盯四个物体（D）。
> 层 = 图上的消息传递轮数：上一层的 o 变成下一层的 hidden，下一层的 q/k 由它现算
> ⇒ 路由规则逐层被改写。头是**宽度**，层是**深度**，两者不可互换。

> **Q3 学的是规则还是实例？**
> 都是，但分层：可学习的 `W_Q/W_K` 是**规则**（怎么打分），跨帧跨图不变；α 是**实例**
> （这一帧这一块该看谁），每次前向现算。图 14-5 的 B/C 两栏就是证据——同一层同一头，
> 只因为 query 内容不同，路由就完全不同（B）或完全一致（C）。

> **Q4 想自己复现？**
> `python3 tools/cap_ch14_siglip_alpha.py`（CPU 十几秒，产 `/tmp/siglip_alpha.npz`）→
> `python3 tools/mk_fig_ch14_alpha_routing.py`。换一张帧、换一个 query（脚本里 `QIDX`）
> 就能看到另一套路由；换 `attn_implementation` 为 sdpa/flash 则抓不到 α（返回 None），
> 这件事本身就印证了上一段。

## 6. 另一条路：整层一次调用 `siglip_*_ffi_fused`（已导出，主链未启用）

`groot_ops@v0.2 ops/torch_ops/fused_mha_out/fused_mha_out_ffi.cc` 里备着两档融合：

- **`siglip_attn_ffi_fused`**（`:319` 起）：python 侧一次调用吃掉
  `qT.reshape[H,D,N,L]` 的转场布局 + attention（09 章"布局货币"的极致形态——
  不再 transpose，直接让下家按上家的布局读）。约束：fp16-only（`:344`）、单 die（`:415`）。
- **`SiglipLayerFfiFusedForward` / `siglip_layer_ffi_fused`**（`:472` 起）：
  **LN1(可带残差 `ln1_has_res`) → QKV(wt_q/k/v+biases, eye) → MHA → out_proj → LN2 →
  MLP(up/down) 一次调用**。硬约束：fp16-only（`:489`）、`N==out_n`（`:490`）、
  `HEADS*D==N`、single-die only（`:519`）。SigLIP 恰好 1152=16×72 全对得上——
  约束就是照它设计的。

**现状**：N1.5 主链热轮 trace 里 `siglip_layer/attn_ffi_fused` 出现 **0 次**——
融合核已随 `.so` 导出，但接入层（`npu/siglip.py`）仍停在 per-op。收益预期见下条 n1.6 板测。

**n1.6 线的演进**（⚠ 出处：n1.6 skill 记录与板测，**非本 checkout 代码**，勿混入 v0.2 事实）：
多相机把注意力变成 cross-image 整段，与 within-image 融合假设不符（cos 掉到 **0.71**），
于是把整层融合**拆回** `siglip_layer_ffi_proj + pad + mha` 三档；3-cam 视觉塔
fused **0.082 s** vs per-op **0.133 s**（cos 0.999984）；另踩实 **o_proj 零 bias 须走 fp32**。
教学点：融合粒度不是越粗越好——**布局假设与数据形状（几路相机、注意力跨不跨图）决定粒度**，
这正是 09 章 §4.3"融合粒度阶梯"在视觉塔上的重演。

## 7. 开关、探针与 golden 对账

| 手段 | 位置/形态 | 用法 |
| --- | --- | --- |
| `GROOT_NPU_QV_TRANSPOSE` | `npu/siglip.py`（pyc 字符串实证） | 切 `unified_mha_forward` ↔ `unified_mha_forward_qv_T` 两种 qv 布局；布局类问题的第一个开关 |
| `NPU_DUMP_SIGLIP` | 同上；落 `/tmp/npu_%s.npy`、`/tmp/npu_vl_embs.npy` | 默认关；开后逐 op 落中间激活，与 GPU golden 逐层对 cos |
| golden 门禁 | `Isaac-GR00T/deployment_scripts/npu/npu_golden_gate.sh` | 换核/换布局/升 transformers 后的一键对账 |
| pyc 考古 | §4.1 的 marshal 片段 | 源码缺失时重建接入清单（只读，无副作用） |

## 8. 本章小结

一条链背下来：`get_action → collate(eagle_前缀) → forward_eagle → extract_feature(唯一调用点)
→ SiglipVisionModel(27 层, transformers 实现) → mlp1 → 按 151669 原位回写 → LM 12 层 → DiT 复用不再回塔`。

三句话带走：

1. **实现与胶水分离**：SigLIP 住在 transformers 里，NPU 接入本质是 patch transformers 的类符号，
   所以版本契约锁 transformers==4.51.3，且必须早于 `AutoModel.from_config`。
2. **当前主链是 per-op**：每层约 7~8 发核（LN、4×gemm_bias、unified_mha、mlp_gelu、残差），
   锚点计数 `conv2d×1 / unified_mha×27`；整层融合 `siglip_layer_ffi_fused` 已导出未启用。
3. **融合粒度跟着布局与数据形状走**：n1.6 因 cross-image 注意力把整层融合拆回三档，
   反而更快更准（0.082 s vs 0.133 s，cos 0.999984）。

## 疑问与批注

（待补充）

## 自己动手

1. **数一遍**：用 §5 的口径自己解析 `trace_current_clean/n15_sd3_chrome_trace.json`
   （json 一次 `json.load`，按 `ph=='X'` 的 `name` 计数），复现
   `conv2d_patch_embed_run_die ×1 / unified_mha_run_die_batch_emb ×27`；
   再回答：为什么 `residual_add_run_die` 只有 25 而不是 54？（提示：两个 kernel 名。）
2. **考古一遍**：对 `gr00t/model/__pycache__/npu_patch.cpython-311.pyc` 跑 §4.1 的 walk，
   找出 `install` 与 `install_backbone_whole` 各自 patch 的符号，并解释为什么后者必须
   在 `AutoModel.from_config`（`eagle_backbone.py:50`）之前发生。
3. **推演一遍**：假设要把 `siglip_layer_ffi_fused` 开上 N1.5 主链，动手前检查清单至少
   应包含哪 5 项？（参考：§6 的四条硬约束 + §7 的 golden 门禁 + 09 章静默失效判据；
   答对 4 条以上再动手。）
