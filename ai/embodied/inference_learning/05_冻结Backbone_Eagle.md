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

（预留：记录问题。）
