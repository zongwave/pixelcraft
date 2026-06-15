# Mixture of Experts (MoE) 原理与 HPU 融合算子实现

> **日期**: 2026-06-15
> **目录**: `ai/llm/inference_opt/`
> **参考代码**: `ai/code/fused_moe.cc`, `ai/code/fused_gate_moe.cc`, `ai/code/test_fused_mixture_of_experts.py`, `ai/code/test_fused_gate_moe.py`

---

## 目录

- [Mixture of Experts (MoE) 原理与 HPU 融合算子实现](#mixture-of-experts-moe-原理与-hpu-融合算子实现)
  - [目录](#目录)
  - [1. MoE 概述](#1-moe-概述)
    - [1.1 为什么需要 MoE](#11-为什么需要-moe)
    - [1.2 核心概念](#12-核心概念)
    - [1.3 典型架构](#13-典型架构)
  - [2. MoE 计算流程详解](#2-moe-计算流程详解)
    - [2.1 整体流程](#21-整体流程)
    - [2.2 步骤一：Router（门控网络）](#22-步骤一router门控网络)
    - [2.3 步骤二：Expert 分配](#23-步骤二expert-分配)
    - [2.4 步骤三：Expert 计算（SwiGLU MLP）](#24-步骤三expert-计算swiglu-mlp)
    - [2.5 步骤四：加权聚合](#25-步骤四加权聚合)
  - [3. 参考实现（NumPy）](#3-参考实现numpy)
    - [3.1 Expert MLP](#31-expert-mlp)
    - [3.2 Sparse MoE](#32-sparse-moe)
  - [4. HPU 融合算子：fused\_moe](#4-hpu-融合算子fused_moe)
    - [4.1 设计目标](#41-设计目标)
    - [4.2 输入输出](#42-输入输出)
    - [4.3 核心参数](#43-核心参数)
    - [4.4 精度变体](#44-精度变体)
    - [4.5 实现架构](#45-实现架构)
  - [5. HPU 融合算子：fused\_gate\_moe](#5-hpu-融合算子fused_gate_moe)
    - [5.1 设计目标](#51-设计目标)
    - [5.2 输入输出](#52-输入输出)
    - [5.3 融合计算图](#53-融合计算图)
    - [5.4 精度变体](#54-精度变体)
    - [5.5 实现架构](#55-实现架构)
  - [6. 量化方案](#6-量化方案)
    - [6.1 FP8 量化（张量级/通道级）](#61-fp8-量化张量级通道级)
    - [6.2 Blockwise FP8 量化](#62-blockwise-fp8-量化)
    - [6.3 动态量化 vs 静态量化](#63-动态量化-vs-静态量化)
  - [7. 分布式并行](#7-分布式并行)
    - [7.1 Expert Parallelism (EP)](#71-expert-parallelism-ep)
    - [7.2 Tensor Parallelism (TP)](#72-tensor-parallelism-tp)
    - [7.3 EP + TP 组合](#73-ep--tp-组合)
  - [8. 测试框架](#8-测试框架)
    - [8.1 测试策略](#81-测试策略)
    - [8.2 验证指标](#82-验证指标)
    - [8.3 参数组合](#83-参数组合)
  - [9. 总结](#9-总结)
    - [9.1 关键设计模式](#91-关键设计模式)
    - [9.2 与 Dense Model 的对比](#92-与-dense-model-的对比)
    - [9.3 文件结构](#93-文件结构)

---

## 1. MoE 概述

### 1.1 为什么需要 MoE

**核心矛盾**：模型容量 vs 计算成本

- 增大模型（更多参数）→ 提升能力 → 但推理成本线性增长
- MoE 思路：**用更多参数，但每次只激活一部分**

**类比**：
- 传统 Dense 模型：一个全才专家处理所有问题
- MoE 模型：一群专才专家，每个问题只找最相关的几位

**效果**：
- 总参数量 = N × expert_params（N 个 expert）
- 每次计算量 ≈ K × expert_params（只激活 top-K 个 expert）
- 典型配置：N=8, K=2 → 4x 参数量，2x 计算量

### 1.2 核心概念

| 概念 | 说明 |
|------|------|
| **Expert** | 独立的 FFN/MLP 子网络，每个 expert 有自己的权重 |
| **Router (Gate)** | 门控网络，决定每个 token 分配给哪些 expert |
| **Top-K Routing** | 每个 token 只激活得分最高的 K 个 expert |
| **Sparse MoE** | 每次只激活部分 expert，保持计算稀疏性 |
| **Load Balancing** | 确保 token 均匀分配到各 expert，避免某些 expert 过载 |
| **Expert Parallelism** | 将不同 expert 分布到不同设备上 |

### 1.3 典型架构

```
输入: [num_tokens, hidden_dim]
         │
         ▼
    ┌─────────────┐
    │   Router    │  ← 门控网络（小规模，全精度）
    │  (Gate)     │
    └──────┬──────┘
           │ routing_weights: [num_tokens, top_k]
           │ routing_table:   [num_tokens, top_k] (expert 索引)
           ▼
    ┌─────────────────────────────────────┐
    │  Expert 0  │  Expert 1  │ ... │  Expert N-1  │
    │  (FFN)     │  (FFN)     │     │  (FFN)        │
    └────────────┴────────────┴─────┴───────────────┘
           │           │                 │
           └───────────┴─────────────────┘
                       │ 加权聚合
                       ▼
         输出: [num_tokens, hidden_dim]
```

---

## 2. MoE 计算流程详解

### 2.1 整体流程

```
输入 hidden_states: [num_tokens, hidden_dim]
                      │
  ────────────────────┼────────────────────
  Step 1: Router      │
                      ▼
              gate_out = hidden_states @ gate_weight          [num_tokens, num_experts]
              weights = softmax(gate_out)                     [num_tokens, num_experts]
              (可选) scores = weights + gate_correction_bias  [num_tokens, num_experts]
              routing_weights, selected_experts = topk(scores, K)  [num_tokens, K], [num_tokens, K]
              (可选) routing_weights /= sum(routing_weights)  # 归一化
                      │
  ────────────────────┼────────────────────
  Step 2: Dispatch    │
                      ▼
              对每个 expert i:
                tokens_i = hidden_states[分配到 expert i 的 token 索引]
                weights_i = routing_weights[对应位置]
                      │
  ────────────────────┼────────────────────
  Step 3: Expert FFN  │
                      ▼
              对每个 expert i:
                h1 = activation(tokens_i @ w1_i)     # up projection
                h2 = tokens_i @ w2_i                 # gate projection
                intermediate = h1 * h2               # SwiGLU 门控
                output_i = intermediate @ w3_i       # down projection
                output_i *= weights_i                # 加权
                      │
  ────────────────────┼────────────────────
  Step 4: Combine     │
                      ▼
              final_hidden_states = sum(output_i)    # 累加所有 expert 输出
              输出: [num_tokens, hidden_dim]
```

### 2.2 步骤一：Router（门控网络）

Router 是一个小型的线性层，负责为每个 token 选择最合适的 expert：

```python
# 1. 线性投影
gate_out = hidden_states @ gate_weight  # [num_tokens, hidden_dim] @ [hidden_dim, num_experts]
                                        # → [num_tokens, num_experts]

# 2. Softmax 归一化
weights = softmax(gate_out, axis=-1)    # [num_tokens, num_experts]

# 3. (可选) 门控修正偏置
if gate_correction_bias is not None:
    scores = weights + gate_correction_bias  # 用于负载均衡
else:
    scores = weights

# 4. Top-K 选择
routing_weights, selected_experts = topk(scores, K, axis=-1)
# routing_weights: [num_tokens, K]  — 每个 token 选中的 expert 权重
# selected_experts: [num_tokens, K] — 每个 token 选中的 expert 索引

# 5. (可选) 归一化 top-k 概率
if norm_topk_prob:
    routing_weights /= sum(routing_weights, axis=-1, keepdim=True)
```

**关键设计**：
- Gate 权重通常用 **fp32** 精度，因为路由决策对精度敏感
- `gate_correction_bias` 是可选的路由修正偏置，用于 **Expert Load Balancing**
- `norm_topk_prob` 控制是否对 top-k 权重做归一化

### 2.3 步骤二：Expert 分配

根据 routing_table 将 token 分发到对应的 expert：

```python
# 构建 expert mask
expert_mask = eye(num_experts)[routing_table]  # [num_tokens, K, num_experts]
expert_mask = expert_mask.transpose(2, 1, 0)   # [num_experts, K, num_tokens]

# 对每个 expert，找出分配到它的 token
for expert_idx in range(num_experts):
    idx, top_x = where(expert_mask[expert_idx])  # top_x: token 索引
    if idx.size == 0:
        continue
    current_state = hidden_states[top_x]          # 取出 token
    current_weights = routing_weights[top_x, idx] # 取出对应权重
    # ... 执行 expert 计算 ...
```

### 2.4 步骤三：Expert 计算（SwiGLU MLP）

每个 expert 是一个独立的 **SwiGLU FFN**（与 LLaMA 风格一致）：

```python
# 非融合权重（3 个独立矩阵）
hidden_states_w1 = activation(hidden_states @ w1)  # up: [*, hidden_dim] → [*, ffn_dim]
hidden_states_w2 = hidden_states @ w2               # gate: [*, hidden_dim] → [*, ffn_dim]
intermediate = hidden_states_w1 * hidden_states_w2  # SwiGLU 门控: element-wise multiply
output = intermediate @ w3                           # down: [*, ffn_dim] → [*, hidden_dim]

# 融合权重（w1 和 w2 合并为 w12）
w12 = concat(w1, w2, axis=1)  # [hidden_dim, 2 * ffn_dim]
hidden_states_w12 = hidden_states @ w12  # [*, 2 * ffn_dim]
hidden_states_w1, hidden_states_w2 = split(hidden_states_w12, 2, axis=-1)
intermediate = activation(hidden_states_w1) * hidden_states_w2
output = intermediate @ w3
```

**权重形状**（非 permuted）：
| 权重 | 形状 | 说明 |
|------|------|------|
| w1 (up) | `[hidden_dim, ffn_dim]` | up projection |
| w2 (gate) | `[hidden_dim, ffn_dim]` | gate projection |
| w3 (down) | `[ffn_dim, hidden_dim]` | down projection |
| w12 (fused) | `[hidden_dim, 2 * ffn_dim]` | w1 和 w2 的拼接 |

**权重形状**（permuted）：
| 权重 | 形状 | 说明 |
|------|------|------|
| w1 (up) | `[ffn_dim, hidden_dim]` | 转置 |
| w2 (gate) | `[ffn_dim, hidden_dim]` | 转置 |
| w3 (down) | `[hidden_dim, ffn_dim]` | 转置 |
| w12 (fused) | `[2 * ffn_dim, hidden_dim]` | 转置后拼接 |

### 2.5 步骤四：加权聚合

```python
final_hidden_states = zeros_like(hidden_states)  # [num_tokens, hidden_dim]

for expert_idx in range(num_experts):
    # ... 获取 expert 输出 current_hidden_states ...
    current_hidden_states *= current_weights      # 加权
    for i, pos in enumerate(top_x):
        final_hidden_states[pos] += current_hidden_states[i]  # 累加
```

---

## 3. 参考实现（NumPy）

### 3.1 Expert MLP

```python
class MixtralBlockSparseMLP_Numpy:
    def __init__(self, w1, w2, w3, activation="silu"):
        self.w1 = w1  # [hidden_dim, ffn_dim]
        self.w2 = w2  # [hidden_dim, ffn_dim]
        self.w3 = w3  # [ffn_dim, hidden_dim]
        self.activation_fn = self.__get_activation_fn(activation)

    def forward(self, hidden_states, compute_amax=False):
        # SwiGLU: SiLU(x @ w1) * (x @ w2) → @ w3
        hidden_states_w1 = self.activation_fn(np.matmul(hidden_states, self.w1))
        hidden_states_w2 = np.matmul(hidden_states, self.w2)
        intermediate = hidden_states_w1 * hidden_states_w2
        output = np.matmul(intermediate, self.w3)
        amax = np.max(np.abs(intermediate)) if compute_amax else None
        return output, amax
```

### 3.2 Sparse MoE

```python
class MixtralSparseMoeRef_Numpy:
    def __init__(self, hidden_dim, num_experts, expert_weights, activation="silu"):
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        w1, w2, w3 = expert_weights
        self.experts = [
            MixtralBlockSparseMLP_Numpy(w1[i], w2[i], w3[i], activation)
            for i in range(num_experts)
        ]

    def forward(self, hidden_states, router_weights, routing_table):
        final_hidden_states = np.zeros_like(hidden_states)
        routing_table = routing_table.astype(np.int64)

        # 构建 expert mask: [num_experts, K, num_tokens]
        expert_mask = np.eye(self.num_experts)[routing_table].transpose(2, 1, 0)

        for expert_idx in range(self.num_experts):
            idx, top_x = np.where(expert_mask[expert_idx])
            if idx.size == 0:
                continue
            # 取出分配到该 expert 的 token
            current_state = hidden_states[top_x].reshape(-1, self.hidden_dim)
            # Expert 前向计算
            current_hidden_states, _ = self.experts[expert_idx].forward(
                current_state, compute_amax=True
            )
            # 加权
            current_hidden_states *= router_weights[top_x, idx, None]
            # 累加
            for i, pos in enumerate(top_x):
                final_hidden_states[pos] += current_hidden_states[i]

        return final_hidden_states.reshape(hidden_states.shape)
```

---

## 4. HPU 融合算子：fused_moe

### 4.1 设计目标

将 MoE 的 Expert 计算部分（步骤三 + 步骤四）融合为一个 HPU Recipe，**不包含 Router**。

**动机**：
- 减少 kernel launch 开销（每个 expert 的 GEMM + activation 原本需要多次 launch）
- 利用 HPU 的 SynapseAI 编译器进行全局优化
- 支持多种精度（BF16 / FP8 / Blockwise FP8）

### 4.2 输入输出

```
输入:
  hidden_states:  [num_tokens, hidden_dim]          — 输入激活
  routing_table:  [num_tokens, top_k]                — 每个 token 选中的 expert 索引 (int32)
  router_weights: [num_tokens, top_k]                — 每个 token 对应的 routing 权重
  gate_up_weights: [num_experts, hidden_dim, ffn_dim] 或 [num_experts, hidden_dim, 2*ffn_dim]
                                                     — 每个 expert 的 w1/w2 权重
  down_weights:   [num_experts, ffn_dim, hidden_dim] — 每个 expert 的 w3 权重
  (可选) scales:  量化缩放因子

输出:
  final_hidden_states: [num_tokens, hidden_dim]      — 输出激活
  (可选) amax_per_expert: [num_experts]              — 中间激活最大值（用于量化校准）
```

### 4.3 核心参数

通过 `ns_MoeKernel::ParamsV4` 结构体传递：

```cpp
struct FusedMoEConfig {
    bool permuted_weights;       // 权重是否转置
    bool fused_gemm;             // w1/w2 是否融合为 w12
    bool measurement_mode;       // 是否计算 amax
    char activation_mode[32];    // 激活函数: selu/gelu/relu/silu
    int32_t num_experts;         // expert 总数
    int32_t experts_min;         // 当前设备负责的 expert 起始索引 (EP)
    int32_t experts_max;         // 当前设备负责的 expert 结束索引 (EP)
    bool dynamic_scale;          // FP8 动态量化
    int32_t block_size;          // Blockwise FP8 的分块大小
    int32_t chunk_size;          // 分块处理大小
};
```

**flags 位域**：
```cpp
moe_params->flags = 
    (permuted_weights ? MOE_FLAGS_PERMUTED_WEIGHTS : 0) |
    (fused_gemm ? MOE_FLAGS_FUSED_GEMM : 0) |
    (measurement_mode ? MOE_FLAGS_CALC_AMAX : 0) |
    (dynamic_scale ? MOE_FLAGS_DYNAMIC_SCALE : 0) |
    (block_size > 0 ? MOE_FLAGS_BLOCKWISE_WEIGHT_QUANTIZATION : 0);
```

### 4.4 精度变体

| 算子名 | 输入精度 | 权重精度 | 计算精度 | 说明 |
|--------|----------|----------|----------|------|
| `mixture_of_experts` | BF16 | BF16 | BF16 | 标准精度 |
| `mixture_of_experts_fp8` | FP8 | FP8 | FP8 | 全 FP8，含 hidden_states 量化 |
| `mixture_of_experts_blockwise_fp8` | BF16 | FP8 (blockwise) | FP8 | 权重分块量化，输入 BF16 |

### 4.5 实现架构

```
FusedMixtureOfExperts (继承 HpuFusedOperator)
│
├── AddNodeMoeForward<T>(inputs, outputs, params)
│   └── 调用 HPU SynapseAPI 添加 MoE 计算节点
│
├── AddNode<T>(ct, config)
│   ├── 计算输入张量数量（含 scales）
│   ├── 创建输入/输出 synTensor
│   └── 调用 AddNodeMoeForward
│
└── FusedMoEKernel (模板函数)
    ├── ConvertTensors: 统一管理张量转换
    ├── OpCacheOperator: 编译缓存
    ├── Compile → Recipe: 编译为 HPU Recipe
    └── RecipeRunner: 执行 Recipe
```

**编译缓存机制**：
```cpp
OpCacheOperator op_info;
op_info.prepareOpInfo<T, FusedMoEConfig>("fused_moe_", inputs_dims, &config);
auto recipe = op_info.GetRecipe();

if (recipe == nullptr) {
    // 首次编译
    FusedMixtureOfExperts op(op_info.datatype_);
    op.AddNode<T>(&ct, config);
    op.Compile();
    op_info.setOp(op);
    recipe = op_info.GetRecipe();
}
// 缓存命中，直接执行
RecipeRunner runner(recipe);
runner.Run(stream, tensors);
```

---

## 5. HPU 融合算子：fused_gate_moe

### 5.1 设计目标

将 **Router + MoE** 整个端到端流程融合为单个 HPU Recipe，从 hidden_states 直接到最终输出。

**动机**：
- 消除 Router 和 MoE 之间的中间数据搬运
- 将 Gate 的 fp32 计算和 MoE 的 bf16/fp8 计算统一调度
- 支持 Gate Correction Bias 等高级路由策略

### 5.2 输入输出

```
输入:
  hidden_states:          [num_tokens, hidden_dim]     — 输入激活 (bf16)
  gate_weights:           [hidden_dim, num_experts]    — Router 权重 (fp32)
  gate_correction_bias:   [1, num_experts]             — 路由修正偏置 (fp32, 可选)
  gate_up_weights:        [num_experts, hidden_dim, 2*ffn_dim]  — w1/w2 融合权重
  down_weights:           [num_experts, ffn_dim, hidden_dim]    — w3 权重
  (可选) scales:          量化缩放因子

输出:
  final_hidden_states:    [num_tokens, hidden_dim]     — 输出激活
```

### 5.3 融合计算图

```
FusedGateMoe.AddNode() 构建的完整计算图:

hidden_states (bf16)
    │
    ├── cast to fp32 ──────────────────────────────────────┐
    │                                                      │
    ▼                                                      │
gate_weights (fp32) ──► GEMM ──► gate_out (fp32)          │
                                │                          │
                                ▼                          │
                           softmax ──► weights (fp32)      │
                                │                          │
                    ┌───────────┼───────────┐              │
                    │           │           │              │
              无 bias     有 bias     有 bias              │
                    │           │           │              │
                    ▼           ▼           │              │
                 topk      add bias         │              │
                    │           │           │              │
                    ▼           ▼           │              │
              routing_weights  topk         │              │
              selected_experts  │           │              │
                    │           ▼           │              │
                    │     index_sample      │              │
                    │           │           │              │
                    └───────────┼───────────┘              │
                                │                          │
                    ┌───────────┼───────────┐              │
                    │           │           │              │
              不归一化     归一化              │              │
                    │     reduce_sum          │              │
                    │     divide              │              │
                    └───────────┼───────────┘              │
                                │                          │
                           cast to bf16                    │
                                │                          │
                    routing_weights (bf16)                 │
                    selected_experts (int32)               │
                                │                          │
                    ┌───────────┼───────────┐              │
                    │  FP8 模式  │  BF16 模式 │              │
                    │           │           │              │
                    ▼           ▼           │              │
              动态/静态量化   直接使用        │              │
                    │           │           │              │
                    └───────────┼───────────┘              │
                                │                          │
                                ▼                          │
                    mixture_of_experts ◄───────────────────┘
                    (hidden_states, selected_experts,
                     routing_weights, gate_up_weights,
                     down_weights, scales...)
                                │
                                ▼
                    final_hidden_states (bf16)
```

### 5.4 精度变体

| 算子名 | 输入精度 | Gate 精度 | MoE 精度 | 说明 |
|--------|----------|-----------|----------|------|
| `fused_gate_moe` | BF16 | fp32 → bf16 | BF16 | 标准精度 |
| `fused_gate_moe_fp8` | BF16 → FP8 | fp32 → bf16 | FP8 | 输入量化后 FP8 计算 |
| `fused_gate_moe_blockwise_fp8` | BF16 | fp32 → bf16 | FP8 (blockwise) | 权重分块量化 |

### 5.5 实现架构

```cpp
class FusedGateMoe : public HpuFusedOperator {
    // 构建完整的 Gate + MoE 计算图
    template <typename T, typename TMoe>
    void AddNode(ConvertTensors& ct, FusedGateMoeParams params) {
        // 1. MoE Gate
        //    hidden_states (bf16) → cast to fp32
        //    → GEMM(gate_weights) → softmax → topk → routing_weights
        //    (可选) gate_correction_bias → add → topk → index_sample
        //    (可选) norm_topk_prob → reduce_sum → divide
        //    → cast to bf16

        // 2. (FP8 模式) hidden_states 量化
        //    动态量化: abs → reduce_max → div → quantize
        //    静态量化: cast to fp8 with scale

        // 3. mixture_of_experts
        //    将 routing_weights, selected_experts, weights 传入 MoE
    }
};
```

**输入张量索引管理**：
```cpp
enum TENSOR_IDS_IN {
    HIDDEN_STATES = 0,    // 输入激活
    GATE_WEIGHT = 1,      // Router 权重
    BIAS_OR_WEIGHTS,      // 2: gate_correction_bias 或权重起始位置
    EOS_TOKEN             // 结束标记
};
```

**权重融合检测**：
```cpp
// 通过形状判断是否 fused_weights
params.fused_gemm = (gate_up_weights_dims[2] == down_weights_dims[1] * 2);
// 如果 gate_up_weights 的最后一维是 down_weights 最后一维的 2 倍，说明是融合权重
```

---

## 6. 量化方案

### 6.1 FP8 量化（张量级/通道级）

**张量级量化**（Tensor-wise）：
```python
def tensorwise_quant_to_fp8(tensor):
    x_abs = paddle.abs(tensor).astype(paddle.float32)
    x_amax = paddle.amax(x_abs)                    # 全局最大值
    x_amax = paddle.clip(x_amax, min=1e-4)
    scale = x_amax / 240.0                          # FP8 最大值 = 240
    x_scaled = (tensor / scale).astype(paddle.float8_e4m3fn)
    return x_scaled, scale
```

**通道级量化**（Channel-wise）：
```python
def channelwise_quant_to_fp8(tensor):
    x_abs = paddle.abs(tensor).astype(paddle.float32)
    x_amax = paddle.amax(x_abs, axis=0)             # 每列独立 scale
    x_amax = paddle.clip(x_amax, min=1e-4)
    scale = x_amax / 240.0
    x_scaled = (tensor / scale).astype(paddle.float8_e4m3fn)
    return x_scaled, scale
```

### 6.2 Blockwise FP8 量化

按 block_size 分块，每块独立量化：

```python
def blockwise_quant_to_fp8(tensorlist, block_size):
    for x in tensorlist:
        # 1. padding 到 block_size 的整数倍
        x_padded = pad(x, block_size)
        # 2. reshape 为 [num_blocks_row, block_size, num_blocks_col, block_size]
        x_view = x_padded.reshape(-1, block_size, -1, block_size)
        # 3. 每块独立计算 scale
        x_amax = amax(abs(x_view), axis=(1, 3))     # [num_blocks_row, num_blocks_col]
        x_scaled = (x_view * (240.0 / x_amax)).astype(fp8)
        # 4. 恢复原始形状
        q_tensor = x_scaled.view_as(x_padded)[:m, :n]
```

### 6.3 动态量化 vs 静态量化

| 特性 | 动态量化 | 静态量化 |
|------|----------|----------|
| Scale 计算 | 运行时计算 | 预计算，固定值 |
| 精度 | 更准确 | 略差 |
| 计算开销 | 额外 abs/max/div | 无额外开销 |
| 适用场景 | 激活值分布变化大 | 激活值分布稳定 |
| 代码路径 | `hidden_states_static_quant=false` | `hidden_states_static_quant=true` |

**动态量化流程**（HPU 算子链）：
```
hidden_states → abs → reduce_max → div(scale = max/240) → quantize
```

**静态量化流程**：
```
hidden_states → cast_to_fp8(scale) → div(reciprocal = 1/scale)
```

---

## 7. 分布式并行

### 7.1 Expert Parallelism (EP)

将 expert 分布到不同设备上，每个设备只负责一部分 expert：

```python
experts_per_rank = num_experts // ep_size
experts_min = ep_rank * experts_per_rank
experts_max = (ep_rank + 1) * experts_per_rank - 1
if ep_rank == ep_size - 1:
    experts_max = num_experts - 1
```

**通信**：EP All-reduce（对 final_hidden_states 做 SUM reduce）

### 7.2 Tensor Parallelism (TP)

将权重矩阵按列/行切分到不同设备：

```python
# 对 up/gate 权重做列切分
up_weights = w[:, tp_rank * (ffn_dim // tp_size) : (tp_rank + 1) * (ffn_dim // tp_size)]
gate_weights = w[:, tp_rank * (ffn_dim // tp_size) : (tp_rank + 1) * (ffn_dim // tp_size)]

# 对 down 权重做行切分
down_weights = w[tp_rank * (ffn_dim // tp_size) : (tp_rank + 1) * (ffn_dim // tp_size), :]
```

**通信**：TP All-reduce（对 final_hidden_states 做 SUM reduce）

### 7.3 EP + TP 组合

```
world_size = ep_size * tp_size
ep_rank = global_rank // tp_size
tp_rank = global_rank % tp_size

TP group: [ep_rank * tp_size + i for i in range(tp_size)]
EP group: [i * tp_size + tp_rank for i in range(ep_size)]
```

**通信顺序**：先 TP All-reduce，再 EP All-reduce

---

## 8. 测试框架

### 8.1 测试策略

```
1. 生成随机测试数据（支持多种 dtype、权重布局）
2. 用 NumPy/Paddle 参考实现计算预期结果
3. 调用 HPU 融合算子
4. 对比融合算子输出与参考输出
```

### 8.2 验证指标

**余弦相似度**：
```python
cos_sim = dot(vec1, vec2) / (norm1 * norm2)
assert cos_sim >= required_similarity
```

**欧几里得相似度**（仅 fused_gate_moe 测试）：
```python
mag_sim = min(norm1 / norm2, norm2 / norm1)
assert mag_sim >= required_similarity
```

**阈值**：
| 精度 | 要求相似度 |
|------|-----------|
| BF16 | > 0.99 |
| FP8 | > 0.99 |
| Blockwise FP8 | > 0.98 |

### 8.3 参数组合

```python
DTYPES = ["bfloat16", "fp8", "blockwise_fp8"]
NUM_TOKENS = [32]
HIDDEN_DIMS = [4096]
FFN_DIMS = [2560]
TOP_K = [2]
NUM_EXPERTS = [8]
SLICE_MAX_EXPERT = [8]
FUSED_WEIGHTS = [True, False]
ACTIVATIONS = ["gelu", "relu", "silu"]
PERMUTED_WEIGHTS = [True, False]
EP_SIZE = [1]
TP_SIZE = [1]
COMPUTE_AMAX = [True, False]       # 仅 BF16
DYNAMIC_SCALE = [True, False]      # 仅 FP8
```

---

## 9. 总结

### 9.1 关键设计模式

| 模式 | 说明 | 代码位置 |
|------|------|----------|
| **OpCacheOperator** | 编译缓存，避免重复编译 | `fused_moe.cc:186-196` |
| **ConvertTensors** | 统一管理张量转换和设备地址 | `fused_moe.cc:149-168` |
| **RecipeRunner** | 执行编译好的 HPU Recipe | `fused_moe.cc:199-200` |
| **Expert Slicing** | 通过 experts_min/max 和 chunk_size 分片 | `FusedMoE.forward()` |
| **Fused Weights** | w1/w2 融合为 w12，减少访存 | `generate_moe_params()` |
| **Permuted Weights** | 权重转置优化内存布局 | `generate_moe_params()` |

### 9.2 与 Dense Model 的对比

| 特性 | Dense Model | MoE Model |
|------|-------------|-----------|
| 参数量 | 1x | N × expert_params + gate_params |
| 每次计算量 | 1x | ≈ K × expert_params + gate_params |
| 典型配置 | — | N=8, K=2 → 4x 参数, 2x 计算 |
| 路由开销 | 无 | 小（gate 是轻量线性层） |
| 分布式友好度 | TP | EP + TP |
| 量化复杂度 | 低 | 高（每个 expert 独立量化） |

### 9.3 文件结构

```
ai/code/
├── fused_moe.cc                    # 基础 MoE 融合算子（C++ HPU）
├── fused_gate_moe.cc               # 融合 Gate + MoE 算子（C++ HPU）
├── test_fused_mixture_of_experts.py # MoE 测试（含 NumPy 参考实现）
└── test_fused_gate_moe.py          # Gate MoE 测试（含 Paddle 参考实现）
```

---

> **最后更新**: 2026-06-15
> **文件位置**: `ai/llm/inference_opt/mixture_of_experts.md`