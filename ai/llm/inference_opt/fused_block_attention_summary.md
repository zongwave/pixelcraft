# Fused Block Attention 实现原理总结

> 本文档基于 `ai/code/fused_block_attention.cc` 代码分析，总结 Intel Habana HPU (Gaudi) 上融合 Block Attention 算子的设计与实现原理。

---

## 目录

- [Fused Block Attention 实现原理总结](#fused-block-attention-实现原理总结)
  - [目录](#目录)
  - [1. 概述](#1-概述)
  - [2. 核心数据结构](#2-核心数据结构)
    - [`FusedBlockAttentionParams`](#fusedblockattentionparams)
  - [3. 类继承体系](#3-类继承体系)
    - [`FusedBlockAttentionBase`](#fusedblockattentionbase)
    - [`FusedMHABlockAttention` vs `FusedGQABlockAttention`](#fusedmhablockattention-vs-fusedgqablockattention)
  - [4. 计算图数据流详解](#4-计算图数据流详解)
    - [4.1 QKV 投影](#41-qkv-投影)
    - [4.2 QKV Split](#42-qkv-split)
    - [4.3 QK RMSNorm（可选）](#43-qk-rmsnorm可选)
    - [4.4 RoPE 位置编码](#44-rope-位置编码)
    - [4.5 KV Cache 写入](#45-kv-cache-写入)
    - [4.6 Block 级 Attention 计算](#46-block-级-attention-计算)
      - [4.6a Q 映射到 Block 空间](#46a-q-映射到-block-空间)
      - [4.6b KV 从 Cache 中索引](#46b-kv-从-cache-中索引)
      - [4.6c QK Score 计算](#46c-qk-score-计算)
      - [4.6d Safe Softmax（跨 Block 聚合）](#46d-safe-softmax跨-block-聚合)
      - [4.6e 加权求和](#46e-加权求和)
      - [4.6f 结果映射回 Batch 空间](#46f-结果映射回-batch-空间)
    - [4.7 输出投影](#47-输出投影)
  - [5. MHA vs GQA 差异](#5-mha-vs-gqa-差异)
  - [6. FP8 支持](#6-fp8-支持)
    - [FP8 数据流](#fp8-数据流)
    - [FP8 应用位置](#fp8-应用位置)
    - [Scale 处理](#scale-处理)
  - [7. 与标准 Attention 的对比](#7-与标准-attention-的对比)
  - [8. 核心设计思想](#8-核心设计思想)
    - [8.1 Block 分页管理](#81-block-分页管理)
    - [8.2 Safe Softmax 跨 Block 聚合](#82-safe-softmax-跨-block-聚合)
    - [8.3 融合编译](#83-融合编译)
    - [8.4 FP8 全链路支持](#84-fp8-全链路支持)
  - [附录：关键代码路径](#附录关键代码路径)
    - [算子注册](#算子注册)
    - [类型分发](#类型分发)
    - [MHA/GQA 选择](#mhagqa-选择)

---

## 1. 概述

`fused_block_attention.cc` 是一个基于 **Intel Habana HPU (Gaudi)** 的融合注意力算子，用于 PaddlePaddle 框架。它将完整的注意力计算流程融合为单个 HPU 编译图（recipe），避免多次 kernel launch 开销。代码支持 **MHA** 和 **GQA** 两种模式，并可选 FP8 精度。

该算子是 PaddlePaddle 在 HPU 上部署 LLaMA 等大模型推理的关键组件，属于"模块级融合"优化策略的一部分——将 Attention 模块中的多个算子（QKV 投影、reshape、matmul、softmax、输出投影等）融合为一个自定义 OP，减少 kernel launch 次数与中间 memory I/O。

---

## 2. 核心数据结构

### `FusedBlockAttentionParams`

配置参数结构体，控制算子的所有行为：

| 字段 | 类型 | 说明 |
|------|------|------|
| `const_params` | `ns_ConstantKernel::Params` | 常量参数（如 scaling factor） |
| `index_select_params` | `ns_GatherKernel::Params` | IndexSelect 操作参数 |
| `reduce_params` | `ns_Reduction::Params` | ReduceMax/ReduceSum 参数 |
| `index_reduce_params` | `ns_IndexReduce::Params` | IndexReduce 参数（用于跨 block 聚合） |
| `rmsnorm_params` | `ns_LayerNormKernel::Params` | RMSNorm 参数 |
| `head_dim` | `int` | 每头维度 |
| `num_head` | `int` | Q 头数 |
| `num_kv_head` | `int` | K/V 头数 |
| `use_neox_style` | `bool` | RoPE 风格（NeoX blockwise vs pairwise） |
| `with_qkv_biases` | `bool` | QKV 是否带 bias |
| `transpose` | `bool` | 权重矩阵是否转置 |
| `use_fp8_embedding` | `bool` | 输入 Embedding 是否 FP8 |
| `use_fp8_kv_cache` | `bool` | KV Cache 是否 FP8 存储 |
| `use_fp8_out_proj` | `bool` | 输出投影是否 FP8 |
| `use_qk_rmsnorm` | `bool` | Q/K 是否额外做 RMSNorm |

---

## 3. 类继承体系

```
HpuFusedOperator
  └── FusedBlockAttentionBase        # 基类：提供混合精度 GEMM 工具方法
        ├── FusedMHABlockAttention   # MHA 模式 (num_head == num_kv_head)
        └── FusedGQABlockAttention   # GQA 模式 (num_head > num_kv_head)
```

### `FusedBlockAttentionBase`

基类提供关键工具方法 `AddNodeMixedPrecisionGemm`，用于在 FP8 和 BF16/FP16 之间透明切换：

```cpp
template <typename T>
void AddNodeMixedPrecisionGemm(bool use_fp8,
                               ConvertTensors& ct,
                               int scale_x_index,
                               int scale_y_index,
                               int reciprocal_scale_x,
                               int reciprocal_scale_y,
                               std::vector<synTensor> inputs,
                               std::vector<synTensor> outputs,
                               synGEMMParams gemm_params,
                               const std::string& suffix);
```

- 当 `use_fp8=true` 时：插入 scale 的 reciprocal 计算 → 调用 `fused_fp8_gemm`
- 当 `use_fp8=false` 时：直接调用 `batchgemm`

### `FusedMHABlockAttention` vs `FusedGQABlockAttention`

两者在以下方面有差异：
- Q 的维度布局
- K/V 的维度布局
- Score 计算的维度
- Softmax 统计量的维度
- 最终映射回 batch 空间的维度处理

---

## 4. 计算图数据流详解

整个算子被编译为一个 HPU recipe，包含以下阶段（以 MHA 为例）：

### 4.1 QKV 投影

```
src [B, L, D] ──Linear/Gemm──→ qkv_out [B, L, (num_head + 2*num_kv_head) * head_dim]
```

- 支持 `transpose` 模式（权重矩阵转置）
- 支持 FP8 混合精度 GEMM（`AddNodeMixedPrecisionGemm`）
- 可选加 bias

**代码位置**：Lines 182-236

### 4.2 QKV Split

```
qkv_out ──Split(axis=1)──→ q_split [B, num_head, head_dim]
                          ├── k_split [B, num_kv_head, head_dim]
                          └── v_split [B, num_kv_head, head_dim]
```

- 沿第 1 维（feature 维）分割为 Q、K、V 三部分
- Q 有 `num_head` 个头，K/V 有 `num_kv_head` 个头（GQA 时 `num_kv_head < num_head`）

**代码位置**：Lines 238-267

### 4.3 QK RMSNorm（可选）

对 Q 和 K 分别做 RMSNorm，使用可学习的 gamma 参数：

```
q_split ──RMSNorm(gamma_q)──→ q_rmsnorm
k_split ──RMSNorm(gamma_k)──→ k_rmsnorm
```

- 输出包含归一化后的 tensor 和 variance tensor
- 通过 `params.use_qk_rmsnorm` 控制是否启用

**代码位置**：Lines 322-369

### 4.4 RoPE 位置编码

```
rotary_embs ──Slice──→ cos_in, sin_in
                      ──Squeeze──→ cos_sq, sin_sq
q_split + sin_sq + cos_sq ──RoPE──→ q_states
k_split + sin_sq + cos_sq ──RoPE──→ k_rope
```

- 从 `rotary_embs` 中 slice 出 cos/sin 分量（第 0 维为 cos，第 1 维为 sin）
- 支持两种模式：
  - **NeoX style (blockwise)**：`ROTARY_POS_EMBEDDING_MODE_BLOCKWISE`
  - **Pairwise**：`ROTARY_POS_EMBEDDING_MODE_PAIRWISE`

**代码位置**：Lines 269-388

### 4.5 KV Cache 写入

```
block_indices + block_offsets ──Concat──→ indices_concat
k_rope ──Scatter(indices_concat)──→ key_cache
v_split ──Scatter(indices_concat)──→ value_cache
```

- 使用 `block_indices` 和 `block_offsets` 拼接为写入索引
- 支持 FP8 转换后写入（`ConvertToFP8` + `Scatter`）
- 使用 `synSectionHandle` 管理 cache 内存的生命周期

**代码位置**：Lines 390-486

### 4.6 Block 级 Attention 计算

这是核心计算部分，实现了 **PagedAttention 风格的 Block-wise Attention**。

#### 4.6a Q 映射到 Block 空间

```
q_states ──Mul(1/√d_k)──→ scaled_q
scaled_q ──Reshape──→ [B, hidden_size]
scaled_q ──Gemm(block_mapping)──→ mapped_q [num_of_block, hidden_size]
mapped_q ──Reshape──→ [num_of_block, num_head, 1, head_dim]
```

- `block_mapping` 是一个 `[num_of_block, B]` 的映射矩阵，将 batch 中的 query 映射到 block 空间
- 每个 block 对应一个独立的注意力计算单元

**代码位置**：Lines 490-545

#### 4.6b KV 从 Cache 中索引

```
key_cache ──IndexSelect(block_list)──→ [num_of_block, block_size, num_kv_head, head_dim]
value_cache ──IndexSelect(block_list)──→ [num_of_block, block_size, num_kv_head, head_dim]
──Transpose(0,2,1,3)──→ [num_of_block, num_kv_head, block_size, head_dim]
```

- `block_list` 指定需要读取哪些 block 的 KV
- Transpose 将维度重排为 `[block, kv_head, seq, head_dim]` 以适配 GEMM

**代码位置**：Lines 549-628

#### 4.6c QK Score 计算

```
mapped_q [num_of_block, num_head, 1, head_dim]
  × K^T [num_of_block, num_kv_head, block_size, head_dim]^T
  = score [num_of_block, num_head, 1, block_size]
score + block_bias (ALiBi 风格偏置)
```

- GQA 模式下，Q 的维度为 `[num_of_block, num_kv_head, ngroups, 1, head_dim]`，K 为 `[num_of_block, num_kv_head, 1, block_size, head_dim]`，通过广播实现 group query 与 key 的匹配

**代码位置**：Lines 630-681

#### 4.6d Safe Softmax（跨 Block 聚合）

实现了 **数值稳定的 block-wise softmax**，这是 PagedAttention 的核心技巧：

```
1. block_max = ReduceMax(score)                    # 每个 block 的局部最大值
2. group_max = IndexReduce(block_groups, block_max) # 同组 block 的全局最大值
3. selected_group_max = IndexSelect(group_max)      # 广播回每个 block
4. block_adjustment = exp(block_max - selected_group_max)  # 调整因子
5. sub_block_max = score - block_max                # 减去局部最大值
6. score_exp = exp(sub_block_max)                   # 指数化
7. block_sums = ReduceSum(score_exp)                # 每个 block 的局部和
8. sum_adjusted = block_sums * block_adjustment     # 调整后的和
9. group_sum = Gemm(block_mapping, sum_adjusted)    # 同组 block 的总和
10. rescale = block_adjustment / max(group_sum, sum_adjusted)  # 重缩放因子
11. softmax_score = score_exp * rescale             # 最终 softmax 结果
```

**关键洞察**：由于每个 block 的 softmax 需要知道同组所有 block 的统计量，这里通过 `IndexReduce`（类似 scatter max）和 `Gemm`（通过 block_mapping 矩阵）来实现跨 block 的统计量聚合。

**数值稳定性原理**：
- 标准 softmax：`softmax(x_i) = exp(x_i) / Σexp(x_j)`
- Block-wise safe softmax：先减局部 max 防溢出，再通过调整因子对齐到全局统计量

**代码位置**：Lines 683-836

#### 4.6e 加权求和

```
softmax_score × V [num_of_block, num_kv_head, block_size, head_dim]
  = attn_out [num_of_block, num_head, 1, head_dim]
```

**代码位置**：Lines 789-809

#### 4.6f 结果映射回 Batch 空间

```
attn_out ──Reshape──→ [num_of_block, hidden_size]
attn_out ──Gemm(block_mapping^T)──→ [B, hidden_size]  # 映射回 batch
```

- 使用 `block_mapping` 的转置将 block 空间的注意力输出映射回 batch 空间

**代码位置**：Lines 838-965

### 4.7 输出投影

```
mapped_attn [B, hidden_size] ──Gemm(linear_weights)──→ out_linear [B, out_features]
```

- 支持 FP8 混合精度

**代码位置**：Lines 947-966

---

## 5. MHA vs GQA 差异

| 方面 | MHA | GQA |
|------|-----|-----|
| Q 维度 | `[num_of_block, num_head, 1, head_dim]` | `[num_of_block, num_kv_head, ngroups, 1, head_dim]` |
| K/V 维度 | `[num_of_block, num_head, block_size, head_dim]` | `[num_of_block, num_kv_head, 1, block_size, head_dim]` |
| Score 维度 | `[num_of_block, num_head, 1, block_size]` | `[num_of_block, num_kv_head, ngroups, 1, block_size]` |
| 选择逻辑 | `num_head == num_kv_head` | `num_head > num_kv_head` |
| KV Cache 大小 | 2 × num_head × block_size × head_dim | 2 × num_kv_head × block_size × head_dim |

GQA 通过 `ngroups = num_head / num_kv_head` 实现分组共享 K/V，K/V 只存储 `num_kv_head` 份，Q 的 `ngroups` 个 head 共享同一组 K/V。

---

## 6. FP8 支持

通过 `AddNodeMixedPrecisionGemm` 模板方法实现 FP8 混合精度：

### FP8 数据流

```
输入 BF16 ──动态缩放──→ FP8 ──FP8 GEMM──→ FP8 输出 ──反缩放──→ BF16
```

### FP8 应用位置

| 位置 | 控制参数 | 说明 |
|------|---------|------|
| Embedding GEMM | `use_fp8_embedding` | src → QKV 投影 |
| KV Cache 存储 | `use_fp8_kv_cache` | K/V 以 FP8 格式存入 cache |
| QK Score | `use_fp8_kv_cache` | Q×K^T 使用 FP8 GEMM |
| Score×V | `use_fp8_kv_cache` | Attention 加权求和 |
| 输出投影 | `use_fp8_out_proj` | 最终线性投影 |

### Scale 处理

- 在 GEMM 前插入 `reciprocal_fwd_f32` 节点计算 scale 的倒数
- 使用 `ConvertToFP8` 节点将 BF16 转换为 FP8（格式为 `syn_type_fp8_143`，即 E4M3）
- 反量化在 GEMM 内部自动完成

---

## 7. 与标准 Attention 的对比

| 特性 | 标准 Attention (PyTorch) | HPU Fused Block Attention |
|------|------------------------|--------------------------|
| 注意力类型 | 单头/多头 Causal Self-Attention | 多头 MHA / GQA |
| 位置编码 | 可学习位置 Embedding / RoPE | RoPE (NeoX/Pairwise) |
| KV Cache | 简单拼接（`torch.cat`） | PagedAttention 风格 Block Cache |
| 数值精度 | FP32 | FP16/BF16 + 可选 FP8 |
| 执行方式 | 逐算子 PyTorch eager | 融合 HPU Recipe（单次 launch） |
| Softmax | 标准 Softmax | Block-wise Safe Softmax |
| 序列管理 | 固定 max_seq_len | Block 分页管理 |
| 内存管理 | 动态分配 | 预分配 Cache + Section 管理 |
| 跨 block 通信 | 无 | IndexReduce + Gemm 聚合 |

---

## 8. 核心设计思想

### 8.1 Block 分页管理

将 KV Cache 划分为固定大小的 block，通过以下辅助张量实现灵活的内存管理和并行计算：

| 张量 | 形状 | 作用 |
|------|------|------|
| `block_list` | `[num_of_block]` | 指定需要读取哪些 block 的 KV |
| `block_mapping` | `[num_of_block, B]` | 将 batch 中的 query 映射到 block 空间 |
| `block_groups` | `[num_of_block]` | 指定每个 block 属于哪个 group（用于跨 block softmax） |
| `block_indices` | `[num_of_block]` | block 在 cache 中的索引位置 |
| `block_offsets` | `[num_of_block]` | block 内的偏移量 |
| `block_bias` | `[num_of_block, block_size]` | 每个 block 的偏置（ALiBi 风格） |

### 8.2 Safe Softmax 跨 Block 聚合

通过 `IndexReduce` + `Gemm` 组合实现跨 block 的 softmax 归一化，避免数值溢出：

- **IndexReduce**：类似 `scatter_max`，将每个 block 的局部最大值聚合到同组 block 的全局最大值
- **Gemm**：通过 `block_mapping` 矩阵实现 batch 空间和 block 空间的双向映射

### 8.3 融合编译

整个注意力计算编译为单个 HPU recipe，减少 host-device 通信开销：

- 一次 `RecipeRunner::Run` 调用完成所有计算
- 中间 tensor 通过 `createTensorNoPresist` 管理，无需 host 侧干预
- 使用 `synSectionHandle` 管理 KV Cache 内存的生命周期

### 8.4 FP8 全链路支持

从 Embedding → KV Cache → Output Projection 均可选 FP8，通过模板参数和条件编译实现零运行时开销的精度切换。

---

## 附录：关键代码路径

### 算子注册

```cpp
PD_BUILD_OP(fused_block_attention)
    .Inputs({"src", "rotary_embs", "key_cache", "value_cache",
             "block_groups", "block_list", "block_mapping", "block_bias",
             "block_indices", "block_offsets", "qkv_weights",
             paddle::Optional("qkv_biases"), "linear_weights",
             paddle::Optional("q_norm_weights"), paddle::Optional("k_norm_weights"),
             paddle::Optional("src_scale"), paddle::Optional("qkv_weights_scale"),
             paddle::Optional("q_scale"), paddle::Optional("k_scale"),
             paddle::Optional("a_scale"), paddle::Optional("v_scale"),
             paddle::Optional("o_linear_scale_x"), paddle::Optional("o_linear_scale_y")})
    .Outputs({"out_linear"})
    .Attrs({"head_dim: int", "num_head: int", "scaling_factor: float",
            "transpose: bool", "use_neox_style: bool", "epsilon: float"})
    .SetKernelFn(PD_KERNEL(FusedBlockAttentionForward))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedBlockAttentionShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedBlockAttentionDtype));
```

### 类型分发

```cpp
if (src.dtype() == phi::DataType::FLOAT16) {
    custom_kernel::FusedBlockAttentionKernel<phi::dtype::float16>(...);
} else if (src.dtype() == phi::DataType::BFLOAT16) {
    custom_kernel::FusedBlockAttentionKernel<phi::dtype::bfloat16>(...);
}
```

### MHA/GQA 选择

```cpp
if (num_head_ == num_kv_head) {
    FusedMHABlockAttention op(guid_prefix, op_info.datatype_);
    op.AddNode<T>(ct, params);
} else {
    FusedGQABlockAttention op(guid_prefix, op_info.datatype_);
    op.AddNode<T>(ct, params);
}
```

---

> **相关文档**：
> - [推理优化总览](README.md) — PaddlePaddle HPU 推理优化"六步走"策略
> - [LLM 推理系统理论](../../transformer/llm-inference-systems.md) — Attention、GQA、FP8 等理论基础
> - [PyTorch Demo](../../demo/decoder_only_demo.py) — Decoder-Only Transformer 极简实现