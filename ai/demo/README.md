# AI Demo 示例代码

本目录包含多个从零实现的 AI 模型示例代码，用于理解 Transformer、MoE 等核心架构的原理。

---

## 目录

- [AI Demo 示例代码](#ai-demo-示例代码)
  - [目录](#目录)
  - [1. Decoder-Only Transformer](#1-decoder-only-transformer)
    - [概述](#概述)
    - [文件](#文件)
    - [代码组件](#代码组件)
    - [运行方式](#运行方式)
    - [演示内容](#演示内容)
  - [2. Block Attention](#2-block-attention)
    - [概述](#概述-1)
    - [文件](#文件-1)
    - [代码组件](#代码组件-1)
    - [运行方式](#运行方式-1)
    - [演示内容](#演示内容-1)
    - [参考文档](#参考文档)
  - [3. Mixture of Experts (MoE)](#3-mixture-of-experts-moe)
    - [概述](#概述-2)
    - [文件](#文件-2)
    - [代码组件](#代码组件-2)
    - [版本演进](#版本演进)
    - [运行方式](#运行方式-2)
    - [演示内容](#演示内容-2)
    - [参考文档](#参考文档-1)
  - [4. 实验脚本](#4-实验脚本)
    - [文件](#文件-3)
    - [实验内容](#实验内容)
    - [运行方式](#运行方式-3)
    - [关键发现](#关键发现)
  - [5. 数据加载管线 (Data Pipeline)](#5-数据加载管线-data-pipeline)
    - [概述](#概述-3)
    - [文件](#文件-4)
    - [代码组件](#代码组件-3)
    - [运行方式](#运行方式-4)
    - [演示内容](#演示内容-3)
    - [参考文档](#参考文档-2)
  - [6. 设计文档](#6-设计文档)
  - [环境要求](#环境要求)
  - [与完整文档的对应关系](#与完整文档的对应关系)

---

## 1. Decoder-Only Transformer

### 概述

从零实现的 Decoder-Only Transformer，用于理解 GPT/LLaMA 系列模型的核心流程和原理。

### 文件

| 文件 | 说明 |
|------|------|
| `decoder_only_demo.py` | 完整实现 + 训练 + 推理演示 |

### 代码组件

| 组件 | 说明 |
|------|------|
| `RMSNorm` | Pre-Norm 归一化（LLaMA 风格） |
| `CausalSelfAttention` | 单头因果自注意力 + KV Cache |
| `SwiGLU_FFN` | 门控前馈网络 |
| `DecoderBlock` | Pre-Norm 结构：Attn + FFN + 残差 |
| `DecoderOnlyTransformer` | 完整模型：Embed → N×Decoder → LM Head |
| `make_arithmetic_dataset` | 等差数列玩具数据集 |

### 运行方式

```bash
python decoder_only_demo.py
```

### 演示内容

1. **训练**：在等差数列数据集上做 Next Token Prediction
2. **推理对比**：有/无 KV Cache 的速度差异
3. **采样多样性**：不同 temperature 下的生成结果
4. **速度基准**：KV Cache 加速比量化

---

## 2. Block Attention

### 概述

从零实现的 Block Attention 机制，用于理解如何通过分块计算来优化长序列的注意力机制。

### 文件

| 文件 | 说明 |
|------|------|
| `block_attention_demo.py` | Block Attention 完整实现 + 训练 + 推理演示 |

### 代码组件

| 组件 | 说明 |
|------|------|
| `BlockCausalSelfAttention` | 分块因果自注意力（Block-wise Attention） |
| `BlockDecoderOnlyTransformer` | 使用 Block Attention 的 Decoder-Only 模型 |

### 运行方式

```bash
python block_attention_demo.py
```

### 演示内容

1. **Block Attention 前向计算**：分块计算注意力，对比标准 Attention 的输出
2. **训练**：在等差数列数据集上训练 Block Attention 模型
3. **推理对比**：Block Attention vs 标准 Attention 的速度差异
4. **KV Cache 兼容性**：验证 Block Attention 与 KV Cache 的配合

### 参考文档

- `ai/llm/inference_opt/fused_block_attention_summary.md` — Block Attention 原理总结
- `ai/transformer/llm-inference-systems.md` — Transformer 推理系统文档

---

## 3. Mixture of Experts (MoE)

### 概述

从零实现的 Sparse Mixture of Experts (MoE) 层，参考 DeepSeek V3 的设计思路。

### 文件

| 文件 | 说明 |
|------|------|
| `moe_demo.py` | MoE 完整实现 + 端到端训练 + 推理演示（v2） |

### 代码组件

| 组件 | 说明 |
|------|------|
| `ExpertMLP` | SwiGLU FFN 作为单个 Expert |
| `Router` | Sigmoid 门控 + Top-K 选择（参考 DeepSeek V3） |
| `SparseMoE` | 完整 MoE 层：Router + Dispatch + Expert + Aggregate + Shared Expert |
| `generate_mixture_of_functions` | 合成数据集生成（可区分的输入分布 + 不同的基函数） |

### 版本演进

| 版本 | Router 类型 | 训练方式 | 关键特性 | MoE vs Dense |
|------|-----------|---------|---------|:-----------:|
| v1 | Softmax | 分阶段（分类器 + 回归） | 基础 MoE 流程 | ❌ Dense 优 |
| v2 | Sigmoid | 端到端 | + Shared Expert + 分组路由 | **✅ MoE 优** |

### 运行方式

```bash
python moe_demo.py
```

### 演示内容

1. **端到端训练**：Router + Expert + Shared Expert 一起通过 MSE Loss 训练
2. **单样本推理**：详细打印每一步的路由信息和输出
3. **批量推理**：统计路由分布和负载均衡情况
4. **与 Dense 模型对比**：激活参数量对齐，公平对比

### 参考文档

- `moe_demo_design.md` — MoE Demo 设计与实现总结（含详细实验数据）
- `ai/llm/inference_opt/mixture_of_experts.md` — MoE 原理与 HPU 融合算子实现
- `ai/code/modeling_deepseek_v3.py` — DeepSeek V3 参考实现

---

## 4. 实验脚本

### 文件

| 文件 | 说明 |
|------|------|
| `moe_experiment.py` | MoE 激活专家比例对比实验 |

### 实验内容

对比不同激活专家比例对 MoE 效果的影响：

| 实验 | Expert 数 | Top-K | 激活比例 | MoE MSE | Dense MSE | 胜出 |
|------|----------|-------|:-------:|:-------:|:---------:|:----:|
| A: 8exp_k2 | 8 | 2 | 25% | **1.449** | 1.548 | ✅ MoE |
| B: 8exp_k4 | 8 | 4 | 50% | **1.386** | 1.769 | ✅ MoE |
| C: 4exp_k2 | 4 | 2 | 50% | **0.024** | 0.105 | ✅ MoE |

### 运行方式

```bash
python moe_experiment.py
```

### 关键发现

1. **所有配置 MoE 均优于 Dense** — v2 端到端训练框架下 MoE 全面领先
2. **激活比例 50% 优于 25%** — MoE 领先优势从 6.4% 扩大到 21.7%
3. **4 expert 效果远好于 8 expert** — 路由问题更简单、每个 expert 数据更多
4. **输入分布混叠是限制路由准确率的根本原因** — 详见 `moe_demo_design.md` 第 3.4 节

---

## 5. 数据加载管线 (Data Pipeline)

### 概述

聚焦训练数据加载的**效率**与**鲁棒性**，不涉及图像处理。以 COCO val2017 为数据源，展示数据完整性校验、样本分析、样本权重、缓存策略、Prefetch 和异常处理。

### 文件

| 文件 | 说明 |
|------|------|
| `data_pipeline_demo.py` | 数据加载管线完整实现（7 个步骤） |
| `data_pipeline_summary.md` | 技术总结文档 |

### 代码组件

| 组件 | 说明 |
|------|------|
| `validate_dataset()` | 数据完整性校验（损坏图片、异常标注、数据缺失） |
| `analyze_annotations()` | 多维度样本分析（类别分布、尺寸、每图物体数、宽高比、位置热力图） |
| `compute_sample_weights()` | 样本权重计算（类别逆频率 + 尺寸权重 + 组合权重） |
| `CocoDataset` | COCO 数据集封装，支持样本权重 |
| `LRUCache` | 线程安全的 LRU 缓存 |
| `CachedDataLoader` | 带缓存的 DataLoader 封装 |

### 运行方式

```bash
python data_pipeline_demo.py
```

### 演示内容

1. **数据完整性校验** — 检测损坏图片、异常 bbox、数据缺失
2. **样本分析** — 5 维度统计（类别、尺寸、密度、宽高比、位置）
3. **样本权重** — 3 种策略（类别逆频率、尺寸、组合）
4. **加载策略对比** — 无缓存 vs LRU 缓存（多 epoch 场景）
5. **Prefetch 展示** — 多 worker 后台预加载
6. **异常处理** — 损坏跳过、标注修复、错误隔离
7. **数据质量报告** — 汇总统计

### 参考文档

- `data_pipeline_summary.md` — 技术总结（含 3 大瓶颈框架和关键工程发现）

---

## 6. 设计文档

| 文件 | 说明 |
|------|------|
| `decoder_only_summary.md` | Decoder-Only Transformer 设计技术原理与实现总结 |
| `block_attention_summary.md` | Block Attention 设计技术原理与实现总结 |
| `moe_demo_design.md` | MoE Demo 设计与实现总结（含 v1/v2 对比、实验数据、分布混叠分析） |
| `data_pipeline_summary.md` | 数据加载管线技术总结 |
| `vibe_coding_journal.md` | Vibe Coding 实践记录 |

---

## 环境要求

- Python >= 3.8
- PyTorch >= 2.0
- 无需额外依赖

## 与完整文档的对应关系

本目录的示例代码与以下文档配套：

| 文档 | 对应 Demo | 说明 |
|------|----------|------|
| `ai/transformer/llm-inference-systems.md` | `decoder_only_demo.py` | Transformer 推理系统 |
| `ai/llm/inference_opt/fused_block_attention_summary.md` | `block_attention_demo.py` | Block Attention 原理 |
| `ai/llm/inference_opt/mixture_of_experts.md` | `moe_demo.py` | MoE 原理与实现 |
| `ai/code/modeling_deepseek_v3.py` | `moe_demo.py` | DeepSeek V3 参考实现 |

文档提供理论深度和数学推导，代码提供可运行的工程实现，两者结合理解效果最佳。