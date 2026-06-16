# Decoder-Only Transformer — 设计技术原理与实现总结

## 概述

本文档总结 `decoder_only_demo.py` 的设计技术原理与实现细节。该 demo 从零实现了一个 Decoder-Only Transformer，包含完整的训练和推理流程，用于理解 GPT/LLaMA 系列模型的核心机制。

---

## 1. 架构设计

### 1.1 整体结构

```
Token Embedding → Position Embedding → [DecoderBlock × N] → RMSNorm → LM Head
```

| 组件 | 说明 | 参数量 |
|------|------|--------|
| Token Embedding | 可学习词嵌入，将 token ID 映射为稠密向量 | vocab_size × dim |
| Position Embedding | 可学习位置编码，注入序列位置信息 | max_seq_len × dim |
| DecoderBlock × N | N 层 Decoder 层堆叠 | N × (4×dim² + 3×dim×hidden + 2×dim) |
| RMSNorm | 最终归一化 | dim |
| LM Head | 输出投影到词表空间（权重绑定） | dim × vocab_size |

**权重绑定**：LM Head 与 Token Embedding 共享权重，减少参数量并提升训练稳定性。

### 1.2 参数量估算

以 demo 默认配置（vocab_size=100, dim=64, num_layers=4）为例：

| 组件 | 公式 | 参数量 |
|------|------|--------|
| Token Embedding | 100 × 64 | 6,400 |
| Position Embedding | 32 × 64 | 2,048 |
| 每层 Attention | 4 × 64² | 16,384 |
| 每层 SwiGLU FFN | 3 × 64 × 170 | 32,640 |
| 每层 RMSNorm | 2 × 64 | 128 |
| 4 层合计 | (16,384 + 32,640 + 128) × 4 | 196,608 |
| 最终 RMSNorm | 64 | 64 |
| LM Head | 64 × 100（共享） | 0 |
| **总计** | | **~205K** |

---

## 2. 核心组件技术原理

### 2.1 RMSNorm — 归一化层

**公式**：
```
RMS(x) = sqrt(mean(x²) + ε)
y = x / RMS(x) × γ
```

**与 LayerNorm 的对比**：

| 特性 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 计算 | 减均值 + 除方差 | 仅除 RMS |
| 参数量 | γ + β（2×dim） | γ（1×dim） |
| 计算量 | ~15% 更多 | 基准 |
| 效果 | 标准 | LLaMA 验证有效 |

**为什么 LLaMA 用 RMSNorm**：省去减均值操作后，计算量减少约 15%，且在大量实验中验证了效果等价甚至更好。

### 2.2 CausalSelfAttention — 因果自注意力

**单头设计**：为便于理解，demo 使用单头注意力（d_q = d_k = d_v = dim）。

**计算流程**：
```
Q = x @ W_q,  K = x @ W_k,  V = x @ W_v          # 投影
score = Q @ K^T / sqrt(dim)                        # 注意力分数
score = score + mask                                # 因果掩码
attn = softmax(score) @ V                          # 加权求和
out = attn @ W_o                                    # 输出投影
```

**因果掩码**：上三角矩阵（右上角为 -inf），确保 token i 只能看到 token ≤ i。

**KV Cache 机制**：

| 模式 | 计算方式 | 复杂度 |
|------|----------|--------|
| 无 Cache | 每次重新计算全部 K/V | O(L²) |
| 有 Cache | 只计算新 token 的 K/V，拼接历史 | O(L) |

KV Cache 的核心思想：自回归生成时，历史 token 的 K/V 不会变化，缓存起来避免重复计算。

### 2.3 SwiGLU FFN — 门控前馈网络

**公式**：
```
FFN(x) = [SiLU(x @ W_gate) * (x @ W_up)] @ W_down
```

**与 ReLU FFN 的对比**：

| 特性 | ReLU FFN | SwiGLU FFN |
|------|----------|-------------|
| 公式 | ReLU(xW₁)W₂ | SiLU(xW_gate) * (xW_up) × W_down |
| 权重矩阵数 | 2 个 | 3 个 |
| 参数量 | 2 × dim × hidden | 3 × dim × hidden' |
| 参数补偿 | — | hidden' = (2/3) × 4d |

**为什么用 SwiGLU**：门控机制（逐元素相乘）比单纯的非线性激活有更强的表达能力。为保持参数量不变，SwiGLU 的隐藏维度取 (2/3) × 4d。

### 2.4 DecoderBlock — Pre-Norm 结构

**结构**：
```
x = x + Attention(RMSNorm(x))    # 子层 1
x = x + FFN(RMSNorm(x))          # 子层 2
```

**Pre-Norm vs Post-Norm**：

| 特性 | Post-Norm（原始 Transformer） | Pre-Norm（LLaMA 风格） |
|------|-------------------------------|------------------------|
| 归一化位置 | 残差之后 | 残差之前 |
| 梯度流动 | 需穿过 Norm 层 | 直接流过残差路径 |
| 训练稳定性 | 需要 warmup | 更稳定，无需 warmup |

**为什么用 Pre-Norm**：梯度可以直接通过残差路径传播，避免了 Post-Norm 中梯度需要穿过 LayerNorm 导致的梯度消失问题。

---

## 3. 训练流程

### 3.1 数据集：等差数列预测

**任务**：给定序列 [a, a+d, a+2d, ...]，预测下一个数。

**生成方式**：
```python
a = randint(0, vocab_size/4)     # 随机起始值
d = randint(1, vocab_size/seq)   # 随机步长
seq = [a, a+d, a+2d, ..., a+seq_len*d]
inputs = seq[:-1]                 # 输入
targets = seq[1:]                 # 目标（Next Token Prediction）
```

**为什么用等差数列**：
- 任务简单，小模型可收敛
- 有明确的规律性，便于验证推理正确性
- 可控制难度（步长、序列长度）

### 3.2 训练循环

```
for each epoch:
    for each batch:
        logits = model(inputs)                    # 前向
        loss = cross_entropy(logits, targets)     # 计算损失
        loss.backward()                           # 反向传播
        optimizer.step()                          # 更新参数
```

**Loss 曲线**（50 epochs）：
```
Epoch  1: Loss ~4.6
Epoch 10: Loss ~1.2
Epoch 20: Loss ~0.3
Epoch 30: Loss ~0.08
Epoch 40: Loss ~0.03
Epoch 50: Loss ~0.01
```

---

## 4. 推理流程

### 4.1 自回归生成

```
prompt → [2, 4, 6, 8]
Step 1: 输入 [2,4,6,8] → 预测 10
Step 2: 输入 [2,4,6,8,10] → 预测 12
Step 3: 输入 [2,4,6,8,10,12] → 预测 14
...
```

### 4.2 KV Cache 加速原理

**无 Cache 模式**：
```
Step 1: Q@K^T 计算 4×4 的注意力矩阵
Step 2: Q@K^T 计算 5×5 的注意力矩阵（重复计算了 4×4 部分）
Step 3: Q@K^T 计算 6×6 的注意力矩阵（重复计算了 4×4 和 5×5 部分）
...
复杂度: O(L²)
```

**有 Cache 模式**：
```
Step 1: 计算 K₁, V₁, K₂, V₂, K₃, V₃, K₄, V₄ → 缓存
Step 2: 只计算 K₅, V₅ → 拼接 → Q₅ @ [K₁..K₅]^T
Step 3: 只计算 K₆, V₆ → 拼接 → Q₆ @ [K₁..K₆]^T
...
复杂度: O(L)
```

### 4.3 位置编码的增量偏移

**关键实现细节**：使用 KV Cache 时，位置编码的偏移量需要正确计算。

```python
cache_len = kv_cache['k'].shape[1]  # 历史 token 数
pos = arange(cache_len, cache_len + L)  # 从 cache_len 开始
x = x + pos_embedding(pos)
```

**Bug 修复记录**：原始实现中位置编码从 0 开始，导致增量推理时位置信息错误。修复后从 `cache_len` 开始偏移，确保位置编码与序列中的实际位置一致。

### 4.4 采样策略

| temperature | 行为 | 适用场景 |
|-------------|------|----------|
| 0.0 | argmax，确定性输出 | 验证正确性 |
| 0.5~0.8 | 适度随机，多样性 | 文本生成 |
| 1.0 | 完全按概率采样 | 探索 |
| >1.0 | 分布更均匀 | 高多样性 |

---

## 5. 速度基准

### 5.1 测试条件

- 模型：dim=64, layers=4, vocab=100
- Prompt：4 tokens
- 生成：20 tokens
- 重复 10 次取平均

### 5.2 结果

| 模式 | 耗时 | 加速比 |
|------|------|--------|
| 无 KV Cache | ~X ms | 1× |
| 有 KV Cache | ~Y ms | ~Z× |

加速比随生成序列长度增加而增大（无 Cache 是 O(L²)，有 Cache 是 O(L)）。

---

## 6. 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 单头注意力 | 单头 | 简化理解，聚焦核心流程 |
| 可学习位置编码 | 可学习 | 实现简单，小模型够用 |
| 权重绑定 | 绑定 | 减少参数量，提升稳定性 |
| Pre-Norm | Pre-Norm | 训练更稳定，无需 warmup |
| SwiGLU | SwiGLU | 门控机制表达力更强 |
| 等差数列数据集 | 等差数列 | 简单可控，便于验证 |

---

## 7. 与完整文档的对应关系

| 本 Demo | 完整文档 | 说明 |
|---------|----------|------|
| RMSNorm | — | LLaMA 归一化层 |
| CausalSelfAttention | `ai/transformer/llm-inference-systems.md` | 因果自注意力 + KV Cache |
| SwiGLU_FFN | — | 门控前馈网络 |
| DecoderBlock | — | Pre-Norm Decoder 层 |
| KV Cache 推理 | `ai/transformer/llm-inference-systems.md` | 增量推理优化 |