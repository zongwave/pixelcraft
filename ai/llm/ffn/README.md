
---


# 📑 目录

* [前馈神经网络（FFN）技术总结](#前馈神经网络ffn技术总结)

  * [🔧 FFN 的结构](#ffn-的结构)
    * [🧠 核心理解](#核心理解)
    * [📐 输入输出维度](#输入输出维度)
    * [🧊 FFN 的计算开销](#ffn-的计算开销)
    * [🎨 FFN 结构图（位置独立前馈网络）](#ffn-结构图位置独立前馈网络)
    * [🧪 PyTorch 伪代码实现](#pytorch-伪代码实现)
    * [✅ 应用示意](#应用示意)
    * [🚀 总结](#总结)
  * [🔧 Gated FFN（门控前馈神经网络）](#gated-ffn门控前馈神经网络总结)
    * [📌 背景介绍](#背景介绍)
    * [🧠 标准 FFN 回顾](#标准-ffn-回顾)
    * [🧊 Gated FFN 基本结构](#gated-ffn-基本结构)
    * [📐 几种常见 Gated FFN 变体](#几种常见-gated-ffn-变体)
    * [🎨 结构图示意](#结构图示意)
    * [🧪 PyTorch 示例代码（以 GEGLU 为例）](#pytorch-示例代码以-geglu-为例)
    * [✅ 优缺点分析](#优缺点分析)
    * [🚀 总结](#总结)


---


📌 前馈神经网络（Feed-Forward Network，简称 FFN）是 Transformer 架构中每个编码器（Encoder）和解码器（Decoder）层的一个核心组成部分，它通常用于对每个位置的特征进行非线性转换与增强。它作用于每个位置（token）**独立地**，不涉及序列中的其他位置，因此常称为“位置独立（position-wise）”的 FFN。

---

## FFN 的结构

Transformer 中的 FFN 模块一般结构如下：

$$
\text{FFN}(x) = \text{Linear}_2(\text{ReLU}(\text{Linear}_1(x)))
$$

即：

```python
output = W2 * ReLU(W1 * x + b1) + b2
```

其中：

* `x`: 输入张量，形状通常为 `[batch_size, seq_len, hidden_dim]`
* `W1`: 第一层线性变换的权重，形状为 `[hidden_dim, ffn_dim]`
* `W2`: 第二层线性变换的权重，形状为 `[ffn_dim, hidden_dim]`
* `ffn_dim`: 通常是 `hidden_dim` 的 4 倍（如 BERT 中为 3072，而 `hidden_dim` 是 768）
* `ReLU`: 激活函数，有时也用 GELU 等替代
* `b1`, `b2`: 偏置项

---

### 核心理解

* FFN 是对**每个 token** 独立地做非线性变换，没有 token 之间的信息交换。
* 它类似于图像中的 1×1 卷积：做的是通道维（特征维）变换，而不改变空间结构（在 NLP 中即序列结构）。

---

### 输入输出维度

假设输入维度为：

* `batch_size = B`
* `seq_len = L`
* `hidden_dim = D`

经过 FFN，输入张量形状为 `[B, L, D]`，输出形状仍为 `[B, L, D]`，因为第二个线性层会投影回原维度。

中间隐藏层（`ffn_dim`）仅在内部使用，形状为 `[B, L, ffn_dim]`。

---

### FFN 的计算开销

* FFN 的计算和参数量主要集中在两个线性层：

  * 第一层：`D × ffn_dim`
  * 第二层：`ffn_dim × D`
* 由于 `ffn_dim` 通常远大于 `D`（如 4×），FFN 是 Transformer 中计算最密集的模块之一，尤其在多层堆叠中。
* 在大模型中有时使用 **Gated FFN** 或 **Mixture of Experts (MoE)** 替代标准 FFN，以实现更高容量与稀疏激活。

---


### FFN 结构图（位置独立前馈网络）

```
        输入 x ∈ [B, L, D]
              │
              ▼
      ┌───────────────────────┐
      │ Linear1: D → ffn_dim  │ ← 参数较大
      └───────────────────────┘
              │
              ▼
         激活函数（ReLU 或 GELU）
              │
              ▼
      ┌───────────────────────┐
      │ Linear2: ffn_dim → D  │ ← 投影回原维度
      └───────────────────────┘
              │
              ▼
        输出 y ∈ [B, L, D]
```

说明：

* `B`: batch size
* `L`: 序列长度（sequence length）
* `D`: 模型维度（如 768）
* `ffn_dim`: 隐藏维度，通常为 `4 * D`（如 3072）

---

### PyTorch 伪代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFFN(nn.Module):
    def __init__(self, hidden_dim=768, ffn_dim=3072, activation='relu'):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, hidden_dim)
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            raise ValueError("Unsupported activation")

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_dim]
        x = self.linear1(x)              # → [B, L, ffn_dim]
        x = self.activation(x)          # 非线性
        x = self.linear2(x)             # → [B, L, hidden_dim]
        return x
```

---

### 应用示意

在 Transformer 中，每一层都会使用这个模块：

```python
class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim=768, ffn_dim=3072):
        super().__init__()
        self.self_attn = MultiHeadAttention(...)
        self.ffn = PositionwiseFFN(hidden_dim, ffn_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = x + self.self_attn(self.norm1(x))  # 残差连接 + 自注意力
        x = x + self.ffn(self.norm2(x))        # 残差连接 + FFN
        return x
```


---

### 总结

| 模块                   | 作用                | 是否引入 token 交互 | 参数量 | 计算密集度 |
| -------------------- | ----------------- | ------------- | --- | ----- |
| Multi-Head Attention | 建立 token 之间的联系    | ✅ 是           | 中等  | 中等    |
| FFN                  | 提升 token 自身特征表达能力 | ❌ 否（位置独立）     | 高   | 高     |

---



## Gated FFN（门控前馈神经网络）总结

---

 * 下面是对 **Gated FFN（门控前馈神经网络）** 的完整总结，涵盖结构原理、数学公式、变体、优劣势和代码示例
---

### 背景介绍

在 Transformer 的标准 FFN 模块中，每个位置上的向量独立地通过两层线性变换和非线性激活进行处理。这种结构简单高效，但存在以下局限：

* 激活冗余：很多 token 的激活并不稀疏或选择性不强。
* 缺乏动态建模能力：所有 token 都通过相同的路径，没有门控或选择机制。

为增强 FFN 的表达能力，研究者提出 **Gated FFN（门控前馈网络）**，常见于 Gated Linear Unit (GLU)、GEGLU、SwiGLU 等结构中。

---

### 标准 FFN 回顾

传统 FFN 结构如下：

```text
FFN(x) = W₂·ReLU(W₁·x + b₁) + b₂
```

其中：

* `W₁`: \[d\_model, d\_ff]
* `W₂`: \[d\_ff, d\_model]
* 激活函数一般为 ReLU/GELU/SwiGLU

---

### Gated FFN 基本结构

**核心思想：使用门控机制，让一部分路径进行信息筛选或调制。**

最通用的形式为：

```text
FFN_Gated(x) = W₂ · (Activation(W₁a·x) ⊙ Gate(W₁b·x))
```

其中：

* `⊙` 表示逐元素乘法（Hadamard product）
* `W₁a` 和 `W₁b` 分别对应主路径和门控路径的权重
* `Gate()` 通常使用 Sigmoid 或 GELU
* 输出通过线性变换 `W₂` 投回原维度

---

### 几种常见 Gated FFN 变体

| 名称         | 结构公式                                          | 特点                     |
| ---------- | --------------------------------------------- | ---------------------- |
| GLU        | `GLU(x) = (W₁a·x) ⊙ sigmoid(W₁b·x)`           | 最基础的门控结构               |
| GEGLU      | `GEGLU(x) = GELU(W₁a·x) ⊙ W₁b·x`              | 更强非线性，OpenAI 使用        |
| SwiGLU     | `SwiGLU(x) = Swish(W₁a·x) ⊙ W₁b·x`            | SwiGLU = x·sigmoid(βx) |
| Gated GELU | `GatedGELU(x) = GELU(W₁a·x) ⊙ sigmoid(W₁b·x)` | 实验中最稳定的一种结构            |

> 💡 注：SwiGLU 被用于 PaLM、GEGLU 用于 GPT-3.5/4 中。

---

### 结构图示意

```
                x
                │
        ┌───────┴────────┐
        │                │
     Linear          Linear
    (W₁a·x)         (W₁b·x)
        │                │
    Activation         Gate
        │                │
        └───────┬────────┘
                ▼
            Elementwise ⊙
                │
              Linear (W₂)
                │
               out
```

---

### PyTorch 示例代码（以 GEGLU 为例）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GEGLUFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1_a = nn.Linear(d_model, d_ff)
        self.fc1_b = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        a = F.gelu(self.fc1_a(x))
        b = self.fc1_b(x)
        x = a * b  # Gating
        return self.fc2(x)
```

---

### 优缺点分析

#### ✅ 优点：

* 增强表示能力：门控机制能筛选激活值，动态调整输出。
* 提高训练稳定性：例如 GELU/Sigmoid 的平滑非线性帮助优化。
* 已在多个大模型中验证有效，如 GPT-4、PaLM。

#### ⚠️ 缺点：

* 参数量略微增加（W₁ 被分成两组）
* 计算量略升高（两次线性变换 + gate 操作）
* 对推理速度略有影响（但可并行优化）

---

### 总结

| 特征   | 标准 FFN         | Gated FFN    |
| ---- | -------------- | ------------ |
| 非线性层 | 单一激活函数         | 激活 + 门控组合    |
| 参数数量 | 少              | 多（W₁a + W₁b） |
| 表达能力 | 中              | 高            |
| 模型代表 | Transformer 原始 | GPT-4、PaLM 等 |

> 🧠 **一句话总结**：Gated FFN 是对传统 FFN 的扩展，引入门控路径以提升选择性激活和模型动态性，已成为现代大型模型结构的主流设计之一。

---





---