# Vibe Coding 实践记录：从零实现 Decoder-Only Transformer 与 Block Attention

> **日期**: 2026-06-15
> **作者**: Zong
> **仓库**: [pixelcraft](https://github.com/zongwave/pixelcraft)
> **目录**: `ai/demo/`

---

## 目录

- [Vibe Coding 实践记录：从零实现 Decoder-Only Transformer 与 Block Attention](#vibe-coding-实践记录从零实现-decoder-only-transformer-与-block-attention)
  - [目录](#目录)
  - [1. 背景与目标](#1-背景与目标)
  - [2. 第一阶段：阅读总结文档](#2-第一阶段阅读总结文档)
    - [2.1 文档概览](#21-文档概览)
    - [2.2 关键收获](#22-关键收获)
  - [3. 第二阶段：实现极简 Decoder-Only Transformer](#3-第二阶段实现极简-decoder-only-transformer)
    - [3.1 设计决策](#31-设计决策)
    - [3.2 核心组件实现](#32-核心组件实现)
    - [3.3 训练与推理验证](#33-训练与推理验证)
  - [4. 第三阶段：实现 Block Attention](#4-第三阶段实现-block-attention)
    - [4.1 背景：为什么需要 Block Attention](#41-背景为什么需要-block-attention)
    - [4.2 设计规划](#42-设计规划)
    - [4.3 核心数据结构：BlockManager](#43-核心数据结构blockmanager)
    - [4.4 实现要点](#44-实现要点)
  - [5. 第四阶段：Bug 排查与修复](#5-第四阶段bug-排查与修复)
    - [5.1 发现不一致](#51-发现不一致)
    - [5.2 根因分析](#52-根因分析)
    - [5.3 修复方案](#53-修复方案)
    - [5.4 验证结果](#54-验证结果)
  - [6. Vibe Coding 方法论总结](#6-vibe-coding-方法论总结)
    - [6.1 什么是 Vibe Coding](#61-什么是-vibe-coding)
    - [6.2 核心原则](#62-核心原则)
    - [6.3 工作流](#63-工作流)
    - [6.4 常见陷阱与应对](#64-常见陷阱与应对)
    - [6.5 适用场景](#65-适用场景)

---

## 1. 背景与目标

**目标**: 理解 Decoder-Only Transformer（GPT/LLaMA 系列）的核心原理，并通过代码实现加深理解。

**具体任务**:
1. 阅读已有的 Transformer 总结文档
2. 从零实现一个极简 Decoder-Only Transformer（训练 + 推理）
3. 在已有实现上添加 Block Attention（PagedAttention 风格的内存分块优化）
4. 验证训练和推理的正确性

**技术栈**: Python + PyTorch，CPU 环境

---

## 2. 第一阶段：阅读总结文档

### 2.1 文档概览

项目 `ai/transformer/` 目录下已有 `llm-inference-systems.md` 文档，系统性地介绍了 LLM 推理系统的核心概念：

| 章节 | 内容 |
|------|------|
| Ch1 | Transformer 架构基础（RMSNorm、SwiGLU、RoPE、GQA） |
| Ch2 | KV Cache 原理与实现 |
| Ch3 | 内存瓶颈分析（KV Cache 占用、计算与访存比） |
| Ch4 | PagedAttention / Block Attention |
| Ch5 | 量化（FP8、INT8、INT4） |
| Ch6 | 系统优化（Continuous Batching、Prefix Caching） |

### 2.2 关键收获

1. **KV Cache 是自回归推理的核心优化**：避免重复计算历史 token 的 K/V
2. **Block Attention 解决内存碎片问题**：预分配固定大小的 block 池，通过 block table 管理
3. **Safe Softmax 是 Block Attention 的关键**：跨 block 聚合时需要正确的 softmax 归一化
4. **位置编码在增量推理中容易出错**：新 token 的位置偏移量需要从 cache 长度计算

---

## 3. 第二阶段：实现极简 Decoder-Only Transformer

### 3.1 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 注意力头数 | 单头 | 简化实现，聚焦核心流程 |
| 位置编码 | 可学习 Embedding | 实现简单，适合小模型 |
| 归一化 | RMSNorm | LLaMA 风格，比 LayerNorm 轻量 |
| FFN | SwiGLU | 现代 LLM 标配，门控机制提升表达力 |
| 权重绑定 | 是 | Embedding 与 LM Head 共享权重 |
| 数据集 | 等差数列预测 | 简单可控，易于验证正确性 |

### 3.2 核心组件实现

```
DecoderOnlyTransformer
├── token_embedding: Embedding(vocab_size, dim)
├── pos_embedding: Embedding(max_seq_len, dim)
├── layers: ModuleList[DecoderBlock × N]
│   └── DecoderBlock
│       ├── norm1: RMSNorm
│       ├── attn: CausalSelfAttention
│       │   ├── q_proj, k_proj, v_proj, out_proj: Linear
│       │   └── forward: QKV → (可选 KV Cache) → Score → Softmax → Output
│       ├── norm2: RMSNorm
│       └── ffn: SwiGLU_FFN
│           ├── w_gate, w_up, w_down: Linear
│           └── forward: SiLU(gate) * up → down
├── norm: RMSNorm
└── lm_head: Linear(dim, vocab_size) [权重绑定]
```

**关键代码片段 — CausalSelfAttention 的 KV Cache 逻辑**:

```python
def forward(self, x, mask=None, kv_cache=None):
    q = self.q_proj(x)
    k = self.k_proj(x)
    v = self.v_proj(x)

    if kv_cache is not None:
        k = torch.cat([kv_cache['k'], k], dim=1)
        v = torch.cat([kv_cache['v'], v], dim=1)

    new_kv_cache = {'k': k, 'v': v}
    # ... attention 计算 ...
    return out, new_kv_cache
```

### 3.3 训练与推理验证

**训练结果**（50 epochs）:
```
Epoch  1/50 | Loss: 26.2031
Epoch 10/50 | Loss: 0.2898
Epoch 20/50 | Loss: 0.1173
Epoch 30/50 | Loss: 0.1104
Epoch 40/50 | Loss: 0.1067
Epoch 50/50 | Loss: 0.1089
```

Loss 从 26 降到 0.1，模型成功学会了等差数列规律。

---

## 4. 第三阶段：实现 Block Attention

### 4.1 背景：为什么需要 Block Attention

原始 KV Cache 的问题：
- **动态内存分配**：每次 `torch.cat` 都可能触发内存重新分配
- **内存碎片**：不同序列的 KV Cache 长度不同，导致碎片化
- **无法预分配**：最大序列长度不确定，无法提前规划内存

Block Attention 的解决方案：
- **预分配静态内存池**：一次性分配固定大小的 block 池
- **分块管理**：通过 block table 映射逻辑位置到物理位置
- **零动态分配**：推理过程中只做读写，不做内存分配

### 4.2 设计规划

在规划阶段，我们讨论了两种方案：

**方案 A：完整 PagedAttention**（复杂）
- 虚拟内存风格的页式管理
- 支持动态分配和释放
- 需要处理缺页、回收等复杂逻辑

**方案 B：极简 Block 管理**（选择）
- 一次性预分配静态内存池
- BlockManager 只维护状态（空闲/已用）
- 不涉及内存分配和释放

最终选择 **方案 B**，原因：
1. 自回归推理中 KV Cache 只增不减，不需要释放
2. 静态池实现简单，易于理解
3. 聚焦于 Block Attention 的核心思想，而非内存管理细节

### 4.3 核心数据结构：BlockManager

```python
class BlockManager:
    def __init__(self, num_blocks, block_size, dim, device):
        # === 静态内存池（一次性分配，永不增长） ===
        self.k_cache = torch.zeros(num_blocks, block_size, dim, device=device)
        self.v_cache = torch.zeros(num_blocks, block_size, dim, device=device)

        # === 状态管理 ===
        self.free_blocks = list(range(num_blocks))  # 空闲 block 索引
        self.block_table = []                        # 逻辑 block → 物理 block
        self.current_length = 0                      # 当前序列长度

    def append(self, k, v):
        """追加一个 token 的 K/V 到静态池中"""
        pos = self.current_length
        block_idx = pos // self.block_size
        offset = pos % self.block_size

        if offset == 0:  # 需要新 block
            phys_block = self.free_blocks.pop(0)
            self.block_table.append(phys_block)

        phys_block = self.block_table[block_idx]
        self.k_cache[phys_block, offset] = k
        self.v_cache[phys_block, offset] = v
        self.current_length += 1

    def gather_kv(self):
        """从静态池中 gather 出连续 K/V"""
        # 遍历 block_table，从每个物理 block 中取出数据
        # 最后一个 block 可能未填满，需要特殊处理
        k_parts = [self.k_cache[phys] for phys in self.block_table[:-1]]
        v_parts = [self.v_cache[phys] for phys in self.block_table[:-1]]
        # 处理最后一个 block（可能未填满）
        remainder = self.current_length % self.block_size
        if remainder > 0:
            k_parts.append(self.k_cache[self.block_table[-1], :remainder])
            v_parts.append(self.v_cache[self.block_table[-1], :remainder])
        return torch.cat(k_parts), torch.cat(v_parts)
```

### 4.4 实现要点

1. **Block 大小选择**：`block_size=4`，每个 block 存储 4 个 token 的 K/V
2. **Block 总数**：`num_blocks = ceil(max_seq_len / block_size) = 8`
3. **每层独立 BlockManager**：每层有自己独立的 K/V，需要独立的 BlockManager
4. **训练时不使用 Block Cache**：训练时 `block_manager=None`，直接使用当前 batch 的 K/V

---

## 5. 第四阶段：Bug 排查与修复

### 5.1 发现不一致

运行 Block Attention demo 后，发现推理结果不一致：

```
[1] 无 Cache 推理:
    生成: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]  ✅

[2] Block Cache 推理:
    生成: [2, 4, 6, 8, 10, 11, 12, 14, 15, 16, 21, 26]  ❌

[3] 结果一致性验证:
    一致: ❌
```

第一个生成 token `10` 正确，之后全部出错。

### 5.2 根因分析

进一步测试发现，**原始 `decoder_only_demo.py` 的 KV Cache 也有同样的 bug**：

```
[1] 无 KV Cache 推理:
    生成: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]  ✅

[2] 有 KV Cache 推理:
    生成: [2, 4, 6, 8, 10, 11, 12, 13, 16, 21, 26, 31]  ❌
```

这说明 bug 不在 Block Attention 实现中，而在**原始 KV Cache 逻辑**中。

**定位过程**：

1. 观察现象：第一个生成 token 正确，后续全部错误
2. 推断：prompt 处理阶段正确，增量推理阶段出错
3. 分析增量推理与全量推理的差异：
   - 全量推理：输入完整序列 `[2,4,6,8,10,...]`，位置编码 `[0,1,2,3,4,...]`
   - 增量推理：输入单个 token `[10]`，位置编码 `[0]` ← **错误！**

**根因**：位置编码偏移量错误。

```python
# ❌ 错误代码
pos = torch.arange(L, device=device)  # L=1 → pos=[0]
# 增量 token 总是得到位置 0，而不是它应有的位置

# ✅ 正确代码
cache_len = kv_caches[0]['k'].shape[1]  # 或 block_managers[0].num_tokens
pos = torch.arange(cache_len, cache_len + L, device=device)
# 增量 token 得到正确的位置偏移
```

**为什么这个 bug 如此隐蔽？**

1. 训练时总是全量计算，不会触发增量路径
2. 无 Cache 推理也是全量计算，不会触发增量路径
3. 只有 KV Cache / Block Cache 推理才会触发增量路径
4. 第一个 token 正确是因为 prompt 处理阶段（全量）正确
5. 后续 token 错误是因为位置编码从 0 开始，模型把新 token 当作序列开头

### 5.3 修复方案

在两个文件中修复相同的问题：

**`decoder_only_demo.py`**:
```python
# 修复前
pos = torch.arange(L, device=device).unsqueeze(0)

# 修复后
cache_len = 0
if kv_caches is not None and kv_caches[0] is not None:
    cache_len = kv_caches[0]['k'].shape[1]
pos = torch.arange(cache_len, cache_len + L, device=device).unsqueeze(0)
```

**`block_attention_demo.py`**:
```python
# 修复前
pos = torch.arange(L, device=device).unsqueeze(0)

# 修复后
cache_len = 0
if block_managers is not None and block_managers[0] is not None:
    cache_len = block_managers[0].num_tokens
pos = torch.arange(cache_len, cache_len + L, device=device).unsqueeze(0)
```

### 5.4 验证结果

修复后，两个 demo 的结果完全一致：

```
[1] 无 Cache 推理:
    生成: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

[2] Block Cache / KV Cache 推理:
    生成: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

[3] 结果一致性验证:
    一致: ✅
```

---

## 6. Vibe Coding 方法论总结

### 6.1 什么是 Vibe Coding

Vibe Coding 是一种**AI 辅助编程**的工作方式：开发者用自然语言描述需求，AI 生成代码，开发者审查、测试、迭代。核心特征是**高频率的"思考-生成-验证"循环**。

### 6.2 核心原则

| 原则 | 说明 | 本次实践中的体现 |
|------|------|------------------|
| **先理解，后编码** | 在写代码前先阅读文档、理解原理 | 先阅读 `llm-inference-systems.md` 再动手 |
| **最小可行实现** | 从最简单的版本开始，逐步迭代 | 先实现单头注意力，再添加 Block Attention |
| **持续验证** | 每步都运行验证，不积累不确定性 | 每次修改后都运行训练+推理验证 |
| **对比验证** | 用已知正确的实现验证新实现 | 无 Cache vs Block Cache 结果对比 |
| **根因分析** | 不满足于表面修复，追查根本原因 | 从 Block Attention 追查到原始 KV Cache 的 bug |

### 6.3 工作流

```
┌─────────────────────────────────────────────────┐
│  1. 需求分析                                     │
│     └─ 用自然语言描述目标（"实现 Block Attention"）│
├─────────────────────────────────────────────────┤
│  2. 知识准备                                     │
│     └─ 阅读相关文档、代码（理解原理和现有实现）    │
├─────────────────────────────────────────────────┤
│  3. 方案设计                                     │
│     └─ 讨论多种方案，选择最简可行方案             │
│     └─ 在 PLAN MODE 中与 AI 讨论                 │
├─────────────────────────────────────────────────┤
│  4. 编码实现                                     │
│     └─ AI 生成代码，人工审查                      │
│     └─ 保持代码简洁、可读、有注释                 │
├─────────────────────────────────────────────────┤
│  5. 运行验证                                     │
│     └─ 运行训练 + 推理，检查结果                  │
│     └─ 对比已知正确的结果                         │
├─────────────────────────────────────────────────┤
│  6. Bug 排查                                     │
│     └─ 发现不一致 → 定位根因 → 修复 → 验证       │
│     └─ 从现象追溯到根因，不放过任何异常           │
├─────────────────────────────────────────────────┤
│  7. 知识沉淀                                     │
│     └─ 记录过程、总结方法（即本文档）             │
└─────────────────────────────────────────────────┘
```

### 6.4 常见陷阱与应对

| 陷阱 | 表现 | 应对方法 |
|------|------|----------|
| **AI 生成代码过于复杂** | 包含不必要的抽象、泛化 | 明确要求"极简实现"，在 PLAN MODE 中讨论简化方案 |
| **增量路径 bug** | 全量计算正确，增量计算错误 | 设计对比验证（无 Cache vs 有 Cache） |
| **位置编码偏移** | 增量推理时位置从 0 开始 | 始终从 cache 长度计算位置偏移 |
| **掩码裁剪错误** | mask 维度与 K/V 长度不匹配 | 统一使用 `cache_len + L` 计算总长度 |
| **层间状态共享** | 不同层错误地共享同一状态 | 每层创建独立的 BlockManager / KV Cache |
| **训练/推理模式混淆** | 训练时错误地使用了 cache | 训练时传入 `block_manager=None` |

### 6.5 适用场景

Vibe Coding 最适合：

1. **原型验证**：快速验证想法是否可行
2. **学习探索**：通过编码加深对原理的理解
3. **工具脚本**：一次性或小规模的工具代码
4. **算法实现**：有明确输入输出的算法

不适合：

1. **生产系统**：需要严格的测试、性能、安全要求
2. **遗留代码维护**：需要深入理解现有系统的复杂交互
3. **安全敏感代码**：加密、认证等需要专业审计

---

> **最后更新**: 2026-06-15
> **文件位置**: `ai/demo/vibe_coding_journal.md`