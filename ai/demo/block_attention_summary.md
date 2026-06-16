# Block Attention — 设计技术原理与实现总结

## 概述

本文档总结 `block_attention_demo.py` 的设计技术原理与实现细节。该 demo 在 Decoder-Only Transformer 的基础上，实现了 Block Attention（PagedAttention 风格的内存分块 KV Cache），用于理解如何通过静态内存池管理来优化长序列推理的内存和性能。

---

## 1. 设计动机

### 1.1 原始 KV Cache 的问题

在 `decoder_only_demo.py` 中，KV Cache 使用 `torch.cat` 动态拼接：

```python
k = torch.cat([kv_cache['k'], k], dim=1)  # 每次生成新 block
v = torch.cat([kv_cache['v'], v], dim=1)
```

**问题**：
1. **动态内存分配**：每次 `torch.cat` 都会分配新的内存块，产生碎片
2. **不可预测的延迟**：内存分配耗时不确定，影响推理延迟稳定性
3. **无法预分配**：无法提前知道最大序列长度，难以做内存规划
4. **批量推理困难**：不同序列长度不同，动态拼接导致形状不一致

### 1.2 Block Attention 的解决思路

参考 PagedAttention（vLLM 的核心技术），将 KV Cache 分块管理：

```
传统 KV Cache:     [K₁, K₂, K₃, K₄, K₅, K₆, ...]  连续内存
Block Attention:   Block 0: [K₁, K₂, K₃, K₄]        静态池
                   Block 1: [K₅, K₆, K₇, K₈]        静态池
                   Block 2: [K₉, K₁₀, ...]           静态池
```

---

## 2. 架构设计

### 2.1 整体结构

```
BlockManager（静态内存池管理器）
    ├── k_cache: [num_blocks, block_size, dim]   ← 一次性分配
    ├── v_cache: [num_blocks, block_size, dim]   ← 一次性分配
    ├── free_blocks: List[int]                    ← 空闲 block 索引
    ├── block_table: List[int]                    ← 逻辑→物理映射
    └── current_length: int                       ← 当前序列长度

BlockKVAttention（使用 Block 分页的注意力）
    ├── Q/K/V 投影（与标准 Attention 相同）
    ├── BlockManager 读写（推理时）
    └── gather_kv() → 连续 K/V → 标准 Attention 计算
```

### 2.2 与原始 Decoder-Only 的差异

| 组件 | 原始 Demo | Block Attention Demo |
|------|-----------|---------------------|
| KV Cache 类型 | 动态 `torch.cat` | 静态内存池 + BlockManager |
| 内存分配 | 每次推理动态分配 | 初始化时一次性分配 |
| 状态管理 | 无 | free_blocks + block_table |
| 注意力计算 | 直接使用拼接后的 K/V | gather 出连续 K/V 再计算 |
| 训练模式 | 不使用 Cache | 不使用 Cache（与原始相同） |

---

## 3. BlockManager 核心实现

### 3.1 静态内存池

```python
# 一次性分配，永不增长
self.k_cache = torch.zeros(num_blocks, block_size, dim)
self.v_cache = torch.zeros(num_blocks, block_size, dim)
```

**物理内存布局**：
```
k_cache: [Block 0: [K₁, K₂, K₃, K₄],
          Block 1: [K₅, K₆, K₇, K₈],
          Block 2: [K₉, K₁₀, K₁₁, K₁₂],
          ...]
```

### 3.2 状态管理

| 状态 | 类型 | 说明 |
|------|------|------|
| `free_blocks` | `List[int]` | 空闲物理 block 索引栈（初始化时包含所有 block） |
| `block_table` | `List[int]` | 逻辑 block i → 物理 block 索引的映射表 |
| `current_length` | `int` | 当前已存储的 token 数 |

**状态流转**：
```
初始化: free_blocks = [0, 1, 2, ..., N-1], block_table = [], current_length = 0

写入 token 0:  block_idx=0, offset=0 → 分配 phys=0, block_table=[0], 写入 k_cache[0,0]
写入 token 1:  block_idx=0, offset=1 → 直接写入 k_cache[0,1]
写入 token 2:  block_idx=0, offset=2 → 直接写入 k_cache[0,2]
写入 token 3:  block_idx=0, offset=3 → 直接写入 k_cache[0,3]
写入 token 4:  block_idx=1, offset=0 → 分配 phys=1, block_table=[0,1], 写入 k_cache[1,0]
...
```

### 3.3 核心操作

**append(k, v)** — 写入一个 token 的 K/V：
```python
pos = current_length
block_idx = pos // block_size    # 逻辑 block 索引
offset = pos % block_size        # block 内偏移

if offset == 0:                  # 新 block 起始
    phys_block = free_blocks.pop(0)   # 从空闲池分配
    block_table.append(phys_block)    # 记录映射

phys_block = block_table[block_idx]
k_cache[phys_block, offset] = k   # 写入静态池
v_cache[phys_block, offset] = v
current_length += 1
```

**gather_kv()** — 从静态池 gather 出连续 K/V：
```python
for i in range(num_full_blocks):
    phys = block_table[i]
    k_parts.append(k_cache[phys])        # 完整 block
    v_parts.append(v_cache[phys])

if remainder > 0:
    phys = block_table[num_full_blocks]
    k_parts.append(k_cache[phys, :remainder])  # 部分 block

return torch.cat(k_parts), torch.cat(v_parts)
```

**reset()** — 重置状态：
```python
free_blocks = list(range(num_blocks))  # 回收所有 block
block_table.clear()
current_length = 0
# 注意：不重置 k_cache/v_cache 内容，下次使用时会被覆盖
```

---

## 4. BlockKVAttention 实现

### 4.1 前向计算

```python
def forward(self, x, mask=None, block_manager=None):
    # 1) 投影 Q, K, V（与标准 Attention 相同）
    q = self.q_proj(x)
    k = self.k_proj(x)
    v = self.v_proj(x)

    # 2) Block KV Cache 处理
    if block_manager is not None:
        # 将当前 token 的 K/V 写入静态池
        for t in range(L):
            block_manager.append(k[0, t], v[0, t])
        # 从静态池 gather 出完整 K/V
        k_full, v_full = block_manager.gather_kv()
    else:
        # 训练模式：不使用 Cache
        k_full, v_full = k, v

    # 3) 标准 Attention 计算
    attn_scores = Q @ K_full^T / sqrt(dim)
    attn_scores = attn_scores + mask
    out = softmax(attn_scores) @ V_full
    return out, block_manager
```

### 4.2 与标准 Attention 的对比

| 方面 | 标准 Attention | BlockKVAttention |
|------|---------------|------------------|
| 训练模式 | 直接使用当前 K/V | 直接使用当前 K/V（相同） |
| 推理模式 | `torch.cat` 拼接历史 | `append()` 写入 + `gather_kv()` 读取 |
| 内存分配 | 每次推理动态分配 | 初始化时一次性分配 |
| 数据流 | K/V 连续存储 | K/V 分块存储，gather 时连续化 |

---

## 5. 训练与推理

### 5.1 训练模式

训练时 `block_manager=None`，BlockKVAttention 退化为标准 Attention：

```python
logits, _ = model(batch_inputs)  # 训练时不传 block_managers
```

### 5.2 推理模式

推理时创建 BlockManager 并传入：

```python
# 初始化 BlockManager
block_managers = [
    BlockManager(num_blocks, block_size, dim, device)
    for _ in range(num_layers)
]

# 第一步：处理 prompt
logits, block_managers = model(prompt, block_managers)

# 后续步：增量推理
for _ in range(max_new_tokens):
    logits, block_managers = model(next_token, block_managers)
```

### 5.3 结果一致性验证

Demo 中包含无 Cache 与 Block Cache 的结果一致性验证：

```python
output_no_cache = model.generate(prompt, use_block_cache=False)
output_with_cache = model.generate(prompt, use_block_cache=True)
assert output_no_cache == output_with_cache  # ✅ 一致
```

---

## 6. 内存分析

### 6.1 内存占用对比

以 demo 配置（dim=64, block_size=4, max_seq_len=32, layers=4）为例：

| 方案 | 每层 Cache 大小 | 总 Cache 大小 |
|------|-----------------|---------------|
| 原始 KV Cache（动态） | 32 × 64 × 2 × 4 = 16 KB | 64 KB |
| Block Cache（静态池） | 8 × 4 × 64 × 2 × 4 = 16 KB | 64 KB |

**两者内存占用相同**，但 Block Cache 的优势在于：
- 内存是**预分配**的，推理过程中无动态分配
- 可以支持**批量推理**中不同序列长度的对齐
- 物理 block 可以**跨序列共享**（PagedAttention 的核心优势，本 demo 未展示）

### 6.2 Block 大小的影响

| Block Size | Block 数 | 内存碎片 | gather 开销 | 适用场景 |
|------------|----------|----------|-------------|----------|
| 小（1~2） | 多 | 少 | 大 | 短序列，精确控制 |
| 中（4~16） | 中 | 中 | 中 | 通用场景 |
| 大（32~64） | 少 | 大 | 小 | 长序列，高吞吐 |

---

## 7. 与 PagedAttention 的对比

| 特性 | 本 Demo | PagedAttention（vLLM） |
|------|---------|----------------------|
| 核心思想 | 静态内存池 + BlockManager | 虚拟内存分页 |
| Block 大小 | 固定（可配置） | 固定（通常 16） |
| 物理 block 共享 | 不支持 | 支持（同一序列内） |
| 跨序列 block 共享 | 不支持 | 支持（beam search） |
| 内存分配策略 | 预分配全部 | 按需分配 + 物理页管理 |
| 注意力计算 | gather 出连续 K/V | 直接使用分块 K/V（block-sparse attention） |

**本 Demo 的定位**：展示 Block Attention 最核心的思想——**静态内存池 + 逻辑到物理的映射**，不涉及 PagedAttention 中更复杂的 block 共享和 block-sparse attention 计算。

---

## 8. 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 训练时不用 Block Cache | 退化为标准 Attention | 训练时序列短，Cache 无意义 |
| 单头注意力 | 单头 | 与原始 demo 保持一致 |
| gather 后做标准 Attention | gather + 标准计算 | 简化实现，聚焦 Block 管理 |
| 不实现 block 共享 | 不共享 | 保持 demo 简洁 |
| 不重置 cache 内容 | 仅重置状态 | 减少不必要的内存操作 |

---

## 9. 与完整文档的对应关系

| 本 Demo | 完整文档 | 说明 |
|---------|----------|------|
| BlockManager | `ai/llm/inference_opt/fused_block_attention_summary.md` | 静态内存池管理 |
| BlockKVAttention | `ai/llm/inference_opt/fused_block_attention_summary.md` | 分块 KV Cache 注意力 |
| gather_kv | `ai/transformer/llm-inference-systems.md` | 逻辑→物理地址转换 |
| PagedAttention 概念 | `ai/llm/inference_opt/fused_block_attention_summary.md` | vLLM 核心技术 |