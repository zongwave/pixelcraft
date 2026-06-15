"""
Block Attention Decoder-Only Transformer 极简示例
==================================================
在 decoder_only_demo.py 的基础上，对 KV Cache 实现 Block Attention（PagedAttention 风格的内存分块）：
  - 预分配静态内存池，避免动态 torch.cat
  - BlockManager 只维护状态（空闲/已用），不管理内存分配释放
  - 推理时通过 block table 索引物理 block，gather 出连续 K/V 做 attention

运行方式：
    python block_attention_demo.py

依赖：torch >= 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


# ============================================================
# 1. RMSNorm — 归一化层（LLaMA 风格，与原始 demo 相同）
# ============================================================
class RMSNorm(nn.Module):
    """RMSNorm: y = x / RMS(x) * gamma"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.gamma


# ============================================================
# 2. SwiGLU FFN — 门控前馈网络（与原始 demo 相同）
# ============================================================
class SwiGLU_FFN(nn.Module):
    """SwiGLU FFN: FFN(x) = [SiLU(x @ W_up) * (x @ W_gate)] @ W_down"""
    def __init__(self, dim: int, hidden_multiple: int = 4):
        super().__init__()
        hidden_dim = int(2 * dim * hidden_multiple / 3)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        hidden = gate * up
        return self.w_down(hidden)


# ============================================================
# 3. BlockManager — 静态内存池管理器
# ============================================================
class BlockManager:
    """
    Block 管理器：管理 KV Cache 的静态内存池。

    核心思想：
      - 在初始化时一次性分配固定大小的物理内存池（k_cache, v_cache）
      - 推理过程中只做读写，不做动态内存分配/释放
      - 通过 free_blocks 列表和 block_table 维护状态

    物理内存布局：
        k_cache: [num_blocks, block_size, dim]  — 静态池
        v_cache: [num_blocks, block_size, dim]  — 静态池

    状态管理：
        free_blocks: List[int] — 空闲物理 block 索引栈
        block_table: List[int] — 逻辑 block i → 物理 block 索引
        current_length: int    — 当前已存储的 token 数
    """
    def __init__(self, num_blocks: int, block_size: int, dim: int, device: torch.device):
        """
        参数:
            num_blocks: 物理 block 总数
            block_size: 每个 block 可存储的 token 数
            dim: 隐藏维度
            device: 设备
        """
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.device = device

        # === 静态内存池（一次性分配，永不增长） ===
        self.k_cache = torch.zeros(num_blocks, block_size, dim, device=device)
        self.v_cache = torch.zeros(num_blocks, block_size, dim, device=device)

        # === 状态管理 ===
        self.free_blocks = list(range(num_blocks))  # 空闲 block 索引
        self.block_table: list[int] = []             # 逻辑 block → 物理 block
        self.current_length = 0                      # 当前序列长度

    def reset(self):
        """重置状态（清空 block_table，回收所有 block）"""
        self.free_blocks = list(range(self.num_blocks))
        self.block_table.clear()
        self.current_length = 0
        # 注意：不重置 k_cache/v_cache 内容，因为下次使用时会被覆盖

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """
        追加一个 token 的 K/V 到静态池中。

        参数:
            k: [dim] — 当前 token 的 K
            v: [dim] — 当前 token 的 V
        """
        pos = self.current_length
        block_idx = pos // self.block_size
        offset = pos % self.block_size

        # 如果需要新 block，从空闲列表中分配
        if offset == 0:
            assert self.free_blocks, f"Block 池已耗尽！num_blocks={self.num_blocks}, current_length={self.current_length}"
            phys_block = self.free_blocks.pop(0)
            self.block_table.append(phys_block)

        # 写入静态池
        phys_block = self.block_table[block_idx]
        self.k_cache[phys_block, offset] = k
        self.v_cache[phys_block, offset] = v
        self.current_length += 1

    def gather_kv(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        从静态池中 gather 出当前序列所有 K/V（连续张量）。

        返回:
            k: [current_length, dim] — 连续的 K
            v: [current_length, dim] — 连续的 V
        """
        if self.current_length == 0:
            return torch.empty(0, self.k_cache.shape[-1], device=self.device), \
                   torch.empty(0, self.v_cache.shape[-1], device=self.device)

        k_parts = []
        v_parts = []
        num_full_blocks = self.current_length // self.block_size
        remainder = self.current_length % self.block_size

        for i in range(num_full_blocks):
            phys = self.block_table[i]
            k_parts.append(self.k_cache[phys])
            v_parts.append(self.v_cache[phys])

        if remainder > 0:
            phys = self.block_table[num_full_blocks]
            k_parts.append(self.k_cache[phys, :remainder])
            v_parts.append(self.v_cache[phys, :remainder])

        return torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0)

    @property
    def num_tokens(self) -> int:
        return self.current_length


# ============================================================
# 4. BlockKVAttention — 使用 Block 分页的因果自注意力
# ============================================================
class BlockKVAttention(nn.Module):
    """
    单头因果自注意力 + Block 分页 KV Cache。

    与原始 CausalSelfAttention 的区别：
      - KV Cache 使用 BlockManager 管理静态内存池
      - 推理时通过 gather_kv() 获取连续 K/V 做 attention
      - 每次前向后将新 token 的 K/V 通过 append() 写入静态池
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        block_manager: BlockManager | None = None,
    ) -> tuple[torch.Tensor, BlockManager | None]:
        """
        参数:
            x: [B, L, D] — 输入序列
            mask: [L, L_total] — 因果掩码
            block_manager: 推理时传入的 BlockManager
        返回:
            out: [B, L, D] — 注意力输出
            block_manager: 更新后的 BlockManager
        """
        B, L, D = x.shape

        # 1) 投影 Q, K, V
        q = self.q_proj(x)  # [B, L, D]
        k = self.k_proj(x)  # [B, L, D]
        v = self.v_proj(x)  # [B, L, D]

        # 2) Block KV Cache 处理
        if block_manager is not None:
            # 将当前 token 的 K/V 写入静态池
            # 注意：推理时 L=1（增量推理），训练时 L>1
            for t in range(L):
                block_manager.append(k[0, t], v[0, t])

            # 从静态池 gather 出完整 K/V
            k_full, v_full = block_manager.gather_kv()  # [L_total, D]
            k_full = k_full.unsqueeze(0)  # [1, L_total, D]
            v_full = v_full.unsqueeze(0)  # [1, L_total, D]
        else:
            # 训练模式：不使用 block cache，直接使用当前 K/V
            k_full = k
            v_full = v

        # 3) 计算注意力分数
        scale = 1.0 / math.sqrt(D)
        attn_scores = torch.matmul(q, k_full.transpose(-2, -1)) * scale

        # 4) 因果掩码
        if mask is not None:
            attn_scores = attn_scores + mask

        # 5) Softmax + 加权求和
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v_full)

        # 6) 输出投影
        out = self.out_proj(out)
        return out, block_manager


# ============================================================
# 5. BlockDecoderBlock — 使用 BlockKVAttention 的 Decoder 层
# ============================================================
class BlockDecoderBlock(nn.Module):
    """
    Decoder Block（Pre-Norm 结构）:
        x = x + BlockAttention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))
    """
    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = BlockKVAttention(dim)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU_FFN(dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        block_manager: BlockManager | None = None,
    ) -> tuple[torch.Tensor, BlockManager | None]:
        attn_out, block_manager = self.attn(self.norm1(x), mask, block_manager)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, block_manager


# ============================================================
# 6. BlockAttentionDecoderOnlyTransformer — 完整模型
# ============================================================
class BlockAttentionDecoderOnlyTransformer(nn.Module):
    """
    Decoder-Only Transformer + Block Attention:
        Token Embedding → Position Embedding → [BlockDecoderBlock × N] → LM Head

    与原始 DecoderOnlyTransformer 的区别：
      - 使用 BlockManager 管理 KV Cache 静态内存池
      - 推理时通过 block table 索引物理 block
    """
    def __init__(
        self,
        vocab_size: int = 100,
        dim: int = 64,
        num_layers: int = 4,
        max_seq_len: int = 32,
        block_size: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.block_size = block_size
        self.num_blocks = (max_seq_len + block_size - 1) // block_size  # 向上取整

        # Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, dim)
        # 可学习位置编码
        self.pos_embedding = nn.Embedding(max_seq_len, dim)

        # Decoder 层堆叠
        self.layers = nn.ModuleList([
            BlockDecoderBlock(dim) for _ in range(num_layers)
        ])

        # 最终归一化
        self.norm = RMSNorm(dim)

        # LM Head（权重绑定）
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        # 因果掩码
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.full((max_seq_len, max_seq_len), float('-inf')), diagonal=1)
        )

    def _create_block_managers(self, device: torch.device) -> list[BlockManager]:
        """为每层创建 BlockManager（共享同一静态内存池配置）"""
        return [
            BlockManager(self.num_blocks, self.block_size, self.dim, device)
            for _ in range(len(self.layers))
        ]

    def forward(
        self,
        tokens: torch.Tensor,
        block_managers: list[BlockManager] | None = None,
    ) -> tuple[torch.Tensor, list[BlockManager] | None]:
        """
        参数:
            tokens: [B, L] — 输入 token IDs
            block_managers: 每层的 BlockManager 列表（推理时使用）
        返回:
            logits: [B, L, vocab_size]
            block_managers: 更新后的 BlockManager 列表
        """
        B, L = tokens.shape
        device = tokens.device

        # 1) Token Embedding
        x = self.token_embedding(tokens)

        # 2) 位置编码（增量推理时使用正确的偏移量）
        cache_len = 0
        if block_managers is not None and block_managers[0] is not None:
            cache_len = block_managers[0].num_tokens
        pos = torch.arange(cache_len, cache_len + L, device=device).unsqueeze(0)
        x = x + self.pos_embedding(pos)

        # 3) 因果掩码裁剪
        total_len = cache_len + L
        mask = self.causal_mask[cache_len:total_len, :total_len]

        # 4) 逐层前向传播
        new_block_managers = block_managers
        for i, layer in enumerate(self.layers):
            layer_bm = block_managers[i] if block_managers else None
            x, new_bm = layer(x, mask, layer_bm)
            if new_block_managers is not None:
                new_block_managers[i] = new_bm

        # 5) 最终归一化
        x = self.norm(x)

        # 6) LM Head → logits
        logits = self.lm_head(x)
        return logits, new_block_managers

    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 20,
        use_block_cache: bool = True,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        自回归生成（支持 Block Attention）。

        参数:
            prompt: [B, L_prompt] — 初始 prompt
            max_new_tokens: 最大生成 token 数
            use_block_cache: 是否使用 Block Cache
            temperature: 采样温度
        返回:
            [B, L_prompt + max_new_tokens] — 完整生成序列
        """
        self.eval()
        B = prompt.shape[0]
        device = prompt.device

        if use_block_cache:
            # === Block Cache 模式：增量推理 ===
            generated = prompt
            block_managers = self._create_block_managers(device)

            with torch.no_grad():
                logits, block_managers = self.forward(prompt, block_managers)
                next_logits = logits[:, -1, :]

            for _ in range(max_new_tokens):
                if temperature > 0:
                    probs = F.softmax(next_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = next_logits.argmax(dim=-1, keepdim=True)

                generated = torch.cat([generated, next_token], dim=1)

                with torch.no_grad():
                    logits, block_managers = self.forward(next_token, block_managers)
                    next_logits = logits[:, -1, :]

        else:
            # === 无 Cache 模式：每次重新计算全部 ===
            generated = prompt
            for _ in range(max_new_tokens):
                if generated.shape[1] > self.max_seq_len:
                    ctx = generated[:, -self.max_seq_len:]
                else:
                    ctx = generated

                with torch.no_grad():
                    logits, _ = self.forward(ctx)
                    next_logits = logits[:, -1, :]

                if temperature > 0:
                    probs = F.softmax(next_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = next_logits.argmax(dim=-1, keepdim=True)

                generated = torch.cat([generated, next_token], dim=1)

        return generated


# ============================================================
# 7. 玩具数据集：等差数列预测（与原始 demo 相同）
# ============================================================
def make_arithmetic_dataset(
    vocab_size: int = 100,
    seq_len: int = 16,
    num_samples: int = 2000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """生成等差数列数据集。"""
    inputs = []
    targets = []
    for _ in range(num_samples):
        a = torch.randint(0, vocab_size // 4, (1,)).item()
        d = torch.randint(1, vocab_size // seq_len, (1,)).item()
        seq = torch.tensor([(a + i * d) % vocab_size for i in range(seq_len + 1)])
        inputs.append(seq[:-1])
        targets.append(seq[1:])
    return torch.stack(inputs), torch.stack(targets)


# ============================================================
# 8. 训练 + 推理演示
# ============================================================
def demo():
    import sys
    import io
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 60)
    print("Block Attention Decoder-Only Transformer Demo")
    print("=" * 60)

    # ---------- 超参数 ----------
    VOCAB_SIZE = 100
    DIM = 64
    NUM_LAYERS = 4
    MAX_SEQ_LEN = 32
    BLOCK_SIZE = 4          # 每个 block 存储 4 个 token 的 K/V
    SEQ_LEN = 16
    BATCH_SIZE = 32
    LR = 3e-4
    NUM_EPOCHS = 50
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n[配置] 设备: {DEVICE}, 隐藏维: {DIM}, 层数: {NUM_LAYERS}")
    print(f"[配置] 词表: {VOCAB_SIZE}, 序列长: {SEQ_LEN}, 批次: {BATCH_SIZE}")
    print(f"[配置] Block 大小: {BLOCK_SIZE}, Block 总数: {(MAX_SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE}")

    # ---------- 数据 ----------
    train_inputs, train_targets = make_arithmetic_dataset(
        VOCAB_SIZE, SEQ_LEN, num_samples=2000
    )
    dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    print(f"[数据] 训练样本数: {len(dataset)}")

    # ---------- 模型 ----------
    model = BlockAttentionDecoderOnlyTransformer(
        vocab_size=VOCAB_SIZE,
        dim=DIM,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        block_size=BLOCK_SIZE,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[模型] 参数量: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # ---------- 训练 ----------
    print("\n" + "=" * 60)
    print("开始训练（Next Token Prediction）")
    print("=" * 60)

    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        num_batches = 0

        for batch_inputs, batch_targets in dataloader:
            batch_inputs = batch_inputs.to(DEVICE)
            batch_targets = batch_targets.to(DEVICE)

            logits, _ = model(batch_inputs)

            loss = F.cross_entropy(
                logits.view(-1, VOCAB_SIZE),
                batch_targets.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{NUM_EPOCHS} | Loss: {avg_loss:.4f}")

    print(f"\n训练完成！最终 Loss: {avg_loss:.4f}")

    # ---------- 推理演示 ----------
    print("\n" + "=" * 60)
    print("推理演示：自回归生成")
    print("=" * 60)

    prompt = torch.tensor([[2, 4, 6, 8]], device=DEVICE)
    print(f"\nPrompt: {prompt[0].tolist()} (期望: 等差数列, 步长=2)")

    # 1) 无 Cache 推理
    print("\n[1] 无 Cache 推理:")
    start = time.time()
    output_no_cache = model.generate(
        prompt, max_new_tokens=8, use_block_cache=False, temperature=0.0
    )
    time_no_cache = time.time() - start
    print(f"    生成: {output_no_cache[0].tolist()}")
    print(f"    耗时: {time_no_cache*1000:.1f} ms")

    # 2) Block Cache 推理
    print("\n[2] Block Cache 推理:")
    start = time.time()
    output_with_cache = model.generate(
        prompt, max_new_tokens=8, use_block_cache=True, temperature=0.0
    )
    time_with_cache = time.time() - start
    print(f"    生成: {output_with_cache[0].tolist()}")
    print(f"    耗时: {time_with_cache*1000:.1f} ms")

    # 3) 结果一致性验证
    print("\n[3] 结果一致性验证:")
    output_no = output_no_cache[0].tolist()
    output_yes = output_with_cache[0].tolist()
    match = output_no == output_yes
    print(f"    无 Cache: {output_no}")
    print(f"    Block Cache: {output_yes}")
    print(f"    一致: {'✅' if match else '❌'}")

    # 4) 随机采样生成
    print("\n[4] 随机采样生成 (temperature=0.8):")
    for i in range(3):
        output = model.generate(
            prompt, max_new_tokens=8, use_block_cache=True, temperature=0.8
        )
        print(f"    第{i+1}次: {output[0].tolist()}")

    # 5) 速度对比
    print("\n" + "-" * 40)
    print("速度对比（生成 20 个 token，重复 10 次取平均）:")
    print("-" * 40)

    # 无 Cache
    times = []
    for _ in range(10):
        start = time.time()
        model.generate(prompt, max_new_tokens=20, use_block_cache=False, temperature=0.0)
        times.append(time.time() - start)
    avg_no_cache = sum(times) / len(times)
    print(f"  无 Cache: {avg_no_cache*1000:.1f} ms (平均)")

    # Block Cache
    times = []
    for _ in range(10):
        start = time.time()
        model.generate(prompt, max_new_tokens=20, use_block_cache=True, temperature=0.0)
        times.append(time.time() - start)
    avg_with_cache = sum(times) / len(times)
    print(f"  Block Cache: {avg_with_cache*1000:.1f} ms (平均)")
    print(f"  加速比: {avg_no_cache / avg_with_cache:.1f}×")

    # 6) Block 内存使用统计
    print("\n" + "-" * 40)
    print("Block 内存使用统计:")
    print("-" * 40)
    num_blocks_total = (MAX_SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
    cache_size_per_layer = num_blocks_total * BLOCK_SIZE * DIM * 2 * 4  # 2 for K/V, 4 for float32
    total_cache_size = cache_size_per_layer * NUM_LAYERS
    print(f"  每层 Block 数: {num_blocks_total}")
    print(f"  每层 Cache 大小: {cache_size_per_layer / 1024:.1f} KB")
    print(f"  总 Cache 大小: {total_cache_size / 1024:.1f} KB")
    print(f"  (对比: 原始 KV Cache 最大 {MAX_SEQ_LEN * DIM * 2 * 4 / 1024:.1f} KB/层)")

    # ---------- 模型结构打印 ----------
    print("\n" + "=" * 60)
    print("模型结构")
    print("=" * 60)
    print(model)

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == '__main__':
    demo()