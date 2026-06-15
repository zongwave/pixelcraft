"""
Decoder-Only Transformer 极简示例
==================================
从零实现一个 Decoder-Only Transformer，包含：
  - RMSNorm、CausalSelfAttention、SwiGLU_FFN、DecoderBlock
  - 训练循环（Next Token Prediction）
  - 推理循环（自回归生成 + KV Cache 演示）

运行方式：
    python decoder_only_demo.py

依赖：torch >= 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


# ============================================================
# 1. RMSNorm — 归一化层（LLaMA 风格）
# ============================================================
class RMSNorm(nn.Module):
    """
    RMSNorm: y = x / RMS(x) * gamma
    相比 LayerNorm 省去了减均值操作，计算量约减少 15%。
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))  # 可学习缩放参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.gamma


# ============================================================
# 2. Causal Self-Attention — 因果自注意力（单头，便于理解）
# ============================================================
class CausalSelfAttention(nn.Module):
    """
    单头因果自注意力：
      - Q, K, V 来自同一序列
      - 因果掩码确保 token i 只能看到 token <= i
      - 推理时支持 KV Cache 增量计算
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Q, K, V 投影（单头，所以 d_q = d_k = d_v = dim）
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        kv_cache: dict | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        参数:
            x: [B, L, D] — 输入序列
            mask: [L, L] — 因果掩码（上三角 -inf）
            kv_cache: 推理时传入的 KV 缓存
                {'k': [B, L_cache, D], 'v': [B, L_cache, D]}
        返回:
            out: [B, L, D] — 注意力输出
            new_kv_cache: 更新后的 KV 缓存
        """
        B, L, D = x.shape

        # 1) 投影 Q, K, V
        q = self.q_proj(x)  # [B, L, D]
        k = self.k_proj(x)  # [B, L, D]
        v = self.v_proj(x)  # [B, L, D]

        # 2) KV Cache 处理（推理时使用）
        if kv_cache is not None:
            # 拼接历史 K, V
            k = torch.cat([kv_cache['k'], k], dim=1)  # [B, L_cache + L, D]
            v = torch.cat([kv_cache['v'], v], dim=1)  # [B, L_cache + L, D]

        new_kv_cache = {'k': k, 'v': v}

        # 3) 计算注意力分数
        #    score = Q @ K^T / sqrt(d_k)
        scale = 1.0 / math.sqrt(D)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, L, L_total]

        # 4) 因果掩码
        if mask is not None:
            # mask 形状 [L, L_total]，只对当前 L 做掩码
            attn_scores = attn_scores + mask

        # 5) Softmax + 加权求和
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, L, L_total]
        out = torch.matmul(attn_weights, v)             # [B, L, D]

        # 6) 输出投影
        out = self.out_proj(out)
        return out, new_kv_cache


# ============================================================
# 3. SwiGLU FFN — 门控前馈网络
# ============================================================
class SwiGLU_FFN(nn.Module):
    """
    SwiGLU FFN:
        FFN(x) = [SiLU(x @ W_up) * (x @ W_gate)] @ W_down
    相比 ReLU-FFN，门控机制提升表达力。
    参数补偿：hidden_dim = (2/3) * 4d 以保持参数量不变。
    """
    def __init__(self, dim: int, hidden_multiple: int = 4):
        super().__init__()
        # SwiGLU 需要 3 个权重矩阵（up, gate, down）
        # 为保持参数量 ≈ 标准 FFN，hidden_dim 取 (2/3) * 4d
        hidden_dim = int(2 * dim * hidden_multiple / 3)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        gate = F.silu(self.w_gate(x))       # SiLU 门控
        up = self.w_up(x)                    # 升维
        hidden = gate * up                   # 逐元素相乘（门控机制）
        return self.w_down(hidden)           # 降维回 D


# ============================================================
# 4. DecoderBlock — 单个 Decoder 层
# ============================================================
class DecoderBlock(nn.Module):
    """
    Decoder Block（Pre-Norm 结构）:
        x = x + Attention(RMSNorm(x))    # 子层 1
        x = x + FFN(RMSNorm(x))          # 子层 2
    Pre-Norm 使梯度直接流过残差路径，训练更稳定。
    """
    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU_FFN(dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        kv_cache: dict | None = None,
    ) -> tuple[torch.Tensor, dict]:
        # 子层 1: 自注意力 + 残差
        attn_out, new_kv_cache = self.attn(self.norm1(x), mask, kv_cache)
        x = x + attn_out

        # 子层 2: FFN + 残差
        x = x + self.ffn(self.norm2(x))
        return x, new_kv_cache


# ============================================================
# 5. DecoderOnlyTransformer — 完整模型
# ============================================================
class DecoderOnlyTransformer(nn.Module):
    """
    Decoder-Only Transformer:
        Token Embedding → Position Embedding → [DecoderBlock × N] → LM Head

    参数量约: N * (4*d^2 + 3*d*hidden + 2*d) + 2*vocab*d
    其中 N=层数, d=隐藏维, hidden=SwiGLU 隐藏维, vocab=词表大小
    """
    def __init__(
        self,
        vocab_size: int = 100,
        dim: int = 64,
        num_layers: int = 4,
        max_seq_len: int = 32,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, dim)
        # 可学习位置编码（简化版，实际可用 RoPE）
        self.pos_embedding = nn.Embedding(max_seq_len, dim)

        # Decoder 层堆叠
        self.layers = nn.ModuleList([
            DecoderBlock(dim) for _ in range(num_layers)
        ])

        # 最终归一化
        self.norm = RMSNorm(dim)

        # LM Head（输出投影到词表）
        # 权重绑定：与 token_embedding 共享权重
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        # 因果掩码（上三角矩阵，-inf 表示不可见）
        # 注册为 buffer，不参与梯度计算
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.full((max_seq_len, max_seq_len), float('-inf')), diagonal=1)
        )

    def forward(
        self,
        tokens: torch.Tensor,
        kv_caches: list[dict] | None = None,
    ) -> tuple[torch.Tensor, list[dict]]:
        """
        参数:
            tokens: [B, L] — 输入 token IDs
            kv_caches: 每层的 KV 缓存列表（推理时使用）
        返回:
            logits: [B, L, vocab_size] — 每个位置的预测 logits
            new_kv_caches: 更新后的 KV 缓存列表
        """
        B, L = tokens.shape
        device = tokens.device

        # 1) Token Embedding
        x = self.token_embedding(tokens)  # [B, L, D]

        # 2) 位置编码（增量推理时使用正确的偏移量）
        cache_len = 0
        if kv_caches is not None and kv_caches[0] is not None:
            cache_len = kv_caches[0]['k'].shape[1]
        pos = torch.arange(cache_len, cache_len + L, device=device).unsqueeze(0)  # [1, L]
        x = x + self.pos_embedding(pos)                     # [B, L, D]

        # 3) 因果掩码裁剪
        #    如果使用 KV Cache，总序列长度 = cache_len + L
        total_len = cache_len + L
        mask = self.causal_mask[cache_len:total_len, :total_len]  # [L, total_len]

        # 4) 逐层前向传播
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            layer_kv = kv_caches[i] if kv_caches else None
            x, new_kv = layer(x, mask, layer_kv)
            new_kv_caches.append(new_kv)

        # 5) 最终归一化
        x = self.norm(x)  # [B, L, D]

        # 6) LM Head → logits
        logits = self.lm_head(x)  # [B, L, vocab_size]
        return logits, new_kv_caches

    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 20,
        use_kv_cache: bool = True,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        自回归生成。

        参数:
            prompt: [B, L_prompt] — 初始 prompt
            max_new_tokens: 最大生成 token 数
            use_kv_cache: 是否使用 KV Cache 加速
            temperature: 采样温度（>0, 越小越确定）
        返回:
            [B, L_prompt + max_new_tokens] — 完整生成序列
        """
        self.eval()
        B = prompt.shape[0]
        device = prompt.device

        if use_kv_cache:
            # === KV Cache 模式：增量推理 ===
            # 第一步：处理 prompt（构建 KV Cache）
            generated = prompt
            kv_caches = [None] * len(self.layers)
            with torch.no_grad():
                logits, kv_caches = self.forward(prompt, kv_caches)
                next_logits = logits[:, -1, :]  # 取最后一个 token 的 logits

            # 后续步：每次只生成 1 个 token
            for _ in range(max_new_tokens):
                # 采样
                if temperature > 0:
                    probs = F.softmax(next_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
                else:
                    next_token = next_logits.argmax(dim=-1, keepdim=True)  # [B, 1]

                generated = torch.cat([generated, next_token], dim=1)

                # 增量推理：只传入最新 token
                with torch.no_grad():
                    logits, kv_caches = self.forward(next_token, kv_caches)
                    next_logits = logits[:, -1, :]

        else:
            # === 无 KV Cache 模式：每次重新计算全部 ===
            generated = prompt
            for _ in range(max_new_tokens):
                # 截断到 max_seq_len
                if generated.shape[1] > self.max_seq_len:
                    ctx = generated[:, -self.max_seq_len:]
                else:
                    ctx = generated

                with torch.no_grad():
                    logits, _ = self.forward(ctx)
                    next_logits = logits[:, -1, :]

                # 采样
                if temperature > 0:
                    probs = F.softmax(next_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = next_logits.argmax(dim=-1, keepdim=True)

                generated = torch.cat([generated, next_token], dim=1)

        return generated


# ============================================================
# 6. 玩具数据集：等差数列预测
# ============================================================
def make_arithmetic_dataset(
    vocab_size: int = 100,
    seq_len: int = 16,
    num_samples: int = 2000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    生成等差数列数据集。
    任务：给定序列 [a, a+d, a+2d, ...]，预测下一个数。

    例如：vocab_size=100, seq_len=8
        输入: [3, 6, 9, 12, 15, 18, 21, 24]
        目标: [6, 9, 12, 15, 18, 21, 24, 27]

    返回:
        inputs: [num_samples, seq_len]
        targets: [num_samples, seq_len]
    """
    inputs = []
    targets = []
    for _ in range(num_samples):
        # 随机选择起始值和步长
        a = torch.randint(0, vocab_size // 4, (1,)).item()
        d = torch.randint(1, vocab_size // seq_len, (1,)).item()

        # 生成序列
        seq = torch.tensor([(a + i * d) % vocab_size for i in range(seq_len + 1)])
        inputs.append(seq[:-1])
        targets.append(seq[1:])

    return torch.stack(inputs), torch.stack(targets)


# ============================================================
# 7. 训练 + 推理演示
# ============================================================
def demo():
    import sys
    import io
    # 确保 stdout 支持 UTF-8
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 60)
    print("Decoder-Only Transformer Demo")
    print("=" * 60)

    # ---------- 超参数 ----------
    VOCAB_SIZE = 100       # 词表大小（数字 0-99）
    DIM = 64               # 隐藏维度
    NUM_LAYERS = 4         # Decoder 层数
    MAX_SEQ_LEN = 32       # 最大序列长度
    SEQ_LEN = 16           # 训练序列长度
    BATCH_SIZE = 32
    LR = 3e-4
    NUM_EPOCHS = 50
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n[配置] 设备: {DEVICE}, 隐藏维: {DIM}, 层数: {NUM_LAYERS}")
    print(f"[配置] 词表: {VOCAB_SIZE}, 序列长: {SEQ_LEN}, 批次: {BATCH_SIZE}")

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
    model = DecoderOnlyTransformer(
        vocab_size=VOCAB_SIZE,
        dim=DIM,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
    ).to(DEVICE)

    # 统计参数量
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
            batch_inputs = batch_inputs.to(DEVICE)    # [B, L]
            batch_targets = batch_targets.to(DEVICE)  # [B, L]

            # 前向传播
            logits, _ = model(batch_inputs)  # [B, L, vocab_size]

            # 计算损失（交叉熵）
            loss = F.cross_entropy(
                logits.view(-1, VOCAB_SIZE),
                batch_targets.view(-1),
            )

            # 反向传播
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

    # 构造 prompt: [2, 4, 6, 8] → 期望生成 [10, 12, 14, ...]
    prompt = torch.tensor([[2, 4, 6, 8]], device=DEVICE)
    print(f"\nPrompt: {prompt[0].tolist()} (期望: 等差数列, 步长=2)")

    # 1) 无 KV Cache 推理
    print("\n[1] 无 KV Cache 推理:")
    start = time.time()
    output_no_cache = model.generate(
        prompt, max_new_tokens=8, use_kv_cache=False, temperature=0.0
    )
    time_no_cache = time.time() - start
    print(f"    生成: {output_no_cache[0].tolist()}")
    print(f"    耗时: {time_no_cache*1000:.1f} ms")

    # 2) 有 KV Cache 推理
    print("\n[2] 有 KV Cache 推理:")
    start = time.time()
    output_with_cache = model.generate(
        prompt, max_new_tokens=8, use_kv_cache=True, temperature=0.0
    )
    time_with_cache = time.time() - start
    print(f"    生成: {output_with_cache[0].tolist()}")
    print(f"    耗时: {time_with_cache*1000:.1f} ms")

    # 3) 随机采样生成（展示多样性）
    print("\n[3] 随机采样生成 (temperature=0.8):")
    for i in range(3):
        output = model.generate(
            prompt, max_new_tokens=8, use_kv_cache=True, temperature=0.8
        )
        print(f"    第{i+1}次: {output[0].tolist()}")

    # 4) 速度对比
    print("\n" + "-" * 40)
    print("速度对比（生成 20 个 token，重复 10 次取平均）:")
    print("-" * 40)

    # 无 KV Cache
    times = []
    for _ in range(10):
        start = time.time()
        model.generate(prompt, max_new_tokens=20, use_kv_cache=False, temperature=0.0)
        times.append(time.time() - start)
    avg_no_cache = sum(times) / len(times)
    print(f"  无 KV Cache: {avg_no_cache*1000:.1f} ms (平均)")

    # 有 KV Cache
    times = []
    for _ in range(10):
        start = time.time()
        model.generate(prompt, max_new_tokens=20, use_kv_cache=True, temperature=0.0)
        times.append(time.time() - start)
    avg_with_cache = sum(times) / len(times)
    print(f"  有 KV Cache: {avg_with_cache*1000:.1f} ms (平均)")
    print(f"  加速比: {avg_no_cache / avg_with_cache:.1f}×")

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