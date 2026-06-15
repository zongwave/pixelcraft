"""
Mixture of Experts (MoE) 极简示例 (v2 — 端到端训练版)
=====================================================
从零实现一个 Sparse MoE 层，参考 DeepSeek V3 的设计思路：

改进点（v1 → v2）:
  1. Sigmoid Router（替代 Softmax）— 每个 expert 独立打分，互不竞争
  2. e_score_correction_bias — 可学习的 expert 偏置，实现负载均衡
  3. Shared Expert — 共享 MLP，所有 token 都经过，提供基础能力
  4. 端到端训练 — Router 和 Expert 一起通过 MSE Loss 训练
  5. 分组路由 — 简化版 grouped top-k，保证负载均衡

运行方式：
    python moe_demo.py

依赖：torch >= 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


# ============================================================
# 1. Expert MLP — 单个 expert 的 SwiGLU FFN
# ============================================================
class ExpertMLP(nn.Module):
    """
    SwiGLU FFN: FFN(x) = [SiLU(x @ W_up) * (x @ W_gate)] @ W_down

    与 LLaMA 风格一致，每个 expert 有独立的 up/gate/down 三个权重矩阵。
    """
    def __init__(self, dim: int, hidden_multiple: int = 4):
        super().__init__()
        # SwiGLU 的中间维度 = 2/3 * dim * hidden_multiple
        hidden_dim = int(2 * dim * hidden_multiple / 3)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: [num_tokens, dim] — 分配到该 expert 的 token
        返回:
            output: [num_tokens, dim] — expert 输出
        """
        gate = F.silu(self.w_gate(x))   # [num_tokens, hidden_dim]
        up = self.w_up(x)               # [num_tokens, hidden_dim]
        hidden = gate * up              # [num_tokens, hidden_dim] 门控
        output = self.w_down(hidden)    # [num_tokens, dim]
        return output


# ============================================================
# 2. Router (Gate) — Sigmoid 门控 + Top-K 选择（参考 DeepSeek V3）
# ============================================================
class Router(nn.Module):
    """
    Router 负责为每个 token 选择最合适的 expert。

    与 v1 的关键区别：
      - 使用 Sigmoid 替代 Softmax：每个 expert 独立打分，互不竞争
      - 添加 e_score_correction_bias：可学习的 expert 偏置，用于负载均衡
      - 支持分组路由：先选组，再在组内选 expert

    流程:
      1. gate_out = x @ gate_weight          — 线性投影到 expert 空间
      2. scores = sigmoid(gate_out)           — 每个 expert 独立打分 [0, 1]
      3. scores_for_choice = scores + bias    — 加偏置用于选择
      4. (可选) 分组路由: 先选 top-k 组，再在组内选 expert
      5. topk_indices = topk(scores_for_choice, K)  — 选 Top-K
      6. topk_weights = scores.gather(indices)      — 用原始 sigmoid 分数作为权重
    """
    def __init__(self, dim: int, num_experts: int, top_k: int = 2,
                 n_group: int = None, topk_group: int = None,
                 norm_topk_prob: bool = True):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob

        # 分组路由配置（参考 DeepSeek V3）
        self.n_group = n_group
        self.topk_group = topk_group

        # Gate 权重: [num_experts, dim]
        self.gate_weight = nn.Parameter(torch.randn(num_experts, dim) * 0.02)

        # e_score_correction_bias: 可学习的 expert 偏置（参考 DeepSeek V3）
        # 用于路由选择时的负载均衡
        self.e_score_correction_bias = nn.Parameter(torch.zeros(num_experts))

    def forward(self, x: torch.Tensor):
        """
        参数:
            x: [num_tokens, dim] — 输入 hidden states
        返回:
            routing_weights:  [num_tokens, top_k] — 选中的 expert 权重（sigmoid 分数）
            selected_experts: [num_tokens, top_k] — 选中的 expert 索引 (int64)
            scores:           [num_tokens, num_experts] — 完整 sigmoid 分数（用于分析）
        """
        # 1. 线性投影（用 float32 保证精度）
        gate_out = F.linear(x.float(), self.gate_weight)  # [num_tokens, num_experts]

        # 2. Sigmoid 归一化（每个 expert 独立打分）
        scores = torch.sigmoid(gate_out)  # [num_tokens, num_experts]

        # 3. 加偏置用于选择（参考 DeepSeek V3）
        scores_for_choice = scores + self.e_score_correction_bias

        # 4. (可选) 分组路由
        if self.n_group is not None and self.topk_group is not None:
            # 将 expert 分成 n_group 组
            experts_per_group = self.num_experts // self.n_group
            # 计算每组得分：每组取 top-2 expert 的分数之和
            group_scores = (
                scores_for_choice.view(-1, self.n_group, experts_per_group)
                .topk(2, dim=-1)[0]
                .sum(dim=-1)
            )  # [num_tokens, n_group]
            # 选 top-k 组
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            # 构建 expert 级别的 mask
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(-1, self.n_group, experts_per_group)
                .reshape(-1, self.num_experts)
            )
            # 未选中的组设为 -inf
            scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))

        # 5. Top-K 选择（用加偏置后的分数选择）
        routing_weights, selected_experts = torch.topk(
            scores_for_choice, self.top_k, dim=-1
        )  # 均为 [num_tokens, top_k]

        # 6. 用原始 sigmoid 分数作为权重（参考 DeepSeek V3）
        routing_weights = scores.gather(1, selected_experts)

        # 7. (可选) 归一化 top-k 权重
        if self.norm_topk_prob:
            routing_weights = routing_weights / (
                routing_weights.sum(dim=-1, keepdim=True) + 1e-20
            )

        return routing_weights, selected_experts, scores


# ============================================================
# 3. Sparse MoE — 完整的稀疏混合专家层（参考 DeepSeek V3）
# ============================================================
class SparseMoE(nn.Module):
    """
    完整的 Sparse MoE 层，包含 Router + Expert 计算 + 聚合 + Shared Expert。

    与 v1 的关键区别：
      - 添加 Shared Expert：所有 token 都经过共享 MLP，提供基础能力
      - 端到端可训练：Sigmoid Router 的梯度可以正常传播

    前向流程:
      1. Router: 计算 routing_weights 和 selected_experts
      2. Dispatch: 构建 expert_mask，将 token 分配到各 expert
      3. Expert FFN: 每个 expert 独立计算分配的 token
      4. Aggregate: 加权聚合所有 expert 的输出
      5. Shared Expert: 所有 token 经过共享 MLP
      6. 合并: output = routed_output + shared_output

    参数:
      dim: int              — 输入/输出维度
      num_experts: int      — expert 总数
      top_k: int            — 每个 token 激活的 expert 数
      hidden_multiple: int  — FFN 中间层倍数
      norm_topk_prob: bool  — 是否归一化 top-k 权重
      n_group: int          — 分组数（None 表示不分组）
      topk_group: int       — 选中的组数
      shared_expert: bool   — 是否使用共享 expert
    """
    def __init__(self, dim: int, num_experts: int, top_k: int = 2,
                 hidden_multiple: int = 4, norm_topk_prob: bool = True,
                 n_group: int = None, topk_group: int = None,
                 shared_expert: bool = True):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k

        # Router
        self.router = Router(dim, num_experts, top_k, n_group, topk_group, norm_topk_prob)

        # Expert 列表
        self.experts = nn.ModuleList([
            ExpertMLP(dim, hidden_multiple)
            for _ in range(num_experts)
        ])

        # Shared Expert（参考 DeepSeek V3）
        self.shared_expert = None
        if shared_expert:
            shared_hidden_dim = int(2 * dim * hidden_multiple / 3)
            self.shared_expert = nn.Sequential(
                nn.Linear(dim, shared_hidden_dim, bias=False),
                nn.SiLU(),
                nn.Linear(shared_hidden_dim, dim, bias=False),
            )

    def forward(self, x: torch.Tensor):
        """
        参数:
            x: [num_tokens, dim] — 输入
        返回:
            output: [num_tokens, dim] — MoE 输出
            aux_info: dict — 辅助信息（用于分析路由行为）
        """
        num_tokens = x.shape[0]

        # ============================================================
        # Step 1: Router — 计算路由权重和 expert 选择
        # ============================================================
        routing_weights, selected_experts, scores = self.router(x)
        # routing_weights:  [num_tokens, top_k]
        # selected_experts: [num_tokens, top_k]
        # scores:           [num_tokens, num_experts]

        # ============================================================
        # Step 2: Dispatch — 构建 expert mask，分配 token
        # ============================================================
        # expert_mask: [num_tokens, top_k, num_experts] — one-hot
        expert_mask = F.one_hot(
            selected_experts, num_classes=self.num_experts
        ).float()  # [num_tokens, top_k, num_experts]

        # 转置为 [num_experts, top_k, num_tokens]
        expert_mask = expert_mask.permute(2, 1, 0)  # [num_experts, top_k, num_tokens]

        # ============================================================
        # Step 3: Expert FFN — 每个 expert 独立计算
        # ============================================================
        output = torch.zeros_like(x)  # [num_tokens, dim]

        # 记录每个 expert 处理的 token 数（用于分析负载均衡）
        tokens_per_expert = torch.zeros(self.num_experts, dtype=torch.long)

        for expert_idx in range(self.num_experts):
            # 找出分配到该 expert 的 token
            mask = expert_mask[expert_idx]  # [top_k, num_tokens]
            idx, top_x = torch.where(mask > 0)

            if idx.numel() == 0:
                continue

            tokens_per_expert[expert_idx] = top_x.numel()

            # 取出 token 和对应权重
            current_state = x[top_x]  # [num_assigned, dim]
            current_weights = routing_weights[top_x, idx]  # [num_assigned]

            # Expert 前向计算
            expert_output = self.experts[expert_idx](current_state)  # [num_assigned, dim]

            # 加权
            expert_output = expert_output * current_weights.unsqueeze(-1)

            # 累加到最终输出
            output.index_add_(0, top_x, expert_output)

        # ============================================================
        # Step 4: Shared Expert — 所有 token 经过共享 MLP
        # ============================================================
        if self.shared_expert is not None:
            shared_output = self.shared_expert(x)  # [num_tokens, dim]
            output = output + shared_output

        # ============================================================
        # Step 5: 收集辅助信息
        # ============================================================
        aux_info = {
            'routing_weights': routing_weights,       # [num_tokens, top_k]
            'selected_experts': selected_experts,      # [num_tokens, top_k]
            'scores': scores,                          # [num_tokens, num_experts]
            'tokens_per_expert': tokens_per_expert,    # [num_experts]
        }

        return output, aux_info


# ============================================================
# 4. 合成数据集生成
# ============================================================
def generate_mixture_of_functions(num_samples: int, dim: int, num_experts: int):
    """
    生成合成数据集，让不同 expert 学习不同的函数模式。

    设计思路:
      - 输入 x 的每个维度从不同的分布中采样，分布类型由 dominant_expert 决定
      - 不同 expert 对应的输入分布不同，使 Router 可以学会区分
      - 不同 expert 学习不同的函数

    参数:
        num_samples: int  — 样本数
        dim: int          — 特征维度
        num_experts: int  — expert 数
    返回:
        x: [num_samples, dim] — 输入特征
        y: [num_samples, dim] — 目标输出
        dominant_expert: [num_samples] — 每个样本对应的主导 expert 索引
    """
    # 为每个样本随机选择主导 expert
    dominant_expert = torch.randint(0, num_experts, (num_samples,))

    # 生成输入: 不同 expert 对应不同的输入分布
    x = torch.zeros(num_samples, dim)

    for i in range(num_samples):
        expert_id = dominant_expert[i].item()

        if expert_id == 0:
            x[i] = torch.rand(dim) * 4.0
        elif expert_id == 1:
            x[i] = -torch.rand(dim) * 4.0
        elif expert_id == 2:
            x[i] = torch.randn(dim) * 2.0
        elif expert_id == 3:
            x[i] = torch.randn(dim) * 0.5
        elif expert_id == 4:
            x[i] = torch.rand(dim) * 4.0 + 2.0
        elif expert_id == 5:
            x[i] = -torch.rand(dim) * 4.0 - 2.0
        elif expert_id == 6:
            x[i] = torch.randn(dim) * 4.0
        elif expert_id == 7:
            x[i] = torch.rand(dim) * 2.0
        else:
            x[i] = torch.randn(dim) * 2.0

    # 构建基函数
    y = torch.zeros(num_samples, dim)

    for i in range(num_samples):
        expert_id = dominant_expert[i].item()

        if expert_id == 0:
            y[i] = torch.sin(x[i])
        elif expert_id == 1:
            y[i] = torch.cos(x[i])
        elif expert_id == 2:
            y[i] = x[i] ** 2 / 4.0
        elif expert_id == 3:
            y[i] = torch.abs(x[i]) / 2.0
        else:
            y[i] = x[i] * (0.5 + 0.1 * expert_id)

    return x, y, dominant_expert


# ============================================================
# 5. 端到端训练
# ============================================================

def train_moe_end_to_end(model: SparseMoE, num_epochs: int = 200,
                         batch_size: int = 64, lr: float = 1e-3,
                         verbose: bool = True):
    """
    端到端训练 MoE（参考 DeepSeek V3 的训练方式）

    Router 和 Expert 一起通过 MSE Loss 训练。
    Sigmoid Router 的梯度可以正常传播（没有离散操作的梯度中断问题）。
    """
    dim = model.dim
    num_experts = model.num_experts

    # 生成训练/测试数据
    num_train = 16384  # 增加数据量，缓解过拟合
    num_test = 2048
    x_train, y_train, dominant_train = generate_mixture_of_functions(
        num_train, dim, num_experts
    )
    x_test, y_test, dominant_test = generate_mixture_of_functions(
        num_test, dim, num_experts
    )

    # 优化所有参数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = {'train_loss': [], 'test_loss': [], 'routing_accuracy': []}

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        perm = torch.randperm(num_train)
        for start in range(0, num_train, batch_size):
            idx = perm[start:start + batch_size]
            x_batch = x_train[idx]
            y_batch = y_train[idx]

            optimizer.zero_grad()
            y_pred, aux_info = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        # 测试
        model.eval()
        with torch.no_grad():
            y_test_pred, test_aux = model(x_test)
            test_loss = loss_fn(y_test_pred, y_test).item()

            # 计算路由准确率
            selected = test_aux['selected_experts']  # [num_test, top_k]
            correct = (selected == dominant_test.unsqueeze(-1)).any(dim=-1)
            routing_accuracy = correct.float().mean().item()

        avg_loss = epoch_loss / num_batches
        history['train_loss'].append(avg_loss)
        history['test_loss'].append(test_loss)
        history['routing_accuracy'].append(routing_accuracy)

        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1:3d}/{num_epochs} | "
                  f"Train Loss: {avg_loss:.6f} | "
                  f"Test Loss: {test_loss:.6f} | "
                  f"Route Acc: {routing_accuracy:.2%}")

    return history


# ============================================================
# 6. 推理验证
# ============================================================
def run_inference_demo(model: SparseMoE, dense_ffn: nn.Module = None):
    """
    运行推理演示，展示 MoE 的完整处理流程。
    """
    dim = model.dim
    num_experts = model.num_experts
    top_k = model.top_k

    print("\n" + "=" * 70)
    print("MoE 推理演示")
    print("=" * 70)

    # ============================================================
    # 6.1 单样本推理 — 详细打印每一步
    # ============================================================
    print("\n" + "-" * 70)
    print("6.1 单样本推理 — 详细流程")
    print("-" * 70)

    x_single, y_single, dominant = generate_mixture_of_functions(1, dim, num_experts)

    model.eval()
    with torch.no_grad():
        y_pred, aux_info = model(x_single)

    print(f"\n输入: x = {x_single[0, :4].tolist()}... (前 4 维)")
    print(f"目标: y = {y_single[0, :4].tolist()}... (前 4 维)")
    print(f"输出: y_pred = {y_pred[0, :4].tolist()}... (前 4 维)")
    print(f"MSE: {F.mse_loss(y_pred, y_single).item():.6f}")

    # 打印路由信息
    routing_weights = aux_info['routing_weights'][0]  # [top_k]
    selected_experts = aux_info['selected_experts'][0]  # [top_k]
    scores = aux_info['scores'][0]  # [num_experts]

    print(f"\n路由信息:")
    print(f"  完整 sigmoid 分数: {[f'{w:.4f}' for w in scores.tolist()]}")
    print(f"  选中的 expert: {selected_experts.tolist()}")
    print(f"  对应权重: {[f'{w:.4f}' for w in routing_weights.tolist()]}")
    print(f"  主导 expert (真实): {dominant[0].item()}")

    # ============================================================
    # 6.2 批量推理 — 展示路由分布
    # ============================================================
    print("\n" + "-" * 70)
    print("6.2 批量推理 — 路由分布分析")
    print("-" * 70)

    num_test = 256
    x_batch, y_batch, dominant_batch = generate_mixture_of_functions(
        num_test, dim, num_experts
    )

    with torch.no_grad():
        y_batch_pred, batch_aux = model(x_batch)

    # 统计每个 expert 被选中的次数
    selected = batch_aux['selected_experts']  # [num_test, top_k]
    expert_counts = torch.zeros(num_experts, dtype=torch.long)
    for k in range(top_k):
        for e in range(num_experts):
            expert_counts[e] += (selected[:, k] == e).sum()

    print(f"\nExpert 负载分布 (共 {num_test} 个 token, top_k={top_k}):")
    total_assignments = expert_counts.sum().item()
    for e in range(num_experts):
        pct = expert_counts[e].item() / total_assignments * 100
        bar = '█' * int(pct / 2) + '░' * (50 - int(pct / 2))
        print(f"  Expert {e}: {expert_counts[e].item():4d} 次 ({pct:5.1f}%) {bar}")

    # 计算路由准确率
    correct = (selected == dominant_batch.unsqueeze(-1)).any(dim=-1)
    accuracy = correct.float().mean().item()
    print(f"\n路由准确率: {accuracy:.2%}")

    # 计算测试集 MSE
    test_mse = F.mse_loss(y_batch_pred, y_batch).item()
    print(f"测试集 MSE: {test_mse:.6f}")

    # ============================================================
    # 6.3 与 Dense 模型对比
    # ============================================================
    if dense_ffn is not None:
        print("\n" + "-" * 70)
        print("6.3 与 Dense 模型对比")
        print("-" * 70)

        dense_ffn.eval()
        with torch.no_grad():
            y_dense_pred = dense_ffn(x_batch)
            dense_mse = F.mse_loss(y_dense_pred, y_batch).item()

        # 计算参数量
        moe_total_params = sum(p.numel() for p in model.parameters())
        moe_active_params = (
            model.router.gate_weight.numel() +  # Router 参数
            model.router.e_score_correction_bias.numel() +  # Bias 参数
            model.top_k * sum(p.numel() for p in model.experts[0].parameters())  # top_k 个 expert
        )
        # 共享 expert 也算在激活参数量中
        if model.shared_expert is not None:
            moe_active_params += sum(p.numel() for p in model.shared_expert.parameters())
        dense_params = sum(p.numel() for p in dense_ffn.parameters())

        print(f"\n对比结果:")
        print(f"  MoE   MSE: {test_mse:.6f}")
        print(f"  Dense MSE: {dense_mse:.6f}")
        print(f"  MoE 总参数量: {moe_total_params:,}")
        print(f"  MoE 激活参数量 (top_k={top_k}): {moe_active_params:,}")
        print(f"  Dense 参数量: {dense_params:,}")

        if test_mse < dense_mse:
            print(f"\n✅ MoE 优于 Dense (MSE 更低)")
        else:
            print(f"\n⚠️  Dense 优于 MoE")

    return {
        'test_mse': test_mse,
        'routing_accuracy': accuracy,
        'expert_counts': expert_counts,
    }


# ============================================================
# 7. 构建 Dense 对比模型
# ============================================================
def build_dense_model(dim: int, num_experts: int, top_k: int,
                      expert_hidden_dim: int, shared_expert: bool = True):
    """
    构建 Dense 模型，使其激活参数量与 MoE 的激活参数量接近。

    MoE 激活参数量 = Router 参数 + Bias 参数 + top_k 个 Expert 参数 + (可选) Shared Expert 参数
    """
    # MoE 激活参数量
    moe_active_params = (
        dim * num_experts +  # Router
        num_experts +        # e_score_correction_bias
        top_k * 3 * dim * expert_hidden_dim  # top_k 个 expert
    )
    if shared_expert:
        moe_active_params += 3 * dim * expert_hidden_dim  # Shared Expert

    # Dense 的 hidden_dim 使总参数量接近 MoE 激活参数量
    dense_hidden_dim = moe_active_params // (3 * dim)
    dense_hidden_dim = max(dense_hidden_dim, dim)

    dense_ffn = nn.Sequential(
        nn.Linear(dim, dense_hidden_dim, bias=False),
        nn.SiLU(),
        nn.Linear(dense_hidden_dim, dim, bias=False),
    )

    return dense_ffn, dense_hidden_dim


def train_dense_model(dense_ffn: nn.Module, dim: int, num_experts: int,
                      num_epochs: int = 200, batch_size: int = 64,
                      lr: float = 1e-3, verbose: bool = True):
    """
    训练 Dense 模型，使用与 MoE 相同的数据和迭代次数。
    """
    num_train = 16384
    x_train, y_train, _ = generate_mixture_of_functions(
        num_train, dim, num_experts
    )

    optimizer = torch.optim.Adam(dense_ffn.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = {'loss': []}

    for epoch in range(num_epochs):
        dense_ffn.train()
        epoch_loss = 0.0
        num_batches = 0

        perm = torch.randperm(num_train)
        for start in range(0, num_train, batch_size):
            idx = perm[start:start + batch_size]
            x_batch = x_train[idx]
            y_batch = y_train[idx]

            optimizer.zero_grad()
            y_pred = dense_ffn(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        history['loss'].append(avg_loss)

        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1:3d}/{num_epochs} | Loss: {avg_loss:.6f}")

    return history


# ============================================================
# 8. 主函数
# ============================================================
def main():
    print("=" * 70)
    print("Mixture of Experts (MoE) 极简示例 v2 — 端到端训练版")
    print("=" * 70)

    # 配置
    dim = 32
    num_experts = 8
    top_k = 2
    hidden_multiple = 4
    num_epochs = 200
    use_shared_expert = True
    use_grouped_routing = True  # 启用分组路由

    # 分组路由配置
    n_group = 4 if use_grouped_routing else None
    topk_group = 2 if use_grouped_routing else None

    print(f"\n配置:")
    print(f"  输入维度: {dim}")
    print(f"  Expert 数: {num_experts}")
    print(f"  Top-K: {top_k}")
    print(f"  训练轮数: {num_epochs}")
    print(f"  共享 Expert: {use_shared_expert}")
    print(f"  分组路由: {use_grouped_routing} (组数={n_group}, 选中组数={topk_group})")

    # 创建模型
    model = SparseMoE(
        dim=dim,
        num_experts=num_experts,
        top_k=top_k,
        hidden_multiple=hidden_multiple,
        norm_topk_prob=True,
        n_group=n_group,
        topk_group=topk_group,
        shared_expert=use_shared_expert,
    )

    # 计算参数量
    expert_hidden_dim = int(2 * dim * hidden_multiple / 3)
    moe_total_params = sum(p.numel() for p in model.parameters())
    moe_active_params = (
        dim * num_experts +  # Router
        num_experts +        # e_score_correction_bias
        top_k * 3 * dim * expert_hidden_dim  # top_k 个 expert
    )
    if use_shared_expert:
        moe_active_params += 3 * dim * expert_hidden_dim  # Shared Expert

    print(f"\n模型参数量:")
    print(f"  MoE 总参数量: {moe_total_params:,}")
    print(f"  MoE 激活参数量 (top_k={top_k}): {moe_active_params:,}")

    # ============================================================
    # 端到端训练
    # ============================================================
    print(f"\n{'=' * 70}")
    print("端到端训练 MoE（Router + Expert 一起训练）")
    print(f"{'=' * 70}")

    start_time = time.time()
    history = train_moe_end_to_end(
        model, num_epochs=num_epochs, verbose=True
    )
    train_time = time.time() - start_time

    print(f"\n训练完成! 耗时: {train_time:.2f}s")
    print(f"最终训练 Loss: {history['train_loss'][-1]:.6f}")
    print(f"最终测试 Loss: {history['test_loss'][-1]:.6f}")
    print(f"最终路由准确率: {history['routing_accuracy'][-1]:.2%}")

    # ============================================================
    # 训练 Dense 对比模型
    # ============================================================
    print(f"\n{'=' * 70}")
    print("训练 Dense 对比模型")
    print(f"{'=' * 70}")

    dense_ffn, dense_hidden_dim = build_dense_model(
        dim, num_experts, top_k, expert_hidden_dim, use_shared_expert
    )
    dense_params = sum(p.numel() for p in dense_ffn.parameters())

    print(f"\nDense 模型参数量: {dense_params:,}")
    print(f"Dense hidden_dim: {dense_hidden_dim}")

    start_time = time.time()
    dense_history = train_dense_model(
        dense_ffn, dim, num_experts,
        num_epochs=num_epochs, verbose=True
    )
    dense_time = time.time() - start_time

    print(f"\nDense 训练完成! 耗时: {dense_time:.2f}s")
    print(f"最终 Loss: {dense_history['loss'][-1]:.6f}")

    # ============================================================
    # 推理演示
    # ============================================================
    results = run_inference_demo(model, dense_ffn)

    print("\n" + "=" * 70)
    print("MoE Demo v2 运行完毕!")
    print("=" * 70)


if __name__ == "__main__":
    main()