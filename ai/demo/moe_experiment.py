"""
MoE 实验脚本 — 对比不同激活专家比例的效果
===========================================
实验目的：验证激活专家与总专家比例对 MoE 效果的影响。

实验配置：
  A. 8 experts, top_k=2 (比例=1/4) — 当前 v2 配置
  B. 8 experts, top_k=4 (比例=1/2) — 提高激活比例
  C. 4 experts, top_k=2 (比例=1/2) — 减少总专家数，保持比例
  D. Dense 模型（参数量对齐各 MoE 配置）

所有配置保持 MoE 激活参数量 ≈ Dense 参数量。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import os

# 添加父目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_demo import (
    SparseMoE, ExpertMLP, Router,
    generate_mixture_of_functions,
    train_moe_end_to_end, train_dense_model,
    build_dense_model, run_inference_demo
)


def run_experiment(config_name: str, num_experts: int, top_k: int,
                   n_group: int = None, topk_group: int = None,
                   num_epochs: int = 200, verbose: bool = True):
    """
    运行单个 MoE 实验配置。

    参数:
        config_name: str — 配置名称
        num_experts: int — expert 总数
        top_k: int — 每个 token 激活的 expert 数
        n_group: int — 分组数
        topk_group: int — 选中的组数
        num_epochs: int — 训练轮数
        verbose: bool — 是否打印详细信息
    返回:
        dict — 实验结果
    """
    dim = 32
    hidden_multiple = 4
    expert_hidden_dim = int(2 * dim * hidden_multiple / 3)  # = 85
    use_shared_expert = True

    print(f"\n{'=' * 70}")
    print(f"实验: {config_name}")
    print(f"{'=' * 70}")
    print(f"  Expert 数: {num_experts}, Top-K: {top_k}, 激活比例: {top_k}/{num_experts} = {top_k/num_experts:.2%}")
    if n_group:
        print(f"  分组: {n_group} 组, 选 {topk_group} 组")

    # 创建 MoE 模型
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
    moe_total_params = sum(p.numel() for p in model.parameters())
    moe_active_params = (
        dim * num_experts +  # Router
        num_experts +        # e_score_correction_bias
        top_k * 3 * dim * expert_hidden_dim +  # top_k 个 expert
        3 * dim * expert_hidden_dim  # Shared Expert
    )

    print(f"\n  MoE 总参数量: {moe_total_params:,}")
    print(f"  MoE 激活参数量 (top_k={top_k}): {moe_active_params:,}")

    # 训练 MoE
    start_time = time.time()
    history = train_moe_end_to_end(
        model, num_epochs=num_epochs, verbose=verbose
    )
    moe_time = time.time() - start_time

    final_train_loss = history['train_loss'][-1]
    final_test_loss = history['test_loss'][-1]
    final_route_acc = history['routing_accuracy'][-1]

    print(f"\n  MoE 训练完成! 耗时: {moe_time:.2f}s")
    print(f"  最终训练 Loss: {final_train_loss:.6f}")
    print(f"  最终测试 Loss: {final_test_loss:.6f}")
    print(f"  最终路由准确率: {final_route_acc:.2%}")

    # 构建并训练 Dense 模型
    dense_ffn, dense_hidden_dim = build_dense_model(
        dim, num_experts, top_k, expert_hidden_dim, use_shared_expert
    )
    dense_params = sum(p.numel() for p in dense_ffn.parameters())

    print(f"\n  Dense 模型参数量: {dense_params:,}")
    print(f"  Dense hidden_dim: {dense_hidden_dim}")

    start_time = time.time()
    dense_history = train_dense_model(
        dense_ffn, dim, num_experts,
        num_epochs=num_epochs, verbose=verbose
    )
    dense_time = time.time() - start_time

    final_dense_loss = dense_history['loss'][-1]
    print(f"\n  Dense 训练完成! 耗时: {dense_time:.2f}s")
    print(f"  最终 Loss: {final_dense_loss:.6f}")

    # 推理验证
    num_test = 256
    x_test, y_test, dominant_test = generate_mixture_of_functions(
        num_test, dim, num_experts
    )

    model.eval()
    dense_ffn.eval()
    with torch.no_grad():
        y_moe_pred, moe_aux = model(x_test)
        y_dense_pred = dense_ffn(x_test)

        moe_mse = F.mse_loss(y_moe_pred, y_test).item()
        dense_mse = F.mse_loss(y_dense_pred, y_test).item()

        # 路由准确率
        selected = moe_aux['selected_experts']
        correct = (selected == dominant_test.unsqueeze(-1)).any(dim=-1)
        route_acc = correct.float().mean().item()

        # 负载分布
        expert_counts = torch.zeros(num_experts, dtype=torch.long)
        for k in range(top_k):
            for e in range(num_experts):
                expert_counts[e] += (selected[:, k] == e).sum()

    print(f"\n  推理结果:")
    print(f"    MoE   MSE: {moe_mse:.6f}")
    print(f"    Dense MSE: {dense_mse:.6f}")
    print(f"    路由准确率: {route_acc:.2%}")
    print(f"    Expert 负载: {expert_counts.tolist()}")

    moe_wins = moe_mse < dense_mse
    print(f"    {'✅ MoE 优于 Dense' if moe_wins else '⚠️  Dense 优于 MoE'}")

    return {
        'config_name': config_name,
        'num_experts': num_experts,
        'top_k': top_k,
        'activation_ratio': top_k / num_experts,
        'n_group': n_group,
        'topk_group': topk_group,
        'moe_total_params': moe_total_params,
        'moe_active_params': moe_active_params,
        'dense_params': dense_params,
        'moe_train_loss': final_train_loss,
        'moe_test_loss': final_test_loss,
        'dense_loss': final_dense_loss,
        'moe_mse': moe_mse,
        'dense_mse': dense_mse,
        'route_acc': route_acc,
        'moe_wins': moe_wins,
        'moe_time': moe_time,
        'dense_time': dense_time,
        'expert_counts': expert_counts.tolist(),
        'train_loss_history': history['train_loss'],
        'test_loss_history': history['test_loss'],
        'route_acc_history': history['routing_accuracy'],
        'dense_loss_history': dense_history['loss'],
    }


def print_comparison_table(results: list):
    """打印所有实验的对比表格"""
    print("\n" + "=" * 100)
    print("实验结果对比汇总")
    print("=" * 100)

    header = f"{'配置':<20} | {'Expert':<8} | {'Top-K':<6} | {'比例':<8} | {'MoE总参':<10} | {'MoE激活':<10} | {'Dense参':<10} | {'MoE MSE':<10} | {'Dense MSE':<10} | {'路由准确率':<10} | {'胜出':<10}"
    sep = "-" * 100

    print(header)
    print(sep)

    for r in results:
        winner = "MoE ✅" if r['moe_wins'] else "Dense ⚠️"
        print(f"{r['config_name']:<20} | "
              f"{r['num_experts']:<8} | "
              f"{r['top_k']:<6} | "
              f"{r['activation_ratio']:.0%}    | "
              f"{r['moe_total_params']:<10,} | "
              f"{r['moe_active_params']:<10,} | "
              f"{r['dense_params']:<10,} | "
              f"{r['moe_mse']:<10.4f} | "
              f"{r['dense_mse']:<10.4f} | "
              f"{r['route_acc']:<8.2%}  | "
              f"{winner:<10}")

    print(sep)
    print()

    # 打印详细对比
    print("详细对比:")
    print("-" * 100)
    for r in results:
        print(f"\n{r['config_name']}:")
        print(f"  Expert={r['num_experts']}, Top-K={r['top_k']}, 比例={r['activation_ratio']:.0%}")
        print(f"  MoE  MSE: {r['moe_mse']:.4f}  |  Dense MSE: {r['dense_mse']:.4f}  |  "
              f"差距: {abs(r['moe_mse'] - r['dense_mse']):.4f} ({'MoE优' if r['moe_wins'] else 'Dense优'})")
        print(f"  路由准确率: {r['route_acc']:.2%}  |  Expert 负载: {r['expert_counts']}")


def main():
    print("=" * 70)
    print("MoE 实验 — 激活专家比例对比")
    print("=" * 70)
    print("\n实验目的: 验证激活专家与总专家比例对 MoE 效果的影响")
    print("所有配置保持 MoE 激活参数量 ≈ Dense 参数量")

    num_epochs = 200
    results = []

    # ============================================================
    # 实验 A: 8 experts, top_k=2 (比例=1/4) — 当前 v2 配置
    # ============================================================
    r_a = run_experiment(
        config_name="A: 8exp_k2(1/4)",
        num_experts=8, top_k=2,
        n_group=4, topk_group=2,
        num_epochs=num_epochs, verbose=True,
    )
    results.append(r_a)

    # ============================================================
    # 实验 B: 8 experts, top_k=4 (比例=1/2)
    # ============================================================
    r_b = run_experiment(
        config_name="B: 8exp_k4(1/2)",
        num_experts=8, top_k=4,
        n_group=4, topk_group=4,  # 所有组都选中
        num_epochs=num_epochs, verbose=True,
    )
    results.append(r_b)

    # ============================================================
    # 实验 C: 4 experts, top_k=2 (比例=1/2)
    # ============================================================
    r_c = run_experiment(
        config_name="C: 4exp_k2(1/2)",
        num_experts=4, top_k=2,
        n_group=2, topk_group=2,  # 所有组都选中
        num_epochs=num_epochs, verbose=True,
    )
    results.append(r_c)

    # ============================================================
    # 打印对比表格
    # ============================================================
    print_comparison_table(results)

    # ============================================================
    # 分析结论
    # ============================================================
    print("\n" + "=" * 70)
    print("实验分析")
    print("=" * 70)

    # 找出最佳配置
    best_moe = min(results, key=lambda r: r['moe_mse'])
    best_dense = min(results, key=lambda r: r['dense_mse'])

    print(f"\nMoE 最佳配置: {best_moe['config_name']} (MSE={best_moe['moe_mse']:.4f})")
    print(f"Dense 最佳配置: {best_dense['config_name']} (MSE={best_dense['dense_mse']:.4f})")

    # 分析激活比例的影响
    print(f"\n激活比例影响分析:")
    for r in sorted(results, key=lambda x: x['activation_ratio']):
        print(f"  比例 {r['activation_ratio']:.0%}: "
              f"MoE={r['moe_mse']:.4f}, Dense={r['dense_mse']:.4f}, "
              f"路由准确率={r['route_acc']:.2%}, "
              f"{'MoE优' if r['moe_wins'] else 'Dense优'}")

    # 分析 MoE 是否优于 Dense
    moe_wins_count = sum(1 for r in results if r['moe_wins'])
    print(f"\nMoE 优于 Dense 的配置数: {moe_wins_count}/{len(results)}")

    print("\n" + "=" * 70)
    print("实验完毕!")
    print("=" * 70)


if __name__ == "__main__":
    main()