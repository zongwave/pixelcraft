#!/usr/bin/env python
"""画 §7 案例的两张图：
   左：老 client 的四个 --preprocess 选项里，三个能过尺寸闸门，只有一个与训练一致；
   右：这个几何 bug 在"五层指标"上的可见度（数值直接读 redmine168_preproc_ab.py 的 json 结果）。

用法：
    python3 tools/mk_fig_preproc_gate.py \
        --json /tmp/preproc_ab_final.json \
        --out  images/common/preproc_gate.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

for f in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",):
    if os.path.exists(f):
        fm.fontManager.addfont(f)
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False

GREEN, RED, AMBER, GREY = "#2e7d32", "#c62828", "#ef6c00", "#546e7a"


def left_panel(ax):
    ax.set_xlim(0, 11.6)
    ax.set_ylim(0, 6.0)
    ax.axis("off")
    ax.set_title("① 入口有 4 个选项，只有 1 个与训练一致", fontsize=12, pad=10, loc="left")
    cols = [0.25, 4.35, 6.05, 7.55, 9.2]
    heads = ["--preprocess 选项", "输出尺寸", "尺寸闸门", "与训练几何"]
    for x, h in zip(cols, heads):
        ax.text(x, 5.45, h, fontsize=10, weight="bold", color="#263238")
    rows = [
        ("pad_and_resize  ← 训练用的", "640×640", "PASS", "一致", GREEN),
        ("crop_and_resize", "640×640", "PASS", "不一致（裁FOV）", AMBER),
        ("resize（非等比拉伸）", "640×640", "PASS", "不一致（压扁）", RED),
        ("none（原样发送）", "1280×800", "REJECT", "—", GREY),
    ]
    y = 4.75
    for name, size, gate, geo, col in rows:
        ax.add_patch(Rectangle((0.10, y - 0.36), 11.35, 0.78, fc="#f5f7f8", ec="#cfd8dc"))
        ax.text(cols[0], y, name, fontsize=10.2, va="center")
        ax.text(cols[1], y, size, fontsize=10.2, va="center")
        ax.text(cols[2], y, gate, fontsize=10.5, va="center", weight="bold",
                color=GREEN if gate == "PASS" else RED)
        ax.text(cols[3], y, geo, fontsize=10.2, va="center", color=col)
        y -= 0.92
    ax.text(0.15, 0.62,
            "⇒ “闸门放行”只证明张量形状合法（3 个选项都能做到），\n"
            "   它不能证明画面几何与训练一致。唯一说真话的那条报错，恰好来自 none。",
            fontsize=10, color="#37474f", linespacing=1.7)


def right_panel(ax, d):
    ax.set_title("② 各指标实际测到的几何效应（相对量，对数轴）", fontsize=12, pad=10, loc="left")
    s = d.get("_summary", {})
    A = d.get("A_pad", {}).get("shape", {})
    B = d.get("B_stretch", {}).get("shape", {})
    pk = s.get("per_key", {}) or {}
    floor = (s.get("noise_floor", {}) or {}).get("A_pad", {}) or {}
    gtA = (s.get("mse_to_ground_truth", {}) or {}).get("A_pad", {}) or {}
    paired = s.get("paired_gt", {}) or {}

    def rel(key, sub="mse_A_B", den="mse_A_gt"):
        v = pk.get("action." + key, {})
        return float(v.get(sub, 0.0)) / (float(v.get(den, 1.0)) + 1e-12)

    one_minus_cos = 1.0 - float(s.get("cos_A_B", 1.0))
    agg = float(s.get("mse_A_B", 0.0)) / (float(gtA.get("mean", 1.0)) + 1e-12)
    arm_r = rel("right_arm")
    arm_l = rel("left_arm")
    paired_pct = abs(float(paired.get("diff_rel_pct", 0.0))) / 100.0
    sweep_pct = abs(float(s.get("sweep_last_over_first", 1.0)) - 1.0)
    seed_rel = float(floor.get("seed_mse", 0.0)) / (float(gtA.get("mean", 1.0)) + 1e-12)

    items = [
        ("尺寸闸门 check_input", 2e-6, "完全看不见：拉伸后仍是 640×640 / 256×256", GREY),
        ("token 数 & seq_len", 2e-6,
         "完全看不见：token %s=%s，seq_len %s=%s"
         % (A.get("vision_tokens"), B.get("vision_tokens"),
            A.get("seq_len"), B.get("seq_len")), GREY),
        ("动作 MSE(A,B)/MSE(A,真值) 全维", agg,
         "只有真值误差的 %.1f%%——被手部维度与采样噪声淹没" % (100 * agg), GREY),
        ("backbone 特征偏离 1-cos", one_minus_cos,
         "余弦 %.4f：特征层几乎无感" % float(s.get("cos_A_B", 0)), AMBER),
        ("同指标但只看手臂维度", max(arm_l, arm_r),
         "右臂 %.1f%% / 左臂 %.1f%%：拆维度才露出来" % (100 * arm_r, 100 * arm_l), AMBER),
        ("配对·对真值误差增幅", paired_pct,
         "B 比 A 差 %+.1f%%（%d/%d 对，t=%.2f）方向对、幅度不显著"
         % (float(paired.get("diff_rel_pct", 0)), int(round(float(paired.get("frac_B_worse", 0))
                                                          * paired.get("n_pairs", 0))),
            paired.get("n_pairs", 0), float(paired.get("t_ratio", 0))), GREEN),
        ("形变扫描 k=1→1.6 误差增幅", sweep_pct,
         "随压扁程度单调上升（r=%.2f）" % float(s.get("sweep_corr_k_vs_err", 0)), GREEN),
        ("对照：换随机种子的噪声地板", seed_rel,
         "采样噪声是这个几何效应的 ~%.0f 倍 ⇒ 单看动作输出根本查不出来"
         % (seed_rel / max(agg, 1e-9)), RED),
    ]
    labels = [i[0] for i in items]
    vals = [i[1] for i in items]
    notes = [i[2] for i in items]
    cols = [i[3] for i in items]
    y = list(range(len(items)))[::-1]
    ax.barh(y, [max(v, 2e-6) for v in vals], color=cols, height=0.58, log=True)
    for yy, v, n in zip(y, vals, notes):
        ax.text(max(v, 2e-6) * 1.45, yy, n, fontsize=8.8, va="center", color="#37474f")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.6)
    ax.set_xscale("log")
    ax.set_xlim(1e-6, 3e3)
    ax.set_xticks([1e-6, 1e-4, 1e-2, 1.0, 100.0])
    ax.set_xticklabels(["≈0\n看不见", "1e-4", "1e-2", "1\n看得见", "100"], fontsize=9)
    ax.set_xlabel("该指标测到的相对变化量（对数刻度）", fontsize=9.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", ls=":", color="#b0bec5", alpha=0.6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="/tmp/preproc_ab_final.json")
    ap.add_argument("--out", default="images/common/preproc_gate.png")
    args = ap.parse_args()
    d = json.load(open(args.json)) if os.path.exists(args.json) else {}
    fig, axes = plt.subplots(1, 2, figsize=(15.6, 5.6))
    left_panel(axes[0])
    right_panel(axes[1], d)
    fig.suptitle("“尺寸对得上”≠“几何对得上”：入口选项与检测梯度（Redmine #168 · 离线 A/B 实测）",
                 fontsize=13.5)
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
