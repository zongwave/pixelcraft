#!/usr/bin/env python
"""第 13 章 N1.7 配图（2026-09-18 实测数据）。

生成：
  images/ch13/n17_conv3d_debug.png    ① 定位链与等价改写示意（退化 Conv3d -> F.linear）
  images/ch13/n17_patch_speedup.png   ② 打补丁前后单步延迟 + 官方 L20 表口径对照
  images/ch13/n17_eval_traj.jpg       ③ N1.7 droid 零样本 GT/Pred 曲线裁格（17 维取 4 行）

用法：
    python3 tools/mk_fig_n17.py --traj1 /tmp/stand_alone_inference/traj_1.jpeg \
                               --traj2 /tmp/stand_alone_inference/traj_2.jpeg \
                               --outdir images/ch13
数值来源：/tmp/n17_standalone2.log（未打补丁）、/tmp/n17_standalone3.log（打补丁）、
/tmp/n17_probe{9,12,14,15}.log（分段计时/栈采样）；官方 L20 行取自 ckpt 自带 README 计时表。
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

for _f in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",):
    if os.path.exists(_f):
        fm.fontManager.addfont(_f)
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False

BLUE, ORANGE, GREEN, GREY, RED, PURPLE = "#1565c0", "#ef6c00", "#2e7d32", "#546e7a", "#c62828", "#6a1b9a"


def box(ax, x, y, w, h, text, fc, ec, fontsize=9.0, bold=False, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color="#111111", fontweight="bold" if bold else "normal", linespacing=1.5)


def arrow(ax, x1, y1, x2, y2, color=GREY, lw=1.4):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))


def fig_conv3d(outdir):
    fig, ax = plt.subplots(figsize=(13.2, 7.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")
    ax.text(50, 54.4, "N1.7 在 ARM 宿主上 40 s/步 的定位链：退化形状的 Conv3d", ha="center",
            fontsize=13.5, fontweight="bold")

    box(ax, 1, 42, 21, 10,
        "观测/数据层（已排除）\nvideo (1,2,180,320,3) x2 视角\ninput_ids (1,277)\npixel_values (1024,1536)\ndata_prep ~0.0000 s/步",
        "#e3f2fd", BLUE, 8.4)
    box(ax, 26, 42, 21, 10,
        "模型内计时\n40.28 s/步 全部落在\nmodel.get_action()\n（脚本自带分段计时）",
        "#e3f2fd", BLUE, 8.4)
    box(ax, 51, 42, 22, 10,
        "faulthandler 栈采样\n每 7 s 采一次，5 次里 4 次停在\nQwen3VL VisionPatchEmbed\nprobe14 单点计时 25.3 s/次",
        "#fff3e0", ORANGE, 8.4)
    box(ax, 77, 42, 22, 10,
        "形状证据\npatch_embed 收到 [1024,3,2,16,16]\n= batch 1024、空间 1x1\n同权重喂视频形输入仅 0.05 s",
        "#ffebee", RED, 8.4)
    arrow(ax, 22.6, 47, 25.6, 47)
    arrow(ax, 47.6, 47, 50.6, 47)
    arrow(ax, 73.6, 47, 76.6, 47)

    box(ax, 1, 31, 98, 8.4,
        "先排除的四个嫌疑（都实测过）：vLLM 共卡争用（SM 占用降到 0% 仍是 38 s）｜ flash_attn（换 SDPA 无变化）｜ "
        "flow-matching 步数（本就是 4 步）｜ cudnn.enabled=False（不改善）\n"
        "结论：不是「共卡慢」，是这条 Conv3d 形状在本机（aarch64 + torch 2.9 + cu128）落进了病态实现",
        "#eceff1", GREY, 9.0)

    box(ax, 1, 8, 30, 19,
        "官方实现（transformers 4.57.3）\n\n"
        "x = hs.view(-1, 3, 2, 16, 16)\n"
        "x = self.proj(x)      # nn.Conv3d\n"
        "    kernel = stride = (2,16,16)\n"
        "输出 [1024,1024,1,1,1] -> squeeze\n\n"
        "kernel == stride => 卷积窗口互不重叠\n=> 每个 patch 是一次独立的线性映射",
        "#fafafa", GREY, 8.8)
    box(ax, 35, 8, 30, 19,
        "等价改写（tools/n17_arm_patch.py）\n\n"
        "x = hs.view(-1, 3*2*16*16)          # (1024,1536)\n"
        "w = self.proj.weight.reshape(1024,-1)\n"
        "out = F.linear(x, w, self.proj.bias)\n\n"
        "monkeypatch Qwen3VLVisionPatchEmbed.forward\n"
        "权重 [1024,3,2,16,16] / 展平后 1536 -> 1024",
        "#e8f5e9", GREEN, 8.8)
    box(ax, 69, 8, 30, 19,
        "必须同序展开的两条约束\n\n"
        "x 展平序改了、w 没改 -> max err 4.07（错）\n"
        "两者同为 (c,t,h,w) 序 -> max err ~9e-4\n     纯浮点约化顺序差\n\n"
        "e2e 复跑 MSE 0.019936 vs 0.019938\n（差异在扩散初噪漂移量级内）",
        "#f3e5f5", PURPLE, 8.8)
    arrow(ax, 31.4, 17.5, 34.6, 17.5, GREEN, 1.8)
    arrow(ax, 65.4, 17.5, 68.6, 17.5, PURPLE, 1.8)

    fig.tight_layout()
    out = os.path.join(outdir, "n17_conv3d_debug.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_speedup(outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.6))

    labels = ["N1.7\n未打补丁", "N1.7\n打补丁后", "N1.6\n(第 4 节)", "N1.5\n(第 3 节)"]
    vals = [40.28, 1.7094, 0.442, 0.569]
    bars = ax1.bar(labels, vals, color=[RED, GREEN, ORANGE, BLUE], width=0.62)
    ax1.set_yscale("log")
    ax1.set_ylabel("单步推理延迟（s/步，对数轴）")
    ax1.set_title("同一台 L20、同一份权重、同一数据\npatch_embed 等价改写的效果（24x）", fontsize=11)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v * 1.15, "%.3g s" % v, ha="center",
                 fontsize=10, fontweight="bold")
    ax1.errorbar([1], [1.7094], yerr=[[1.7094 - 1.506], [2.0583 - 1.7094]], fmt="none",
                 ecolor=GREY, capsize=5, lw=1.2)
    ax1.text(1, 2.5, "min 1.51 / P90 1.94 / max 2.06", ha="center", fontsize=8, color=GREY)
    ax1.annotate("", xy=(1.0, 5.2), xytext=(0.0, 28),
                 arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.4))
    ax1.text(0.5, 8.0, "24x", ha="center", fontsize=12, fontweight="bold", color=GREY)
    ax1.set_ylim(0.2, 140)
    ax1.grid(axis="y", ls=":", alpha=0.4)
    fig.text(0.012, 0.015, "左：N1.5/N1.6 柱只是量级参照（不同代、不同动作空间），用来说明补丁后落在哪一档",
             fontsize=7.5, color=GREY)

    names = ["官方 L20\nPyTorch Eager", "官方 L20\ntorch.compile", "官方 L20\nTRT 全管线", "本机实测\n补丁后 eager"]
    ms = [140.3, 73.1, 42.8, 1709.4]
    bars2 = ax2.bar(names, ms, color=[GREY, GREY, PURPLE, RED], width=0.62)
    ax2.set_yscale("log")
    ax2.set_ylabel("单步延迟（ms，对数轴）")
    ax2.set_title("口径对照：官方 L20 计时表 vs 本机自测\n（本机是 ARM 宿主 + 共享 GPU，不代表 L20 上限）", fontsize=11)
    for b, v in zip(bars2, ms):
        ax2.text(b.get_x() + b.get_width() / 2, v * 1.15, "%.4g ms" % v, ha="center",
                 fontsize=9.5, fontweight="bold")
    ax2.set_ylim(20, 8000)
    ax2.grid(axis="y", ls=":", alpha=0.4)
    fig.text(0.52, 0.015, "右：官方表出自 ckpt 自带 README（含 compile/TRT 优化）；本机为 eager + 两视角真实视频，解码在宿主 CPU",
             fontsize=7.3, color=GREY)

    fig.tight_layout(rect=(0, 0.035, 1, 1))
    out = os.path.join(outdir, "n17_patch_speedup.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_traj(traj1, traj2, outdir):
    """droid tag 共 17 维：eef_9d(0-8) + gripper_position(9) + joint_position(10-16)。

    取 4 行有代表性的：5 = eef rot6d 分量（chunk 锯齿最明显）、9 = gripper（近方波，
    几乎全对）、10 = joint_1（关节，跟踪最好）、16 = joint_7（尾段可见跟丢，诚实展示误差）。
    注：不裁第 0 行，因为原图 suptitle 与第一个子图重叠（脚本自身的小瑕疵）。
    行定位靠检测每个子图上下 spine 的横线（jpeg 顶上有 suptitle，等分裁会串位）。
    """
    import numpy as np
    from PIL import Image, ImageDraw

    pick = [5, 9, 10, 16]

    def rowstrip(path):
        pil = Image.open(path)
        g = np.asarray(pil.convert("L")).astype(np.int16)
        dark = (g[:, 150:700] < 200).mean(axis=1)
        ys = np.where(dark > 0.7)[0]
        groups, st, prev = [], ys[0], ys[0]
        for y in ys[1:]:
            if y - prev > 3:
                groups.append((st + prev) // 2)
                st = y
            prev = y
        groups.append((st + prev) // 2)
        tops = groups[0::2]
        pitch = tops[1] - tops[0]  # 子图行距；窗口 = [top-30, top-30+pitch]，既不含下一行标题也不含 x 轴标签之外的空白
        return pil, [(max(0, t - 30), max(0, t - 30) + pitch) for t in tops]

    p1, sp1 = rowstrip(traj1)
    p2, sp2 = rowstrip(traj2)
    def cut(pil, spans, i):
        a, b = spans[i]
        c = pil.crop((0, a, pil.size[0], b))
        if i == 0:  # 第 0 行窗口会带进整图的 suptitle，抹掉
            ImageDraw.Draw(c).rectangle([0, 0, c.size[0], 34], fill="white")
        return c

    r1 = [cut(p1, sp1, i) for i in pick]
    r2 = [cut(p2, sp2, i) for i in pick]
    w = p1.size[0]
    rh = r1[0].size[1]
    pad = 30
    grid = Image.new("RGB", (2 * w + 20, len(pick) * rh + pad), "white")
    d = ImageDraw.Draw(grid)
    d.text((w // 2 - 70, 8), "traj 1  (MSE 0.00330)", fill="black")
    d.text((w + 20 + w // 2 - 70, 8), "traj 2  (MSE 0.03657)", fill="black")
    y = pad
    for i in range(len(pick)):
        grid.paste(r1[i], (0, y))
        grid.paste(r2[i], (w + 20, y))
        y += rh
    grid = grid.resize((1300, int(1300 * grid.size[1] / grid.size[0])))
    out = os.path.join(outdir, "n17_eval_traj.jpg")
    grid.save(out, quality=84)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj1", default="/tmp/stand_alone_inference/traj_1.jpeg")
    ap.add_argument("--traj2", default="/tmp/stand_alone_inference/traj_2.jpeg")
    ap.add_argument("--outdir", default="images/ch13")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    print(fig_conv3d(a.outdir))
    print(fig_speedup(a.outdir))
    if os.path.exists(a.traj1) and os.path.exists(a.traj2):
        print(fig_traj(a.traj1, a.traj2, a.outdir))
    else:
        print("skip traj crop:", a.traj1, a.traj2)
