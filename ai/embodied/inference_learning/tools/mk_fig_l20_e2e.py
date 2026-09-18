#!/usr/bin/env python
"""第 13 章配图：L20 上 N1.5/N1.6 端到端推理实测。

生成：
  images/ch13/l20_run_matrix.png   ① 双版本并存环境矩阵（表格式）
  images/ch13/l20_run_paths.png    ② 四条已实测执行路径（泳道流程）
  images/ch13/l20_latency.png      ③ 单步延迟实测对比（含官方 compile 参照）
  images/ch13/n16_eval_arms.jpg    ④ N1.6 实测 GT/Pred 曲线（29 维前 12 维裁格）
  images/ch13/n15_eval_arms.jpg    ⑤ N1.5 实测 GT/Pred 曲线（前 3 通道节选）

用法：
    python3 tools/mk_fig_l20_e2e.py \
        --n15-jpeg /tmp/n15_eval_traj.jpeg \
        --n16-jpeg /tmp/stand_alone_inference/traj_0.jpeg \
        --outdir   images/ch13
数值来源：/tmp/n16_e2e.log、/tmp/n15_lat_inproc.log、/tmp/n15_zmq.log、
/tmp/n15_e2e.log、/tmp/n16_openloop.log、/tmp/n16_zmq.log（2026-09-17/18 实测）。
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

for f in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",):
    if os.path.exists(f):
        fm.fontManager.addfont(f)
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False

BLUE, ORANGE, GREEN, GREY, RED = "#1565c0", "#ef6c00", "#2e7d32", "#546e7a", "#c62828"


def box(ax, x, y, w, h, text, fc, ec, fontsize=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color="#263238", weight="bold" if bold else "normal", linespacing=1.55)


def arrow(ax, x1, y1, x2, y2, color=GREY):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4))


def fig_matrix(path):
    fig, ax = plt.subplots(figsize=(11.6, 5.8))
    ax.set_xlim(0, 11.6); ax.set_ylim(0, 6.4); ax.axis("off")
    ax.set_title("N1.5 / N1.6 双版本并存 · 同一台 L20 机器、同一个 conda 环境",
                 fontsize=13, pad=12, loc="left")
    cols = [0.25, 1.75, 6.55]
    for x, h, c in zip(cols, ["", "N1.5 主线（课程基准）", "N1.6 实跑（本章新增）"],
                       [GREY, BLUE, ORANGE]):
        ax.text(x, 5.82, h, fontsize=10.5, weight="bold", color=c)
    rows = [
        ("代码检出", "Isaac-GR00T/  @ groot_ops_patch_n15\n（不移动、不 stash）",
         "git branch n1.6 n1.6-release(=ead5283)\ngit worktree add ../Isaac-GR00T-n1.6 n1.6"),
        ("import gr00t\n如何解析", "pip -e editable 安装\n→ finder 指向 Isaac-GR00T/",
         "PYTHONPATH=$PWD 前插覆盖 editable\n（finder 是 meta_path.append，路径查找先赢）"),
        ("conda 环境", "gr00t（torch 2.9 + transformers 4.51.3，\n后者恰与 n1.6 的 pin 相同）",
         "同一个 gr00t 共用；仅补装 lmdb==1.7.5\n（Eagle 动态模块硬依赖，纯增量）"),
        ("权重（离线）", "~/.cache/huggingface/.../GR00T-N1.5-3B\n快照 869830fc…（5.1 G）",
         "/home/ft/wzong/models/GR00T-N1.6-3B（6.2 G）\nHF×ModelScope 双源逐 shard MD5 一致"),
        ("离线开关", "HF_HUB_OFFLINE=1", "HF_HUB_OFFLINE=1  TRANSFORMERS_OFFLINE=1"),
        ("跑前自检", "python -c \"import gr00t; print(gr00t.__file__)\"\n应打印 Isaac-GR00T/gr00t/__init__.py",
         "同一条命令，必须打印 Isaac-GR00T-n1.6/…\n若打印成 n1.5 路径 = PYTHONPATH 忘了"),
    ]
    y = 5.35
    for name, a, b in rows:
        hh = 0.82
        ax.add_patch(Rectangle((0.10, y - hh / 2), 11.35, hh, fc="#f5f7f8", ec="#cfd8dc"))
        ax.text(cols[0], y, name, fontsize=9.4, va="center", weight="bold",
                color="#37474f", linespacing=1.4)
        ax.text(cols[1], y, a, fontsize=8.7, va="center", color=BLUE, linespacing=1.5)
        ax.text(cols[2], y, b, fontsize=8.7, va="center", color=ORANGE, linespacing=1.5)
        y -= hh + 0.06
    ax.text(0.15, 0.12, "⇒ 原则：不动 n1.5 的检出/环境/依赖；n1.6 全部用 git worktree + PYTHONPATH + 增量包装出来，任何一步都能整体删除回滚。",
            fontsize=9.4, color=RED)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_paths(path):
    fig, ax = plt.subplots(figsize=(11.8, 7.0))
    ax.set_xlim(0, 11.8); ax.set_ylim(0, 7.4); ax.axis("off")
    ax.set_title("四条已在 L20 实测跑通的路径（每版两条：单进程 / ZMQ 服务化）",
                 fontsize=13, pad=12, loc="left")
    ax.add_patch(Rectangle((0.05, 3.85), 11.6, 3.0, fc="#e3f2fd", alpha=0.35))
    ax.add_patch(Rectangle((0.05, 0.55), 11.6, 3.0, fc="#fff3e0", alpha=0.4))
    ax.text(0.15, 6.55, "N1.5", fontsize=12, weight="bold", color=BLUE)
    ax.text(0.15, 3.25, "N1.6", fontsize=12, weight="bold", color=ORANGE)

    box(ax, 0.55, 5.5, 2.5, 0.85, "路径① 单进程开环\nscripts/eval_policy.py\n--model-path <快照目录>", "#ffffff", BLUE, 8.6)
    box(ax, 3.95, 5.5, 2.2, 0.85, "Gr00tPolicy\n进程内直连 GPU", "#ffffff", BLUE, 8.8)
    box(ax, 7.0, 5.5, 4.45, 0.85, "实测：150 步 MSE=0.0391（arms_only 14 维）\n单步 0.569 s ≈1.8 Hz（探针 tools/l20_latency_probe.py）", "#e8f5e9", GREEN, 8.6)
    arrow(ax, 3.05, 5.92, 3.95, 5.92); arrow(ax, 6.15, 5.92, 7.0, 5.92)

    box(ax, 0.55, 4.2, 2.5, 0.85, "路径② 服务化\ninference_service.py\n--server --model-path …", "#ffffff", BLUE, 8.6)
    box(ax, 3.95, 4.2, 2.2, 0.85, "ZMQ :5555\nMsgSerializer 打包", "#ffffff", BLUE, 8.8)
    box(ax, 7.0, 4.2, 4.45, 0.85, "client（eval_policy 不带 --model-path）48 步 MSE=0.0352\n单步 0.596 s：序列化+网络只 +27 ms（≈5%）", "#e8f5e9", GREEN, 8.6)
    arrow(ax, 3.05, 4.62, 3.95, 4.62); arrow(ax, 6.15, 4.62, 7.0, 4.62)

    box(ax, 0.55, 2.25, 2.5, 0.95, "路径① 单进程开环\nscripts/deployment/\nstandalone_inference_script.py", "#ffffff", ORANGE, 8.3)
    box(ax, 3.95, 2.25, 2.2, 0.95, "Gr00tPolicy(n1.6)\nPYTHONPATH 保证\nimport 到新版包", "#ffffff", ORANGE, 8.3)
    box(ax, 7.0, 2.25, 4.45, 0.95, "实测：3×25 步平均 MSE=1.178（全身 29 维）\n单步 0.442 s（min .395 / P90 .475）；首跑含 triton autotune", "#e8f5e9", GREEN, 8.6)
    arrow(ax, 3.05, 2.72, 3.95, 2.72); arrow(ax, 6.15, 2.72, 7.0, 2.72)

    box(ax, 0.55, 0.85, 2.5, 0.95, "路径② 服务化\ngr00t/eval/run_gr00t_server.py\n--embodiment-tag GR1", "#ffffff", ORANGE, 8.3)
    box(ax, 3.95, 0.85, 2.2, 0.95, "ZMQ :5555\nPolicyClient", "#ffffff", ORANGE, 8.8)
    box(ax, 7.0, 0.85, 4.45, 0.95, "client：open_loop_eval.py 不带 --model-path\n32 步 MSE=1.358；open_loop 进程内 100 步 MSE=1.214", "#e8f5e9", GREEN, 8.6)
    arrow(ax, 3.05, 1.32, 3.95, 1.32); arrow(ax, 6.15, 1.32, 7.0, 1.32)

    ax.text(0.15, 0.14, "⇒ 两代\"服务化\"包法不同（n1.5: RobotInferenceServer / n1.6: PolicyServer），但请求旅程与第 07 章讲的一致；两版服务勿同时占 5555 端口。",
            fontsize=9.2, color=RED)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_latency(path):
    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    labels = ["N1.5\n进程内\n(chunk 16×denoise 4)",
              "N1.5\nZMQ 服务化\n(同 +MsgPack 往返)",
              "N1.6\nstandalone\n(异步预取+horizon 8)"]
    vals = [0.569, 0.596, 0.442]
    cols = [BLUE, "#64b5f6", ORANGE]
    bars = ax.bar(labels, vals, color=cols, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f} s\n({1/v:.1f} Hz)",
                ha="center", fontsize=10, weight="bold")
    ax.axhline(0.038, ls="--", color=RED, lw=1.3)
    ax.text(0.98, 0.775, "官方 README 参照线（红虚线）：H100 + torch.compile = 38 ms。\n那是编译模式 + H100；本机是 L20 + pytorch eager，勿直接混比",
            fontsize=8.8, color=RED, ha="right", va="top")
    ax.set_ylabel("单次 get_action / 推理步 平均耗时（s）")
    ax.set_ylim(0, 0.80)
    ax.set_title("L20（GPU3，与其他租户共卡）实测单步延迟 · denoising=4 · 去掉首步 warmup",
                 fontsize=12, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.02, -0.24, "warmup 首步：N1.5 1.41 s / N1.6 1.69 s（图外）。ZMQ 化只加 ≈27 ms（第 07 章\"传输不改变数值\"\n在延迟维度同样成立）。N1.6 首次运行另有 triton autotune 数分钟（一次性，图外）。",
            transform=ax.transAxes, fontsize=8.8, color=GREY)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_crops(n15_jpeg, n16_jpeg, outdir):
    from PIL import Image
    im = Image.open(n16_jpeg)
    w, h = im.size
    row_h = h // 29
    rows = [im.crop((0, i * row_h, w, (i + 1) * row_h)) for i in range(12)]
    grid = Image.new("RGB", (3 * w, 4 * row_h), "white")
    for i, r in enumerate(rows):
        grid.paste(r, ((i % 3) * w, (i // 3) * row_h))
    grid = grid.resize((1440, int(1440 * 4 * row_h / (3 * w))))
    grid.save(os.path.join(outdir, "n16_eval_arms.jpg"), quality=82)
    im2 = Image.open(n15_jpeg)
    w2, h2 = im2.size
    row2 = h2 // 14
    crop = im2.crop((0, 0, w2, 3 * row2)).resize((900, int(900 * 3 * row2 / w2)))
    crop.save(os.path.join(outdir, "n15_eval_arms.jpg"), quality=82)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n15-jpeg", default="/tmp/n15_eval_traj.jpeg")
    ap.add_argument("--n16-jpeg", default="/tmp/stand_alone_inference/traj_0.jpeg")
    ap.add_argument("--outdir", default="images/ch13")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    fig_matrix(os.path.join(a.outdir, "l20_run_matrix.png"))
    fig_paths(os.path.join(a.outdir, "l20_run_paths.png"))
    fig_latency(os.path.join(a.outdir, "l20_latency.png"))
    fig_crops(a.n15_jpeg, a.n16_jpeg, a.outdir)
    print("done ->", a.outdir)
