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

BLUE, ORANGE, GREEN, GREY, RED, PURPLE = "#1565c0", "#ef6c00", "#2e7d32", "#546e7a", "#c62828", "#6a1b9a"


def box(ax, x, y, w, h, text, fc, ec, fontsize=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color="#263238", weight="bold" if bold else "normal", linespacing=1.55)


def arrow(ax, x1, y1, x2, y2, color=GREY):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4))


def fig_matrix(path):
    fig, ax = plt.subplots(figsize=(14.4, 6.2))
    ax.set_xlim(0, 14.4); ax.set_ylim(0, 6.4); ax.axis("off")
    ax.set_title("N1.5 / N1.6 / N1.7 三代并存 · 同一台 L20（ARM 宿主）、同一个仓库，互不侵入",
                 fontsize=13, pad=12, loc="left")
    cols = [0.22, 1.85, 6.15, 10.35]
    for x, h, c in zip(cols, ["", "N1.5 主线（课程基准）", "N1.6 实跑", "N1.7 实跑（本章新增）"],
                       [GREY, BLUE, ORANGE, PURPLE]):
        ax.text(x, 5.85, h, fontsize=10.5, weight="bold", color=c)
    rows = [
        ("代码检出", "Isaac-GR00T/  @ groot_ops_patch_n15\n（不移动、不 stash）",
         "git branch n1.6 n1.6-release(=ead5283)\ngit worktree add ../Isaac-GR00T-n1.6 n1.6",
         "git branch n1.7 n1.7-release(=23ace64)\ngit worktree add ../Isaac-GR00T-n1.7 n1.7"),
        ("import gr00t\n如何解析", "pip -e editable 安装\n→ finder 指向 Isaac-GR00T/",
         "PYTHONPATH=$PWD 前插覆盖 editable\n（finder 是 meta_path.append，路径查找先赢）",
         "同法 PYTHONPATH=$PWD:$TOOLS\n补丁用 runpy 注入，仓库一行不改"),
        ("conda 环境", "gr00t（torch 2.9 + transformers 4.51.3，\n后者恰与 n1.6 的 pin 相同）",
         "同一个 gr00t 共用；仅补装 lmdb==1.7.5\n（Eagle 动态模块硬依赖，纯增量）",
         "gr00t_n17 = clone gr00t + transformers 4.57.3\n（4.51.3 无 Qwen3VLForConditionalGeneration）"),
        ("权重（离线）", "~/.cache/huggingface/.../GR00T-N1.5-3B\n快照 869830fc…（5.1 G）",
         "/home/ft/wzong/models/GR00T-N1.6-3B（6.2 G）\nHF×ModelScope 双源逐 shard SHA256 一致",
         "…/GR00T-N1.7-3B-modelscope（6.9 G，1031 张量）\n骨干 gated → §5.1 结构占位等价跑通"),
        ("离线开关", "HF_HUB_OFFLINE=1", "HF_HUB_OFFLINE=1  TRANSFORMERS_OFFLINE=1",
         "同 n1.6 两个开关 + 必带 --video-backend decord\n（torchcodec 0.11 与 torch 2.9 ABI 不匹配）"),
        ("跑前自检", 'python -c "import gr00t; print(gr00t.__file__)"\n应打印 Isaac-GR00T/gr00t/__init__.py',
         "同一条命令，必须打印 Isaac-GR00T-n1.6/…\n若打印成 n1.5 路径 = PYTHONPATH 忘了",
         "同一条命令须打印 Isaac-GR00T-n1.7/…\n另加 n17_arm_patch.apply() 应返回 1"),
    ]
    y = 5.38
    for name, a, b, c in rows:
        hh = 0.82
        ax.add_patch(Rectangle((0.10, y - hh / 2), 14.15, hh, fc="#f5f7f8", ec="#cfd8dc"))
        ax.text(cols[0], y, name, fontsize=9.2, va="center", weight="bold",
                color="#37474f", linespacing=1.4)
        ax.text(cols[1], y, a, fontsize=8.1, va="center", color=BLUE, linespacing=1.5)
        ax.text(cols[2], y, b, fontsize=8.1, va="center", color=ORANGE, linespacing=1.5)
        ax.text(cols[3], y, c, fontsize=8.1, va="center", color=PURPLE, linespacing=1.5)
        y -= hh + 0.06
    ax.text(0.15, 0.12, "⇒ 原则：不动 n1.5 的检出/环境/依赖。n1.6 用 worktree + PYTHONPATH + 增量包装出来；n1.7 再额外克隆一个 conda 环境 + 占位骨干 + runpy 补丁——任何一步都能整体删除回滚。",
            fontsize=9.0, color=RED)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_paths(path):
    fig, ax = plt.subplots(figsize=(11.8, 9.2))
    ax.set_xlim(0, 11.8); ax.set_ylim(0, 9.6); ax.axis("off")
    ax.text(0.05, 9.32, "五条已在 L20 实测跑通的路径（N1.5 / N1.6 各两条；N1.7 一条进程内）",
            fontsize=13, weight="bold", color="#111111")
    ax.add_patch(Rectangle((0.05, 6.6), 11.6, 2.5, fc="#e3f2fd", alpha=0.35))
    ax.add_patch(Rectangle((0.05, 3.5), 11.6, 2.8, fc="#fff3e0", alpha=0.4))
    ax.add_patch(Rectangle((0.05, 1.0), 11.6, 1.9, fc="#f3e5f5", alpha=0.45))
    ax.text(0.15, 8.82, "N1.5", fontsize=12, weight="bold", color=BLUE)
    ax.text(0.15, 5.92, "N1.6", fontsize=12, weight="bold", color=ORANGE)
    ax.text(0.15, 2.62, "N1.7", fontsize=12, weight="bold", color=PURPLE)

    box(ax, 0.55, 7.75, 2.5, 0.85, "路径① 单进程开环\nscripts/eval_policy.py\n--model-path <快照目录>", "#ffffff", BLUE, 8.6)
    box(ax, 3.95, 7.75, 2.2, 0.85, "Gr00tPolicy\n进程内直连 GPU", "#ffffff", BLUE, 8.8)
    box(ax, 7.0, 7.75, 4.45, 0.85, "实测：150 步 MSE=0.0391（arms_only 14 维）\n单步 0.569 s ≈1.8 Hz（探针 tools/l20_latency_probe.py）", "#e8f5e9", GREEN, 8.6)
    arrow(ax, 3.05, 8.17, 3.95, 8.17); arrow(ax, 6.15, 8.17, 7.0, 8.17)

    box(ax, 0.55, 6.68, 2.5, 0.85, "路径② 服务化\ninference_service.py\n--server --model-path …", "#ffffff", BLUE, 8.6)
    box(ax, 3.95, 6.68, 2.2, 0.85, "ZMQ :5555\nMsgSerializer 打包", "#ffffff", BLUE, 8.8)
    box(ax, 7.0, 6.68, 4.45, 0.85, "client（eval_policy 不带 --model-path）48 步 MSE=0.0352\n单步 0.596 s：序列化+网络只 +27 ms（≈5%）", "#e8f5e9", GREEN, 8.6)
    arrow(ax, 3.05, 7.10, 3.95, 7.10); arrow(ax, 6.15, 7.10, 7.0, 7.10)

    box(ax, 0.55, 4.95, 2.5, 0.95, "路径③ 单进程开环\nscripts/deployment/\nstandalone_inference_script.py", "#ffffff", ORANGE, 8.3)
    box(ax, 3.95, 4.95, 2.2, 0.95, "Gr00tPolicy(n1.6)\nPYTHONPATH 保证\nimport 到新版包", "#ffffff", ORANGE, 8.3)
    box(ax, 7.0, 4.95, 4.45, 0.95, "实测：3×25 步平均 MSE=1.178（全身 29 维）\n单步 0.442 s（min .395 / P90 .475）；首跑含 triton autotune", "#e8f5e9", GREEN, 8.6)
    arrow(ax, 3.05, 5.42, 3.95, 5.42); arrow(ax, 6.15, 5.42, 7.0, 5.42)

    box(ax, 0.55, 3.62, 2.5, 0.95, "路径④ 服务化\ngr00t/eval/run_gr00t_server.py\n--embodiment-tag GR1", "#ffffff", ORANGE, 8.3)
    box(ax, 3.95, 3.62, 2.2, 0.95, "ZMQ :5555\nPolicyClient", "#ffffff", ORANGE, 8.8)
    box(ax, 7.0, 3.62, 4.45, 0.95, "client：open_loop_eval.py 不带 --model-path\n32 步 MSE=1.358；open_loop 进程内 100 步 MSE=1.214", "#e8f5e9", GREEN, 8.6)
    arrow(ax, 3.05, 4.09, 3.95, 4.09); arrow(ax, 6.15, 4.09, 7.0, 4.09)

    box(ax, 0.55, 1.55, 2.5, 0.95, "路径⑤ 单进程开环\nstandalone_inference_script.py\n+ n17_arm_patch（runpy 注入）", "#ffffff", PURPLE, 8.0)
    box(ax, 3.95, 1.55, 2.2, 0.95, "Gr00tN1d7 + Qwen3VL 骨干\n占位骨干（§5.1）\n--video-backend decord", "#ffffff", PURPLE, 7.8)
    box(ax, 7.0, 1.55, 4.45, 0.95, "实测：droid 2×25 步 MSE=0.0199（17 维，零样本）\n单步 1.709 s；同一命令未打补丁 40.28 s（§5.4）", "#e8f5e9", GREEN, 8.3)
    arrow(ax, 3.05, 2.02, 3.95, 2.02); arrow(ax, 6.15, 2.02, 7.0, 2.02)
    ax.text(3.35, 1.22, "注：N1.7 的 ZMQ 服务化路径本章未实测（入口与 n1.6 同名同参 run_gr00t_server.py），不宣称已验证。",
            fontsize=8.4, color=RED)

    ax.text(0.15, 0.62, "⇒ 三代\"服务化\"包法不同（n1.5: RobotInferenceServer / n1.6-n1.7: PolicyServer），但请求旅程与第 07 章讲的一致；任何两版服务都勿同时占 5555 端口。",
            fontsize=9.0, color=RED)
    ax.text(0.15, 0.22, "⇒ 五条路径的 MSE 分属四套口径（14 / 29 / 17 维 + 不同数据集），只能竖着看同版本，不能横着比大小——见 §6 解读纪律 1。",
            fontsize=9.0, color=RED)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_latency(path):
    fig, ax = plt.subplots(figsize=(11.4, 5.2))
    labels = ["N1.5\n进程内\n(chunk16×denoise4)",
              "N1.5\nZMQ 服务化\n(同+MsgPack 往返)",
              "N1.6\nstandalone\n(异步预取+horizon8)",
              "N1.7\nstandalone\n(补丁后)",
              "N1.7\nstandalone\n(补丁前，退化 Conv3d)"]
    vals = [0.569, 0.596, 0.442, 1.7094, 40.2765]
    cols = [BLUE, "#64b5f6", ORANGE, PURPLE, RED]
    bars = ax.bar(labels, vals, color=cols, width=0.55)
    ax.set_yscale("log")
    for b, v in zip(bars, vals):
        hz = "%.2f Hz" % (1 / v) if v > 2 else "%.1f Hz" % (1 / v)
        ax.text(b.get_x() + b.get_width() / 2, v * 1.14,
                "%.3f s\n(%s)" % (v, hz), ha="center", fontsize=9.2, weight="bold")
    ax.axhline(0.038, ls="--", color="#8e24aa", lw=1.2)
    ax.text(-0.46, 45, "紫虚线 = 官方 README 参照线：H100 + torch.compile = 38 ms\n"
                       "（编译模式 + H100，与本机 L20 + eager 不可混比）",
            fontsize=8.4, color="#8e24aa", ha="left", va="top")
    ax.set_yticks([0.05, 0.1, 0.5, 1, 5, 10, 50])
    ax.set_yticklabels(["0.05", "0.1", "0.5", "1", "5", "10", "50"])
    ax.errorbar([3], [1.7094], yerr=[[1.7094 - 1.506], [2.0583 - 1.7094]], fmt="none",
                ecolor=GREY, capsize=4, lw=1.1)
    ax.set_ylabel("单次 get_action 平均耗时（s，对数轴）")
    ax.set_ylim(0.02, 200)
    ax.set_title("L20（GPU3，与其他租户共卡；宿主为 aarch64）实测单步延迟 · denoising=4 · 去掉首步 warmup",
                 fontsize=11.5, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls=":", alpha=0.35)
    ax.text(0.02, -0.30, "warmup 首步：N1.5 1.41 s / N1.6 1.69 s（图外）。ZMQ 化只加 ≈27 ms（第 07 章\"传输不改变数值\"在延迟维度同样成立）；\n"
                         "N1.6 首跑另有 triton autotune 数分钟（一次性，图外）；N1.7 两根柱是同机同权重同数据，差别只在 §5.4 那个退化 Conv3d 补丁。\n"
                         "五根柱口径各异（维度 14/29/17、数据集不同）：延迟可以横着看量级，MSE 不可以。",
            transform=ax.transAxes, fontsize=8.2, color=GREY)
    fig.tight_layout(rect=(0, 0.10, 1, 1)); fig.savefig(path, dpi=150); plt.close(fig)


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
