#!/usr/bin/env python3
"""第 9 章配图（图 9-7）：perf trace 三轨对比——旧融合链(0908) vs 当前热轮 vs 当前冷轮。

数据来源（本机 scp 落盘的三份 pytorch/LPU chrome trace，脚本直接解析原文件）：
  OLD  ~/wzong/workspace/embodied/logs/trace_3cam_2iter_ae_fused_k67/wzong_sd3_2iter_chrome_trace.json
  CUR  ~/wzong/workspace/embodied/logs/trace_current_clean/n15_sd3_chrome_trace.json
  NEW  ~/wzong/workspace/embodied/logs/wz_prof_current_20260930_145846/trace/n15_sd3_chrome_trace.json

产物：images/ch09/npu_perfetto_three_way_compare.png
用法：python3 tools/mk_fig_ch09_perfetto_compare.py [OLD CUR NEW]
"""
import collections
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch

for f in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",):
    if os.path.exists(f):
        fm.fontManager.addfont(f)
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False

HOME = os.path.expanduser("~")
DEF = [
    f"{HOME}/wzong/workspace/embodied/logs/trace_3cam_2iter_ae_fused_k67/wzong_sd3_2iter_chrome_trace.json",
    f"{HOME}/wzong/workspace/embodied/logs/trace_current_clean/n15_sd3_chrome_trace.json",
    f"{HOME}/wzong/workspace/embodied/logs/wz_prof_current_20260930_145846/trace/n15_sd3_chrome_trace.json",
]
paths = sys.argv[1:4] if len(sys.argv) >= 4 else DEF
LABELS = ["OLD  0908 旧融合链\nwzong_sd3 (3cam, 2 iter)",
          "CUR  当前·热轮\nn15_sd3 (clean)",
          "NEW  当前·冷轮\nn15_sd3 (0930 首 infer)"]
TRACE_COLOR = ["#c0392b", "#1e8449", "#5d6d7e"]
K, H2D, D2H = "#3b6ea5", "#e67e22", "#c0392b"


def load(p):
    d = json.load(open(p))
    return d["traceEvents"] if isinstance(d, dict) else d


def analyze(ev):
    dev = sorted((e["ts"], e["dur"], e["name"], e["cat"]) for e in ev
                 if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy"))
    t0 = dev[0][0]
    busy = 0.0
    s0 = e0 = dev[0][0]
    for ts, d, n, c in dev:
        if ts > e0:
            busy += e0 - s0
            s0 = ts
        e0 = max(e0, ts + d)
    busy += e0 - s0
    span = dev[-1][0] + dev[-1][1] - t0
    rt = collections.defaultdict(float)
    rt_n = collections.Counter()
    for e in ev:
        if e.get("ph") == "X" and e.get("cat") == "runtime":
            rt[e["name"]] += e["dur"]
            rt_n[e["name"]] += 1
    kt = collections.defaultdict(float)
    for ts, d, n, c in dev:
        kt[n] += d
    h2d_ms = sum(d for ts, d, n, c in dev if c == "gpu_memcpy" and "HtoD" in n) / 1e3
    h2d_n = sum(1 for ts, d, n, c in dev if c == "gpu_memcpy" and "HtoD" in n)
    return dict(dev=dev, t0=t0, span=span / 1e3, busy=busy / 1e3,
                util=100 * busy / span, n_k=sum(1 for x in dev if x[3] == "kernel"),
                h2d_ms=h2d_ms, h2d_n=h2d_n,
                clone_ms=kt.get("KernelCloneTranspose", 0) / 1e3,
                sync_ms=rt.get("evStreamSynchronize", 0) / 1e3,
                sync_n=rt_n.get("evStreamSynchronize", 0),
                conf_ms=rt.get("evConfigureCall", 0) / 1e3,
                conf_n=rt_n.get("evConfigureCall", 0))

A = [analyze(load(p)) for p in paths]
for lab, p, a in zip(LABELS, paths, A):
    print(f"{lab.splitlines()[0]:>6}: span={a['span']:8.1f}ms busy={a['busy']:6.1f}ms "
          f"util={a['util']:4.1f}% k={a['n_k']} HtoD={a['h2d_n']}/{a['h2d_ms']:.0f}ms "
          f"clone={a['clone_ms']:.0f}ms sync={a['sync_n']}/{a['sync_ms']:.0f}ms "
          f"conf={a['conf_ms']:.0f}ms")

fig = plt.figure(figsize=(15.5, 11.2), dpi=150)
gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1.25], hspace=0.52, wspace=0.22,
                      left=0.055, right=0.975, top=0.90, bottom=0.055)

# ---------- rows 1-3: device timelines ----------
for i, (lab, a) in enumerate(zip(LABELS, A)):
    ax = fig.add_subplot(gs[i, :])
    for ts, d, n, c in a["dev"]:
        x = (ts - a["t0"]) / 1e3
        col = K if c == "kernel" else (H2D if "HtoD" in n else D2H)
        ax.vlines(x, 0, 1, color=col, lw=0.6, alpha=0.85)
    ax.set_xlim(-a["span"] * 0.01, a["span"] * 1.01)
    ax.set_ylim(-0.35, 1.35)
    ax.set_yticks([])
    ax.set_xlabel("device time (ms)", fontsize=8.5)
    ax.tick_params(labelsize=8)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"{lab.replace(chr(10), ' ')}   |   span={a['span']:.1f} ms   busy={a['busy']:.1f} ms"
                 f"   利用率={a['util']:.1f}%   kernels={a['n_k']}",
                 fontsize=10, loc="left", color=TRACE_COLOR[i], weight="bold")
    if i == 0:
        ax.text(50, 1.22, "947 个小簇、~100 个 ≈40 ms host 同步 gap（768 次 evStreamSynchronize）"
                          "  |  702×HtoD=174 ms 按迭代重搬  |  2× 38 ms CloneTranspose = 迭代边界"
                          "（iter1≈4.45 s / iter2≈0.74 s）",
                fontsize=8, color="#7b241c")
    if i == 1:
        ax.axvspan(3.4, 58.8, color="#1e8449", alpha=0.08)
        ax.text(31, 1.2, "prologue 55.4 ms（siglip2 视觉塔 + qwen3 LM + patch_embed）",
                fontsize=8, ha="center", color="#145a32")
        for j, t in enumerate([63.6, 87.8, 110.6, 134.3]):
            ax.axvspan(t, t + 13.8, color="#1e8449", alpha=0.16)
        ax.text(105, 0.45, "DiT 去噪 ×4 步（每步 51 op：adaln_qkv×16 + mlp_gelu_norm×16 + fused_mha_out×16 …）",
                fontsize=8, ha="center", color="#145a32")
        ax.text(105, -0.28, "步间 host gap ≈5.9 ms/次；簇内 gap 5.2 ms/步 → 簇内利用率仅 ~55%",
                fontsize=8, ha="center", color="#6e2c00")
    if i == 2:
        ax.annotate("同一批 617 kernel / busy 80.2 ms（与热轮逐 op 对齐），\n被冷 host 路径摊到 1388 ms：evConfigureCall 616 次 = 988 ms",
                    xy=(600, 0.3), xytext=(430, 1.15), fontsize=8, color="#2e4053",
                    arrowprops=dict(arrowstyle="->", color="#2e4053", lw=1))

# ---------- row 4 left: counters ----------
ax = fig.add_subplot(gs[3, 0])
metrics = [("e2e span", [a["span"] for a in A], "ms"),
           ("device busy", [a["busy"] for a in A], "ms"),
           ("HtoD", [a["h2d_ms"] for a in A], "ms"),
           ("CloneTranspose", [a["clone_ms"] for a in A], "ms"),
           ("StreamSync", [a["sync_ms"] for a in A], "ms"),
           ("ConfigureCall", [a["conf_ms"] for a in A], "ms")]
import numpy as np
xpos = np.arange(len(metrics))
w = 0.26
for j, a in enumerate(A):
    vals = [m[1][j] for m in metrics]
    vals = [max(v, 0.005) for v in vals]
    b = ax.bar(xpos + (j - 1) * w, vals, w, color=TRACE_COLOR[j], alpha=0.9)
    for r, v in zip(b, vals):
        if v > 0.006:
            ax.text(r.get_x() + r.get_width() / 2, v * 1.15, f"{v:.0f}" if v >= 1 else f"{v:.2f}",
                    ha="center", fontsize=6.6, color=TRACE_COLOR[j])
ax.set_yscale("log")
ax.set_xticks(xpos)
ax.set_xticklabels([m[0] for m in metrics], fontsize=8.5)
ax.set_ylabel("ms (log)", fontsize=8.5)
ax.tick_params(labelsize=7.5)
ax.set_title("host/device 账目对比（profiler 抬高绝对值，只比同栏差）", fontsize=10, loc="left", weight="bold")
ax.grid(axis="y", ls=":", alpha=0.4)

# ---------- row 4 right: CUR anatomy ----------
ax = fig.add_subplot(gs[3, 1])
segs = [("prologue busy 47.1", 47.1, "#1e8449"),
        ("gap 8.3", 8.3, "#d5dbdb"),
        ("step1 busy 7.5", 7.5, "#2ecc71"), ("gap 6.3", 6.3, "#d5dbdb"),
        ("step2 busy 7.5", 7.5, "#2ecc71"), ("gap 6.3", 6.3, "#d5dbdb"),
        ("step3 busy 7.5", 7.5, "#2ecc71"), ("gap 6.3", 6.3, "#d5dbdb"),
        ("step4 busy 7.5", 7.5, "#2ecc71"), ("tail+gap 13.6", 13.6, "#d5dbdb")]
left = 0
for name, v, c in segs:
    ax.barh(0, v, left=left, color=c, edgecolor="white")
    if v > 6:
        ax.text(left + v / 2, 0, name, ha="center", va="center", fontsize=6.4,
                color="white" if c != "#d5dbdb" else "#2c3e50", rotation=0)
    left += v
ax.set_xlim(0, left)
ax.set_ylim(-0.55, 2.3)
ax.set_yticks([])
ax.set_xlabel("ms（热轮 e2e = 152.9 ms：busy 80.1 + gap 72.8）", fontsize=8.5)
ax.tick_params(labelsize=7.5)
for s in ("left", "right", "top"):
    ax.spines[s].set_visible(False)
ax.set_title("当前热轮解剖：prologue 36% > 去噪 busy 30% > 各类 gap 34%", fontsize=10, loc="left", weight="bold")
ax.text(0, 1.7, "下一步收益排序：① prologue(55 ms, gemm_bias×121 最大项)  ② 步间 gap ~18 ms  "
                "③ 簇内 launch gap ~21 ms（graph 化/流水提交）",
        fontsize=8, color="#145a32")
ax.text(0, 0.85, "冷轮同 workload：busy 不变(80.2 ms)，e2e 1388 ms —— 差的 1.23 s 全在 host 首帧路径，\n"
                 "属首 infer 延迟（warmup 可消），不是稳态回退。", fontsize=8, color="#2e4053")

handles = [Patch(color=K, label="kernel"), Patch(color=H2D, label="Memcpy HtoD"),
           Patch(color=D2H, label="Memcpy DtoH")]
fig.legend(handles=handles, loc="upper right", ncol=3, fontsize=8.5, frameon=False,
           bbox_to_anchor=(0.98, 0.985))
fig.suptitle("图 9-7 · perf trace 三轨对比：旧融合链(2026-09-08) vs 当前热轮/冷轮(2026-09-30)，同一 SD3-DiT 主链 n15_sd3",
             fontsize=12.5, weight="bold", y=0.965)
fig.text(0.055, 0.925, "解析自三份 chrome trace 原文件（Perfetto 可直接打开逐条核对）；"
                       "口径纪律：单样本 + profiler 扰动，只支撑结构性结论（次数/逐 kernel 时长/gap 位置），不支撑 wall-clock 结论。",
         fontsize=8, color="#555")

out = os.path.join(os.path.dirname(__file__), "..", "images", "ch09", "npu_perfetto_three_way_compare.png")
fig.savefig(os.path.normpath(out), facecolor="white")
print("saved:", os.path.normpath(out))
