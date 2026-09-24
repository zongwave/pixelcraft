#!/usr/bin/env python
"""第 06 章配图（动图）：DiT = 按观测现画地图的绘图师（课堂类比）。

复用 tools/mk_fig_ch06_denoise_paths.py 的玩具流匹配机器（解析可算的最优速度场），
外加"观测"这一维：用簇的先验权重(prior)代表不同观测——
  观测A 壶在右 → 地图偏右盆；观测B 壶在左 → 地图偏左盆。
两幕连播：每幕 t 从 0→1 放一支"箭头电影"，粒子（当前动作草稿 x_k）顺箭头漂 K=16 步入盆。

生成：images/ch06/dit_cartographer.gif
注：不用 FuncAnimation+PillowWriter（部分环境输出重复帧）；逐帧 PNG→PIL 拼装，
    白底不透明 + disposal=2。验证请用 ffmpeg 解码（PIL ImageSequence 对差帧合成显示失真）。
用法：python3 tools/mk_gif_ch06_cartographer.py [--width 440] [--fps 8]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

for f in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",):
    if os.path.exists(f):
        fm.fontManager.addfont(f)
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False

BLUE, ORANGE, GREY, RED = "#1565c0", "#ef6c00", "#546e7a", "#c62828"
rng = np.random.default_rng(3)
N = 260
data = np.vstack([np.column_stack([rng.normal(2.2, .35, N), rng.normal(0, .35, N)]),
                  np.column_stack([rng.normal(-2.2, .35, N), rng.normal(0, .35, N)])])
def field(x_t, t, prior_right):
    t = min(t, 1.0 - 1e-6)
    eps = (x_t - t * data) / (1 - t)
    logw = -0.5 * (eps ** 2).sum(-1)
    logw[:N] += np.log(prior_right)       # data 前半为右簇(x=+2.2)
    logw[N:] += np.log(1 - prior_right)   # 后半为左簇
    logw -= logw.max()
    w = np.exp(logw)
    w /= w.sum()
    return w @ ((data - x_t) / (1 - t))


def euler(eps0, K, pr):
    x, tr = eps0.copy(), [eps0.copy()]
    for k in range(K):
        x = x + (1.0 / K) * field(x, k / K, pr)
        tr.append(x.copy())
    return np.array(tr)


GX, GY = np.meshgrid(np.linspace(-3.4, 3.4, 9), np.linspace(-2.3, 2.3, 7))
K = 16
scenes = [("观测 A：壶在右 → 发一张偏右的地图", 0.85, np.array([0.2, 0.6])),
          ("观测 B：壶在左 → 发一张偏左的地图", 0.15, np.array([-0.2, -0.6]))]
paths = [(title, pr, euler(eps0, K, pr)) for title, pr, eps0 in scenes]

fig, ax = plt.subplots(figsize=(4.6, 3.9), dpi=100)
ax.set_xlim(-3.6, 3.6); ax.set_ylim(-2.6, 2.8); ax.set_aspect("equal"); ax.axis("off")


def draw(i):
    ax.clear(); ax.set_xlim(-3.6, 3.6); ax.set_ylim(-2.6, 2.8)
    ax.set_aspect("equal"); ax.axis("off")
    per = K + 3                                   # 每幕：K 个运动帧 + 3 帧定格
    si = min(i // per, len(paths) - 1)
    k = i - si * per
    title, pr, tr = paths[si]
    kt = min(k, K)
    for (px, py) in zip(GX.ravel(), GY.ravel()):
        v = field(np.array([px, py]), kt / K, pr)
        n = max(np.hypot(*v), 1e-9)
        ax.annotate("", xy=(px + .3 * v[0] / n, py + .3 * v[1] / n), xytext=(px, py),
                    arrowprops=dict(arrowstyle="-|>", color=GREY, alpha=.45, lw=.7))
    ax.scatter(data[:N, 0], data[:N, 1], s=3, c=GREY, alpha=.25)
    ax.scatter(data[N:, 0], data[N:, 1], s=3, c=GREY, alpha=.25)
    ax.plot(tr[:kt + 1, 0], tr[:kt + 1, 1], color=ORANGE, lw=1.8)
    ax.plot(*tr[kt], "o", ms=7, mfc="none", mec=RED, mew=1.6)
    ax.text(-3.5, 2.55, title, fontsize=9.5, color=BLUE)
    ax.text(-3.5, -2.45, f"t = {kt}/16    灰箭头=绘图师现画的地图   ○=动作草稿 x_k",
            fontsize=8, color=GREY)


ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "images", "ch06", "dit_cartographer.gif"))
ap.add_argument("--fps", type=int, default=8)
a = ap.parse_args()

# 逐帧渲染为 PNG 内存块再拼 GIF（FuncAnimation+PillowWriter 在部分环境会输出重复帧，绕开）
from PIL import Image
import io as _io
frames = []
for i in range((K + 3) * len(paths)):
    draw(i)
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    buf.seek(0)
    frames.append(Image.open(buf).convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=128))
    buf.close()
out = os.path.abspath(a.out)
frames[0].save(out, save_all=True, append_images=frames[1:],
               duration=int(1000 / a.fps), loop=0, optimize=False, disposal=2)
mb = os.path.getsize(out) / 1e6
print(f"saved: {out} ({mb:.1f} MB, {len(frames)} frames @ {a.fps}fps)")
