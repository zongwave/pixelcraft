#!/usr/bin/env python
"""第 06 章配图：流匹配迭代去噪 = 河道选边（课堂论述的玩具模型实证）。

玩具设定（与正文 §2/§3 思想实验一一对应）：
  数据分布 = 2D 两个高斯点簇（左右两个"合理动作模式"，多模态）；
  "完美网络" = 解析可算的最优速度场 u(x_t,t)=E[x-x_noise | x_t]
             在数据集离散点上加权：w_i ∝ exp(-||(x_t - t*x_i)/(1-t)||^2 / 2)
                                          v = Σ w_i (x_i - x_t)/(1-t)
  对该场做精确到离散步的欧拉积分 K=1/4/16，直接观察课堂结论：
    ① K=1：所有轨迹一步落到全局均值（mode averaging 回归，正是回归头的死法）；
    ② K=4/16：轨迹早期"选边"进入某一 basin，后期只做落点精修；
    ③ 选边时刻 t_commit（簇后验首次>95%）的分布集中在轨迹前半段。

生成：images/ch06/fm_euler_paths.png
用法：python3 tools/mk_fig_ch06_denoise_paths.py
"""
import os

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

for f in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",):
    if os.path.exists(f):
        fm.fontManager.addfont(f)
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False

BLUE, ORANGE, GREEN, GREY, RED, PURPLE = "#1565c0", "#ef6c00", "#2e7d32", "#546e7a", "#c62828", "#6a1b9a"

rng = np.random.default_rng(7)
N = 400
data = np.vstack([np.column_stack([rng.normal(2.2, .35, N), rng.normal(0, .35, N)]),
                  np.column_stack([rng.normal(-2.2, .35, N), rng.normal(0, .35, N)])])
right = np.arange(len(data)) < N          # 右簇掩码


def weights(x_t, t):
    """完美网络的后验权重 w_i(t)（对单个点 x_t）。t->1 时退化为最近邻。"""
    t = min(t, 1.0 - 1e-6)
    eps = (x_t - t * data) / (1 - t)
    logw = -0.5 * ((eps ** 2).sum(-1))
    logw -= logw.max()
    w = np.exp(logw)
    return w / w.sum()


def field(x_t, t):
    w = weights(x_t, t)
    v = (data - x_t) / (1 - min(t, 1.0 - 1e-6))
    return w @ v, w


def euler(eps0, K):
    """返回轨迹点列、每步前簇后验、(t_commit, 终点簇)。"""
    x, traj, pr = eps0.copy(), [eps0.copy()], []
    commit, done = None, None
    for k in range(K):
        t = k / K
        v, w = field(x, t)
        pr.append(float(w[right].sum()))
        x = x + (1.0 / K) * v
        traj.append(x.copy())
        p = pr[-1]
        if commit is None and (p > 0.95 or p < 0.05):
            commit, done = (k + 1) / K, (p > 0.5)
    return np.array(traj), pr, commit, done


starts = rng.normal(0, 1.0, (14, 2))
starts = starts[np.abs(starts[:, 0]) + np.abs(starts[:, 1]) > 0.6][:12]

fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.9), dpi=150)

# ── ① K=16：河道选边（指路牌全场 + 轨迹分色：选边段/精修段）──
ax = axes[0]
ax.scatter(data[:, 0], data[:, 1], s=4, c=GREY, alpha=.28, label="真实动作数据(两模式)")
gx, gy = np.meshgrid(np.linspace(-3.6, 3.6, 11), np.linspace(-2.6, 2.6, 9))
t_view = 0.25
for (px, py) in zip(gx.ravel(), gy.ravel()):
    v, _ = field(np.array([px, py]), t_view)
    n = max(np.hypot(*v), 1e-9)
    ax.annotate("", xy=(px + .34 * v[0] / n, py + .34 * v[1] / n), xytext=(px, py),
                arrowprops=dict(arrowstyle="-|>", color=GREY, alpha=.5, lw=.8))
commits = []
for e in starts:
    tr, pr, c, d = euler(e, 16)
    ci = max(int(round((c if c else .5) * 16)), 1)
    ax.plot(tr[:ci, 0], tr[:ci, 1], color=ORANGE, lw=1.7)
    ax.plot(tr[ci:, 0], tr[ci:, 1], color=BLUE, lw=1.2, alpha=.8)
    ax.plot(*tr[0], "o", ms=4, mfc="none", mec=RED, mew=1.2)
    commits.append((c, d))
ax.axvline(0, color=RED, ls="--", lw=1, alpha=.65)
ax.text(0, 2.75, "t≈0.25 的指路牌(灰色箭头)", fontsize=9, color=GREY, ha="left", va="top",
        bbox=dict(fc="white", ec="none", alpha=.7, pad=1.5))
ax.text(-3.5, -2.35, "橙=选边段(前中段定命运)\n蓝=精修段(只调落点)\n红圈=噪声起点 ε",
        fontsize=9, va="bottom", bbox=dict(fc="#fff8e1", ec=ORANGE, alpha=.9, pad=3))
ax.set_title("① K=16 欧拉积分：沿速度场顺流而下，入哪个 basin 出哪个动作", fontsize=10.5)
ax.set_xlim(-4, 4); ax.set_ylim(-3, 3); ax.set_aspect("equal"); ax.axis("off")

# ── ② K=1：完美一步 = 全局均值（mode averaging）──
ax = axes[1]
ax.scatter(data[:, 0], data[:, 1], s=4, c=GREY, alpha=.28)
for e in starts:
    tr, *_ = euler(e, 1)
    ax.annotate("", xy=tr[-1], xytext=tr[0],
                arrowprops=dict(arrowstyle="-|>", color=RED, alpha=.75, lw=1.3))
m = data.mean(0)
ax.plot(*m, "X", ms=15, c=RED, mew=2, zorder=5)
ax.annotate("终点恒为条件均值 E[x|obs]\n(此玩具=两簇中心中点)\n→ 哪个模式都不是", xy=m,
            xytext=(0, -2.35), ha="center", fontsize=9.5, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED))
ax.set_title("② K=1 思想实验实跑：所有轨迹一步精准落在均值上\n(完美 1 步 = 回归头的死法)", fontsize=10.5)
ax.set_xlim(-4, 4); ax.set_ylim(-3, 3); ax.set_aspect("equal"); ax.axis("off")

# ── ③ 选边时刻分布：前段投票，后段陪跑 ──
ax = axes[2]
cs = [c for c, _ in commits if c is not None]
ax.hist(cs, bins=np.arange(0.0, 1.01, 0.125), color=PURPLE, alpha=.8, edgecolor="white")
ax.axvline(0.5, color=RED, ls="--", lw=1.2)
ax.text(0.52, ax.get_ylim()[1] * .92, "t=0.5\n(16 步的第 8 步)", fontsize=9, color=RED)
ax.set_xlabel("选边完成时刻  t_commit(簇后验首次 > 95%)")
ax.set_ylabel("轨迹条数")
ax.set_title("③ 12 条轨迹的『选边』全部完成于 t≤0.56（前中段）\n此后速度场只剩精修落点，无人再改主意", fontsize=10.5)
ax.set_xticks(np.arange(0, 1.01, 0.125))
ax.set_xticklabels([f"{v:g}" for v in np.arange(0, 1.01, .125)], fontsize=8)
ax.grid(axis="y", alpha=.3)

fig.suptitle("流匹配迭代去噪 = 河道放船：完美速度场下的 K=1 / K=16 与选边时刻(第 06 章课堂实证)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.93))
out = os.path.join(os.path.dirname(__file__), "..", "images", "ch06", "fm_euler_paths.png")
fig.savefig(os.path.abspath(out))
print("saved:", out)
print("t_commits:", sorted(c for c, _ in commits if c is not None))
