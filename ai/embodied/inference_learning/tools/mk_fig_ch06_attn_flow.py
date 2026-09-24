#!/usr/bin/env python
"""第 06 章配图：action head 的 attention 处理流程（维度/来源/用途/缓存属性）。

依据 2026-09-24 实测：官方 GR00T-N1_5-3B config + 开源 release 代码(5e3ef5e)：
vl 侧 2048（project_to_dim=null）；DiT 16 层 interleave（偶 cross/奇 self）32头x48=1536；
vl_self_attention 4 层 32x64；num_inference_timesteps=4；
sa_embs=[state|future(32)|action]（tag n1.5-release，非首发布 5e3ef5e）。
生成：images/ch06/attn_flow.png
用法：python3 tools/mk_fig_ch06_attn_flow.py
"""
import os

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

for f in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",):
    if os.path.exists(f):
        fm.fontManager.addfont(f)
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False

ORANGE, BLUE, GREEN, GREY, PURPLE = "#ef6c00", "#1565c0", "#2e7d32", "#455a64", "#6a1b9a"
BG_G, BG_O, BG_B, BG_P = "#e8f5e9", "#fff3e0", "#e3f2fd", "#f3e5f5"

fig, ax = plt.subplots(figsize=(16.6, 9.2), dpi=140)
ax.set_xlim(0, 166); ax.set_ylim(0, 92); ax.axis("off")


def box(x, y, w, h, main, sub="", fc="white", ec=GREY, fs=7.5, sfs=5.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4", fc=fc, ec=ec, lw=1.3))
    dy = 2.0 if sub else 0
    ax.text(x + w / 2, y + h / 2 + dy, main, ha="center", va="center", fontsize=fs)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 2.6, sub, ha="center", va="center", fontsize=sfs, color="#37474f")


def arr(x1, y1, x2, y2, c=GREY, ls="-", lw=1.5, head=True):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>" if head else "-", mutation_scale=11,
                                 color=c, lw=lw, linestyle=ls))


def poly(pts, c=ORANGE, ls="--", lw=1.7):
    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:-1]):
        arr(x1, y1, x2, y2, c=c, ls=ls, lw=lw, head=False)
    arr(pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1], c=c, ls=ls, lw=lw, head=True)


ax.text(83, 89.8, "action head 的 Attention 流程：条件流(绿,K步恒定) · 草稿流(橙,每步变) · 时间流(蓝)",
        ha="center", fontsize=12, fontweight="bold")
ax.text(83, 87.2, "绿=K步恒定(可缓存)   橙=每步变   蓝=时间   紫=掩码   |   数字为 GR00T-N1.5-3B 官方 ckpt 实测",
        ha="center", fontsize=7, color="#37474f")

# ── 条件流（顶部横向）──
box(2, 75.5, 16, 9, "观测", "client 相机帧+指令", fc="#eceff1")
box(22, 75.5, 18, 9, "Eagle VLM\n(冻结, ch05)", "select_layer=12\nproject_to_dim=null", fc="#eceff1", sfs=5.2)
box(44, 75.5, 22, 9, "backbone_features\n[B,296,2048]", "来源:第12层 hidden state", fc=BG_G, ec=GREEN)
box(44, 64.5, 22, 7, "attn_mask [B,296]", "backbone 同路输出(padding 标记)", fc=BG_P, ec=PURPLE, sfs=5.2)
box(70, 75.5, 20, 10, "vlln + vl_self_attn\n4 x self-attn(32x64)", "现场①:token 互通气", fc=BG_G, ec=GREEN, sfs=5.4)
box(94, 75.5, 18, 10, "vl_embs\n[B,296,2048]\nK步恒定, 锁存", "cross 层 K/V 之源", fc=BG_G, ec=GREEN, sfs=5.4)
for x1, x2 in [(18, 21.5), (40, 43.5), (66, 69.5), (90, 93.5)]:
    arr(x1, 80, x2, 80)
arr(45, 75.3, 45, 71.8)
ax.text(38.5, 73.2, "mask", fontsize=5.5, color=PURPLE)

# ── 草稿流（中部横向）──
box(2, 44, 16, 9, "动作草稿 x_k\n[B,16,32]", "k=0 纯噪声, k>0 上步Euler", fc=BG_O, ec=ORANGE, sfs=5.4)
box(22, 43, 26, 11, "action_encoder(+t桶)\nMultiEmbodiment(本体私有)", "16x32 -> 16 token, 1536", fc=BG_O, ec=ORANGE)
box(52, 44, 16, 9, "+pos_embedding\n(可学习)", "DiT块内无位置编码", fc=BG_O, ec=ORANGE)
box(72, 43.5, 40, 10, "sa_embs = [state | future(32) | action]\n[B, T_q=N_s+32+16, 1536]", "future = 32 个可学习 token(按 tag n1.5-release)", fc=BG_O, ec=ORANGE)
box(22, 30, 16, 8, "state\n[B,N_s,d]", "关节角等本体传感", fc=BG_O, ec=ORANGE, fs=7)
box(42, 30, 22, 8, "state_encoder\nMLP(本体私有)", "state 投成 N_s 个 1536", fc=BG_O, ec=ORANGE, fs=7)
for x1, x2 in [(18, 21.5), (48, 51.5), (68, 71.5)]:
    arr(x1, 48.5, x2, 48.5)
arr(30, 38, 30, 42.9)
arr(53, 38, 53, 42.9)
arr(64, 34, 72, 43.3)

# ── 时间流（底部横向）──
box(34, 13, 14, 9, "t 桶\n0..999", "0.999 -> 桶", fc=BG_B, ec=BLUE, fs=7)
box(52, 13, 14, 9, "Timestep\nEncoder", "正弦256+MLP", fc=BG_B, ec=BLUE, fs=7)
box(72, 13, 14, 9, "temb\n[B,1536]", "AdaLN 之源", fc=BG_B, ec=BLUE, fs=7)
arr(48, 17.5, 51.5, 17.5)
arr(66, 17.5, 71.5, 17.5)
# t 的第二路：竖直穿过 state / state_encoder 之间的通道
arr(40, 42.9, 40, 22.3, c=ORANGE, ls=":", lw=1.2)
ax.text(41, 24.2, "t 亦入 action_encoder(第2路)", fontsize=5.4, color=ORANGE, ha="left")

# ── DiT 竖列 ──
ax.add_patch(FancyBboxPatch((120, 39.5), 44, 46, boxstyle="round,pad=0.7", fc="#fafafa", ec=GREY, lw=1.7))
ax.text(142, 83.4, "DiT 16 层交替 (interleave_self_attention=True)", ha="center", fontsize=8.2, fontweight="bold")
ax.text(142, 81.2, "条件流只进偶数 cross 层(8/16)：K/V 恒定 -> 可缓存", ha="center", fontsize=6.2, color=GREEN)
for i in range(8):
    yy = 76.6 - i * 5.15
    box(122, yy, 19, 4.0, f"L{2*i} cross  Q<-x  KV<-vl", "", fc=BG_B, ec=BLUE, fs=5.4)
    box(143, yy, 19, 4.0, f"L{2*i+1} self  QKV<-x", "", fc=BG_O, ec=ORANGE, fs=5.4)
    if i < 7:
        arr(152, yy, 152, yy - 1.15)
box(122, 33.0, 40, 4.6, "action_decoder(本体私有) -> 速度场 v [B,16,32] = pred[:,-16:]", "", fc=BG_O, ec=ORANGE, fs=5.8)
box(122, 26.3, 40, 4.6, "每层：AdaLN(norm1)<-temb | FF=GEGLU | dropout0.2 仅训练态", "", fc=BG_P, ec=PURPLE, fs=5.4)
box(122, 18.5, 40, 6.5, "单层 cross: Q=to_q(x) 32头x48 ; K/V=to_k/v(vl) 2048->1536",
    "softmax(QK^T/sqrt(48)) V | 8 层 KV 可缓存(K,V 恒定) | mask 实际未启用", fc="white", ec=GREY, fs=5.6, sfs=5.4)

# ── 三条入流 + 掩码 ──
arr(112, 80, 121.5, 80.5, c=GREEN, lw=2.2)
box(85.5, 55.5, 31, 9, "[!] 实核：mask 传而不达",
    "训练 forward 里传了 mask，但 DiT.forward 给每层下发 None，\n块内 encoder_attention_mask 参数又被注释掉\n=> padding token 照样参与 cross-attn（B=1 无 padding 时无害）",
    fc=BG_P, ec=PURPLE, fs=6.6, sfs=4.8)
arr(66, 68, 85.1, 61.5, c=PURPLE, ls="--", lw=1.2)
arr(112, 48.5, 121.5, 50.6, c=ORANGE, lw=2.2)
ax.text(117, 52.5, "贯穿", fontsize=5.8, color=ORANGE, ha="center")
arr(86, 17.5, 121.5, 28.4, c=BLUE, lw=1.6)

# ── Euler 回流（沿底部）──
poly([(121.6, 35.4), (118, 35.4), (118, 12), (10, 12), (10, 43.6)], c=ORANGE, ls="--", lw=1.7)
ax.text(48, 9.8, "Euler: x_{k+1} = x_k + dt*v   (K=4 圈, 每圈 = 4 次 DiT 前向)", fontsize=7.2, color=ORANGE)

fig.tight_layout()
out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "images", "ch06", "attn_flow.png"))
fig.savefig(out)
print("saved:", out)
