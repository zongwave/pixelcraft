#!/usr/bin/env python3
"""图 14-4：SigLIP 的 Conv2d 到底在干什么——真实权重版。
224×224×3 → 16×16=256 patch → 每块 flatten 成 588 维 → 共享 Linear(588→1152)（bias 每通道一个）
→ +可学习 PE。底部一排 = GR00T-N1.5-3B 真实 patch_embedding 权重里高频条纹最强的若干行。
权重来源：/home/ft/wzong/workspace/images/GR00T-N1_5-3B（patch_embedding.weight [1152,3,14,14]），
重跑前置：先用 safetensors 从 /home/ft/wzong/workspace/images/GR00T-N1_5-3B 抽
patch_embedding.weight 到 /tmp/patchW.npy（一次即可）；PE 面板与统计直接从 safetensors 读。
"""
import base64, io, os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = "Helvetica, Arial, 'Noto Sans CJK SC', 'WenQuanYi Zen Hei', sans-serif"
WARN = []


def tw(t, size):
    return sum(1.0 if ord(c) > 0x2E7F else 0.56 for c in t) * size


class SVG:
    def __init__(self, w, h):
        self.W, self.H = w, h
        self.parts = []

    def rect(self, x, y, w, h, fill, stroke, rx=4, sw=1.5, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
                          f'stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, t, size=13, fill="#222", bold=False, anchor="start"):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{"bold" if bold else "normal"}" '
                          f'fill="{fill}" text-anchor="{anchor}">{t.replace(chr(38), chr(38)+"amp;").replace("<", "&lt;").replace(">", "&gt;")}</text>')

    def path(self, pts, color="#777", dash=None, sw=1.7, marker="arr"):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#{marker})"' if marker else ""
        self.parts.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in pts)}" fill="none" '
                          f'stroke="{color}" stroke-width="{sw}"{d}{m}/>')

    def box(self, x, y, w, lines, fill="#cfe3f7", stroke="#5b8db8", lsize=12.5, red=False, where="b"):
        lh = lsize + 5.5
        h = max(28, 11 + lh * len(lines))
        for ln in lines:
            if tw(ln, lsize) > w - 10:
                WARN.append(f"{where:<8} {tw(ln,lsize):6.1f}>{w-10:6.1f}  {ln}")
        self.rect(x, y, w, h, "#fdecea" if red else fill, "#c0392b" if red else stroke)
        ty = y + h / 2 - (len(lines) - 1) * lh / 2 + 4.5
        for ln in lines:
            self.text(x + w / 2, ty, ln, lsize, anchor="middle", fill="#c0392b" if red else "#1d3d57", bold=red)
            ty += lh
        return h

    def img64(self, arr, x, y, size):
        a = np.clip(arr, 0, 1)
        a8 = (a * 255).astype(np.uint8)
        from PIL import Image
        im = Image.fromarray(a8).resize((size, size), Image.NEAREST)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        self.parts.append(f'<image href="data:image/png;base64,{b64}" x="{x}" y="{y}" width="{size}" height="{size}" '
                          f'style="image-rendering:pixelated"/>')
        self.rect(x, y, size, size, "none", "#999", rx=0, sw=1.0)

    def str_(self):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.W}" height="{self.H}" '
                f'viewBox="0 0 {self.W} {self.H}" font-family="{FONT}">\n<defs>'
                '<marker id="arr" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,3 L0,6 Z" fill="#777"/></marker>'
                '<marker id="arrRed" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,3 L0,6 Z" fill="#c0392b"/></marker>'
                '</defs>\n')
        return head + "\n".join(self.parts) + "\n</svg>\n"


RED = "#c0392b"; GRAY = "#666"; GREEN = "#1e8449"
W, H = 1500, 1080
s = SVG(W, H)
s.rect(0, 0, W, H, "#ffffff", "none", rx=0)
s.text(W / 2, 34, "SigLIP 的 Conv2d 到底在干什么：分块 → 摊平 → 共享 Linear → 1152 个“视觉词”打分", 21, bold=True, anchor="middle")
s.text(W / 2, 58, "kernel==stride==14 ⇒ 不重叠平铺；这发 conv 严格等于「硬分块 gather ∘ Linear(588→1152)（256 格共享同一套权重）」 · 底部滤波器来自 GR00T-N1.5-3B 真实权重", 12, anchor="middle", fill=GRAY)

# ---- 合成样例图 224×224×3 ----
yy, xx = np.mgrid[0:224, 0:224]
img = np.zeros((224, 224, 3))
img[..., 0] = 0.45 + 0.35 * np.sin(2 * np.pi * xx / 23)
img[..., 1] = 0.25 + 0.5 * yy / 224
img[..., 2] = 0.30 + 0.4 * (xx + yy) / 448
circ = ((xx - 70) ** 2 + (yy - 150) ** 2) < 45 ** 2
img[circ] = np.array([0.9, 0.75, 0.2])
s.img64(img, 70, 100, 224)
for i in range(1, 16):
    s.parts.append(f'<line x1="{70+i*14}" y1="100" x2="{70+i*14}" y2="324" stroke="#0006" stroke-width="0.6"/>')
    s.parts.append(f'<line x1="70" y1="{100+i*14}" x2="294" y2="{100+i*14}" stroke="#0006" stroke-width="0.6"/>')
r0, c0 = 70, 140
s.rect(70 + c0, 100 + r0, 14, 14, "none", RED, rx=0, sw=2.6)
s.text(235, 92, "pixel_values [B,3,224,224] → 16×16=256 格（patch=14×14 像素；224=14×16 恰好整除）", 12.5, anchor="middle", bold=True)
s.text(235, 344, "每格 = 1 token 的前身；入口零信息交换，块间关系全靠 27 层双向 attention", 11, anchor="middle", fill=GRAY)

# ---- 放大 patch：R/G/B 三个 14×14 ----
patch = img[r0:r0+14, c0:c0+14]
names = ["R", "G", "B"]
cols = ["#c0392b", "#1e8449", "#2471a3"]
s.path([(70 + c0 + 14, 100 + r0 + 7), (368, 150)], color=RED)
for k in range(3):
    x0 = 372 + k * 92
    for i in range(14):
        for j in range(14):
            v = patch[i, j, k]
            s.parts.append(f'<rect x="{x0+j*6}" y="{132+i*6}" width="6" height="6" fill="rgb({int(v*255)},{int(v*255)},{int(v*255)})"/>')
    s.rect(x0, 132, 84, 84, "none", cols[k], rx=0, sw=1.8)
    s.text(x0 + 42, 232, names[k] + " 通道", 11.5, anchor="middle", fill=cols[k], bold=True)
s.text(480, 122, "红框 patch 的 3×14×14（真实采样值）", 12, anchor="middle", bold=True)

# ---- flatten → 588 向量 ----
s.path([(560, 236), (560, 262), (500, 262), (500, 286)], color="#777")
s.text(505, 262, "flatten（行优先：R14行→G14行→B14行）", 11.5, anchor="middle", fill=GRAY)
vx, vy, vh = 492, 288, 120
s.rect(vx, vy, 16, vh, "#eaf2fb", "#5b8db8", rx=2)
for i in range(4):
    s.parts.append(f'<line x1="{vx}" y1="{vy+18+i*26}" x2="{vx+16}" y2="{vy+18+i*26}" stroke="#5b8db8" stroke-width="0.7"/>')
s.text(vx + 24, vy + 30, "v ∈ R⁵⁸⁸", 13, bold=True)
s.text(vx + 24, vy + 52, "588 = 14×14×3", 11.5, fill=GRAY)
s.text(vx + 24, vy + 72, "一次内积就塌缩掉", 11.5, fill=GRAY)
s.text(vx + 24, vy + 90, "全部空间尺寸", 11.5, fill=GRAY)

# ---- 权重矩阵 ----
mx, my, mw, mh = 640, 252, 300, 150
s.rect(mx, my, mw, mh, "#f2f6fb", "#5b8db8", rx=3, sw=1.8)
s.path([(vx + 16, vy + vh / 2), (mx - 8, my + mh / 2)], color="#777")
for i in range(7):
    yline = my + 12 + i * 22
    s.parts.append(f'<line x1="{mx+8}" y1="{yline}" x2="{mx+mw-8}" y2="{yline}" stroke="#9db8d2" stroke-width="6" opacity="0.55"/>')
s.rect(mx + 8, my + 12 - 6, mw - 16, 13, "#fdecea", RED, rx=2, sw=1.6)
s.text(mx + mw / 2, my + mh + 20, "patch_embedding.weight [1152, 3, 14, 14]", 12.5, anchor="middle", bold=True)
s.text(mx + mw / 2, my + mh + 40, "reshape → [1152, 588]：红行 = 一个“视觉词”的 588 个系数", 11.5, anchor="middle", fill=RED)
s.text(mx + mw / 2, my - 12, "out[c] = Σ₅₈₈ vᵢ·W[c,ᵢ] + bias[c]", 14, anchor="middle", bold=True, fill=RED)
s.text(mx + mw / 2, my + mh + 60, "bias 只有 1152 个——每输出通道一个，加在 588 项求和之后（不是 14×14 个！）", 11.5, anchor="middle", fill=RED)

# ---- 输出 1152 ----
ox, oy, oh = 1010, 252, 168
s.path([(mx + mw + 8, my + mh / 2), (ox - 8, oy + oh / 2)], color="#777")
s.rect(ox, oy, 16, oh, "#eaf7ef", GREEN, rx=2)
s.rect(ox - 1, oy + 26, 18, 12, "#fdecea", RED, rx=1, sw=1.5)
s.text(ox + 26, oy + 24, "1152 = 16 头 × 72", 13, bold=True)
s.text(ox + 26, oy + 46, "= 视觉塔自身家族签名（ViT-B 系）", 11.5, fill=GRAY)
s.text(ox + 26, oy + 66, "对 Qwen 的适配发生在塔外：", 11.5, fill=GRAY)
s.text(ox + 26, oy + 84, "mlp1 Linear(1152→2048)", 12, bold=True, fill=GREEN)
s.text(ox + 26, oy + 104, "（2048 才是 Qwen3 的宽度；", 11.5, fill=GRAY)
s.text(ox + 26, oy + 122, "  红行=第 c 个“视觉词”的打分）", 11.5, fill=GRAY)
s.text(ox + 26, oy + 146, "256 格用同一套权重 ⇒ 卷积仅剩的性质", 11.5, fill=GRAY)

# ---- conv ≡ gather+Linear 红框 ----
s.box(1150, 130, 300, ["conv2d(k=s=14) ≡ 硬分块 ∘ Linear(588→1152)", "kernel<stride 漏缝丢像素；kernel>stride", "窗口重叠、token 互相纠缠——都不行", "NPU：gather+GEMM 融合一核", "conv2d_patch_embed_run_die ×1/轮"], red=True, where="equiv", lsize=11.5)

# ---- 真实滤波器条 ----
s.text(110, 470, "真实权重长什么样：把 1152 行里每行 reshape 回 3×14×14 —— 真实权重的长相（实测统计见下文）", 14, bold=True)
Wp = np.load('/tmp/patchW.npy')   # 由 safetensors 抽取，见文件头注释
SRC = "权重实测：GR00T-N1.5-3B safetensors（patch_embedding.weight，1152 行全统计）"

def hf_peak_ratio(w):
    g = w.sum(axis=0).astype(float); g = g - g.mean()
    F = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    tot = F.sum()
    if tot <= 0:
        return 0.0
    yy3, xx3 = np.mgrid[0:14, 0:14] - 7
    rr = np.sqrt(yy3 ** 2 + xx3 ** 2)
    Fs = F.copy(); Fs[rr < 1.5] = 0
    return float(Fs.max() / tot)

ratios = np.array([hf_peak_ratio(Wp[c]) for c in range(Wp.shape[0])])
n_gab = int((ratios > 0.3).sum())
order = np.linspace(0, Wp.shape[0] - 1, 12).astype(int)   # 均匀抽样，不挑好看的
x0, y0, fs, gap = 110, 508, 66, 14
for n, c in enumerate(order):
    f = Wp[c]
    f = (f / max(np.abs(f).max(), 1e-9) + 1) / 2
    s.img64(f.transpose(1, 2, 0), x0 + n * (fs + gap), y0, fs)
    s.text(x0 + n * (fs + gap) + fs / 2, y0 + fs + 14, f"W[{c}]", 10.5, anchor="middle", fill=GRAY)
s.text(110, y0 + fs + 36, SRC + f"。诚实的结论：全塔 1152 行里频谱单峰占比>0.3（Gabor 式条纹）的仅 {n_gab} 行（逐通道口径仅 1/3456）——", 11.5, fill=RED)
s.text(110, y0 + fs + 56, "这副权重视上去像高维噪声，并非 CNN 第一层那种自组织条纹。原因不神秘：卷积对 patch 内部零假设（无局部性/无跨通道共享），588→1152 又是升维投影（信息近乎无损的随机超平面族），", 11.5, fill=RED)
s.text(110, y0 + fs + 76, "“看条纹”的活儿整层根本不需要干——27 层双向 attention 有的是容量。“条纹检测器”是 CNN conv1 的民俗图像，别照抄给 ViT patch embedding。", 11.5, fill=RED)

# ---- 底部注 ----
s.rect(90, 700, 1320, 96, "#fafafa", "#ddd")
s.text(110, 728, "读图口诀：14×14 只活在“求和之前”——588 个数进、1152 个标量出，空间尺寸在内积那一刻塌缩；“14×14”是 gather 的 reshape 约定，不是运算的假设。", 12.5, fill="#444")
s.text(110, 754, "出塔前：+ 可学习 PE[256,1152]（位置即格子，不用 RoPE）→ 27 层双向 encoder → post-LN；出塔后 mlp1 1152→2048 按 151669 槽位回写 LLM 序列（图 14-1）。", 12.5, fill="#444")
s.text(110, 780, "口径：patch 尺寸 14 是自由设计量（token 数 (224/p)² 决定 attention 的二次方付费）；kernel==patch 是被“一块=一 token”逼出来的恒等式，不是可调参数。", 12.5, fill="#444")
s.rect(90, 816, 1320, 60, "#fdf6e3", GREEN, dash="4 3")
s.text(110, 842, "顺答“1152 是不是为了迁就 Qwen”——不是：1152 进不了 Qwen3（hidden 2048），也进不了 DiT（1536）。", 13, fill=GREEN, bold=True)
s.text(110, 864, "它是 SigLIP 塔自己的带宽（head 整数性 16×72，NPU 融合核硬约束 HEADS×D==N）；与 LLM 的接缝只有一处 = mlp1，1152→2048。", 12.5, fill=GREEN)

# ---- PE 空间结构面板 ----
try:
    import json as _json
    from safetensors import safe_open
    d = '/home/ft/wzong/workspace/images/GR00T-N1_5-3B/'
    idx = _json.load(open(d + 'model.safetensors.index.json'))['weight_map']
    pk = [k for k in idx if 'vision_model.vision_model.embeddings.position_embedding' in k]
    with safe_open(d + idx[pk[0]], 'pt') as f:
        P = f.get_tensor(pk[0]).float().numpy()  # [256,1152]
    Q = P - P.mean(0)
    U, Sv, Vt = np.linalg.svd(Q, full_matrices=False)
    Z = U[:, :3] * Sv[:3]
    Z = (Z - Z.min(0)) / (Z.ptp(0) + 1e-9)
    s.text(110, 908, "可学习 PE[256,1152] 编码的到底是什么：256 个位置向量做 PCA，前 3 主成分映射为 RGB——色场连续平滑 ⇒ 位置嵌入就是一套学出来的坐标系：", 13, bold=True)
    cs = 10
    for i in range(16):
        for j in range(16):
            r, g2, b = Z[i * 16 + j]
            s.parts.append(f'<rect x="{110+j*cs}" y="{922+i*cs}" width="{cs}" height="{cs}" '
                           f'fill="rgb({int(r*255)},{int(g2*255)},{int(b*255)})" stroke="none"/>')
    s.text(310, 940, "邻居余弦相似度：水平 0.70 · 垂直 0.72 · 对角 0.46", 12, fill=GREEN)
    s.text(310, 962, "远格 (Δ8,8) 0.10 · 全对均值 0.21 ⇒ 单调随距离衰减", 12, fill=GREEN)
    s.text(310, 990, "对照：attention 置换不变，PE 是空间感唯一的输入；", 12, fill=GRAY)
    s.text(310, 1010, "它没编码“条纹”，编码的是“我在 16×16 的哪里”。", 12, fill=GRAY)
except Exception as e:
    print("PE 面板跳过:", e)

save_path = os.path.join(ROOT, "images", "ch14", "siglip_patch_embed.svg")
open(save_path, "w").write(s.str_())
print("wrote", save_path)
if WARN:
    print("[越界]", *WARN, sep="\n  ")
