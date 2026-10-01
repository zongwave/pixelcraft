#!/usr/bin/env python3
"""图 14-5：SigLIP 的“注意”到底住在哪一格——真实权重 + 真实 demo 帧的 α 路由图。

数据来源（都是真的，不是示意）：
  * /tmp/siglip_alpha.npz —— 由 tools/cap_ch14_siglip_alpha.py 生成：
    GR00T-N1.5-3B 真实 SigLIP 权重（剥前缀 backbone.eagle_model.vision_model. 后
    直接 load 进 transformers 的 SiglipVisionModel，missing=0）在真实 demo 帧
    （demo_data/cube_to_bowl_5 前相机第 0 帧，ffmpeg 抽帧后 resize 224×224）上，
    eager attention + output_attentions=True 抓到的 27 层 × 16 头 α（softmax 之后）。
  * 四张热力小图与右侧连线图都是同一 query patch（行优先下标 120 = 第 7 行第 8 列）
    在四个不同 (层, 头) 上的**同一行 α**。

前置：先跑一次 tools/cap_ch14_siglip_alpha.py（CPU 十几秒）。
渲染：python3 tools/mk_fig_ch14_alpha_routing.py &&
      rsvg-convert -w 1500 images/ch14/siglip_alpha_routing.svg -o /tmp/f145.png
"""
import base64
import io
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = "/tmp/siglip_alpha.npz"
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
        if tw(t, size) > self.W - 20:
            WARN.append(f"{tw(t,size):7.1f}>{self.W-20}  {t}")
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{"bold" if bold else "normal"}" '
                          f'fill="{fill}" text-anchor="{anchor}">{t.replace(chr(38), chr(38)+"amp;").replace("<", "&lt;").replace(">", "&gt;")}</text>')

    def line(self, x1, y1, x2, y2, color="#777", sw=1.5, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                          f'stroke-width="{sw}"{d}/>')

    def path(self, pts, color="#777", dash=None, sw=1.7, marker="arr"):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#{marker})"' if marker else ""
        self.parts.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in pts)}" fill="none" '
                          f'stroke="{color}" stroke-width="{sw}"{d}{m}/>')

    def label(self, x, y, t, size=11, fill="#ffffff"):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="bold" fill="{fill}" '
                          f'stroke="#000000" stroke-opacity="0.55" stroke-width="2.6" '
                          f'paint-order="stroke" text-anchor="middle">{t}</text>')

    def img64(self, arr, x, y, size):
        a8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        from PIL import Image
        im = Image.fromarray(a8).resize((size, size), Image.NEAREST)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        self.parts.append(f'<image href="data:image/png;base64,{b64}" x="{x}" y="{y}" width="{size}" '
                          f'height="{size}" style="image-rendering:pixelated"/>')

    def str_(self):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.W}" height="{self.H}" '
                f'viewBox="0 0 {self.W} {self.H}" font-family="{FONT}">\n<defs>'
                '<marker id="arr" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto" '
                'markerUnits="strokeWidth"><path d="M0,0 L8,3 L0,6 Z" fill="#777"/></marker>'
                '<marker id="arrRed" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto" '
                'markerUnits="strokeWidth"><path d="M0,0 L8,3 L0,6 Z" fill="#c0392b"/></marker>'
                '</defs>\n')
        return head + "\n".join(self.parts) + "\n</svg>\n"


RED = "#c0392b"; GRAY = "#666"; GREEN = "#1e8449"; BLUE = "#1f618d"

d = np.load(NPZ)
A = d["attn"].astype(np.float32)      # [27,16,256,256]
FRAME = d["frame"].astype(np.float32)  # [224,224,3] in 0..1
Q = int(d["qidx"]); QR, QC = divmod(Q, 16)
ENT = -(A * np.log(A + 1e-9)).sum(-1)


def objlab(rc):
    """按 demo 帧目视标的物体归属（读图辅助，非模型输出）。"""
    r, c = rc
    if 3 <= r <= 7 and 1 <= c <= 4:
        return "橙球"
    if 5 <= r <= 7 and 6 <= c <= 7:
        return "立方体"
    if 6 <= r <= 11 and 9 <= c <= 15:
        return "绿碗"
    if r >= 9 and 3 <= c <= 8:
        return "机械臂"
    return "桌面"


# 四个展示用的 (层, 头)：弥散 / 局部 / 汇 / 多物体
PANELS = [(0, 4, "A 弥散头", "≈全局平均池化"),
          (1, 2, "B 局部头", "只看紧邻格"),
          (12, 3, "C 注意力汇", "256 个 query 全指向同一格"),
          (1, 13, "D 多物体头", "同时摊到四个物体")]
FAN = (1, 13)   # 右侧连线图用哪一头

W, H = 1500, 992
s = SVG(W, H)
s.rect(0, 0, W, H, "#ffffff", "none", rx=0)
s.text(W / 2, 34, "SigLIP 的“注意”住在哪一格：同一个 patch 的 256 个权重，真实权重 + 真实 demo 帧实测", 21,
       bold=True, anchor="middle")
s.text(W / 2, 58,
       "GR00T-N1.5-3B 权重 · cube_to_bowl_5 前相机第 0 帧 · eager attention 抓 27 层 ×16 头 softmax 之后的 α"
       " · 四张小图与右图是同一行 α 的不同画法", 12, anchor="middle", fill=GRAY)

# ---------------- 左：输入帧 + query 高亮 ----------------
FX, FY, FS = 60, 108, 224
s.text(FX + FS / 2, 98, "输入帧 224×224 → 256 格", 12.5, anchor="middle", bold=True)
s.img64(FRAME, FX, FY, FS)
for i in range(1, 16):
    s.line(FX + i * 14, FY, FX + i * 14, FY + FS, "#00000044", 0.6)
    s.line(FX, FY + i * 14, FX + FS, FY + i * 14, "#00000044", 0.6)
s.rect(FX + QC * 14, FY + QR * 14, 14, 14, "none", RED, rx=0, sw=2.8)
s.text(FX + FS / 2, FY + FS + 20, f"query = 第 {Q} 格（行优先）= 第 {QR} 行第 {QC} 列", 12,
       anchor="middle", fill=RED, bold=True)
s.text(FX + FS / 2, FY + FS + 40, "本格内容：立方体右下缘旁的桌面", 11.5, anchor="middle", fill=GRAY)
s.text(FX + FS / 2, FY + FS + 60, "它自己不特殊——四张表给它发了不同指令", 11.5, anchor="middle", fill=GRAY)

# 物体名目视标注（左图旁，小字）
for (r, c), nm in [((6, 2), "橙球"), ((7, 6), "立方体"), ((9, 12), "绿碗"), ((12, 5), "机械臂")]:
    s.label(FX + c * 14 + 7, FY + r * 14 + 7, nm)

# ---------------- 中：四张 α 热力表 ----------------
s.text(660, 88, "同一个 query 的一行 α（256 个数，和为 1）——四种截然不同的分配", 13,
       anchor="middle", bold=True)
CELL = 9


def cmap(v):
    """v∈[0,1] → 白→橙→红→暗红。开方压缩让 0.02 量级也看得见。"""
    v = float(np.clip(v, 0, 1)) ** 0.5
    stops = [(0.0, (252, 252, 250)), (0.30, (253, 208, 120)), (0.62, (230, 110, 50)),
             (1.0, (120, 12, 26))]
    for i in range(1, len(stops)):
        if v <= stops[i][0]:
            a0, c0 = stops[i - 1]; a1, c1 = stops[i]
            t = (v - a0) / max(a1 - a0, 1e-9)
            return tuple(int(c0[k] + t * (c1[k] - c0[k])) for k in range(3))
    return stops[-1][1]


hx0, hy0, GW = 330, 116, 16 * CELL
for k, (L, h, nm, sub) in enumerate(PANELS):
    row, col = divmod(k, 2)
    x = hx0 + col * 340
    y = hy0 + row * 232
    row_a = A[L, h, Q, :]
    top1 = int(row_a.argmax())
    s.text(x, y - 12, f"{nm}  L{L} h{h}", 12.5, bold=True, fill=BLUE)
    s.text(x + 128, y - 12, sub, 11.5, fill=GRAY)
    for r in range(16):
        for c in range(16):
            rgb = cmap(row_a[r * 16 + c])
            s.parts.append(f'<rect x="{x+c*CELL}" y="{y+r*CELL}" width="{CELL}" height="{CELL}" '
                           f'fill="rgb({rgb[0]},{rgb[1]},{rgb[2]})" stroke="none"/>')
    s.rect(x, y, GW, GW, "none", "#999", rx=0, sw=1.2)
    tr, tc = divmod(top1, 16)
    s.rect(x + tc * CELL, y + tr * CELL, CELL, CELL, "none", RED, rx=0, sw=2.2)
    s.rect(x + QC * CELL, y + QR * CELL, CELL, CELL, "none", BLUE, rx=0, sw=2.0)
    ent = ENT[L, h, Q]
    t8 = np.sort(row_a)[-8:].sum()
    s.text(x + GW + 12, y + 14, f"top-1 = {row_a[top1]:.3f} → 格 {tr},{tc}（{objlab((tr,tc))}）", 11.5, fill=RED)
    s.text(x + GW + 12, y + 34, f"熵 = {ent:.2f}（均匀 = ln256 = 5.55）", 11.5, fill=GRAY)
    s.text(x + GW + 12, y + 54, f"top-8 合计 = {t8:.2f}", 11.5, fill=GRAY)
    if k == 0:
        s.text(x + GW + 12, y + 78, "1/n 的均匀表 ⇒ 这一头在“取全图平均”", 11, fill=GRAY)
    if k == 1:
        s.text(x + GW + 12, y + 78, "全图 256 个 query 的目标两两不同", 11, fill=GRAY)
        s.text(x + GW + 12, y + 96, "（239 个不同目标）⇒ 逐位置的局部滤波", 11, fill=GRAY)
    if k == 2:
        s.text(x + GW + 12, y + 78, "全图 256 个 query 的 top-1 都是它", 11, fill=RED)
        s.text(x + GW + 12, y + 96, "⇒ 这一头退化成一个“背景锚点”", 11, fill=RED)
    if k == 3:
        s.text(x + GW + 12, y + 78, "质量摊在 4 个物体上（见右图连线）", 11, fill=GRAY)
s.text(hx0 + 320, hy0 + 396, "蓝框 = query 自身所在格；红框 = 该行 top-1 所在格；色标对 256 个 α 开方压缩", 11.5,
       anchor="middle", fill=GRAY)

# ---------------- 右：连线路由图（D 头的 top-8） ----------------
GX, GY, GS = 1040, 130, 192
s.text(GX + GS / 2, 118, f"同一行 α 换画法：{FAN[0]} 层 {FAN[1]} 头 的 top-8 连线", 12.5,
       anchor="middle", bold=True)
s.img64(FRAME, GX, GY, GS)
cs = GS / 16.0
for i in range(1, 16):
    s.line(GX + i * cs, GY, GX + i * cs, GY + GS, "#00000033", 0.5)
    s.line(GX, GY + i * cs, GX + GS, GY + i * cs, "#00000033", 0.5)
row_a = A[FAN[0], FAN[1], Q, :]
tg = np.argsort(-row_a)[:8]
qx, qy = GX + (QC + .5) * cs, GY + (QR + .5) * cs
amax = row_a[tg[0]]
for t in tg:
    tr, tc = divmod(int(t), 16)
    tx, ty = GX + (tc + .5) * cs, GY + (tr + .5) * cs
    v = float(row_a[t])
    sw = 0.8 + 4.2 * (v / amax)
    s.parts.append(f'<line x1="{qx}" y1="{qy}" x2="{tx}" y2="{ty}" stroke="#c0392b" '
                   f'stroke-width="{sw:.2f}" stroke-opacity="0.75" stroke-linecap="round"/>')
    s.parts.append(f'<circle cx="{tx}" cy="{ty}" r="{3+5*v/amax:.1f}" fill="#c0392b" fill-opacity="0.9"/>')
s.rect(GX + QC * cs, GY + QR * cs, cs, cs, "none", BLUE, rx=0, sw=2.4)
s.text(GX + GS / 2, GY + GS + 20, "线宽/点径 ∝ α；蓝框 = query（它自己不参与连线）", 11.5,
       anchor="middle", fill=GRAY)
ly = GY + GS + 44
s.text(GX, ly, "top-8 明细（α 与所在物体）：", 12, bold=True)
for i, t in enumerate(tg):
    tr, tc = divmod(int(t), 16)
    tag = "自身" if int(t) == Q else objlab((tr, tc))
    s.text(GX + 6, ly + 20 + i * 18, f"α={row_a[t]:.3f}  格 {tr},{tc}  {tag}", 11.5,
           fill=RED if i < 3 else GRAY)

# ---------------- 底部三块 ----------------
BY = 546
s.rect(50, BY, 690, 250, "#f6f9fc", "#5b8db8")
s.text(70, BY + 26, "α 是什么：softmax 之后的那一行 256 个数（图 14-3 里 ⊗→softmax→⊗ 三格的核心）", 14,
       bold=True, fill=BLUE)
for i, ln in enumerate([
    "s_j = (q · k_j) / √72        j = 1..256  —— 本头 head_dim=72，72 是 1152/16",
    "α = softmax(s)              ⇒ 256 个非负数、和恰为 1",
    "o = Σ_j α_j · v_j           ⇒ 新的 1152 维向量送回残差流",
]):
    s.text(80, BY + 54 + i * 24, ln, 13)
s.line(70, BY + 132, 720, BY + 132, "#cfe0ef", 1.0)
for i, ln in enumerate([
    "三角色分工：q「我要找什么」· k「我这儿有什么、快来找我」· v「选中我就把我带走」",
    "“和为 1”是全部机关：这是一笔**竞争性预算**——给 (6,8) 多分了，给 (11,0) 就少了。",
    "所以 attention 的“注意”不是“算得多”，而是“同一份预算下按内容重新分配”。",
    "多头 = 16 笔预算同时并行（16 张互不相同的分配表）；27 层 = 图上 27 轮消息传递：",
    "    上一层的 o 变成下一层的输入 hidden，下一层的 q/k 由它现算 ⇒ 路由规则逐层改写。",
]):
    s.text(80, BY + 158 + i * 22, ln.replace("**", ""), 12.5)

s.rect(770, BY, 330, 250, "#fbf7ef", "#c8a45c")
s.text(790, BY + 26, "静态查表 vs 动态路由", 14, bold=True, fill="#8a6d1f")
s.rect(786, BY + 44, 148, 92, "#ffffff", "#ddd")
s.text(860, BY + 64, "卷积 / 查表", 12.5, anchor="middle", bold=True)
s.text(860, BY + 86, "权重与输入无关", 11, anchor="middle", fill=GRAY)
s.text(860, BY + 104, "学一次用到底", 11, anchor="middle", fill=GRAY)
s.text(860, BY + 122, "每格算法恒定", 11, anchor="middle", fill=GRAY)
s.rect(944, BY + 44, 148, 92, "#ffffff", "#c8a45c")
s.text(1018, BY + 64, "attention", 12.5, anchor="middle", bold=True)
s.text(1018, BY + 86, "权重由 q、k 现算", 11, anchor="middle", fill=GRAY)
s.text(1018, BY + 104, "每层每头每 query", 11, anchor="middle", fill=GRAY)
s.text(1018, BY + 122, "都有一张新表", 11, anchor="middle", fill=GRAY)
for i, ln in enumerate([
    "本图就是证据：B 头 256 个 query 各看自己的邻格",
    "（239 个不同目标）；C 头 256 个 query 又全被",
    "同一格 (11,0) 吸走——同层同头，只因内容不同",
    "而路由不同 ⇒ 表不是查出来的，是当场算的。",
    "推论：学的是规则（W_Q/W_K 怎么打分），",
    "算的是实例（本帧本格的这一行分配）。",
]):
    s.text(790, BY + 154 + i * 17, ln, 11.5)

s.rect(1130, BY, 320, 250, "#f4faf5", GREEN)
s.text(1150, BY + 26, "α 数学上存在，硬件上从不物化", 14, bold=True, fill=GREEN)
for i, ln in enumerate([
    "一层：16 张 256×256 表",
    "一塔：27 层 = 432 张 / 图",
    "α 个数 = 28,311,552 / 图（fp16 存下 54 MiB）",
    "三相机 batch 一趟 ⇒ 85 M 个 α / 162 MiB",
    "flash（online-softmax）分块滚动归一，",
    "整张 α 从不落进任何内存 —— 只存在寄存器里",
]):
    s.text(1150, BY + 52 + i * 21, ln, 12.5)
s.line(1140, BY + 186, 1440, BY + 186, "#bfe0c8", 1.0)
s.text(1150, BY + 208, "NPU 落点：一次发射吃掉整格运算", 12.5, bold=True, fill=GREEN)
s.text(1150, BY + 228, "unified_mha_run_die_batch_emb ×27（=27 层）", 12, fill=GREEN)

# 底部实测指纹条
s.rect(50, 816, 1400, 150, "#fafafa", "#ddd")
s.text(70, 842, "顺手量的三个事实（同一份 α，全 256 个 query 统计，不是挑出来的）", 13.5, bold=True)
for i, ln in enumerate([
    "① 头分工是真的：432 个头里，最弥散的熵 5.51（≈均匀 5.545，等价全局平均池化）；最尖锐的熵 0.01（top-1=0.999，几乎是一次“取值”）。",
    "② 局部性是 U 型不是单调：top-1 目标到 query 的曼哈顿距离中位数 L0=12 → L9=3 → L21=12；“≤2 格”占比 0.12 → 0.43 → 0.06 ⇒ 浅层先补局部、深层几乎全远程。",
    "③ 视觉塔里也有“注意力汇”：L12/L18/L20/L21/L22/L26 至少 9 个头把**全部 256 个 query** 的 top-1 投到同一格 (11,0)（左缘桌面）。它不是 bug——",
    "    softmax 需要一个“谁都不像”的格子来倾倒多余质量，与 LLM 里首个 token 吸走注意力的 sink 同源；读论文看到 attention map 全图发亮时先查汇。",
]):
    s.text(70, 870 + i * 22, ln.replace("**", ""), 12.5, fill="#333" if i < 3 else GRAY)

out = os.path.join(ROOT, "images", "ch14", "siglip_alpha_routing.svg")
open(out, "w").write(s.str_())
print("wrote", out)
if WARN:
    print("[越界]", *WARN, sep="\n  ")
