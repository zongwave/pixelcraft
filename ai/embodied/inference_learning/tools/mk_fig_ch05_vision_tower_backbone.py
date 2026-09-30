#!/usr/bin/env python3
"""第 5 章配图：GR00T backbone 两图（自绘，全部数字来自 config + ckpt 权重头 + tools/token_ledger_eagle.py 实测）。

生成：
  images/ch05/siglip_vision_tower.svg    图 5-1 视觉塔 SigLIP 结构框图（§2.5/§2.6 口径）
  images/ch05/backbone_eagle_vlm.svg     图 5-2 Eagle backbone 全景（视觉道+文本道→Qwen3 前 12 层→backbone_features）

数值来源（2026-09-29 实测）：
  vendored eagle2_hg_model/config.json（vision_config / text_config）
  GR00T-N1_5-3B model-0000x-of-00003.safetensors 权重头
  tools/token_ledger_eagle.py（token 账复跑）
  09 章 §3.5 NPU 补丁表（op 归属对号）

用法：python3 tools/mk_fig_ch05_vision_tower_backbone.py
"""
import os
import sys

FONT = "Helvetica, Arial, 'Noto Sans CJK SC', 'WenQuanYi Zen Hei', sans-serif"


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def tw(s, size):
    """粗略估宽：CJK 记 1.0 em，拉丁/数字记 0.56 em。用于排版自检。"""
    w = 0.0
    for ch in s:
        w += 1.0 if ord(ch) > 0x2E7F else 0.56
    return w * size


WARN = []


def chk(where, s_, size, avail):
    w = tw(s_, size)
    if w > avail:
        WARN.append("%-14s %7.1f > %.1f  %s" % (where, w, avail, s_))


class SVG:
    def __init__(self, w, h):
        self.W, self.H = w, h
        self.parts = []

    def rect(self, x, y, w, h, fill, stroke, rx=8, sw=1.6, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                          f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=13, fill="#222", bold=False, anchor="start", color=None):
        w = "bold" if bold else "normal"
        c = color or fill
        self.parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{w}" fill="{c}" '
            f'text-anchor="{anchor}">{esc(s)}</text>')

    def circle(self, cx, cy, r, fill, stroke, sw=1.6):
        self.parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" '
                          f'stroke="{stroke}" stroke-width="{sw}"/>')

    def line(self, x1, y1, x2, y2, color="#555", dash=None, sw=1.8):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                          f'stroke-width="{sw}"{d}/>')

    def arrow(self, x1, y1, x2, y2, color="#555", dash=None, sw=1.8, marker="arr"):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                          f'stroke-width="{sw}"{d} marker-end="url(#{marker})"/>')

    def box(self, x, y, w, lines, fill, stroke, title=None, tsize=13.5, tcolor="#222",
            lsize=12.5, lcolor="#444", pad=12, where="box"):
        lh = 18
        n = len(lines)
        h = pad * 2 + (20 if title else 0) + lh * n
        if title:
            chk(where + ":title", title, tsize, w - 2 * pad)
        for ln in lines:
            chk(where + ":line", ln, lsize, w - 2 * pad)
        self.rect(x, y, w, h, fill, stroke)
        cy = y + pad + 4
        if title:
            self.text(x + w / 2, cy + 12, title, tsize, bold=True, anchor="middle", color=tcolor)
            cy += 20
        for ln in lines:
            self.text(x + w / 2, cy + lh - 5, ln, lsize, anchor="middle", color=lcolor)
            cy += lh
        return y + h

    def str_(self):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.W}" height="{self.H}" '
                f'viewBox="0 0 {self.W} {self.H}" font-family="{FONT}">\n<defs>'
                '<marker id="arr" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,3 L0,6 Z" fill="#555"/></marker>'
                '<marker id="arrRed" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,3 L0,6 Z" fill="#c0392b"/></marker>'
                '<marker id="arrGreen" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,3 L0,6 Z" fill="#1e8449"/></marker>'
                '<marker id="arrBlue" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,3 L0,6 Z" fill="#1f6feb"/></marker>'
                '</defs>\n')
        return head + "\n".join(self.parts) + "\n</svg>\n"


OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images", "ch05")
os.makedirs(OUT, exist_ok=True)

BLUE, BLUE_L = "#1f6feb", "#eaf2fd"
RED, RED_L = "#c0392b", "#fdecea"
GREEN, GREEN_L = "#1e8449", "#eafaf1"
ORANGE, ORANGE_L = "#b9770e", "#fdf6e3"
PURPLE, PURPLE_L = "#7d3c98", "#f4ecf7"
GRAY, GRAY_L = "#888", "#f2f2f2"


# ============================================================================
# 图 5-1：SigLIP 视觉塔结构框图
# ============================================================================
def fig_vision_tower():
    W, H = 1260, 880
    s = SVG(W, H)
    s.rect(0, 0, W, H, "#fafafa", "none")
    s.text(W / 2, 40, "GR00T-N1.5 视觉塔本体：SigLIP ViT（so400m 系）结构框图", 23,
           bold=True, anchor="middle")
    s.text(W / 2, 66, "数字来源：eagle2_hg_model/config.json + 官方 ckpt 权重头实测（2026-09-29）· 单 tile 视角 · demo 单相机 = 3 tile 并行同构",
           12.5, anchor="middle", color="#666")

    CX, BW = 60, 700          # 主流程列
    RX, RW = 800, 410         # 右侧注释列
    mx = CX + BW / 2

    y = 92
    y = s.box(CX, y, BW, [
        "一个 tile = 224×224×3（tiling 后每 tile 定尺缩进）",
        "demo 单相机 224×224 → 1×1 网格 1 个 tile（use_thumbnail，1×1 时缩略图不计）"],
        "#fff", "#aaa", title="输入：单 tile", where="A1") + 16
    s.arrow(mx, y - 16, mx, y + 2); y += 4

    y = s.box(CX, y, BW, [
        "Conv2d kernel 14×14 / stride 14，权重 [1152, 3, 14, 14]",
        "224 / 14 = 16 ⇒ 16×16 = 256 个 patch，展平 3·14·14 = 588 → 投到 1152"],
        BLUE_L, BLUE, title="① patch-embed（视觉塔唯一的卷积）", tcolor=BLUE, where="A2") + 16
    s.arrow(mx, y - 16, mx, y + 2); y += 4

    y = s.box(CX, y, BW, [
        "学习式绝对 PE，权重 [256, 1152]：无 CLS 位、无 RoPE",
        "tile 定尺 224 ⇒ 位置格固定 16×16，任何分辨率都先归格，无需插值"],
        BLUE_L, BLUE, title="② + 位置编码（embeddings = patch + PE）", tcolor=BLUE, where="A3") + 16
    s.arrow(mx, y - 16, mx, y + 2); y += 4

    # —— ③ ×27 block 放大框 ——
    bh = 250
    s.rect(CX, y, BW, bh, "#ffffff", BLUE, 12, 2.2)
    s.text(CX + BW / 2, y + 28, "③ × 27 Transformer block（pre-LN · encoder-only · 无 causal mask）",
           15, bold=True, anchor="middle", color=BLUE)
    iy = y + 48
    sw_, gap = 130, 24
    x1 = CX + 40
    # 行1：LN1 → MHSA → ⊕
    s.rect(x1, iy, sw_, 46, GRAY_L, "#999"); s.text(x1 + sw_ / 2, iy + 29, "LayerNorm₁", 12.5, anchor="middle")
    s.arrow(x1 + sw_, iy + 23, x1 + sw_ + gap, iy + 23)
    x2 = x1 + sw_ + gap
    s.rect(x2, iy, 250, 46, BLUE_L, BLUE); s.text(x2 + 125, iy + 20, "MHSA · 16 头 · 1152", 12.5, anchor="middle", color=BLUE)
    s.text(x2 + 125, iy + 38, "建模代码强制 flash_attention_2", 11, anchor="middle", color="#666")
    s.arrow(x2 + 250, iy + 23, x2 + 250 + gap, iy + 23)
    cx_plus1 = x2 + 250 + gap + 22
    s.circle(cx_plus1, iy + 23, 13, "#fff", "#555", 1.8)
    s.text(cx_plus1, iy + 28, "⊕", 15, anchor="middle", bold=True)
    s.line(CX + 24, iy + 23, CX + 24, iy + 130, "#999", dash="5,4")
    s.line(CX + 24, iy + 23, x1, iy + 23, "#999", dash="5,4")
    s.arrow(CX + 24, iy + 130, cx_plus1 - 15, iy + 130, "#999", dash="5,4")
    s.line(cx_plus1, iy + 36, cx_plus1, iy + 130, "#999", dash="5,4", sw=1.4)
    s.text(CX + 24, iy + 78, "残差", 11, color="#999", anchor="middle")
    iy += 126
    # 行2：LN2 → MLP → ⊕
    s.rect(x1, iy, sw_, 46, GRAY_L, "#999"); s.text(x1 + sw_ / 2, iy + 29, "LayerNorm₂", 12.5, anchor="middle")
    s.arrow(x1 + sw_, iy + 23, x1 + sw_ + gap, iy + 23)
    s.rect(x2, iy, 250, 46, GREEN_L, GREEN); s.text(x2 + 125, iy + 20, "MLP 1152 → 4304 → 1152", 12.5, anchor="middle", color=GREEN)
    s.text(x2 + 125, iy + 38, "激活 gelu_pytorch_tanh", 11, anchor="middle", color="#666")
    s.arrow(x2 + 250, iy + 23, x2 + 250 + gap, iy + 23)
    cx_plus2 = cx_plus1
    s.circle(cx_plus2, iy + 23, 13, "#fff", "#555", 1.8)
    s.text(cx_plus2, iy + 28, "⊕", 15, anchor="middle", bold=True)
    s.text(cx_plus2 + 30, iy + 28, "→ 下一层输入 256×1152", 12, color="#555")
    chk("A3-box", "③ × 27 Transformer block（pre-LN · encoder-only · 无 causal mask）", 15, BW - 40)
    chk("A3-side", "→ 下一层输入 256×1152", 12, CX + BW - (cx_plus2 + 30) - 12)
    y += bh + 16
    s.arrow(mx, y - 16, mx, y + 2); y += 4

    y = s.box(CX, y, BW, [
        "encoder 尾部的全局 LayerNorm（ckpt 键 post_layernorm.*，pre-LN 架构的出口归一）"],
        GRAY_L, "#999", title="④ post_layernorm", where="A4") + 16
    s.arrow(mx, y - 16, mx, y + 2); y += 4

    y = s.box(CX, y, BW, [
        "→ mlp1 = Linear(1152 → 2048)，逐 patch 单发直通（§2.6，权重 [2048,1152] 终审）",
        "→ 256 个 2048 维视觉 token，挤进语言塔门口的 image_pad 位（图 5-2）"],
        ORANGE_L, ORANGE, title="输出：256 × 1152 / tile（无 CLS · 无洗牌）", tcolor=ORANGE, where="A5")
    chk("A-canvas", "", 12, 10)  # 占位，高度检查在下方

    # —— 右列：NPU 补丁对号 ——
    ry = 92
    ry = s.box(RX, ry, RW, [
        "conv2d_patch_embed        → ①",
        "NPU_SiglipVisionEmbeddings → ①+②",
        "NPU_SiglipVisionAttention  → MHSA",
        "NPU_SiglipVisionMLP        → MLP",
        "（融合核 mlp_gelu 吃的正是 gelu_pytorch_tanh）"],
        GREEN_L, GREEN, title="09 §3.5 补丁表在本图对号入座", tcolor=GREEN, lsize=12, where="A-NPU")
    s.line(RX - 8, 92, RX - 8, ry, GREEN, sw=3)

    ry = s.box(RX, ry + 20, RW, [
        "PE 固定在 16×16 格、patch 定尺 14×14：",
        "几何畸变（如 02 §7 的内容拉伸 1.6×）不改变",
        "任何张量形状——错误内容被摊进正确的格子，",
        "形状级监控天然失明，必须往像素/内容层查",
        "（02 §7.4 事故 → 这里的机制版解释）"],
        RED_L, RED, title="为什么这套结构对畸变「形状失明」", tcolor=RED, lsize=12.5, where="A-blind")

    ry = s.box(RX, ry + 20, RW, [
        "tune_visual = True：视觉塔可训（本节图内全蓝）",
        "桥 mlp1 属 tune_projector 训练的投影件",
        "推理期两者都 eval + 冻结在 ckpt 权重上"],
        "#fff", "#aaa", title="冻结开关（§3）", lsize=12.5, where="A-frozen")

    if y > H - 10 or ry > H - 10:
        WARN.append("canvas height: main=%d right=%d > H-10=%d" % (y, ry, H - 10))
    return s, W, H


# ============================================================================
# 图 5-2：Eagle backbone 全景（双道 → 拼接 → Qwen3 前 12 层 → backbone_features）
# ============================================================================
def fig_backbone():
    W, H = 1560, 1080
    s = SVG(W, H)
    s.rect(0, 0, W, H, "#fafafa", "none")
    s.text(W / 2, 40, "EagleBackbone 前向全景：图像 + 文本 → backbone_features [B, 296, 2048]", 23,
           bold=True, anchor="middle")
    s.text(W / 2, 66, "口径 = vendored eagle2_hg_model（n1.5-release 与 main 零 diff）· token 账实测 tools/token_ledger_eagle.py · 示例 = demo 单相机单 tile",
           12.5, anchor="middle", color="#666")

    LW = 1120                   # 左区宽
    RX, RW = 1180, 350          # 右注释列

    # —— 视觉道泳道 ——
    s.rect(30, 88, LW, 258, BLUE_L, BLUE, 12, 2)
    s.text(50, 116, "视觉道（image → visual tokens）", 15, bold=True, color=BLUE)
    bw, gap, by, bh = 196, 28, 136, 178
    xs = [50 + i * (bw + gap) for i in range(5)]
    cells = [
        ("输入图像 ×N", ["demo：单相机 1 张", "224×224×3"], "#fff", "#aaa", "#222"),
        ("动态 tiling", ["tile=224 · 按纵横比挑网格", "max 12 · 1×1 时缩略图不计", "demo = 1 tile"], "#fff", "#aaa", "#222"),
        ("SigLIP 视觉塔", ["27 层 × 1152 × 16 头", "结构细节 = 图 5-1", "tune_visual=True 可训"], BLUE_L, BLUE, BLUE),
        ("256 patch token / tile", ["无 CLS · 无洗牌", "downsample=0.5 是死字段"], "#fff", "#aaa", "#222"),
        ("mlp1 = Linear(1152→2048)", ["逐 patch 单发 · 无空间聚合", "tune_projector 投影件"], ORANGE_L, ORANGE, ORANGE),
    ]
    for i, (t, lns, f, st, tc) in enumerate(cells):
        bh_i = s.box(xs[i], by, bw, lns, f, st, title=t, tsize=12.5, tcolor=tc, lsize=11.5,
                     pad=10, where="B-vis%d" % i)
        if i < 4:
            s.arrow(xs[i] + bw, by + 60, xs[i + 1] - 4, by + 60, marker="arrBlue", color=BLUE)

    # —— 文本道泳道 ——
    s.rect(30, 366, LW, 150, ORANGE_L, ORANGE, 12, 2)
    s.text(50, 394, "文本道（chat 模板 + 指令 → text tokens）", 15, bold=True, color=ORANGE)
    t_cells = [(50, 330, "chat 模板 26 + tokenize(指令) 14", ["demo 指令 14 token ⇒ 共 40"]),
               (430, 300, "token embedding", ["词表 → 2048 维"]),
               (780, 240, "40 × 2048", ["文本 token"])]
    for x, w, t, lns in t_cells:
        s.box(x, 412, w, lns, "#fff", "#aaa", title=t, tsize=12.5, lsize=11.5, pad=10,
              where="B-txt")
    s.arrow(380, 445, 426, 445); s.arrow(730, 445, 776, 445)

    # —— 拼接 ——
    my = 544
    m = s.box(270, my, 640, [
        "image_pad（id=151669）原位替换为 256 个视觉 token",
        "seq = 256×tiles + 模板26 + 指令14 = 296 ⇒ 进塔 [B, 296, 2048]"],
        "#fff", "#555", title="拼接：一个序列喂进语言塔", tsize=14, where="B-merge")
    # 视觉道：沿右侧空廊（1020~1150 之间）正交下行，从右缘进拼接框，不再斜穿文本道
    s.line(1044, 218, 1044, 584, BLUE, sw=1.8)
    s.arrow(1044, 584, 916, 584, color=BLUE, marker="arrBlue")
    s.text(1054, 560, "视觉 256×2048", 11, color=BLUE)
    chk("B-merge-note", "视觉 256×2048", 11, 1176 - 1054)
    # 文本道：竖直一段进拼接框顶缘
    s.arrow(900, 476, 900, my - 2, color=ORANGE)

    # —— Qwen3 大框 ——
    qy = m + 18
    qw, qh = 760, 208
    qx = 180
    s.rect(qx, qy, qw, qh, PURPLE_L, PURPLE, 12, 2.2)
    s.text(qx + qw / 2, qy + 28, "Qwen3 语言塔：config 共 28 层，gr00t select_layer=12 只用前 12 层",
           14.5, bold=True, anchor="middle", color=PURPLE)
    # 层条：12 实心 + 剪刀 + 16 灰
    lx, ly, lw_, lh_, lg = qx + 36, qy + 48, 30, 26, 4
    for i in range(12):
        s.rect(lx + i * (lw_ + lg), ly, lw_, lh_, PURPLE, PURPLE, 3, 0)
    cutx = lx + 12 * (lw_ + lg) + 6
    s.text(cutx + 10, ly + 20, "✂", 18, anchor="middle", color=RED)
    gx = cutx + 34
    for i in range(8):
        s.rect(gx + i * (lw_ + lg), ly, lw_, lh_, GRAY_L, "#bbb", 3, 0)
        s.rect(gx + i * (lw_ + lg), ly + lh_ + 4, lw_, lh_, GRAY_L, "#bbb", 3, 0)
    s.text(qx + qw / 2, ly + lh_ * 2 + 26, "后 16 层整段裁掉，从不执行（不是「跳过执行」，是根本没实例化进来）",
           12, anchor="middle", color="#999")
    s.text(qx + qw / 2, qy + qh - 42, "GQA 16Q/8KV · head_dim 128 · FFN 6144 SwiGLU · RoPE θ=10⁶",
           12.5, anchor="middle", color="#555")
    s.text(qx + qw / 2, qy + qh - 20, "tune_llm = False：全冻结 + eval（set_frozen_modules_to_eval_mode）",
           12.5, anchor="middle", color=PURPLE)
    chk("B-qwen-title", "Qwen3 语言塔：config 共 28 层，gr00t select_layer=12 只用前 12 层", 14.5, qw - 40)
    chk("B-qwen-cut", "后 16 层整段裁掉，从不执行（不是「跳过执行」，是根本没实例化进来）", 12, qw - 40)

    # —— 输出链 ——
    oy = qy + qh + 30
    o_cells = [(qx - 20, 250, "hidden_states[12]", ["output_hidden_states=True 取第 12 层"]),
               (qx + 270, 230, "eagle_linear = Identity", ["project_to_dim=null ⇒ 不投影"]),
               (qx + 540, 260, "backbone_features + mask",
                 ["[B, 296, 2048] · mask 传而不达", "（06 §6.8.5：掩码进接口但未进注意力）"])]
    for x, w, t, lns in o_cells:
        s.box(x, oy, w, lns, "#fff", "#555", title=t, tsize=12.5, lsize=11.5, pad=10, where="B-out")
    s.arrow(qx + qw / 2, qy + qh + 2, qx + qw / 2, oy - 2)
    s.arrow(qx - 20 + 250, oy + 40, qx + 270 - 4, oy + 40)
    s.arrow(qx + 270 + 230, oy + 40, qx + 540 - 4, oy + 40)
    s.box(qx + 540, oy + 108, 250, ["当 cross-attn 的 K/V（06 §6.8）"],
          GREEN_L, GREEN, title="下游 → DiT action head", tsize=12, lsize=11.5, pad=8, where="B-dit")
    s.arrow(qx + 540 + 125, oy + 96, qx + 540 + 125, oy + 104, marker="arrGreen", color=GREEN)

    # —— 右列注释 ——
    ry = 88
    ry = s.box(RX, ry, RW, [
        "单相机 224²：256+26+14 = 296",
        "空指令：282 · 三相机：768+45 = 813",
        "（模板不随图数套 26，每加图约半成）",
        "640×640 → 10 tiles → 2586",
        "640×400 → 7 tiles → 1818"],
        "#fff", "#aaa", title="token 账（复跑 tools/token_ledger_eagle.py）", lsize=12, where="B-ledger")
    ry = s.box(RX, ry + 18, RW, [
        "视觉塔 conv2d + 3×NPU_Siglip*（图 5-1）",
        "Qwen3 层 6 个补丁（Attention/MLP 等）",
        "mlp1 → NPULinear（gemm_bias）",
        "09 §3.5 · v0.2 交付口径"],
        GREEN_L, GREEN, title="NPU 补丁落点（本章全蓝/橙处）", tcolor=GREEN, lsize=12, where="B-npu")
    ry = s.box(RX, ry + 18, RW, [
        "一次 get_action 视觉+语言塔只跑一次",
        "296 token 全序列进 12 层 ≈ prefill；",
        "去噪循环复用其特征 —— 与 06 §6.7",
        "prefill/decode 同构结论的出处"],
        "#fff", "#aaa", title="推理期视角（09 §7 对账前置）", lsize=12, where="B-infer")
    ry = s.box(RX, ry + 18, RW, [
        "eagle_ 前缀剥掉后交 VLM；image_sizes 剔除",
        "输出键固定 backbone_features /",
        "backbone_attention_mask（04 validate_data）"],
        "#fff", "#aaa", title="forward 接口约定", lsize=12, where="B-api")

    if oy + 178 > H - 10 or ry > H - 10:
        WARN.append("canvas height: out=%d right=%d vs H-10=%d" % (oy + 178, ry, H - 10))
    return s, W, H


def save(name, s, W, H):
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        f.write(s.str_())
    print("wrote %s (%dx%d)" % (path, W, H))


s1, w1, h1 = fig_vision_tower()
s2, w2, h2 = fig_backbone()

if WARN:
    print("排版越界自检未通过：")
    for w in WARN:
        print("  " + w)
    sys.exit(1)

save("siglip_vision_tower.svg", s1, w1, h1)
save("backbone_eagle_vlm.svg", s2, w2, h2)
print("OK: 2 figures, self-check passed")
