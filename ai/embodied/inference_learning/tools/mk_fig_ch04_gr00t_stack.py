#!/usr/bin/env python3
"""第 04 章配图（图 4-1）：gr00t 三件套前向框架图——对标 llama_decoder.png 的画法。

竖排主链 + 嵌套虚线容器（GR00T_N1_5 / 脑1 VLM / 脑2 ActionHead / 各"×N"层容器）
+ 蓝色模块盒 + 红框关键机制。数字均 n1.5-release 实测（04 §1.1 / 05 §2.5 / 06 / 14 章）。

生成：images/ch04/gr00t_three_stack_framework.svg
用法：python3 tools/mk_fig_ch04_gr00t_stack.py
"""
import os
import sys

FONT = "Helvetica, Arial, 'Noto Sans CJK SC', 'WenQuanYi Zen Hei', sans-serif"


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def tw(s, size):
    return sum(1.0 if ord(c) > 0x2E7F else 0.56 for c in s) * size


WARN = []


def chk(where, s_, size, avail):
    w = tw(s_, size)
    if w > avail:
        WARN.append("%-14s %7.1f > %.1f  %s" % (where, w, avail, s_))


class SVG:
    def __init__(self, w, h):
        self.W, self.H = w, h
        self.parts = []

    def rect(self, x, y, w, h, fill, stroke, rx=6, sw=1.6, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                          f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=13, fill="#222", bold=False, anchor="start"):
        w = "bold" if bold else "normal"
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{w}" '
                          f'fill="{fill}" text-anchor="{anchor}">{esc(s)}</text>')

    def path(self, pts, color="#777", dash=None, sw=1.7, marker="arr"):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        p = " ".join(f"{x},{y}" for x, y in pts)
        self.parts.append(f'<polyline points="{p}" fill="none" stroke="{color}" '
                          f'stroke-width="{sw}"{d} marker-end="url(#{marker})"/>')

    def node(self, cx, cy, label, r=13, stroke="#555", fill="#fff", color="#333", size=13):
        self.parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" '
                          f'stroke="{stroke}" stroke-width="1.6"/>')
        self.text(cx, cy + 4.5, label, size, fill=color, bold=True, anchor="middle")

    def box(self, x, y, w, lines, fill="#cfe3f7", stroke="#5b8db8", lsize=12.5,
            lcolor="#1d3d57", where="b", red=False):
        """蓝盒（参考图同款）：多行、垂直居中，返回 (y_bottom, x, y_top)。"""
        lh = lsize + 5.5
        h = max(30, 12 + lh * len(lines))
        for ln in lines:
            chk(where, ln, lsize, w - 8)
        self.rect(x, y, w, h, "#fdecea" if red else fill, "#c0392b" if red else stroke)
        ty = y + h / 2 - (len(lines) - 1) * lh / 2 + 4.5
        for ln in lines:
            self.text(x + w / 2, ty, ln, lsize, anchor="middle",
                      fill="#c0392b" if red else lcolor, bold=red)
            ty += lh
        return y + h


OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images", "ch04")
os.makedirs(OUT, exist_ok=True)

BLUE_L, BLUE_S = "#cfe3f7", "#5b8db8"
RED = "#c0392b"
GRAY = "#666"

W, H = 1500, 1900
s = SVG(W, H)
s.rect(0, 0, W, H, "#ffffff", "none", rx=0)
s.text(W / 2, 34, "GR00T-N1.5 前向框架图：一个 VLM（ViT 嵌在其内）+ 一个 DiT", 22, bold=True, anchor="middle")
s.text(W / 2, 58, "画法对标 llama_decoder.png：竖排数据流 + 虚线容器=类/模块边界 + 蓝盒=算子/子模块 + 红=关键机制。数字 n1.5-release 实测（04 §1.1 / 05 §2.5 / 06 / 14 章）",
       12, anchor="middle", fill="#777")

CX = 300          # 主链中心线
mx = CX

def arr_down(y1, y2, x=None, label=None, color="#777", dash=None):
    x = x or mx
    s.path([(x, y1), (x, y2)], color=color, dash=dash)
    if label:
        s.text(x + 8, (y1 + y2) / 2 + 4, label, 11.5, fill=GRAY)

# ---------------- 入口 ----------------
y = s.box(CX - 120, 76, 240, ["obs 输入", "3 cam 224² + state + 语言"], fill="#fff",
          stroke="#999", where="start")
arr_down(y + 2, 152, label="inputs")
y2 = s.box(CX - 280, 152, 560, ["collate_fn：图像 resize/normalize；tokenize；key 加 eagle_ 前缀",
                                "(gr00t/model/transforms.py:55-83)"],
           fill="#f2f2f2", stroke="#999", where="collate")

# ---------------- GR00T_N1_5 大容器 ----------------
s.rect(60, 228, 1380, 1406, "none", "#1f6feb", rx=18, dash="7 5", sw=2)
s.text(W / 2, 250, "GR00T_N1_5  (gr00t_n1.py) —— 双脑容器", 15, bold=True, fill="#1f6feb", anchor="middle")

# ---------------- 脑1 ----------------
s.rect(90, 262, 1320, 842, "none", "#c0392b", rx=16, dash="6 5", sw=1.8)
s.text(700, 286, "脑1 · EagleBackbone = Eagle-2.5-VLM（整体冻结；每轮 infer 只跑 1 遍，产出即 backbone_features）",
       14.5, bold=True, fill=RED)

# ---- SigLIP 容器 ----
s.rect(120, 302, 640, 356, "none", "#b9770e", rx=12, dash="5 4", sw=1.8)
s.text(440, 322, "SiglipVisionModel（“ViT”·实现住在 transformers 4.51.3，14 章）", 13,
       bold=True, fill="#b9770e", anchor="middle")
pv = s.box(CX - 130, 334, 260, ["pixel_values [B,3,224,224]"], where="pv")
arr_down(pv + 2, 388)
cv = s.box(CX - 130, 388, 260, ["Conv2d 3→1152, k14×14 s14", "→ 16×16 = 256 patch"], where="conv")
arr_down(cv + 2, 452)
pe = s.box(CX - 130, 452, 260, ["+ 可学习 PE [256,1152]（无 CLS）"], where="pe")
arr_down(pe + 2, 512)
enc = s.box(CX - 130, 512, 260, ["27 × SiglipEncoderLayer"], where="enc", lsize=14)
arr_down(enc + 2, 572)
pln = s.box(CX - 130, 572, 260, ["post-LN → [B, n_img, 1152]"], where="pln")
# EncoderLayer 展开
s.path([(CX + 132, 534), (472, 534)], color="#999", dash="4 3")
s.rect(472, 342, 268, 296, "none", "#999", rx=10, dash="4 3")
s.text(606, 360, "1 层展开（pre-norm）", 12, bold=True, anchor="middle", fill="#555")
ln1 = s.box(496, 370, 220, ["LayerNorm"], where="ln1", lsize=12)
arr_down(ln1 + 2, 414, x=606)
mha = s.box(496, 414, 220, ["MHA（flash-attn2）", "q/k/v·o proj"], where="mha", lsize=12)
s.node(606, 478, "+")
s.path([(496, 434), (466, 434), (466, 478), (591, 478)], color="#999")
arr_down(434 + 12, 463, x=606)
arr_down(493, 512, x=606)
ln2 = s.box(496, 512, 220, ["LayerNorm"], where="ln2", lsize=12)
arr_down(ln2 + 2, 554, x=606)
mlp = s.box(496, 554, 220, ["MLP fc1 1152→4304", "gelu_tanh · fc2 →1152"], where="mlp", lsize=12)
s.node(606, 618, "+")
s.path([(606, 596), (606, 603)], color="#999")
s.path([(496, 574), (466, 574), (466, 618), (591, 618)], color="#999")
# ---- mlp1 ----
arr_down(pln + 2, 680, label="extract_feature 唯一调用点（:311）", )
mp1 = s.box(CX - 130, 680, 260, ["mlp1（桥）1152 → 2048", "extract_feature :334"], where="mlp1")

# ---- 文本分支 ----
t1 = s.box(770, 334, 250, ["input_ids（含 image_token", "151669 占位 ×256×n_cam）"],
           fill="#f2f2f2", stroke="#999", where="ids")
arr_down(t1 + 2, 424, x=895)
t2 = s.box(770, 424, 250, ["LM embedding → [B,296,2048]"], fill="#f2f2f2", stroke="#999", where="emb")
s.path([(895, 462), (895, 756), (440, 756)], color="#999")

# ---- 原位替换（红，对标 KV Update 的位置）----
rep = s.box(CX - 140, 730, 280, ["原位替换（红框机制）", "img_embeds → 151669 位置 (:247)"],
            red=True, where="rep")
arr_down(mp1 + 2, 728)
s.text(CX, 796, "inputs_embeds [B,296,2048] 图文混合（demo 296 = 256+26+14）", 12,
       anchor="middle", fill=GRAY)
arr_down(802, 830)

# ---- Qwen3 容器 ----
s.rect(120, 830, 640, 190, "none", "#7d3c98", rx=12, dash="5 4", sw=1.8)
s.text(440, 850, "Qwen3 语言塔（“LLM”·加载时 layers.pop 裁到前 12 层，:56-58）", 13,
       bold=True, fill="#7d3c98", anchor="middle")
q12 = s.box(CX - 140, 866, 280, ["12 × Qwen3DecoderLayer"], where="q12", lsize=14)
s.text(CX, 930, "※ 无 lm_head、不生成 token——", 12, anchor="middle", fill=RED)
s.text(CX, 948, "只 tap hidden_states[12] 当条件特征", 12, anchor="middle", fill=RED)
s.text(CX, 972, "（与视觉塔 config 里那个 select_layer 同名不同物！05/14 章）", 11.5,
       anchor="middle", fill=GRAY)
s.path([(CX + 142, 890), (770, 890)], color="#999", dash="4 3")
s.rect(770, 856, 610, 150, "none", "#999", rx=10, dash="4 3")
s.text(1075, 876, "1 层展开（pre-norm, RoPE, GQA）", 12, bold=True, anchor="middle", fill="#555")
qx = 786
for lab, wdt in [("RMSNorm", 84), ("Self-Attn", 92), ("+", 26), ("RMSNorm", 84), ("SwiGLU MLP", 106), ("+", 26)]:
    if lab == "+":
        s.node(qx + 13, 912, "+", r=13)
        qx += 26 + 16
    else:
        s.box(qx, 896, wdt, [lab], where="qx", lsize=11.5)
        qx += wdt + 16
s.path([(1271, 912), (1271, 950)], color="#999")
s.text(1075, 972, "trace: qwen3_attention_run_die ×12 + mlp_swiglu ×12（09 章）", 11,
       anchor="middle", fill=GRAY)

# ---- 出脑1 ----
arr_down(1020 + 8, 1052)
el = s.box(CX - 130, 1052, 260, ["eagle_linear（adapter）", "→ backbone_features [B,296,2048]"],
           where="el")
s.text(700, 1070, "= 脑2 全部 cross-attn 的 K/V：4 步去噪固定不变 → 可缓存（06 §6.7）", 13,
       bold=True, fill=RED)
s.parts.append('<polyline points="700,1082 1235,1082" fill="none" stroke="#c0392b" stroke-width="1.7" stroke-dasharray="7 4"/>')

# ---------------- 脑2 ----------------
s.rect(90, 1124, 1320, 486, "none", "#1e8449", rx=16, dash="6 5", sw=1.8)
s.text(700, 1148, "脑2 · FlowmatchingActionHead（DiT·可训练；“K iterations”去噪循环在这里）",
       14.5, bold=True, fill="#1e8449")
at = s.box(CX - 130, 1166, 260, ["noise a_t [B,16,32]（randn）"], where="at")
arr_down(at + 2, 1226)
ip = s.box(CX - 130, 1226, 260, ["in_proj 32→1536（+state token）"], where="ip")
st = s.box(560, 1226, 200, ["state 32 → encoder → 1536"], fill="#f2f2f2", stroke="#999", where="st")
s.path([(558, 1246), (434, 1246)], color="#999")
s.text(660, 1278, "horizon=16, action_dim=32", 11, anchor="middle", fill=GRAY)
arr_down(ip + 2, 1310)
tb = s.box(560, 1306, 150, ["t → sinusoidal", "→ AdaLN 调制"], fill="#f2f2f2", stroke="#999", where="tb")
s.path([(558, 1334), (434, 1334)], color="#999")
dit = s.box(CX - 130, 1310, 260, ["16 × DiT Block", "hidden 1536 · 偶数层 cross"], where="dit", lsize=14)
s.path([(432, 1374), (500, 1374), (500, 1444), (768, 1444)], color="#999", dash="4 3")
s.rect(770, 1180, 620, 316, "none", "#999", rx=10, dash="4 3")
s.text(1080, 1200, "1 层 DiT Block 展开（06 §6.8：三种注意力各归其位）", 12, bold=True,
       anchor="middle", fill="#555")
dx = 790
seq = [("AdaLN", 76), ("Self-Attn", 92), ("+", 26), ("Cross-Attn", 96), ("+", 26)]
for lab, wdt in seq:
    if lab == "+":
        s.node(dx + 13, 1240, "+", r=13)
        dx += 26 + 16
    else:
        s.box(dx, 1222, wdt, [lab], where="dx", lsize=11.5)
        dx += wdt + 16
s.path([(1157, 1255), (1157, 1280), (875, 1280), (875, 1298)], color="#999")
s.path([(1113, 1335), (1113, 1358)], color="#999")
d2x = 830
s.box(d2x, 1300, 90, ["AdaLN"], where="d2", lsize=11.5)
s.path([(d2x + 92, 1320), (d2x + 110, 1320)], color="#999")
s.box(d2x + 112, 1300, 130, ["FF GEGLU ×4"], where="d2b", lsize=11.5)
s.node(d2x + 270, 1320, "+")
s.path([(d2x + 244, 1320), (d2x + 255, 1320)], color="#999")
s.box(d2x, 1384, 200, ["cross K/V = 脑1 特征", "4 步不变 → 只缓存 K/V", "（8/16 层，约 14 MiB）"],
      red=True, where="kv")
s.path([(1235, 1086), (1235, 1410), (1034, 1410)], color=RED, dash="7 4", marker="arrRed")
s.text(1080, 1478, "self：动作 16 token 互相看；cross：查图文背景（vl_embs 2048→1536，32 头×48）",
       11.5, anchor="middle", fill=GRAY)
arr_down(dit + 2, 1492)
op = s.box(CX - 130, 1492, 260, ["out_proj 1536→32（速度场 v）"], where="op")
# 去噪循环（红，对标循环箭头）
s.path([(168, 1512), (106, 1512), (106, 1186), (166, 1186)], color=RED, dash="7 4", marker="arrRed")
s.text(112, 1436, "×4 步", 13.5, bold=True, fill=RED)
s.text(112, 1453, "复用 cross K/V", 10.5, fill=RED)
arr_down(op + 2, 1552)
ap = s.box(CX - 130, 1552, 260, ["action_pred [B,16,32]"], where="ap", lsize=14)

# ---------------- 出口 ----------------
arr_down(ap + 2, 1652)
dn = s.box(CX - 150, 1652, 300, ["反归一化（transforms 逆变换，02 章）"], fill="#f2f2f2",
           stroke="#999", where="dn")
arr_down(dn + 2, 1724)
s.box(CX - 120, 1724, 240, ["action → 机器人"], fill="#fff", stroke="#999", where="end")

# ---------------- 图例 ----------------
ly = 1800
s.rect(90, ly - 8, 1320, 68, "#fafafa", "#ddd")
s.box(110, ly + 4, 120, ["模块/算子"], where="lg1", lsize=11.5)
s.box(260, ly + 4, 120, ["关键机制"], red=True, where="lg2", lsize=11.5)
s.rect(420, ly + 2, 26, 26, "none", "#999", dash="4 3")
s.text(456, ly + 21, "虚线容器 = 类/模块边界（对标 LlamaModel/LlamaAttention）", 12, fill="#555")
s.text(456, ly + 46, "冻结 = 权重不更新（N1.5 基座 VLM 全冻，12 章）；可训练 = DiT + adapter；×1/×4 = 每轮 infer 执行次数（trace 可验证，14 章 §3.2）",
       12, fill="#555")


def str_(self):
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.W}" height="{self.H}" '
            f'viewBox="0 0 {self.W} {self.H}" font-family="{FONT}">\n<defs>'
            '<marker id="arr" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,3 L0,6 Z" fill="#777"/></marker>'
            '<marker id="arrRed" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,3 L0,6 Z" fill="#c0392b"/></marker>'
            '</defs>\n')
    return head + "\n".join(self.parts) + "\n</svg>\n"


SVG.str_ = str_
p = os.path.join(OUT, "gr00t_three_stack_framework.svg")
open(p, "w").write(s.str_())
print("wrote", p, "%dx%d" % (W, H))
if WARN:
    print("[越界自检] %d 处:" % len(WARN))
    for w in WARN:
        print(" ", w)
    sys.exit(1)
print("OK: no overflow")
