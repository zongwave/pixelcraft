#!/usr/bin/env python3
"""第 9 章附录配图：L20 vs E200 硬件账本 + gr00t e2e 瓶颈分解。

生成：images/ch09/l20_e200_bottleneck_ledger.svg

数值来源（全部 2026-09-28 实测/实查）：
  L20:  nvml/本地基准（PyTorch 0.686s、TRT fp16 0.103s、backbone 0.228、head 0.385、transform 0.031）
        峰值 BF16 119.5 TFLOPS、864 GB/s（官方规格 + nvml 92SM@2520MHz 核对）
  E200: 板端 ev-qual 日志（FP16 256 / INT8 512 TOPS / 带宽理论 960、实测 771/747 GB/s）
        ev-smi（PCIe Gen5 x16、48GB @18000Mbps）、groot_ops CHANGELOG v0.1/v0.2/v0.2-fix
用法：python3 tools/mk_fig_ch09_l20_e200_ledger.py
"""
import math, os

W, H = 1560, 1050
FONT = "Helvetica, Arial, 'Noto Sans CJK SC', 'WenQuanYi Zen Hei', sans-serif"
OUT = os.path.join(os.path.dirname(__file__), "..", "images", "ch09", "l20_e200_bottleneck_ledger.svg")

def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

class SVG:
    def __init__(self):
        self.parts = []
    def rect(self, x, y, w, h, fill, stroke="none", rx=8, sw=1.4, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        st = f' stroke="{stroke}" stroke-width="{sw}"{d}' if stroke != "none" else ""
        self.parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"{st}/>')
    def text(self, x, y, s, size=13, bold=False, anchor="start", fill="#222"):
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{"bold" if bold else "normal"}" fill="{fill}" text-anchor="{anchor}">{esc(s)}</text>')
    def line(self, x1, y1, x2, y2, color="#999", sw=1.2, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{sw}"{d}/>')

s = SVG()
s.parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
s.parts.append(f'<rect width="{W}" height="{H}" fill="#fbfbfd"/>')
s.text(W/2, 40, "L20 vs E200：硬件账本与 gr00t N1.5 e2e 瓶颈分解（2026-09-28 实测）", 22, bold=True, anchor="middle", fill="#111")

# ============ 左面板：e2e 时间账本（log 横条） ============
PX, PY, PW = 60, 80, 700
s.rect(PX, PY, PW, 620, "#ffffff", "#d8d8de")
s.text(PX+16, PY+28, "① 一次 get_action 的墙钟账本（log 刻度，单位 s）", 15, bold=True, fill="#01579b")
bars = [  # (label, value, color, note)
    ("L20 · 原生 PyTorch (bf16)", 0.686, "#e57373", "达成算力 ≈ 峰值 2%"),
    ("E200 单Die · 原生算子链", 0.826, "#ef9a9a", "v0.1 之前基线"),
    ("E200 单Die · v0.1", 0.697, "#ffb74d", "发射/拷贝大清理，−16%"),
    ("E200 单Die · v0.2", 0.140, "#66bb6a", "整块融合 64 发射/轮，5.9×"),
    ("E200 单Die · v0.2-fix", 0.113, "#2e7d32", "busy 51%，host 往返成大头"),
    ("L20 · TensorRT fp16", 0.103, "#42a5f5", "达成算力 ≈ 峰值 16%（6.6×）"),
    ("E200 双Die·往返清零(推算)", 0.070, "#1565c0", "设备时间×0.5 + 常驻化"),
    ("物理下限（权重带宽）", 0.007, "#9e9e9e", "3B×2B ÷ ~0.9TB/s"),
]
X0, XW = PX+230, PW-260
LO, HI = math.log10(0.005), math.log10(1.2)
def xpos(v): return X0 + XW * (math.log10(v) - LO) / (HI - LO)
by = PY + 60
s.line(xpos(0.01), PY+48, xpos(0.01), PY+560, "#ccc", 1, "3,3")
s.line(xpos(0.1), PY+48, xpos(0.1), PY+560, "#ccc", 1, "3,3")
s.line(xpos(1.0), PY+48, xpos(1.0), PY+560, "#ccc", 1, "3,3")
s.text(xpos(0.01), PY+575, "0.01", 11, anchor="middle", fill="#888")
s.text(xpos(0.1), PY+575, "0.1", 11, anchor="middle", fill="#888")
s.text(xpos(1.0), PY+575, "1.0", 11, anchor="middle", fill="#888")
RX = PX + PW - 14          # 面板右缘
for lab, v, col, note in bars:
    s.text(X0-10, by+15, lab, 12.5, anchor="end")
    s.rect(X0, by, xpos(v)-X0, 22, col, rx=4)
    vx = f"{v:.3f}" + ("…" if v == 0.070 else "")
    if xpos(v) + 240 < RX:            # 右侧放得下：标注在条外
        s.text(xpos(v)+8, by+16, vx, 12, bold=True, fill="#333")
        s.text(xpos(v)+62, by+16, note, 11.5, fill="#777")
    else:                              # 长条：数值与标注收进条内右对齐
        s.text(xpos(v)-10, by+16, note + "  " + vx, 12, bold=True, anchor="end", fill="#ffffff")
    by += 46
s.rect(PX+16, PY+588, PW-32, 40, "#fff8e1", "#f9a825", rx=6)
s.text(PX+30, PY+613, "读法：两条 0.1s 级蓝/绿条贴在一起 —— 半张 E200 ≈ 整张 L20-TRT；而它们离灰色物理下限都还有 ~10×。", 12.5, fill="#795548")

# ============ 右上：硬件规格卡 ============
QX, QY, QW = 800, 80, 700
s.rect(QX, QY, QW, 330, "#ffffff", "#d8d8de")
s.text(QX+16, QY+28, "② 硬件规格对照（本机 nvml · 板端 ev-qual/ev-smi 实查）", 15, bold=True, fill="#01579b")
cols = [
    ("", "", ""),
    ("NVIDIA L20 (Ada·92SM@2.52GHz)", "EVAS E200/E100 (Epoch·32核·双Die)", ""),
    ("BF16 稠密  119.5 TF", "BF16  256 TF（实测 255.9，99.97%）", "#1565c0"),
    ("INT8  239 TOPS", "INT8  512 TOPS · INT4 1024", "#1565c0"),
    ("GDDR6 864 GB/s", "DDR 理论 960 / 实测读771·写747 GB/s", "#2e7d32"),
    ("PCIe Gen4 x16 · 350W", "PCIe Gen5 x16 + 16×XLink · 边缘功耗档", "#2e7d32"),
    ("L2 96 MB", "L1 56MB(AM8+MM48) + L2 72MB", "#555"),
    ("算力/带宽 138 FLOP/B", "算力/带宽 267 FLOP/B —— 更挑食", "#e65100"),
]
ry = QY + 52
for i, (a, b, c) in enumerate(cols):
    if i == 1:
        s.text(QX+30, ry+13, a, 12.5, bold=True, fill="#333")
        s.text(QX+360, ry+13, b, 12.5, bold=True, fill="#333")
        s.line(QX+20, ry+22, QX+QW-20, ry+22, "#bbb")
    elif i > 1:
        s.text(QX+30, ry+13, a, 12.5, fill=c or "#444")
        s.text(QX+360, ry+13, b, 12.5, fill=c or "#444")
    ry += 32
s.text(QX+20, QY+305, "注：E200 规格为 ev-qual 整卡口径（block_num 2）；gr00t 当前仅用单 Die。", 11.5, fill="#888")

# ============ 右下：瓶颈分解 ============
BY_ = 440
s.rect(QX, BY_, QW, 530, "#ffffff", "#d8d8de")
s.text(QX+16, BY_+28, "③ 时间到底花在哪（归因）", 15, bold=True, fill="#01579b")
# L20 PyTorch 分解堆叠条
bx, bw = QX+30, QW-360
s.text(bx, BY_+58, "L20 PyTorch 0.686 s 分解", 12.5, bold=True)
segs = [("head 0.385", 0.385, "#ef5350"), ("backbone 0.228", 0.228, "#ff9800"), ("其他 0.072", 0.072, "#bdbdbd")]
cx = bx
for lab, v, col in segs:
    w = bw * v / 0.686
    s.rect(cx, BY_+66, w, 26, col, rx=3)
    s.text(cx+w/2, BY_+84, lab, 11.5, anchor="middle", fill="#fff" if col != "#bdbdbd" else "#333")
    cx += w
s.text(QX+30, BY_+116, "DiT 4 步 ≈ 960 次算子发射：发射/Python/分发开销 ≥ 计算本身", 12, fill="#c62828")
s.text(QX+30, BY_+134, "TRT 把这头“前端税”清算掉 → 6.6×；但也只用到峰值 16%。", 12, fill="#333")
# E200 分解
s.text(bx, BY_+170, "E200 v0.2 0.140 s 分解", 12.5, bold=True)
s.rect(bx, BY_+178, bw*0.51, 26, "#66bb6a", rx=3)
s.rect(bx+bw*0.51, BY_+178, bw*0.49, 26, "#ba68c8", rx=3)
s.text(bx+bw*0.255, BY_+196, "device busy 51%", 11.5, anchor="middle", fill="#fff")
s.text(bx+bw*0.755, BY_+196, "host↔device 往返 49%", 11.5, anchor="middle", fill="#fff")
s.text(QX+30, BY_+228, "to_lpu 累计 6148ms、page_kv 1691ms；FFI 单次固定成本 60–150 µs（GPU launch ~5 µs）", 12, fill="#6a1b9a")
# 攻击次序
oy = BY_ + 258
opts = [
    ("1", "砍发射/往返", "整块融合(960→64 发射)、权重/KV 常驻、去 page_kv", "#c62828"),
    ("2", "吃满双 Die", "K/N 切分算子已在库；设备时间 ~60ms ⇒ e2e ~0.07s", "#e65100"),
    ("3", "最后才砍算力", "INT8 512TOPS=2× 红利；但 left_hand cos 0.992 ⇒ 逐通道验证", "#2e7d32"),
]
for n, t, d, c in opts:
    s.rect(QX+30, oy, 26, 26, c, rx=13)
    s.text(QX+43, oy+18, n, 14, bold=True, anchor="middle", fill="#fff")
    s.text(QX+70, oy+12, t, 13, bold=True, fill=c)
    s.text(QX+240, oy+12, d, 12, fill="#444")
    oy += 40
s.rect(QX+30, oy+6, QW-60, 54, "#e8f5e9", "#66bb6a", rx=6)
s.text(QX+46, oy+28, "反证实验：KV-cache 省 90 GFLOP、设备 −6.8 ms/step，e2e 纹丝不动(0.113 vs 0.114)", 12.5, bold=True, fill="#1b5e20")
s.text(QX+46, oy+48, "⇒ 发射 bound 的系统里先砍算力=白砍；往返清零后这些收益会一次性兑现。", 12.5, fill="#1b5e20")

s.text(W/2, H-16, "来源：L20 nvml+本地基准(20·去3warmup) · E200 ev-qual/ev-smi 实查 + groot_ops CHANGELOG v0.1–v0.2-fix · 单Die口径已标注", 11.5, anchor="middle", fill="#888")
s.parts.append("</svg>")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write("\n".join(s.parts))
print("wrote", OUT)
