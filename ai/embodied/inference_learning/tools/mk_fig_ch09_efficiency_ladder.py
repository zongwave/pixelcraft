#!/usr/bin/env python3
"""图 9-10：有效算力阶梯（左）× 分相气泡瀑布（右）——同一条 148.7 ms 的墙钟，两种看法。

数据全部来自 tools/trace_phase_ledger.py 对同一份 trace 的输出：
  logs/trace_current_clean/n15_sd3_chrome_trace.json（N1.5 单轮热轮、三相机、E100 单 Die）
左：每发 kernel 的"有效算力" = FLOPs（权重形状 × 实测序列长度）÷ trace 实测 µs。
右：每个相的 busy（设备在算）+ 气泡（host 没发射够）= span，另加相外空隙；合计 = 整轮墙钟。
结论三句话写在底部：阶梯跨 54×、busy 只有 26% 不是空转、DiT 的账在发射侧不在算力侧。
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = "Helvetica, Arial, 'Noto Sans CJK SC', 'WenQuanYi Zen Hei', sans-serif"
WARN = []


def tw(t, size):
    return sum(1.0 if ord(c) > 0x2E7F else 0.56 for c in t) * size


class SVG:
    def __init__(self, w, h):
        self.W, self.H = w, h
        self.p = []

    def rect(self, x, y, w, h, fill, stroke="none", rx=3, sw=1.2, dash=None, op=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        o = f' fill-opacity="{op}"' if op else ""
        self.p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}{o}/>')

    def text(self, x, y, t, size=13, fill="#222", bold=False, anchor="start", rot=None):
        if anchor == "start" and x + tw(t, size) > self.W - 6:
            WARN.append(f"{x + tw(t,size):7.1f}>{self.W}  {t}")
        r = f' transform="rotate({rot} {x:.1f} {y:.1f})"' if rot else ""
        t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.p.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
                      f'font-weight="{"bold" if bold else "normal"}" fill="{fill}" '
                      f'text-anchor="{anchor}"{r}>{t}</text>')

    def line(self, x1, y1, x2, y2, c="#888", sw=1.2, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.p.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                      f'stroke="{c}" stroke-width="{sw}"{d}/>')

    def str_(self):
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.W}" height="{self.H}" '
                f'viewBox="0 0 {self.W} {self.H}" font-family="{FONT}">\n' + "\n".join(self.p) + "\n</svg>\n")


RED, GRAY, GREEN, BLUE, PURPLE, ORANGE = "#c0392b", "#666", "#1e8449", "#1f618d", "#6c3483", "#b9770e"
W, H = 1700, 1046
s = SVG(W, H)
s.rect(0, 0, W, H, "#ffffff")
s.text(W / 2, 30, "一轮 148.7 ms 的两种看法：左看「每发值多少算力」，右看「这段时间是设备在算还是在等 host」",
       20, bold=True, anchor="middle")
s.text(W / 2, 53, "同一份 trace（N1.5 单轮热轮 n15_sd3_chrome_trace.json，三相机、E100 单 Die）；"
                  "FLOPs = 权重形状 × 实测序列长度，µs = trace 实测；复跑 tools/trace_phase_ledger.py",
       12, anchor="middle", fill=GRAY)

TONE = {"SigLIP": BLUE, "Qwen3": GREEN, "vl": PURPLE, "DiT": RED}
# (标签, TFLOPS, 塔, 备注)
LAD = [
    ("SigLIP MLP fc1+fc2（M=768）", 63.6, "SigLIP", "27 发 · 240 µs · 15.2 GFLOP"),
    ("vl 塔 FF 2048→8192→2048（M=296）", 42.5, "vl", "4 发 · 468 µs · 19.9 GFLOP"),
    ("Qwen3 MLP swiglu（M=296）", 30.7, "Qwen3", "12 发 · 727 µs · 22.4 GFLOP"),
    ("SigLIP q/k/v/out_proj（M=768）", 26.7, "SigLIP", "109 发 · 76 µs · 2.0 GFLOP"),
    ("Qwen3 QKV+RMSNorm+RoPE（M=296）", 21.4, "Qwen3", "11 发 · 232 µs · 5.0 GFLOP"),
    ("Qwen3 o_proj（M=296）", 16.8, "Qwen3", "12 发 · 148 µs · 2.5 GFLOP"),
    ("vl 塔 QKV 一合（M=296）", 16.6, "vl", "4 发 · 447 µs · 7.4 GFLOP"),
    ("DiT FFN 1536→6144→1536（M=49）", 9.9, "DiT", "64 发 · 186 µs · 1.85 GFLOP"),
    ("DiT cross QKV（KV MISS，M_kv=296）", 8.5, "DiT", "8 发 · 466 µs · 3.96 GFLOP"),
    ("SigLIP attention（3 相机合批 256²）", 7.8, "SigLIP", "27 发 · 117 µs · 0.91 GFLOP"),
    ("DiT self attn + to_out（49²）", 5.9, "DiT", "32 发 · 42 µs · 0.25 GFLOP"),
    ("vl 塔 attn + to_out（296²）", 4.0, "vl", "4 发 · 806 µs · 3.2 GFLOP"),
    ("DiT cross attn + to_out（49×296）", 3.4, "DiT", "32 发 · 93 µs · 0.32 GFLOP"),
    ("DiT self QKV（adaLN 调制后 ×3）", 2.8, "DiT", "32 发 · 250 µs · 0.69 GFLOP"),
    ("DiT cross QKV（KV HIT，只剩 q）", 1.3, "DiT", "24 发 · 184 µs · 0.23 GFLOP"),
    ("Qwen3 attention（causal + GQA）", 1.2, "Qwen3", "12 发 · 304 µs · 0.36 GFLOP"),
]
AX, AY0, AROW, ARAIL = 470, 96, 40, 560
TOP = 63.6
s.text(40, 82, "① 有效算力阶梯：同一块 Die，一轮之内跨 54 倍", 16, bold=True)
s.line(AX, AY0 - 8, AX, AY0 + AROW * len(LAD) - 6, "#bbb")
yv = AX + ARAIL * 9.0 / TOP
s.line(yv, AY0 - 14, yv, AY0 + AROW * len(LAD), RED, 1.6, "5,4")
for i, (lab, tf, tow, note) in enumerate(LAD):
    y = AY0 + i * AROW
    bw = ARAIL * tf / TOP
    s.rect(AX, y, max(bw, 3), 22, TONE[tow], op=0.85 if i < 8 else 0.55)
    s.text(AX - 10, y + 16, lab, 12.5, anchor="end", fill="#222")
    s.text(AX + bw + 8, y + 12, "%.1f TFLOPS" % tf, 12.5, bold=True, fill=TONE[tow])
    s.text(AX + bw + 8, y + 27, note, 11, fill=GRAY)
s.text(AX + 4, AY0 + AROW * len(LAD) + 24,
       "红色竖线 = 整轮平均 9.0 TFLOPS（1.33 TFLOP ÷ 148.7 ms 墙钟）= 阶梯顶的 14%", 12, bold=True, fill=RED)
for i, (tow, c) in enumerate([("SigLIP", BLUE), ("Qwen3", GREEN), ("vl 塔", PURPLE), ("DiT 动作头", RED)]):
    x = 1235 + i * 108
    s.rect(x, AY0 + AROW * len(LAD) + 12, 14, 14, c)
    s.text(x + 20, AY0 + AROW * len(LAD) + 24, tow, 12.5)

# ---------------- 右：分相气泡瀑布 ----------------
BX, BY0, BROW = 1215, 150, 44
BSC = 2.70          # ms → px
PH = [("SigLIP 27 层", 20.33, 23.40, 230), ("Qwen3 12 层", 19.32, 20.29, 98),
      ("vl_self_attn 4 层", 7.44, 13.69, 35), ("DiT 去噪步 1", 9.93, 22.42, 65),
      ("DiT 去噪步 2", 7.64, 21.00, 65), ("DiT 去噪步 3", 7.68, 21.74, 65),
      ("DiT 去噪步 4", 7.62, 18.34, 58)]
s.text(1180, 82, "② 分相账本：实心 = 设备在算，淡色 = 气泡", 15.5, bold=True)
s.text(1180, 104, "（气泡 = span − busy = 这段时间设备没拿到发射）", 11.5, fill=GRAY)
s.text(1180, 126, "busy 80.0 ms｜相内气泡 60.9 ms｜相外 7.8 ms", 12.5, fill=GRAY)
s.text(1180, 144, "相按 kernel 边界自动切分（±1 层粘连，不影响总量）", 11, fill=GRAY)
for i, (lab, busy, span, nk) in enumerate(PH):
    y = BY0 + i * BROW
    tow = "SigLIP" if "SigLIP" in lab else ("Qwen3" if "Qwen3" in lab else ("vl" if lab.startswith("vl") else "DiT"))
    s.text(BX - 12, y + 17, "%s（%d kernel）" % (lab, nk), 12.5, anchor="end")
    s.rect(BX, y, span * BSC, 24, "#f4dcd8", "#c9a09a", rx=2)
    s.rect(BX, y, busy * BSC, 24, TONE[tow], op=0.88, rx=2)
    bub = 100 * (1 - busy / span)
    s.text(BX + span * BSC + 8, y + 12, "span %.1f｜busy %.1f｜气泡 %.0f%%" % (span, busy, bub), 12,
           fill="#8e3b31" if bub > 30 else GRAY, bold=bub > 30)
y = BY0 + len(PH) * BROW
s.text(BX - 12, y + 17, "相外空隙（不属于任何 kernel）", 12.5, anchor="end", fill=GRAY)
s.rect(BX, y, 2.28 * BSC, 20, "#ddd", "#aaa")
s.text(BX + 2.28 * BSC + 6, y + 15, "vl→DiT 2.28 ms ＋ 每步之间 1.82/1.81/1.89 ms ＝ 合计 7.8 ms", 12, fill=GRAY)
y += BROW
s.text(BX - 12, y + 17, "整轮墙钟", 13.5, anchor="end", bold=True)
s.rect(BX, y, 80.0 * BSC, 26, "#2c3e50", rx=2)
s.rect(BX + 80.0 * BSC, y, 60.9 * BSC, 26, "#e6b0aa", "#c0392b", rx=2)
s.rect(BX + 140.9 * BSC, y, 7.8 * BSC, 26, "#ddd", "#aaa", rx=2)
s.text(BX + 80.0 * BSC / 2, y + 18, "busy 80.0", 12.5, fill="#fff", bold=True, anchor="middle")
s.text(BX + (80.0 + 60.9 / 2) * BSC, y + 18, "气泡 60.9", 12.5, fill="#7b241c", bold=True, anchor="middle")
s.text(BX + 148.7 * BSC, y - 6, "整轮 = 148.7 ms", 13, bold=True, anchor="end", fill=RED)
yy = y + 44
for t_, c_ in [("busy 80.0 ms（54%）", "#2c3e50"), ("相内气泡 60.9 ms（41%）", "#c0392b"),
               ("相外空隙 7.8 ms（5%）", "#999")]:
    s.text(BX, yy, "■ " + t_, 12.5, fill=c_)
    yy += 20

# ---------------- 底部结论 ----------------
BY = 800
s.rect(36, BY, W - 72, 210, "#f7fbff", "#9ec5e8", rx=6, sw=1.5)
s.text(56, BY + 26, "三句结论（对应 §10.7）", 15, bold=True, fill="#1a5276")
BL = [
    ("① 算力不是瓶颈，形状才是。",
     ["阶梯顶 63.6（SigLIP MLP，M=768）→ 阶梯底 1.2（Qwen3 attention），跨 54 倍。全轮有用算力 1.33 TFLOP，按阶梯顶折算只要 21.0 ms，而 busy 已经花掉 80.0 ms",
      "⇒ busy 里只有 26% 不是空转。DiT 全部条目压在 1–10 TFLOPS：M=49 行、head_dim 48，tile 天生打不满（06 章 §6.8.6 的维度账，在这里变成钱）。"], RED),
    ("② 墙钟的大头不在 kernel 里。",
     ["148.7 ms 中 60.9 ms 是相内气泡、7.8 ms 是相外空隙；DiT 四步 span 83.5 ms（墙钟 56%）却只有 32.9 ms 是 busy ⇒ 动作头的账记在发射侧，不记在算力侧",
      "旁证：616 次 evLaunchKernel 花 19.8 ms host（32 µs/发 = 墙钟的 13%）；617 发 kernel 里 219 发 <50 µs、70 发 <20 µs；§9 那次 −6.8 ms/step 的 KV 缓存 e2e 归零，就是这个机制造成的。"], BLUE),
    ("③ 所以 KPI 是这两个数，不是 FLOPS。",
     ["每轮发射次数（617 kernel / 616 launch）与常驻字节数。已兑现的最大一笔正是这条：布局契约统一后 KernelCloneTranspose 136 发 134.6 ms → 0 发、",
      "750 次 memcpy 174 ms → 23 次 147 µs（两份 trace 对账见 §10.7.3）；融合只是这条路的载体，不是收益本身。"], PURPLE),
]
yy = BY + 44
for h_, ls, c_ in BL:
    s.text(56, yy, h_, 13, bold=True, fill=c_)
    s.text(56, yy + 19, ls[0], 12.3, fill="#333")
    s.text(56, yy + 37, ls[1], 12.3, fill="#333")
    yy += 56
if WARN:
    print("\n".join(WARN[:25]))
out = os.path.join(ROOT, "images", "ch09", "efficiency_ladder_bubble.svg")
open(out, "w", encoding="utf-8").write(s.str_())
print("wrote", out, os.path.getsize(out), "bytes; warn=%d" % len(WARN))
