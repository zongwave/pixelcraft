#!/usr/bin/env python3
"""第 9 章配图：一个 DiT block 的融合粒度阶梯 + 一次 get_action 的 host 发射账。

生成：
  images/ch09/dit_fusion_granularity.svg

口径：groot_ops tag v0.1(24 op) / v0.2(30 op) / v0.4(36 op)（git ls-tree ops/torch_ops 实测）；
      gr00t 侧 file:line 取自 Isaac-GR00T@v0.2 (57ca788)。

用法：python3 tools/mk_fig_ch09_dit_granularity.py
"""
import os

W, H = 1560, 1180
FONT = "Helvetica, Arial, 'Noto Sans CJK SC', 'WenQuanYi Zen Hei', sans-serif"


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def tw(s, size):
    w = 0.0
    for ch in s:
        w += 1.0 if ord(ch) > 0x2E7F else 0.56
    return w * size


WARN = []


def chk(where, s_, size, avail):
    w = tw(s_, size)
    if w > avail:
        WARN.append("%-10s %6.1f > %.1f  %s" % (where, w, avail, s_))


class SVG:
    def __init__(self):
        self.parts = []

    def rect(self, x, y, w, h, fill, stroke, rx=8, sw=1.6, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                          f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=13, bold=False, anchor="start", color="#222"):
        w = "bold" if bold else "normal"
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{w}" '
                          f'fill="{color}" text-anchor="{anchor}">{esc(s)}</text>')

    def arrow(self, x1, y1, x2, y2, color="#555", sw=1.8, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                          f'stroke-width="{sw}"{d} marker-end="url(#arr)"/>')

    def svg(self):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
                f'viewBox="0 0 {W} {H}" font-family="{FONT}">\n<defs>'
                '<marker id="arr" markerWidth="10" markerHeight="10" refX="7" refY="3" '
                'orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,3 L0,6 Z" fill="#555"/>'
                '</marker></defs>\n')
        return head + "\n".join(self.parts) + "\n</svg>\n"


s = SVG()
s.rect(0, 0, W, H, "#fcfcfc", "none", 0, 0)
s.text(W / 2, 40, "一个 DiT block 的融合粒度阶梯：960 次发射 → 64 次，然后瓶颈搬家到 host 往返",
       23, bold=True, anchor="middle")
s.text(W / 2, 65, "16 层 × 4 去噪步 = 64 个 block 执行。左=原生 aten；中=v0.1/v0.2 块内三融合；"
                  "右=v0.2 单 Die 整块一次 FFI。列内已标注 op 属于哪个 tag。",
       13, anchor="middle", color="#666")

CW, GAP, X0 = 470, 40, 45
cols = []
for i, (title, sub, fill, stroke, tcol) in enumerate([
    ("原生 PyTorch（H20 / aten 基线）", "≈15 次发射/层 → ≈960 次/轮", "#fdecea", "#c0392b", "#a3271d"),
    ("块内三融合（groot_ops v0.1 → v0.2）", "3 次发射/层 → 192 次/轮", "#fff6e8", "#b9770e", "#95590a"),
    ("整块一次 FFI（v0.2 单 Die）", "1 次发射/层 → 64 次/轮", "#eef8f0", "#1e8449", "#16693a"),
]):
    x = X0 + i * (CW + GAP)
    cols.append((x, title, sub, fill, stroke, tcol))

top = 118
for x, title, sub, fill, stroke, tcol in cols:
    chk("col-title", title, 15, CW - 24)
    chk("col-sub", sub, 12, CW - 24)
    s.rect(x, top, CW, 40, fill, stroke, 8, 1.6)
    s.text(x + CW / 2, top + 17, title, 15, bold=True, anchor="middle", color=tcol)
    s.text(x + CW / 2, top + 33, sub, 12, anchor="middle", color="#777")


def chips(x, y, w, items, fill, stroke, tcol, h=26, size=12.5):
    for it in items:
        chk("chip", it, size, w - 20)
        s.rect(x, y, w, h, fill, stroke, 5, 1.2)
        s.text(x + w / 2, y + h / 2 + 4.5, it, size, anchor="middle", color=tcol)
        y += h + 5
    return y


yb = top + 58
native = ["LayerNorm", "linear to_q / to_k / to_v（3 次）", "matmul QK^T → softmax",
          "matmul softmax·V", "linear to_out[0]", "residual add",
          "LayerNorm（norm3）", "linear fc1 → GELU → fc2", "residual add",
          "＋穿插 contiguous / permute / clone（未计入）"]
yb2 = chips(cols[0][0], yb, CW, native, "#fff", "#c9a29c", "#5c2a22")

mid = ["adaln_qkv　norm + AdaLN 调制 + QKV 三投影",
       "fused_mha_out　MHA + out proj（残差折进 bias）",
       "mlp_gelu_norm　norm3(LN+残差) + plain-GELU FFN",
       "", "v0.2 相对 v0.1 新增 6 个 op：",
       "adaln_qkv · adaln_qkv_mha_out · fused_mha_out",
       "linear_qkv · mlp_gelu_norm · dit_block",
       "", "另：头尾 gemm_bias(W1)+mlp_silu（action_encoder）",
       "　　　mlp_relu（CategorySpecificMLP decoder）"]
yb3 = chips(cols[1][0], yb, CW, [m if m else "　" for m in mid], "#fff", "#d9bb84", "#5c4310")

big = ["torch_evo.dit_block_fused(...)",
       "action.py:773　一次 FFI 吃完整个 block",
       "kernel 内部只是编排两个已测 launcher：",
       "　launch_adaln_qkv_mha_out",
       "　launch_mlp_gelu_norm_die_single",
       "dit_block_ffi.cc:205-225 把形状写成硬校验：",
       "　out_k=[M_kv,N_k]　out_v=[N_q,M_kv]（V 转置）",
       "　wt_v 必须 [K_kv,N_q]，错即 TVM_FFI_THROW",
       "守卫（action.py:703-736）：单 Die + 2D +",
       "plain-GELU + n%16，否则静默回退分段路径"]
ytop = yb
for i, it in enumerate(big):
    hh = 44 if i == 0 else 26
    chk("big", it, 13 if i == 0 else 12.5, CW - 20)
    s.rect(cols[2][0], ytop, CW, hh, "#d8f0de" if i == 0 else "#fff",
           "#1e8449" if i == 0 else "#9fcdb0", 6, 2 if i == 0 else 1.2)
    s.text(cols[2][0] + CW / 2, ytop + hh / 2 + 5, it, 13.5 if i == 0 else 12.5,
           bold=(i == 0), anchor="middle", color="#12572f")
    ytop += hh + 5

bot = max(yb2, yb3, ytop) + 8
s.arrow(X0 + 10, top - 14, X0 + CW * 3 + GAP * 2 - 10, top - 14, "#888", 2)
s.text(X0 + (CW * 3 + GAP * 2) / 2, top - 24, "融合粒度 ↑　·　host→device 发射次数 ↓　·　守卫/校验越写越硬",
       12.5, anchor="middle", color="#777")

# ---------------- 发射账 & 瓶颈搬家 ----------------
s.rect(X0, bot, CW * 3 + GAP * 2, 108, "#f4f7fb", "#5b7fa6", 10, 1.8)
s.text(X0 + 22, bot + 26, "一次 get_action 的账（4 步 × 16 层 = 64 block）", 15, bold=True, color="#2c5170")
for ln, dy in [("960 → 192 → 64 次发射：省下来的不是 FLOP，是 host→device 往返与 aten 中间张量的 contiguous/permute。", 50),
               ("但 v0.2 稳态 0.14 s/round 的同时，profile 里 to_lpu 累计 6148 ms、page_kv 1691 ms，远大于单轮耗时——", 72),
               ("两个数同时成立恰恰说明：剩余瓶颈在 host↔device 往返（page_kv 把 k/v 搬回 host 重排成 paged 再传上去），不在算力。", 94)]:
    chk("strip1", ln, 12.5, CW * 3 + GAP * 2 - 44)
    s.text(X0 + 22, bot + dy, ln, 12.5, color="#33506b")

y2 = bot + 124
s.rect(X0, y2, CW * 3 + GAP * 2, 172, "#fdecea", "#c0392b", 10, 2)
s.text(X0 + 22, y2 + 27, "整块融合没有做的（§6.7 唯一未落地项）：cross-attention 的 K/V 每步重投",
       15.5, bold=True, color="#a3271d")
for ln, dy in [("· cross 层的 K/V 源 = backbone feature，一次 get_action 的 4 步内恒定 ⇒ 数学上可缓存。", 55),
               ("· kernel 仍每步吃 encoder + wt_k + wt_v：action.py:743 每次喂 encoder_l、:752 kv_from_nh1 = 0，", 79),
               ("　dit_block_ffi.cc:214-219 每步重写 out_k / out_v ⇒ 4 步 × 8 个 cross 层 × 2 投影 = 64 次，其中 48 次白算（≈90 GFLOP）。", 101),
               ("· 整块融合后它被吞进 dit_block_fused 内部 ⇒ 不再出现在 profiler 顶层，从「可见的慢」变成「隐形的慢」。", 125),
               ("· 前置条件已就位：process_backbone_output 在循环外（flow_matching_action_head.py:352）、_ditbufs 常驻（action.py:754）。", 149)]:
    chk("strip2", ln, 12.5, CW * 3 + GAP * 2 - 44)
    s.text(X0 + 22, y2 + dy, ln, 12.5, color="#8e2b21")

H = y2 + 172 + 28

out = os.path.join(os.path.dirname(__file__), "..", "images", "ch09", "dit_fusion_granularity.svg")
with open(out, "w", encoding="utf-8") as f:
    f.write(s.svg())
print("written:", os.path.normpath(out))
print("overflow warnings:", len(WARN))
for w in WARN:
    print("  ", w)
