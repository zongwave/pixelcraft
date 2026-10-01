#!/usr/bin/env python3
"""图 9-9：三处 attention × NPU kernel——分界画在四条轴上，不画在"哪个塔"上。

事实来源（全部实测/源码，无推测）：
  * 核与次数：logs/trace_current_clean/n15_sd3_chrome_trace.json（N1.5 单轮热轮）
      unified_mha_run_die_batch_emb ×27 / qwen3_attention_run_die ×12 /
      fused_mha_out_run_die_single_vraw_qraw ×68（= DiT 16×4 步 64 + vl_self_attention 4）/
      adaln_qkv_run_die ×64 / gemm_norm_rope_run_die_single ×12 / linear_qkv_run_fused ×4
    另一份 3cam/2iter trace：unified_mha_run_die_batch ×54=27×2、qwen3_attention ×24=12×2。
  * 契约：groot_ops ops/torch_ops/{unified_mha,qwen3_attention,fused_mha_out}/ 头注释与 FFI 校验。
  * 形状：06 章 §6.8.6 实测维度账本（backbone 296、DiT 32×48、vl 32×64、Qwen3 16Q/8KV×128）。
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
        self.parts = []

    def rect(self, x, y, w, h, fill, stroke, rx=4, sw=1.5, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
                          f'stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, t, size=13, fill="#222", bold=False, anchor="start"):
        if tw(t, size) > self.W - 20:
            WARN.append(f"{tw(t,size):7.1f}  {t}")
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{"bold" if bold else "normal"}" '
                          f'fill="{fill}" text-anchor="{anchor}">{t.replace(chr(38), chr(38)+"amp;").replace("<", "&lt;").replace(">", "&gt;")}</text>')

    def path(self, pts, color="#777", dash=None, sw=1.7, marker="arr"):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#{marker})"' if marker else ""
        self.parts.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in pts)}" fill="none" '
                          f'stroke="{color}" stroke-width="{sw}"{d}{m}/>')

    def str_(self):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.W}" height="{self.H}" '
                f'viewBox="0 0 {self.W} {self.H}" font-family="{FONT}">\n<defs>'
                '<marker id="arr" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto" '
                'markerUnits="strokeWidth"><path d="M0,0 L8,3 L0,6 Z" fill="#777"/></marker>'
                '</defs>\n')
        return head + "\n".join(self.parts) + "\n</svg>\n"


RED = "#c0392b"; GRAY = "#666"; GREEN = "#1e8449"; BLUE = "#1f618d"; PURPLE = "#6c3483"
W, H = 1500, 1010
s = SVG(W, H)
s.rect(0, 0, W, H, "#ffffff", "none", rx=0)
s.text(W / 2, 32, "三处 attention，三个 kernel：分界线画在四条轴上，不画在「哪个塔」上", 21, bold=True, anchor="middle")
s.text(W / 2, 56, "核与次数 = trace 实测（N1.5 单轮热轮 n15_sd3_chrome_trace.json）；布局契约 = groot_ops "
                  "ops/torch_ops 头注释与 FFI 校验；形状 = 06 章 §6.8.6 维度账本", 12, anchor="middle", fill=GRAY)

COLS = [
    dict(x=40, title="① SigLIP 视觉塔（图 14-3/14-5）", tone="#eaf2fb", stroke="#5b8db8",
         up=("QKV 从哪来", ["4 × gemm_bias（q/k/v 各一发 + out_proj 一发）",
                            "linear_qkv_run_fused 有导出，视觉塔 0 次（那 4 次在 vl 塔）"]),
         core=("unified_mha_run_die_batch_emb", "×27  = 27 层，一发吃掉 ⊗→softmax→⊗V"),
         down=("出口", ["gemm_bias(out_proj 1152→1152)", "上一层 hidden = 下一层的 q/k 来源"]),
         attrs=[("头", "16 × 72 = 1152（HEADS×D==N 是硬约束）"),
                ("L_Q / L_KV", "256 / 256；batch 维 = 相机数 N"),
                ("mask", "无：256 格两两可见（双向）"),
                ("头型", "MHA：K 的 head 数 == Q 的 head 数"),
                ("scale", "host 侧预乘进 Q^T（核里没有 scale 实参）"),
                ("V 的 ones 行", "host 侧显式补 → [H, D+1, L_KV]"),
                ("tile B", "128（partial_softmax 编译期模板）；L_Q 须是它"),
                ("", "的倍数：256 = 2×128 ⇒ 免补零"),
                ("KV 复用", "无：三张图都是新观测，每轮重算")]),
    dict(x=530, title="② Qwen3 backbone（图 5-3）", tone="#f4ecf7", stroke=PURPLE,
         up=("QKV 从哪来", ["gemm_norm_rope_run_die_single ×12",
                            "RMSNorm + RoPE + QK-norm + QKV 三合一，一发射"]),
         core=("qwen3_attention_run_die", "×12  = 12 层，核内 causal + GQA"),
         down=("出口", ["gemm_bias(o_proj 2048→2048)",
                        "另有 Qwen3LayerFfiFused：八步一发射（整层）"]),
         attrs=[("头", "16 Q / 8 KV × 128 = 2048"),
                ("L_Q / L_KV", "296 / 296（prefill 全长，一遍过）"),
                ("mask", "下三角：核内生成阈值，仅对角块走"),
                ("", "masked matmul（DEQUANT_SCALAR），future→≈−inf"),
                ("头型", "GQA：q 头 h → kv 头 h/group，多一层分组索引"),
                ("scale", "FFI 实参（核内 DEQUANT_SCALAR 注入）"),
                ("布局", "Q/K/V 全 token-major（gemm_norm_rope 直出）"),
                ("", "V 不补行：ones 行由核内 V_mm 第 D 行生成"),
                ("tile B", "64（调用实参，与 DiT 同）；L_Q 尾块由核内处理"),
                ("KV 复用", "无：单轮 prefill，本来就没有「过去」")]),
    dict(x=1020, title="③ DiT action head（图 6-1）+ vl 塔", tone="#f4faf5", stroke=GREEN,
         up=("QKV 从哪来", ["adaln_qkv_run_die ×64（=16 块 × 4 欧拉步）",
                            "adaLN 调制 + QKV 三合一；vl 塔走 linear_qkv ×4"]),
         core=("fused_mha_out_run_die_single_vraw_qraw", "×68 = DiT 64 ＋ vl_self_attention 4"),
         down=("出口", ["to_out 的 gemm 折进同一次发射（无独立出口）",
                        "再往上：dit_block_fused 把整块一次发射（§4.3 图 9-4）"]),
         attrs=[("头", "32 × 48 = 1536（vl 塔 32 × 64 = 2048）"),
                ("L_Q", "T_q = N_s + 32(future) + 16(action)，补零到 64"),
                ("", "的倍数（qT pad 常驻缓冲，§B3/#130）"),
                ("L_KV", "cross 296（backbone）/ self T_q"),
                ("mask", "无：cross 全长可见、self 无未来可言"),
                ("头型", "MHA"),
                ("K/V 换源", "cross 与 self **同一个核**：L_Q/L_KV 是运行时"),
                ("", "参数，换 K/V 只是换指针，核里没有「条件」概念"),
                ("tile B", "64（DiT）/ 128（vl 塔）"),
                ("KV 复用", "cross 的 K/V 跨 4 个欧拉步复用（8/16 层 ≈14 MiB）")]),
]

for c in COLS:
    x, w = c["x"], 440
    s.rect(x, 78, w, 30, c["tone"], c["stroke"])
    s.text(x + w / 2, 98, c["title"], 14, bold=True, anchor="middle", fill=c["stroke"])
    y = 120
    for tag, lines in [("up", None), ("core", None), ("down", None)]:
        blk = c[tag]
        if tag == "core":
            s.rect(x, y, w, 54, "#fdecea", RED, sw=2.2)
            s.text(x + w / 2, y + 22, blk[0], 13.5, bold=True, anchor="middle", fill=RED)
            s.text(x + w / 2, y + 42, blk[1], 11.5, anchor="middle", fill=RED)
            y += 54
        else:
            hgt = 18 + 19 * len(blk[1])
            s.rect(x, y, w, hgt, "#fafafa", "#ccc")
            s.text(x + 10, y + 16, blk[0], 11.5, bold=True, fill=GRAY)
            for i, ln in enumerate(blk[1]):
                s.text(x + 128, y + 16 + i * 19, ln.replace("**", ""), 11.5)
            y += hgt
        if tag != "down":
            s.path([(x + w / 2, y + 2), (x + w / 2, y + 16)], color="#888")
            y += 18
    y += 12
    s.rect(x, y, w, 18 + 21 * len(c["attrs"]), "#ffffff", c["stroke"], dash="4 3")
    for i, (k, v) in enumerate(c["attrs"]):
        yy = y + 22 + i * 21
        if k:
            s.text(x + 10, yy, k, 11.5, bold=True, fill=c["stroke"])
        s.text(x + 132, yy, v.replace("**", ""), 11.5, fill=RED if "同一个核" in v or "参数，换" in v else "#333")

# ---------------- 底部三块 ----------------
BY = 640
s.rect(40, BY, 470, 330, "#f6f9fc", BLUE)
s.text(58, BY + 26, "同一个族，三份拷贝（源码自述）", 14, bold=True, fill=BLUE)
for i, ln in enumerate([
    "unified_mha_kernel.ac：",
    "  「统一 MHA flash attention——三 site 通用」",
    "qwen3_attention_kernel.ac：",
    "  「结构镜像 unified_mha，增量 = GQA 分组",
    "    + 硬件因果 mask」",
    "attention_prefill_kernel.ac：",
    "  「结构对齐 unified_mha」（bf16 type-generic 版）",
    "",
    "共享的是骨架：32 核 / 4 cluster、K·V 驻 cluster L2 由 4 核复用、",
    "online-softmax 滚动归一。没共享成一份二进制的原因有三条：",
    "① tile B 是编译期模板（DiT·Qwen3 64 / SigLIP·vl 128）；",
    "② 布局契约不同（scale 与 ones 行在 host 还是核内）；",
    "③ mask/GQA 分支会拖慢不需要它的那两路。",
]):
    s.text(58, BY + 52 + i * 20, ln, 12, fill="#333" if i < 8 else GRAY)

s.rect(530, BY, 450, 330, "#fbf7ef", "#c8a45c")
s.text(548, BY + 26, "真正的分化在 attention 的上下游", 14, bold=True, fill="#8a6d1f")
rows = [("SigLIP", "4× gemm_bias", "unified_mha", "gemm_bias(out)"),
        ("Qwen3", "gemm_norm_rope", "qwen3_attn", "gemm_bias(o_proj)"),
        ("DiT", "adaln_qkv", "fused_mha_out", "to_out 折入"),
        ("vl 塔", "linear_qkv", "fused_mha_out", "to_out 折入")]
s.text(548, BY + 52, "入口融合", 11.5, bold=True, fill=GRAY)
s.text(660, BY + 52, "attention 本体", 11.5, bold=True, fill=GRAY)
s.text(820, BY + 52, "出口融合", 11.5, bold=True, fill=GRAY)
for i, r in enumerate(rows):
    yy = BY + 74 + i * 20
    s.text(548, yy, r[0], 11.5, bold=True)
    s.text(600, yy, r[1], 11.5, fill="#8a6d1f")
    s.text(712, yy, r[2], 11.5, fill=RED)
    s.text(820, yy, r[3], 11.5, fill=GRAY)
s.parts.append(f'<line x1="548" y1="{BY+58}" x2="962" y2="{BY+58}" stroke="#e4d7b8"/>')
for i, ln in enumerate([
    "三行的**入口**与**出口**都不一样，而这与 attention 本体无关：",
    "SigLIP 的 q/k/v 是同一份 hidden 的三次投影；Qwen3 要把 RMSNorm+",
    "RoPE+QK-norm 一起折进 QKV；DiT 的 QKV 必须先被 adaLN 的",
    "scale/shift 调制过。**融合的机会长在边界上，不长在运算上。**",
    "再往粗走一格就是整层/整块融合：Qwen3LayerFfiFused（八步一发射）、",
    "dit_block_fused（一块一发射）、siglip_layer_ffi_fused（已导出未启用）。",
    "这是 §4.3 融合粒度阶梯（960→192→64 次发射）在三个塔上的重演。",
]):
    s.text(548, BY + 172 + i * 21, ln.replace("**", ""), 12, fill="#333")

s.rect(1000, BY, 460, 330, "#fafafa", "#bbb")
s.text(1018, BY + 26, "三条判据与一条反面教材", 14, bold=True)
for i, ln in enumerate([
    "① 分界画在四条轴上：causal? × GQA? × 布局契约 × tile B。",
    "    DiT 与 SigLIP 在四轴上同类（非因果 MHA），只差 tile B、batch",
    "    布局与出口融合；Qwen3 在两轴上跳出去，才必须独立成核。",
    "② 「三处 attention」≠「三个核」：DiT self 与 cross 共用一个核；",
    "    2048→1536 的维度差在建模块时就焊死在 W_K/W_V 里（06 章",
    "    §6.8.2），核只认「非因果 MHA、hd=48、后面串一发 gemm」。",
    "③ 语义不进核：cross「看 backbone」这件事在 kernel 里没有任何",
    "    痕迹——换 K/V 只是换指针。语义住在权重和调用点里。",
    "",
    "反面教材：vLLM 血统的 paged 那一套（batch_attention 带",
    "k_tab/v_tab/idx/cu/su + causal 实参、reshape_and_cache_flash）",
    "在主链 trace 里出现 **0 次**。它们为「多轮对话 + 变长 KV cache」",
    "而生，而 gr00t 每轮面对的是一个全新观测（06 章 §6.8.8）——",
    "把 LLM 推理的标配照搬进来，只会买来一堆没人调用的算子。",
]):
    s.text(1018, BY + 52 + i * 20, ln.replace("**", ""), 12,
           fill=GREEN if i == 10 else ("#333" if i < 9 else GRAY), bold=(i == 10))

out = os.path.join(ROOT, "images", "ch09", "attention_kernel_mapping.svg")
open(out, "w").write(s.str_())
print("wrote", out)
if WARN:
    print("[越界]", *WARN, sep="\n  ")
