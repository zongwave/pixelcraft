#!/usr/bin/env python3
"""三件套"单层计算流程图"（严格对标 llama_decoder.png 的运算图语法），一脚本三图：

  images/ch05/qwen3_llm_stack.svg    图 5-3  LLM：12×Qwen3DecoderLayer 的逐运算展开
  images/ch14/siglip_vit_stack.svg   图 14-3 ViT：27×SiglipEncoderLayer 的逐运算展开 + NPU kernel 落点
  images/ch06/dit_stack.svg          图 6-1  DiT：16×DiT Block 的逐运算展开（AdaLN/cross-KV/GEGLU）

语法与 llama_decoder.png 一致：
  左侧竖排主干 = hidden_states 逐级下行；残差从左侧绕行走 ⊕（"Residual add"）；
  中间 = 真实运算：qkv 三投影分叉 → Q/K/V → ⊗ 打分 → softmax → ⊗V → 投影 → 从下方折回 ⊕；
  MLP 同样展开为分步运算，输出折回第二个 ⊕。差异点用红框标注（无 mask / 无 lm_head / K/V 冻结…）。
"""
import os
import sys

FONT = "Helvetica, Arial, 'Noto Sans CJK SC', 'WenQuanYi Zen Hei', sans-serif"
WARN = []


def tw(t, size):
    return sum(1.0 if ord(c) > 0x2E7F else 0.56 for c in t) * size


class SVG:
    def __init__(self, w, h):
        self.W, self.H = w, h
        self.parts = []

    def rect(self, x, y, w, h, fill, stroke, rx=6, sw=1.6, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
                          f'stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, t, size=13, fill="#222", bold=False, anchor="start"):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{"bold" if bold else "normal"}" '
                          f'fill="{fill}" text-anchor="{anchor}">{t.replace(chr(38),chr(38)+"amp;").replace("<","&lt;").replace(">","&gt;")}</text>')

    def path(self, pts, color="#777", dash=None, sw=1.7, marker="arr"):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#{marker})"' if marker else ""
        self.parts.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in pts)}" fill="none" '
                          f'stroke="{color}" stroke-width="{sw}"{d}{m}/>')

    def circ(self, x, y, label, r=12, color="#555", fill="#fff", fs=13):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{color}" stroke-width="1.7"/>')
        self.text(x, y + 4.5, label, fs, fill="#333", bold=True, anchor="middle")

    def box(self, x, y, w, lines, fill="#cfe3f7", stroke="#5b8db8", lsize=12.5, lcolor="#1d3d57",
            red=False, where="b"):
        lh = lsize + 5.5
        h = max(28, 11 + lh * len(lines))
        for ln in lines:
            if tw(ln, lsize) > w - 8:
                WARN.append(f"{where:<10} {tw(ln,lsize):6.1f}>{w-8:6.1f}  {ln}")
        self.rect(x, y, w, h, "#fdecea" if red else fill, "#c0392b" if red else stroke)
        ty = y + h / 2 - (len(lines) - 1) * lh / 2 + 4.5
        for ln in lines:
            self.text(x + w / 2, ty, ln, lsize, anchor="middle",
                      fill="#c0392b" if red else lcolor, bold=red)
            ty += lh
        return h

    def str_(self):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.W}" height="{self.H}" '
                f'viewBox="0 0 {self.W} {self.H}" font-family="{FONT}">\n<defs>'
                '<marker id="arr" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,3 L0,6 Z" fill="#777"/></marker>'
                '<marker id="arrRed" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,3 L0,6 Z" fill="#c0392b"/></marker>'
                '</defs>\n')
        return head + "\n".join(self.parts) + "\n</svg>\n"


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RED = "#c0392b"
BLUE = "#1f6feb"
GRAY = "#666"
GREEN = "#1e8449"
PURPLE = "#7d3c98"
ORANGE = "#b9770e"

SP = 250          # 主干 x
RB = 215          # 残差旁路 x


def header(s, title, sub):
    s.rect(0, 0, s.W, s.H, "#ffffff", "none", rx=0)
    s.text(s.W / 2, 34, title, 21, bold=True, anchor="middle")
    s.text(s.W / 2, 58, sub, 12, anchor="middle", fill="#777")


def save(s, rel):
    p = os.path.join(ROOT, "images", rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").write(s.str_())
    print("wrote", p)


def spine_seg(s, y1, y2):
    s.path([(SP, y1), (SP, y2)], marker=None)


def residual(s, y_top, y_node, label="Residual add"):
    s.path([(SP, y_top), (RB, y_top), (RB, y_node), (SP - 13, y_node)], color="#999")
    s.text(RB - 10, y_node + 4, label, 12.5, bold=True, fill="#444", anchor="end")


def plus(s, y):
    s.circ(SP, y, "+", r=13, color="#555")


# ============================================================================
# 图 5-3 · Qwen3 语言塔：单层逐运算
# ============================================================================
def fig_llm():
    W, H = 1500, 1290
    s = SVG(W, H)
    header(s, "Qwen3 语言塔（三件套里的 “LLM”）：12 × Qwen3DecoderLayer 逐运算展开",
           "对标 llama_decoder.png 的画法：主干下行 + 左侧残差 ⊕ + 中间 qkv/打分/MLP 全展开 · hidden 2048 · 16Q/8KV(GQA)×128 · inter 6144（05 章 §4）")

    hb = s.box(SP - 160, 76, 320, ["inputs_embeds [B,296,2048] 图文混合"], fill="#fff", stroke="#999")
    s.text(SP, 140, "（256×n_img 槽位已被 mlp1 视觉嵌入填过，05 图 5-2）", 11.5, anchor="middle", fill=GRAY)
    s.path([(SP, 76 + hb), (SP, 166)])

    s.rect(60, 166, 1380, 940, "none", BLUE, rx=16, dash="7 5", sw=2)
    s.text(750, 190, "Qwen3Model —— Eagle 的 language_model（整体冻结；原 28 层被 pop 到 12：eagle_backbone.py:56-58）",
           14, bold=True, fill=BLUE, anchor="middle")

    C = 250  # 层容器内容基准
    s.rect(95, C - 20, 1330, 470, "none", RED, rx=12, dash="6 5", sw=1.8)
    s.text(760, C - 2, "12 × Qwen3DecoderLayer —— 框内是一层的完整计算（对标 LlamaDecoderLayer 框）", 13,
           bold=True, fill=RED, anchor="middle")

    spine_seg(s, C + 12, C + 262)
    # ---- attention 分支 ----
    t1 = C + 40
    s.path([(SP, t1), (275, t1)], marker=None)
    s.box(278, t1 - 14, 140, ["Qwen3RMSNorm"], where="n1")
    s.path([(418, t1), (438, t1)])
    s.text(430, t1 - 8, "hidden_states", 11, fill=GRAY, anchor="middle")
    yq, yk, yv = C + 55, C + 110, C + 165
    s.path([(438, yq), (438, yv)], color="#999", marker=None)
    s.path([(438, yq), (458, yq)])
    s.path([(438, yk), (458, yk)])
    s.path([(438, yv), (458, yv)])
    s.box(460, yq - 14, 120, ["q_proj"], where="qp")
    s.box(460, yk - 14, 120, ["k_proj"], where="kp")
    s.box(460, yv - 14, 120, ["v_proj"], where="vp")
    s.text(588, yq - 2, "Q", 12, fill=GRAY)
    s.text(588, yk - 2, "K", 12, fill=GRAY)
    s.path([(580, yq), (598, yq)])
    s.path([(580, yk), (598, yk)])
    s.box(600, yq - 14, 150, ["q_norm（头级RMS）"], where="qn")
    s.box(600, yk - 14, 150, ["k_norm（头级RMS）"], where="kn")
    s.path([(750, yq), (768, yq)])
    s.path([(750, yk), (768, yk)])
    s.box(770, yq - 14, 100, ["RoPE θ=1e6"], where="rp", lsize=11.5)
    s.box(770, yk - 14, 100, ["RoPE"], where="rp2")
    ym = C + 82
    s.path([(870, yq), (893, ym - 6)], color="#999")
    s.path([(870, yk), (893, ym + 6)], color="#999")
    s.circ(905, ym, "×")
    s.path([(917, ym), (938, ym)])
    s.box(940, ym - 18, 125, ["softmax", "(QKᵀ/√128)·V"], where="sm", lsize=11.5)
    s.box(868, C + 8, 260, ["causal mask：仍在——核内置 causal GQA", "无 KV cache：每轮 infer 一次 prefill"], red=True, where="mask", lsize=11)
    s.path([(1000, C + 44), (1000, ym - 20)], color=RED, dash="5 4")
    # V 绕下 → ⊗V
    ymv = C + 165
    s.path([(580, yv), (1090, yv), (1090, ymv + 12)], color="#999", marker=None)
    s.path([(1065, ym + 18), (1065, ymv - 22), (1090, ymv - 12)], color="#999", marker=None)
    s.path([(1002, ym + 18), (1002, ymv - 34), (1078, ymv - 8)], color="#999")
    s.circ(1090, ymv, "×")
    s.text(1112, ymv - 6, "attention_weights", 10.5, fill=GRAY)
    s.path([(1077, ymv), (1062, ymv)])
    s.text(1070, ymv + 16, "attention_outputs", 10.5, fill=GRAY)
    s.box(940, ymv - 14, 120, ["o_proj 2048→2048"], where="op", lsize=11)
    y1 = C + 230
    s.path([(940, ymv), (310, ymv), (310, y1), (SP + 14, y1)], color="#999")
    s.box(350, ymv + 18, 170, ["All reduce：没有——单 die"], fill="#f2f2f2", stroke="#999", where="ar", lsize=11)
    plus(s, y1)
    residual(s, C + 22, y1)

    # ---- MLP 分支 ----
    t2 = C + 270
    s.path([(SP, t2), (275, t2)], marker=None)
    s.box(278, t2 - 14, 140, ["Qwen3RMSNorm"], where="n2")
    s.path([(418, t2), (438, t2)])
    yg, yu = C + 256, C + 336
    s.path([(438, yg), (438, yu)], color="#999", marker=None)
    s.path([(438, yg), (458, yg)])
    s.path([(438, yu), (458, yu)])
    s.box(460, yg - 14, 110, ["gate 2048→6144"], where="g", lsize=11)
    s.box(460, yu - 14, 110, ["up 2048→6144"], where="u", lsize=11)
    s.path([(570, yg), (588, yg)])
    s.box(590, yg - 14, 80, ["SiLU"], where="silu")
    yx = C + 296
    s.path([(670, yg), (688, yg), (688, yx - 13)], color="#999")
    s.path([(570, yu), (700, yu), (700, yx + 13)], color="#999")
    s.circ(694, yx, "×")
    s.path([(706, yx), (728, yx)])
    s.box(730, yx - 14, 110, ["down →2048"], where="dn")
    y2 = C + 390
    s.path([(730, yx), (330, yx), (330, y2), (SP + 14, y2)], color="#999")
    plus(s, y2)
    residual(s, y1 + 14, y2)
    spine_seg(s, y2 + 14, C + 430)

    # ---- 出塔 ----
    s.path([(SP, C + 450), (SP, 742)])
    s.box(150, 742, 1200, ["HF 约定：hidden_states[12] = 第 12 层之后、最终 model.norm 之前——backbone tap 的正是这一支；",
                           "model.norm 照跑但输出被丢弃。整塔每轮 infer 只跑 1 遍（trace 锚点：qwen3_attention_run_die ×12）。"],
          fill="#f2f2f2", stroke="#999", where="norm", lsize=12)
    tap = s.box(SP - 160, 812, 320, ["tap：hidden_states[12] [B,296,2048]"], red=True, where="tap")
    s.path([(SP, 800), (SP, 810)])
    el = s.box(SP - 160, 866, 320, ["eagle_linear → backbone_features"], where="el")
    s.path([(SP, 812 + tap + 2), (SP, 864)], marker=None)
    s.text(SP + 180, 890, "= 脑2 全部 cross-attn 的 K/V（06 章 / 图 4-1 红线）", 12.5, bold=True, fill=RED)
    s.box(620, 830, 700, ["③ 没有 lm_head、没有采样：这 12 层的唯一使命是第 12 层结束时交出 hidden_states——",
                          "不是生成器，是“截断的特征提取器”（04 §1.1）"], red=True, where="red3", lsize=12)

    s.rect(90, 940, 1320, 78, "#fafafa", "#ddd")
    s.text(110, 966, "NPU 落点对照（trace 单轮，09 §10.6）：qwen3_attention_run_die ×12 · mlp_swiglu_run_die ×12 ·", 12.5, fill="#555")
    s.text(110, 990, "rmsnorm_run_die ×25（=12×2+1，含被丢弃的 final norm）· gemm_norm_rope_run_die_single ×12（qkv+qk-norm+rope 合一核）", 12.5, fill="#555")
    s.box(SP - 160, 1050, 320, ["→ DiT 的 cross K/V（图 6-1）"], fill="#fff", stroke="#999", where="out")
    s.path([(SP, 1018), (SP, 1048)])
    save(s, "ch05/qwen3_llm_stack.svg")


# ============================================================================
# 图 14-3 · SigLIP：单层逐运算 + NPU kernel 落点
# ============================================================================
def fig_vit():
    W, H = 1500, 1230
    s = SVG(W, H)
    header(s, "SigLIP 视觉塔（三件套里的 “ViT”）：27 × SiglipEncoderLayer 逐运算 + NPU 落点",
           "对标 llama_decoder.png 的画法 · 27 层/1152/16头×72/inter4304 · 实现住 transformers 4.51.3（14 章 §2）· 红标签=当前主链发射的 LPU kernel（trace 单轮）")

    hb = s.box(SP - 160, 76, 320, ["pixel_values：每图 224×224×3（3 相机同构）"], fill="#fff", stroke="#999")
    s.path([(SP, 76 + hb), (SP, 158)])

    s.rect(60, 158, 1380, 722, "none", ORANGE, rx=16, dash="7 5", sw=2)
    s.text(750, 182, "SiglipVisionModel（冻结）—— 实现住 transformers 4.51.3，gr00t 零 class Siglip* 定义", 14,
           bold=True, fill=ORANGE, anchor="middle")

    pc = s.box(SP - 150, 196, 300, ["PatchEmbed Conv2d 3→1152, k14 s14", "→ 16×16 = 256 token/图"], where="pc")
    s.box(990, 196, 430, ["conv2d_patch_embed_run_die ×1（唯一识别视觉塔的锚点核）"], red=True, where="t0", lsize=11.5)
    s.path([(988, 216), (402, 216)], color=RED, dash="5 4")
    s.path([(SP, 196 + pc), (SP, 250)])
    pe = s.box(SP - 150, 250, 300, ["+ 可学习 PE [256,1152]（无 RoPE、无 CLS）"], where="pe")
    s.path([(SP, 250 + pe), (SP, 322)])

    C = 352
    s.rect(95, C - 22, 860, 452, "none", RED, rx=12, dash="6 5", sw=1.8)
    s.text(525, C - 4, "27 × SiglipEncoderLayer（pre-norm · 双向）", 13, bold=True, fill=RED, anchor="middle")

    spine_seg(s, C + 12, C + 240)
    t1 = C + 40
    s.path([(SP, t1), (278, t1)], marker=None)
    s.box(280, t1 - 14, 140, ["LayerNorm (eps 1e-6)"], where="l1", lsize=11.5)
    s.box(990, t1 - 52, 430, ["layernorm_run_die（ln1+ln2 共用；共享核全模型 ×68，", "计数不能整除 27）"], red=True, where="t1", lsize=11.5)
    s.path([(988, t1 - 26), (424, t1 - 12)], color=RED, dash="5 4")
    s.path([(420, t1), (438, t1)])

    yq, yk, yv = C + 55, C + 110, C + 165
    s.path([(438, yq), (438, yv)], color="#999", marker=None)
    s.path([(438, yq), (458, yq)])
    s.path([(438, yk), (458, yk)])
    s.path([(438, yv), (458, yv)])
    s.box(460, yq - 14, 90, ["q_proj"], where="qp")
    s.box(460, yk - 14, 90, ["k_proj"], where="kp")
    s.box(460, yv - 14, 90, ["v_proj"], where="vp")
    s.text(556, yq - 2, "Q", 12, fill=GRAY)
    s.text(556, yk - 2, "K", 12, fill=GRAY)
    ym = C + 82
    s.path([(550, yq), (593, ym - 6)], color="#999")
    s.path([(550, yk), (593, ym + 6)], color="#999")
    s.circ(605, ym, "×")
    s.path([(617, ym), (633, ym)])
    s.box(635, ym - 18, 130, ["softmax", "(QKᵀ/√72)·V"], where="sm", lsize=11.5)
    s.box(722, C - 20, 232, ["没有 causal mask（双向）·没有 qk-norm", "没有 RoPE——位置在塔头 PE 已加"], red=True, where="mask", lsize=11)
    s.path([(838, C + 43), (632, ym - 14)], color=RED, dash="5 4")
    ymv = C + 165
    s.path([(550, yv + 7), (550, yv + 28), (780, yv + 28), (780, ymv + 14)], color="#999", marker=None)
    s.path([(700, ym + 18), (700, ymv - 30), (770, ymv - 10)], color="#999")
    s.circ(780, ymv, "×")
    s.text(798, ymv - 8, "attn_weights→", 10, fill=GRAY)
    s.path([(767, ymv), (752, ymv)])
    s.box(635, ymv - 14, 115, ["out proj"], where="op")
    y1 = C + 225
    s.path([(635, ymv), (310, ymv), (310, y1), (SP + 14, y1)], color="#999")
    plus(s, y1)
    residual(s, C + 22, y1)

    t2 = C + 262
    s.path([(SP, t2), (278, t2)], marker=None)
    s.box(280, t2 - 14, 140, ["LayerNorm"], where="l2")
    s.path([(420, t2), (438, t2)])
    yf = C + 262
    s.path([(438, yf), (458, yf)])
    s.box(460, yf - 14, 130, ["fc1 →4304"], where="fc1")
    s.path([(590, yf), (610, yf)])
    s.box(612, yf - 14, 175, ["gelu_pytorch_tanh"], where="gelu")
    s.path([(787, yf), (805, yf)])
    s.box(807, yf - 14, 120, ["fc2 →1152"], where="fc2")
    y2 = C + 330
    s.path([(807, yf), (330, yf), (330, y2), (SP + 14, y2)], color="#999")
    plus(s, y2)
    residual(s, y1 + 14, y2)
    spine_seg(s, y2 + 14, C + 410)

    # NPU 标签（右侧列）
    TX = 990
    s.box(TX, C + 50, 430, ["gemm_bias_run_die ×4/层（q,k,v,o；共享核 ×129）"], red=True, where="tq", lsize=11.5)
    s.path([(TX - 2, C + 62), (554, yq + 2)], color=RED, dash="5 4")
    s.box(TX, C + 90, 430, ["unified_mha_run_die_batch_emb ×27 —— 与层数一一对应", "（主链仍 per-op 的铁证；GROOT_NPU_QV_TRANSPOSE 切布局）"],
          red=True, where="tm", lsize=11.5)
    s.path([(TX - 2, C + 106), (768, ym + 4)], color=RED, dash="5 4")
    s.box(TX, C + 152, 430, ["residual_add_run_die（全模型 ×25 < 名义 2×27：部分残差已折进邻居核）"], red=True, where="tr", lsize=11.5)
    s.path([(TX - 2, C + 176), (268, y1 - 2)], color=RED, dash="5 4")
    s.box(TX, C + 246, 430, ["mlp_gelu_run_die（ffi 带 residual_enabled：可折入下方 ⊕）"], red=True, where="tm2", lsize=11.5)
    s.path([(TX - 2, C + 258), (929, yf - 2)], color=RED, dash="5 4")

    s.path([(SP, 784), (SP, 798)], marker=None)
    pl = s.box(SP - 160, 800, 320, ["post-LN → last_hidden_state", "[B, 256×n_img, 1152]"], where="pl")
    s.box(990, 804, 430, ["post_layernorm（vision_transformer_forward_wrapper 的 NPU 调用点）"], red=True, where="t7", lsize=11.5)
    s.path([(988, 822), (522, 822)], color=RED, dash="5 4")

    s.box(SP - 230, 900, 460, ["mlp1 1152→2048（出塔）→ 按 151669 原位回写 LLM 序列（图 14-1）"], fill="#f2f2f2", stroke="#999", where="mp")
    s.path([(SP, 848), (SP, 898)])

    s.rect(90, 958, 1320, 66, "#fdf6e3", ORANGE, dash="4 3")
    s.text(110, 982, "另一条路（已导出、未上主链）：siglip_layer_ffi_fused = LN1+QKV+MHA+out_proj+LN2+MLP 一次调用（27 层→27 发）；", 12.5, fill=GREEN)
    s.text(110, 1004, "硬约束 fp16-only · N==out_n · HEADS*D==N · 单 die；当前主链 trace 中出现 0 次（14 章 §6）。", 12.5, fill=GREEN)

    s.rect(90, 1044, 1320, 132, "#fafafa", "#ddd")
    s.text(110, 1070, "与 llama_decoder.png 逐格同构：pre-norm、双 ⊕ 残差、qkv 三叉、⊗打分→softmax→⊗V→投影折回——同一套读图法。", 12.5, fill="#444")
    s.text(110, 1094, "ViT 的“五无”：无 causal mask（双向）· 无 RoPE（可学习 PE）· 无 CLS/尾头（post-LN 即出塔）· 无 KV cache 概念 ·", 12.5, fill="#444")
    s.text(110, 1116, "非 SwiGLU（fc1+gelu_tanh+fc2）、LayerNorm 而非 RMSNorm。", 12.5, fill="#444")
    s.text(110, 1146, "计数纪律（14 章 §5）：conv2d_patch_embed(×1)/unified_mha_batch_emb(×27) 唯一识别视觉塔；layernorm/gemm_bias/mlp_gelu 为共享核，计数不按 27 硬拆。", 12.5, fill=RED)
    save(s, "ch14/siglip_vit_stack.svg")


# ============================================================================
# 图 6-1 · DiT：单层逐运算（AdaLN + cross/self + GEGLU）
# ============================================================================
def fig_dit():
    W, H = 1500, 1500
    s = SVG(W, H)
    header(s, "DiT（三件套里的 “DiT”）：16 × DiT Block 逐运算 · self/cross 交替 ×4 欧拉步",
           "对标 llama_decoder.png 的画法 · 16 层/hidden 1536/32头×48/GEGLU inner 4×（06 §6.8）· cross_attention_dit.py 实核：一块只有一个注意力")

    s.rect(90, 74, 1320, 56, "#f2f2f2", "#999", dash="6 4")
    s.text(110, 96, "循环外前置（每次 get_action 只 1 遍，4 步恒定）：vl_embs [B,296,2048] = backbone 输出 → vl_self_attention", 12.5, fill="#444")
    s.text(110, 118, "（4 层 × 32 头 × 64）→ 供偶数层 cross 的 K/V（2048→1536 投影，可缓存，06 §6.7）", 12.5, fill="#444")

    s.rect(60, 150, 1380, 1140, "none", RED, rx=16, dash="7 5", sw=2)
    s.text(750, 174, "×4 欧拉步：本容器每轮 infer 跑 4 遍（a ← a + dt·v，flow_matching_action_head.py:404）——逐步变化的只有 a_t 与 temb",
           13.5, bold=True, fill=RED, anchor="middle")

    hb = s.box(SP - 160, 190, 320, ["a_t [B,16,32]（第 1 步 = randn）", "+ t 正弦编码 concat（t 副路）"], where="at")
    s.path([(SP, 190 + hb), (SP, 278)])
    e1 = s.box(SP - 160, 278, 320, ["action_encoder：W2 (3072→1536) + swish"], where="enc")
    s.path([(SP, 278 + e1), (SP, 336)])
    e2 = s.box(SP - 160, 336, 320, ["+ 可学习 position_embedding（只加 16 动作）"], where="pos")
    s.box(700, 300, 430, ["concat：state 1 + future_tokens 32（可学习 nn.Embedding）", "+ action 16 → sa_embs [B,49,1536]"],
          fill="#f2f2f2", stroke="#999", where="cat", lsize=11.5)
    s.path([(698, 340), (412, 380)], color="#999")
    s.path([(SP, 336 + e2), (SP, 396)])
    sab = s.box(SP - 160, 396, 320, ["sa_embs [B,49,1536]（草稿流）"], where="sa")

    C = 480
    s.path([(SP, 396 + sab), (SP, C + 12)], marker=None)
    s.rect(95, C - 34, 1330, 520, "none", GREEN, rx=12, dash="6 5", sw=1.8)
    s.text(760, C - 14, "16 × DiT Block（interleave：偶数层 cross / 奇数层 self——一块只有一个注意力）", 13,
           bold=True, fill=GREEN, anchor="middle")

    # temb 支
    tb = s.box(1050, C - 6, 212, ["temb [B,1536]（TimestepEncoder）"], fill="#f2f2f2", stroke="#999", where="tb", lsize=11)
    s.path([(1050, C + 8), (1012, C + 8)], marker=None)
    s.box(900, C - 6, 110, ["SiLU"], where="ts", lsize=11)
    s.path([(898, C + 8), (860, C + 8)], marker=None)
    s.box(680, C - 8, 180, ["Linear(1536→3072)→scale|shift"], where="tl", lsize=10.5)
    s.path([(760, C + 22), (760, C + 40), (SP + 190, C + 40), (SP + 190, C + 52)], color="#999", dash="4 3")

    spine_seg(s, C + 12, C + 290)
    t1 = C + 66
    s.path([(SP, t1), (278, t1)], marker=None)
    s.box(280, t1 - 16, 210, ["AdaLN：LN(x)×(1+scale)+shift"], where="ad", lsize=11.5)
    s.path([(490, t1), (508, t1)])
    s.text(500, t1 - 9, "xn", 11, fill=GRAY, anchor="middle")

    yq, yk, yv = C + 76, C + 130, C + 184
    ym = C + 102
    s.path([(508, yq), (508, yv)], color="#999", marker=None)
    s.path([(508, yq), (528, yq)])
    s.box(530, yq - 14, 90, ["to_q"], where="tq")
    s.text(626, yq - 2, "Q", 12, fill=GRAY)
    s.box(530, yk - 14, 250, ["偶层：to_k/to_v(vl_embs 2048→1536)", "奇层：to_k/to_v(xn 自身)"], red=True, where="kv", lsize=11)
    s.path([(620, yq), (798, yq), (816, ym - 6)], color="#999")
    s.path([(780, yk - 8), (816, ym + 7)], color="#999")
    s.circ(830, ym, "×")
    s.path([(842, ym), (858, ym)])
    s.box(860, ym - 18, 130, ["softmax", "(QKᵀ/√48)·V"], where="sm", lsize=11.5)
    ymv = C + 200
    s.path([(925, ym + 25), (925, ymv - 35), (985, ymv - 12)], color="#999")
    s.path([(700, yk + 30), (700, ymv), (982, ymv)], color="#999", marker=None)
    s.circ(996, ymv, "×")
    s.text(1014, ymv - 8, "×V", 10.5, fill=GRAY)
    tob = s.box(930, ymv + 34, 130, ["to_out"], where="to", lsize=11.5)
    s.path([(996, ymv + 14), (996, ymv + 32)], color="#999")
    yout = ymv + 34 + tob / 2
    y1 = C + 300
    s.path([(930, yout), (310, yout), (310, y1), (SP + 14, y1)], color="#999")
    s.text(585, 748, "attention_outputs：QK 矩阵极小 49×296——成本在 K/V 投影那两刀", 10.5, fill=GRAY)
    plus(s, y1)
    residual(s, C + 20, y1)

    t2 = C + 356
    spine_seg(s, y1 + 15, t2 - 17)
    s.path([(SP, t2), (278, t2)], marker=None)
    s.box(280, t2 - 16, 200, ["norm3 = LayerNorm", "（普通 LN，不吃 temb）"], where="l3", lsize=11)
    s.path([(492, t2), (510, t2)])
    s.box(512, t2 - 16, 250, ["GEGLU：fc → 2×6144，左半过 gelu"], where="gg", lsize=11.5)
    s.circ(800, t2, "×")
    s.path([(764, t2), (786, t2)])
    s.box(814, t2 - 14, 170, ["fc_out 6144→1536"], where="fo", lsize=11.5)
    y2 = C + 420
    s.path([(814, t2), (330, t2), (330, y2), (SP + 14, y2)], color="#999")
    plus(s, y2)
    residual(s, y1 + 14, y2)
    spine_seg(s, y2 + 15, C + 452)

    s.box(1090, C + 220, 320, ["K/V 冻结红利：偶层的 K/V 在 4 步之间完全不变", "→ 只缓存 K/V（8/16 层，约 14 MiB、省 ~90 GFLOP）"], red=True, where="kvc", lsize=11.5)
    s.path([(1088, C + 235), (784, C + 150)], color=RED, dash="6 4")
    s.box(1090, t2 - 40, 320, ["块内无位置编码（pos_embed=null）、无 mask；", "顺序信息只在塔头的 position_embedding 与 temb"], fill="#f2f2f2", stroke="#999", where="nop", lsize=11.5)

    s.path([(SP, C + 458), (SP, 970)])
    po = s.box(SP - 175, 970, 350, ["出口：proj_out_1(SiLU(temb))→shift/scale；", "norm_out×(1+scale)+shift → proj_out_2 →1024"], where="po")
    sl = s.box(SP - 175, 1042, 350, ["取尾 16 token：pred[:, -action_horizon:]"], fill="#f2f2f2", stroke="#999", where="sl")
    s.path([(SP, 970 + po), (SP, 1040)])
    dc = s.box(SP - 175, 1094, 350, ["action_decoder（CategorySpecific MLP 1024→32）→ v"], where="dc")
    s.path([(SP, 1042 + sl), (SP, 1092)])
    eu = s.box(SP - 175, 1146, 350, ["a ← a + dt·v（欧拉一步）"], red=True, where="eu")
    s.path([(SP, 1094 + dc), (SP, 1144)])
    s.path([(SP + 177, 1164), (1345, 1164), (1345, 214), (SP + 162, 214)], color=RED, dash="7 4", marker="arrRed")
    s.text(1352, 500, "×4", 15, bold=True, fill=RED)

    s.path([(SP, 1175), (SP, 1318)], marker=None)
    s.box(SP - 195, 1320, 390, ["去噪完成的 actions [B,16,32] → 反归一化（图 4-1 出口）"], fill="#fff", stroke="#999", where="out")
    s.rect(90, 1380, 1320, 86, "#fafafa", "#ddd")
    s.text(110, 1410, "NPU 落点（trace 单轮，含 4 步，09 §10.6）：adaln_qkv_run_die ×64（=16×4，AdaLN 与 QKV 投影同核融合）·", 12.5, fill="#555")
    s.text(110, 1434, "fused_mha_out ×68（=16×4+4，多的 4 次是 vl_self_attention）· mlp_gelu_norm ×64 · linear_qkv_run_fused ×4（条件流 qkv 三合一）", 12.5, fill="#555")
    save(s, "ch06/dit_stack.svg")


fig_llm()
fig_vit()
fig_dit()
if WARN:
    print("\n[越界自检] %d 处:" % len(WARN))
    for w in WARN:
        print(" ", w)
    sys.exit(1)
print("OK: no overflow")
