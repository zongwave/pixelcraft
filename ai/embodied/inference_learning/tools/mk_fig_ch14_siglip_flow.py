#!/usr/bin/env python3
"""第 14 章配图：SigLIP 视觉塔的调用链与 NPU 融合算子接入（两图）。

生成：
  images/ch14/siglip_call_flow.svg      图 14-1 SigLIP 在 gr00t 里的完整调用链 + NPU 替换点
  images/ch14/siglip_layer_ops_map.svg  图 14-2 一个 SiglipEncoderLayer 的三栏映射
                                        （原生结构 | NPU per-op 算子 | 整层融合）

数值/事实来源（2026-09-30 实测，均为 file:line 核对）：
  Isaac-GR00T@n1.5-release：gr00t_n1.py:172 / transforms.py:55-83 /
    eagle_backbone.py:50-58,99-111 / eagle2_hg_model/modeling_eagle2_5_vl.py:22,110-112,237,311-338
  transformers 4.51.3：site-packages/transformers/models/siglip/modeling_siglip.py
  gr00t/transformers_npu/npu/__pycache__/siglip.cpython-311.pyc（源码仅存 pyc，符号从中提取）
  groot_ops@v0.2(7df9a04)：ops/torch_ops/*/*_ffi.cc（siglip_layer/attn_ffi_fused 约束行号见章文）
  trace：logs/trace_current_clean/n15_sd3_chrome_trace.json（09 章 §10.6 CUR 热轮，单轮计数）

用法：python3 tools/mk_fig_ch14_siglip_flow.py
"""
import os
import sys

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
        WARN.append("%-16s %7.1f > %.1f  %s" % (where, w, avail, s_))


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

    def line(self, x1, y1, x2, y2, color="#555", dash=None, sw=1.8):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                          f'stroke-width="{sw}"{d}/>')

    def arrow(self, x1, y1, x2, y2, color="#555", dash=None, sw=1.8, marker="arr"):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                          f'stroke-width="{sw}"{d} marker-end="url(#{marker})"/>')

    def box(self, x, y, w, lines, fill, stroke, title=None, tsize=13.5, tcolor="#222",
            lsize=12.5, lcolor="#444", pad=10, where="box", dash=None, align="center"):
        lh = 17
        n = len(lines)
        h = pad * 2 + (20 if title else 0) + lh * n
        if title:
            chk(where + ":title", title, tsize, w - 2 * pad)
        for ln in lines:
            chk(where + ":line", ln, lsize, w - 2 * pad)
        self.rect(x, y, w, h, fill, stroke, dash=dash)
        cy = y + pad + 4
        if title:
            self.text(x + w / 2, cy + 12, title, tsize, bold=True, anchor="middle", color=tcolor)
            cy += 20
        for ln in lines:
            if align == "left":
                self.text(x + pad, cy + lh - 5, ln, lsize, color=lcolor)
            else:
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


OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images", "ch14")
os.makedirs(OUT, exist_ok=True)

BLUE, BLUE_L = "#1f6feb", "#eaf2fd"
RED, RED_L = "#c0392b", "#fdecea"
GREEN, GREEN_L = "#1e8449", "#eafaf1"
ORANGE, ORANGE_L = "#b9770e", "#fdf6e3"
PURPLE, PURPLE_L = "#7d3c98", "#f4ecf7"
GRAY, GRAY_L = "#888", "#f2f2f2"


# ============================================================================
# 图 14-1：SigLIP 在 gr00t 里的完整调用链 + NPU 替换点
# ============================================================================
def fig_call_flow():
    W, H = 1300, 880
    s = SVG(W, H)
    s.rect(0, 0, W, H, "#fafafa", "none")
    s.text(W / 2, 36, "图 14-1 · SigLIP 视觉塔在 gr00t 中的调用链与 NPU 替换点（N1.5 实测 file:line）",
           22, bold=True, anchor="middle")
    s.text(W / 2, 60, "左列 = 一次 get_action 的调用顺序（SigLIP 只跑一遍）；红框 = transformers_npu 在 T0 换掉的类 / 前向里改调的 kernel",
           12.5, anchor="middle", color="#666")

    CX, BW = 50, 660
    RX, RW = 760, 500
    mx = CX + BW / 2

    y = 84
    y = s.box(CX, y, BW, [
        "Gr00tN1Model.get_action  (gr00t_n1.py:172)",
        "obs：3 相机 224×224 + state + 语言指令"],
        "#fff", "#aaa", title="① 入口", where="F1")
    s.arrow(mx, y - 14, mx, y + 2); y += 6

    y = s.box(CX, y, BW, [
        "collate_fn：图像 resize/normalize → pixel_values，",
        "文本 tokenize → input_ids（image_token=151669 占位）",
        "collate_fn 里加 eagle_ 前缀  (gr00t/model/transforms.py:55-83)"],
        "#fff", "#aaa", title="② 数据变换", where="F2")
    s.arrow(mx, y - 14, mx, y + 2); y += 6

    y = s.box(CX, y, BW, [
        "剥 eagle_ 前缀 → self.eagle_model(**input)  (eagle_backbone.py:100-111)",
        "LM 已在 __init__ 裁到前 12 层（layers.pop :56-58，两个 select_layer 陷阱见 §3.3）"],
        BLUE_L, BLUE, title="③ EagleBackbone.forward_eagle", tcolor=BLUE, where="F3")
    s.arrow(mx, y - 14, mx, y + 2); y += 6

    y = s.box(CX, y, BW, [
        "Eagle2_5_VLModel.forward → extract_feature(pixel_values)",
        "(modeling_eagle2_5_vl.py:237/:358 → 定义 :311-338)",
        "★ 这是 SigLIP 全仓库唯一调用点——一次 get_action 只跑 1 遍"],
        ORANGE_L, ORANGE, title="④ extract_feature", tcolor=ORANGE, where="F4")
    s.arrow(mx, y - 14, mx, y + 2); y += 6

    yh = y
    y = s.box(CX, y, BW, [
        "transformers 4.51.3 siglip/modeling_siglip.py（gr00t 零 class Siglip* 定义）",
        "patch conv(14×14/s14) → +可学习PE[256,1152] → 27×SiglipEncoderLayer",
        "→ post-LN → 每图 256 token ×1152（27层/16头/inter4304/无CLS）"],
        RED_L, RED, title="⑤ SiglipVisionModel（本章主角）", tcolor=RED, where="F5")
    s.arrow(mx, y - 14, mx, y + 2); y += 6

    y = s.box(CX, y, BW, [
        "select_layer==-1 → last_hidden_state (modeling:312-322)",
        "mlp1: 1152→2048 (:334-338) → 按 image_token_index=151669",
        "原位替换进 LLM 输入序列 (:247)"],
        "#fff", "#aaa", title="⑥ 回写 LLM 序列", where="F6")
    s.arrow(mx, y - 14, mx, y + 2); y += 6

    y = s.box(CX, y, BW, [
        "Qwen3 前 12 层 → output_hidden_states",
        "取 hidden_states[select_layer] → eagle_linear (:109-110)",
        "NPU：qwen3_attention_run_die×12 + mlp_swiglu×12（09 章，非本章）"],
        PURPLE_L, PURPLE, title="⑦ LM 出 backbone_features", tcolor=PURPLE, where="F7")
    s.arrow(mx, y - 14, mx, y + 2); y += 6

    y = s.box(CX, y, BW, [
        "FlowmatchingActionHead：DiT 去噪 4 步全部复用 ⑦ 的输出，",
        "不再回视觉塔 —— 所以 SigLIP 每轮 infer 只出现 1 次"],
        "#fff", "#aaa", title="⑧ Action Head（06 章）", where="F8")

    # 右侧：NPU 替换点，三个红/绿虚线框
    s.text(RX + RW / 2, 100, "transformers_npu 的 SigLIP 接入（npu/siglip.py）", 14.5,
           bold=True, anchor="middle", color=RED)
    yr = 116
    yr = s.box(RX, yr, RW, [
        "T0 换类符号（必须早于 AutoModel.from_config，",
        "  eagle_backbone.py:50-51）：",
        "SiglipEncoderLayer → encoder_layer_forward_wrapper",
        "SiglipMultiheadAttentionPoolingHead（npu_patch.install）"],
        RED_L, RED, title="注入点：patch 类符号", tcolor=RED, lsize=12, where="R1", dash="6 4")
    s.arrow(RX - 8, yr - 60, CX + BW + 6, yh + 42, color=RED, dash="6 4", marker="arrRed")
    yr += 12
    yr = s.box(RX, yr, RW, [
        "NPU_SiglipVisionEmbeddings → conv2d_patch_embed_run_die ×1",
        "NPU_SiglipAttention → gemm_bias(QKV/O) +",
        "  unified_mha_run_die_batch_emb ×27（=27 层铁证）",
        "NPU_SiglipMLP → mlp_gelu_run_die（ffi 带 residual_enabled）",
        "+ layernorm_buf / residual_add_buf / post_layernorm",
        "版本契约：transformers 4.51.3（09 章 T0 静默失效框架）"],
        RED_L, RED, title="⑤ 内部逐结构替换（per-op，当前主链）", tcolor=RED,
        lsize=12, where="R2", dash="6 4", align="left")
    s.arrow(RX - 8, yr - 70, CX + BW + 6, yh + 58, color=RED, dash="6 4", marker="arrRed")
    yr += 12
    yr = s.box(RX, yr, RW, [
        "siglip_layer_ffi_fused：LN1+QKV+MHA+out_proj+LN2+MLP 一次调用",
        "硬约束：fp16-only / N==out_n / HEADS*D==N / 单 die",
        "当前 N1.5 主链未启用：热轮 trace 中出现 0 次（§5）"],
        GREEN_L, GREEN, title="另一条路：整层融合（groot_ops 已导出）", tcolor=GREEN,
        lsize=12, where="R3", dash="6 4", align="left")
    s.arrow(RX - 8, yr - 40, CX + BW + 6, yh + 88, color=GREEN, dash="6 4", marker="arrGreen")
    yr += 12
    yr = s.box(RX, yr, RW, [
        "GROOT_NPU_QV_TRANSPOSE：unified_mha_forward ↔ _qv_T 两布局",
        "NPU_DUMP_SIGLIP：中间激活落 /tmp/npu_*.npy（默认关）",
        "golden 对账：deployment_scripts/npu/npu_golden_gate.sh"],
        GRAY_L, GRAY, title="开关与探针（§7）", tcolor="#555", lsize=12, where="R4", align="left")

    s.text(W / 2, H - 20, "trace 计数 = trace_current_clean/n15_sd3_chrome_trace.json 单轮 infer（09 章 §10.6 CUR 热轮口径）",
           12, anchor="middle", color="#777")
    save("siglip_call_flow.svg", s, W, H)


# ============================================================================
# 图 14-2：一个 SiglipEncoderLayer 的三栏映射
# ============================================================================
def fig_layer_ops_map():
    W, H = 1300, 860
    s = SVG(W, H)
    s.rect(0, 0, W, H, "#fafafa", "none")
    s.text(W / 2, 36, "图 14-2 · 一个 SiglipEncoderLayer（×27）：原生结构 | NPU per-op（当前主链） | 整层融合（已导出未启用）",
           20.5, bold=True, anchor="middle")
    s.text(W / 2, 60, "左：transformers 4.51.3 SiglipEncoderLayer（pre-norm）；中：npu/siglip.py wrapper 改调的 torch_evo kernel（计数=热轮 trace 单轮）；右：groot_ops@v0.2 fused_mha_out_ffi.cc",
           12, anchor="middle", color="#666")

    C1X, C2X, C3X, CW = 40, 460, 900, 380
    y0 = 88
    s.text(C1X + CW / 2, y0, "① 原生（GPU / flash-attn2）", 15, bold=True, anchor="middle", color="#444")
    s.text(C2X + CW / 2, y0, "② NPU per-op（当前主链）", 15, bold=True, anchor="middle", color=RED)
    s.text(C3X + CW / 2, y0, "③ 整层融合 siglip_layer_ffi_fused", 15, bold=True, anchor="middle", color=GREEN)
    y0 += 16

    mids = []

    def row(y, l1, l2, l3, fill1="#fff", fill3=GREEN_L, stroke3=GREEN):
        h1 = s.box(C1X, y, CW, l1, fill1, "#aaa", lsize=12, where="L1")
        h2 = s.box(C2X, y, CW, l2, RED_L, RED, lsize=12, where="L2")
        h3 = s.box(C3X, y, CW, l3, fill3, stroke3, lsize=12, where="L3")
        b = max(h1, h2, h3)
        mid = y + (b - y) / 2
        s.arrow(C1X + CW, mid, C2X - 2, mid, color=RED, marker="arrRed")
        s.arrow(C2X + CW, mid, C3X - 2, mid, color=GREEN, marker="arrGreen")
        return b + 12

    y = row(y0, ["LayerNorm（eps=1e-6）"],
            ["layernorm_run_die（ln1）", "共享核：全模型 ×68"],
            ["LN1（ln1_has_res 可带残差）"])
    y = row(y, ["MultiheadAttention", "q/k/v proj + softmax(QKᵀ/√d)V + out proj", "（16 头 × 72 = 1152）"],
            ["gemm_bias_run_die ×4/层（q,k,v,o）", "共享核：全模型 ×129", "unified_mha_run_die_batch_emb", "  ×27 —— 与层数一一对应", "GROOT_NPU_QV_TRANSPOSE 选布局"],
            ["QKV：wt_q/k/v+biases+eye", "→ MHA（qv 布局内化，", "   免 transpose）→ out_proj"],
            )
    y = row(y, ["+ 残差（x + attn_out）"],
            ["residual_add_run_die", "全模型 ×25 < 名义 2×27：", "部分残差已折进邻居 kernel"],
            ["（融合在 ln1_has_res /", "  mlp residual_enabled 内）"])
    y = row(y, ["LayerNorm"],
            ["layernorm_run_die（ln2）"],
            ["LN2"])
    y = row(y, ["MLP：fc1(1152→4304)", "→ gelu_pytorch_tanh → fc2(→1152)"],
            ["mlp_gelu_run_die（独立核）", "ffi 有 residual_enabled 参数", "（可把下方残差折进来）"],
            ["up(gelu_pytorch_tanh)/down"])
    ye = row(y, ["+ 残差（x + mlp_out）"],
             ["residual_add_run_die"],
             ["（层输出 = 下一层输入）"])

    # 底部说明
    y = ye + 24
    s.rect(C1X, y, W - 2 * C1X, 118, "#fff", "#bbb")
    s.text(C1X + 16, y + 26, "怎么读这张图：", 13.5, bold=True)
    s.text(C1X + 16, y + 48, "· 左→中：npu/siglip.py 的三个 NPU_* 类 + encoder_layer_forward_wrapper 逐个结构换 kernel（per-op，发射次数≈结构数）；"
                             "「共享核」的计数不能整除 27，因为 LM/DiT 也用同名 kernel（09 章纪律：名+@op 归组、trace 为准）。", 12)
    s.text(C1X + 16, y + 70, "· 中→右：siglip_attn_ffi_fused / siglip_layer_ffi_fused（fused_mha_out_ffi.cc :319/:472）把整层折成 1~2 次发射——"
                             "N1.5 主链 trace 中出现 0 次；n1.6 线板测 3-cam 视觉塔 fused 0.082 s vs per-op 0.133 s（cos 0.999984，出处见 §6）。", 12)
    s.text(C1X + 16, y + 92, "· 塔尾另有 post_layernorm（vision_transformer_forward_wrapper 的 NPU 调用点）；mlp1(1152→2048) 不在 SigLIP 融合算子清单内。", 12)
    save("siglip_layer_ops_map.svg", s, W, H)


def save(name, s, W, H):
    p = os.path.join(OUT, name)
    with open(p, "w") as f:
        f.write(s.str_())
    print("wrote", p, "%dx%d" % (W, H))


if __name__ == "__main__":
    fig_call_flow()
    fig_layer_ops_map()
    if WARN:
        print("\n[文本越界自检] %d 处：" % len(WARN))
        for w in WARN:
            print(" ", w)
        sys.exit(1)
    print("OK: no overflow")
