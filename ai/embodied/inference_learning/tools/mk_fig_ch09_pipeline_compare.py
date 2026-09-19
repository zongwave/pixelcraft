#!/usr/bin/env python3
"""第 9 章配图：gr00t e2e 原生 PyTorch 逐算子 pipeline vs E200 NPU 融合算子 pipeline（最终交付 v0.2）。

生成：
  images/ch09/npu_pipeline_native_vs_fused.svg

数值来源：groot_ops CHANGELOG v0.1/v0.2、Isaac-GR00T docs/performance/gr00t_npu_*.md、
Redmine #162（板端 e2e/精度/胶水统计）。风格参照 #162 附件图 gr00t_e2e_pipeline_before_after.svg。

用法：python3 tools/mk_fig_ch09_pipeline_compare.py
"""
import os

W, H = 1560, 1210
FONT = "Helvetica, Arial, 'Noto Sans CJK SC', 'WenQuanYi Zen Hei', sans-serif"

def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

class SVG:
    def __init__(self):
        self.parts = []

    def rect(self, x, y, w, h, fill, stroke, rx=8, sw=1.6, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=13, fill="#222", bold=False, anchor="start", color=None):
        w = "bold" if bold else "normal"
        c = color or fill
        self.parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{w}" fill="{c}" text-anchor="{anchor}">{esc(s)}</text>')

    def arrow(self, x1, y1, x2, y2, color="#555", dash=None, sw=1.8, marker="arr"):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d} marker-end="url(#{marker})"/>')

    def box(self, x, y, w, lines, fill, stroke, title=None, tsize=13, tcolor="#222", lsize=12, lcolor="#444", pad=12):
        lh = 17
        n = len(lines)
        h = pad * 2 + (18 if title else 0) + lh * n
        self.rect(x, y, w, h, fill, stroke)
        cy = y + pad + 4
        if title:
            self.text(x + w / 2, cy + 11, title, tsize, bold=True, anchor="middle", color=tcolor)
            cy += 18
        for ln in lines:
            self.text(x + w / 2, cy + lh - 4, ln, lsize, anchor="middle", color=lcolor)
            cy += lh
        return y + h

    def str(self):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
                f'viewBox="0 0 {W} {H}" font-family="{FONT}">\n<defs>'
                '<marker id="arr" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,3 L0,6 Z" fill="#555"/></marker>'
                '<marker id="arrRed" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,3 L0,6 Z" fill="#c0392b"/></marker>'
                '<marker id="arrGreen" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,3 L0,6 Z" fill="#1e8449"/></marker>'
                '</defs>\n')
        return head + "\n".join(self.parts) + "\n</svg>\n"


s = SVG()
s.rect(0, 0, W, H, "#fafafa", "none")
s.text(W / 2, 42, "gr00t e2e infer Pipeline：原生 PyTorch 逐算子  vs  E200 NPU 融合算子（最终交付）", 24, bold=True, anchor="middle")
s.text(W / 2, 68, "单 Die · 数值来源：groot_ops CHANGELOG v0.1/v0.2 与 Redmine #162 板端实测（L20 交叉机 golden 比对）", 13, anchor="middle", color="#666")

LX, RX, CW = 30, 800, 730   # 两列 x 与宽

# ---------- 左列：原生 PyTorch ----------
y = 95
s.rect(LX, y, CW, 1080, "#fdecea", "#e6b0aa", 12, 2)
s.text(LX + CW / 2, y + 32, "原生 PyTorch 逐算子（未融合 / NPU 接入基线）", 18, bold=True, anchor="middle", color="#c0392b")
y += 52
y = s.box(LX + 25, y, CW - 50, ["输入：图像 ×N + 文本 + 本体 state"], "#fff", "#aaa8", lcolor="#333") + 14
s.arrow(LX + CW / 2, y - 14, LX + CW / 2, y + 2); y += 4

y = s.box(LX + 25, y, CW - 50, [
    "SigLIP 视觉：conv2d patch_embed + L×[ 3×linear(QKV) → softmax(QKᵀ+mask)V",
    "→ out proj → LN → fc1/GELU/fc2 ]　—— 全 aten 逐算子，穿插 permute/contiguous"],
    "#f5f5f5", "#999", title="Backbone（eagle2.5 VL）· SigLIP", tcolor="#333") + 12
s.arrow(LX + CW / 2, y - 12, LX + CW / 2, y + 2); y += 4

y = s.box(LX + 25, y, CW - 50, [
    "Qwen3 语言：L×[ rmsnorm → qkv+RoPE → attention → o_proj → rmsnorm → SwiGLU ]",
    "每层 attention/linear/norm/mlp 各自独立发射"],
    "#f5f5f5", "#999", lcolor="#444") + 10
s.box(LX + 25, y, CW - 50, ["vl_embs（视觉+语言 embedding）· 每轮 H2D 上载"], "#fff", "#aaa8", lcolor="#333")
y += 52
s.arrow(LX + CW / 2, y - 8, LX + CW / 2, y + 2); y += 6

s.rect(LX + 20, y, CW - 40, 470, "#fdf3e7", "#e59866", 10, 1.8)
s.text(LX + CW / 2, y + 26, "Action Head（DiT flow-matching · get_action）", 15, bold=True, anchor="middle", color="#b9770e")
y2 = y + 40
y2 = s.box(LX + 45, y2, CW - 230, [
    "initial noise = torch.randn",
    "CPU 生成 + H2D（同步阻塞设备）"], "#fadbd8", "#e74c3c", lcolor="#7b241c") + 12
s.arrow(LX + CW / 2 - 60, y2 - 12, LX + CW / 2 - 60, y2 + 2); y2 += 6

y2b = s.box(LX + CW - 175, y + 40, 140, [
    "权重 H2D", "~1.65 GB/轮"], "#fff", "#c0392b", tsize=1, lcolor="#c0392b")
s.arrow(LX + CW - 105, yb_h := (y + 40) + y2b - y - 40, LX + CW - 105, y + 460, "#c0392b", dash="6,5", marker="arrRed")

y2 = s.box(LX + 45, y2, CW - 230, [
    "4×Euler × 16 DiT 层（每步过一遍）：",
    "adaLN → 3×linear(QKV) → attention → project_out",
    "adaLN → fc1 → GELU → fc2　(逐算子 + 布局重排)"], "#fadbd8", "#e74c3c", lcolor="#7b241c") + 12
s.arrow(LX + CW / 2 - 60, y2 - 12, LX + CW / 2 - 60, y2 + 2); y2 += 6

y2 = s.box(LX + 45, y2, CW - 230, ["actions += dt · pred_velocity（mul+add 各一次发射）"], "#f5f5f5", "#999", lcolor="#444") + 14

s.rect(LX + 40, y2, CW - 80, 74, "#fff", "#c0392b", 8, 1.6)
s.text(LX + CW / 2, y2 + 22, "aten 胶水：copy_ 5554 次/轮 · contiguous 432.8ms · permute/reshape/clone 频发", 13, bold=True, anchor="middle", color="#c0392b")
s.text(LX + CW / 2, y2 + 46, "to_head_major_k 963 次/轮 · evMemcpy 605 次（99% 来自接入层）", 13, anchor="middle", color="#943126")
y2 += 90
s.arrow(LX + CW / 2 - 60, y2 - 12, LX + CW / 2 - 60, y2 + 2); 
y2 = s.box(LX + 45, y2, CW - 230, ["action_pred"], "#eaf0fa", "#8da9d8", lcolor="#333")

s.rect(LX + 25, 1010, CW - 50, 130, "#fff", "#c0392b", 8, 1.6)
s.text(LX + 45, 1035, "指标（step1 稳态，单 Die）", 14, bold=True, color="#c0392b")
s.text(LX + 45, 1060, "e2e ≈ 0.826 s/轮　·　device busy ≈ 13%（host/H2D bound，设备大量闲置）", 13, color="#641e16")
s.text(LX + 45, 1085, "权重每轮 H2D 上载 + CPU randn H2D + 大量 aten reshape/clone/transpose/cat", 13, color="#641e16")
s.text(LX + 45, 1110, "golden 比对 cos ≈ 0.99338（L20 原生 PyTorch 为参考基准）", 13, color="#641e16")

# ---------- 右列：NPU 融合算子 ----------
y = 95
s.rect(RX, y, CW, 1080, "#eafaf1", "#a9dfbf", 12, 2)
s.text(RX + CW / 2, y + 32, "E200 NPU 融合算子 + 权重设备驻留（v0.1 → v0.2 最终交付）", 18, bold=True, anchor="middle", color="#1e8449")
y += 52
y = s.box(RX + 25, y, CW - 50, ["输入：图像 ×N + 文本 + 本体 state"], "#fff", "#aaa8", lcolor="#333") + 14
s.arrow(RX + CW / 2, y - 14, RX + CW / 2, y + 2); y += 4

y = s.box(RX + 25, y, CW - 50, [
    "SigLIP：conv2d_patch_embed（加载期权重转置 12ms→2ms）+ unified_mha + mlp_gelu",
    "跨图 cross-image host-FFI 整层编排（kernel clone/transpose 消除）"],
    "#d5f5e3", "#27ae60", title="Backbone（eagle2.5 VL）· SigLIP", tcolor="#145a32") + 12
s.arrow(RX + CW / 2, y - 12, RX + CW / 2, y + 2); y += 4

y = s.box(RX + 25, y, CW - 50, [
    "Qwen3：gemm_norm_rope + qwen3_attention + mlp_swiglu（整层 FFI，发射次数大幅下降）",
    "（v0.2：胶水 aten 未收 → view 缓存，794→504 次）"],
    "#d5f5e3", "#27ae60", lcolor="#145a32") + 10
s.box(RX + 25, y, CW - 50, ["vl_embs · 设备驻留（免每轮 H2D）"], "#fff", "#aaa8", lcolor="#333")
y += 52
s.arrow(RX + CW / 2, y - 8, RX + CW / 2, y + 2); y += 6

s.rect(RX + 20, y, CW - 40, 470, "#f4fdf6", "#82c99b", 10, 1.8)
s.text(RX + CW / 2, y + 26, "Action Head（DiT flow-matching · get_action）", 15, bold=True, anchor="middle", color="#1e8449")
y2 = y + 40
y2 = s.box(RX + 45, y2, CW - 230, [
    "initial noise：设备侧 randn_normal（#161）",
    "免 CPU randn + 同步 H2D"], "#d5f5e3", "#27ae60", lcolor="#145a32") + 12
s.arrow(RX + CW / 2 - 60, y2 - 12, RX + CW / 2 - 60, y2 + 2); y2 += 6

yb_h = s.box(RX + CW - 175, y + 40, 140, [
    "权重设备驻留", "H2D ≈ 0（稳态）"], "#fff", "#1e8449", tsize=1, lcolor="#1e8449")
s.arrow(RX + CW - 105, (y + 40) + yb_h - y - 40, RX + CW - 105, y + 460, "#1e8449", dash="6,5", marker="arrGreen")

y2 = s.box(RX + 45, y2, CW - 230, [
    "4×Euler × 16 DiT 层 · 每层 3 个大融合：",
    "adaln_qkv（norm+QKV）+ fused_mha_out（attn+o_proj）",
    "+ mlp_gelu_norm（norm+FFN+gelu）"], "#d5f5e3", "#27ae60", lcolor="#145a32") + 12
s.arrow(RX + CW / 2 - 60, y2 - 12, RX + CW / 2 - 60, y2 + 2); y2 += 6

y2 = s.box(RX + 45, y2, CW - 230, [
    "头 dit_action_head（2 次发射，tau 折入 bias，消 cat/swish）",
    "尾 dit_action_tail（1 次发射，含 Euler：macc scale-add）"], "#d5f5e3", "#27ae60", lcolor="#145a32") + 14

s.rect(RX + 40, y2, CW - 80, 74, "#fff", "#1e8449", 8, 1.6)
s.text(RX + CW / 2, y2 + 22, "效果：contiguous −88% · permute −93% · to_head_major_k 963 → 0", 13, bold=True, anchor="middle", color="#1e8449")
s.text(RX + CW / 2, y2 + 46, "布局直出（head-major QKV / token-major K,O）消除跨算子数据搬运", 13, anchor="middle", color="#145a32")
y2 += 90
s.arrow(RX + CW / 2 - 60, y2 - 12, RX + CW / 2 - 60, y2 + 2)
y2 = s.box(RX + 45, y2, CW - 230, ["action_pred"], "#eaf0fa", "#8da9d8", lcolor="#333")

s.rect(RX + 25, 1010, CW - 50, 130, "#fff", "#1e8449", 8, 1.6)
s.text(RX + 45, 1035, "指标（step1 稳态，单 Die）", 14, bold=True, color="#1e8449")
s.text(RX + 45, 1060, "e2e 0.826 → 0.697（v0.1）→ 0.14 s 稳态（−83%，5.9×）　·　device busy 51%（87/171ms）", 13, color="#145a32")
s.text(RX + 45, 1085, "融合算子主导：adaln_qkv / fused_mha_out / mlp_gelu_norm / mlp_swiglu / dit_action_head|tail", 13, color="#145a32")
s.text(RX + 45, 1110, "golden 比对 cos 0.99999+（n1.6 达 0.9999999，逐位一致验证）", 13, color="#145a32")

s.text(W / 2, 1195, "参考：Redmine #162/#170 · groot_ops CHANGELOG v0.1/v0.2 · Isaac-GR00T docs/performance/gr00t_npu_*.md", 12, anchor="middle", color="#888")

out = os.path.join(os.path.dirname(__file__), "..", "images", "ch09", "npu_pipeline_native_vs_fused.svg")
with open(out, "w") as f:
    f.write(s.str())
print("written:", os.path.normpath(out))
