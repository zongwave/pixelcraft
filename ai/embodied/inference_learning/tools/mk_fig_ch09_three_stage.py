#!/usr/bin/env python3
"""第 9 章配图：NPU 融合算子接入 gr00t 的「三级入口」时间轴（对照 tag v0.2 实测代码）。

生成：
  images/ch09/npu_three_stage.svg

口径：Isaac-GR00T@v0.2 (57ca788) + groot_ops@v0.2 (7df9a04)。全部 file:line 已逐条核对。
与 npu_patch_framework.svg（n1.6 空间分层栈）互补：本图是**时间轴**（何时做什么）。

用法：python3 tools/mk_fig_ch09_three_stage.py
"""
import os

W, H = 1560, 1360
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
        WARN.append("%-12s %6.1f > %.1f  %s" % (where, w, avail, s_))


class SVG:
    def __init__(self):
        self.parts = []

    def rect(self, x, y, w, h, fill, stroke, rx=8, sw=1.6, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                          f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=13, fill="#222", bold=False, anchor="start", color=None):
        w = "bold" if bold else "normal"
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{w}" '
                          f'fill="{color or fill}" text-anchor="{anchor}">{esc(s)}</text>')

    def arrow(self, x1, y1, x2, y2, color="#555", sw=1.8, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                          f'stroke-width="{sw}"{d} marker-end="url(#arr)"/>')

    def cell(self, x, y, w, h, title, lines, fill, stroke, tcolor="#222"):
        self.rect(x, y, w, h, fill, stroke, 8, 1.5)
        self.text(x + 14, y + 24, title, 13.5, bold=True, color=tcolor)
        yy = y + 46
        for ln in lines:
            self.text(x + 14, yy, ln, 12, color="#3a3a3a")
            yy += 19
        return y + h

    def strip(self, x, y, w, h, lines, fill, stroke, color="#333", size=12.5, bold=False):
        self.rect(x, y, w, h, fill, stroke, 6, 1.4)
        yy = y + 21
        for ln in lines:
            self.text(x + 16, yy, ln, size, color=color, bold=bold)
            yy += 19

    def svg(self):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
                f'viewBox="0 0 {W} {H}" font-family="{FONT}">\n<defs>'
                '<marker id="arr" markerWidth="10" markerHeight="10" refX="7" refY="3" '
                'orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,3 L0,6 Z" fill="#555"/></marker>'
                '</defs>\n')
        return head + "\n".join(self.parts) + "\n</svg>\n"


s = SVG()
s.rect(0, 0, W, H, "#fcfcfc", "none")
s.text(W / 2, 40, "NPU 融合算子怎么接进 gr00t：三级入口时间轴（Isaac-GR00T v0.2 + groot_ops v0.2）",
       23, bold=True, anchor="middle")
s.text(W / 2, 65, "空间分层栈见 npu_patch_framework.svg（n1.6 视角）；本图回答的是「何时做哪一步」——"
                  "接入 = 换叶子 + 守布局，三步各有一个静默失效点", 13, anchor="middle", color="#666")

# ---------------- 时间轴 ----------------
ty = 92
s.arrow(60, ty + 16, W - 60, ty + 16, "#999", 2.2)
stations = [
    (170, "T0 安装 / 绑定期", "import 期 · from_pretrained 之前 · 一次", "#1f6feb"),
    (700, "T1 权重期", "load_state_dict 后 · 首次 forward 懒触发", "#b9770e"),
    (1230, "T2 前向期", "每次 get_action · 每个被换的叶子", "#1e8449"),
]
for cx, t1, t2, col in stations:
    s.parts.append(f'<circle cx="{cx}" cy="{ty + 16}" r="8" fill="{col}"/>')
    s.text(cx, ty - 4, t1, 15, bold=True, anchor="middle", color=col)
    s.text(cx, ty + 44, t2, 12, anchor="middle", color="#777")

X0, CW = 30, 1500
GAP = 26


def panel(y, title, subtitle, fill, stroke, tcol, cells, footer):
    """高度按内容自适应；副标题右对齐；逐行做越界自检。"""
    n = len(cells)
    inner = CW - 48
    cw = (inner - GAP * (n - 1)) / n
    avail = cw - 30
    maxlines = max(len(cl) for _, cl in cells)
    for ct, cl in cells:
        chk("cell-title", ct, 13.5, avail)
        for ln in cl:
            chk("cell-line", ln, 12, avail)
    ch = 46 + 19 * maxlines + 10
    h = 44 + ch + 16 + (60 if footer else 0)
    s.rect(X0, y, CW, h, "#ffffff", stroke, 12, 2)
    s.rect(X0, y, 8, h, fill, fill, 0, 0)
    s.text(X0 + 26, y + 27, title, 17, bold=True, color=tcol)
    s.text(X0 + CW - 26, y + 27, subtitle, 12.5, anchor="end", color="#8a8a8a")
    cy = y + 44
    for i, (ct, cl) in enumerate(cells):
        s.cell(X0 + 24 + i * (cw + GAP), cy, cw, ch, ct, cl, fill, stroke, tcol)
    if footer:
        s.strip(X0 + 24, cy + ch + 14, CW - 48, 46, footer, "#f7f7f7", "#bbb")
    return y + h


y = 150
y = panel(y, "① 安装/绑定期", "gr00t/transformers_npu/{__init__,patch,register}.py",
          "#eaf2fd", "#1f6feb", "#1a5fb4",
          [("install() 只有五步  __init__.py:29-33",
            ["init_lpu → register_all → apply_patches",
             "→ _verify_patch_bindings → auto_report",
             "入口二 hijack_policy_load：包住 _load_model",
             "红线：晚于 from_pretrained ⇒ 静默走原生链"]),
           ("两种替换语义  register.py:6-8",
            ["「M.Cls」整类替换（Qwen3Attention）",
             "「M.Cls.method」方法包装（get_action）",
             "函数名尾缀 wrapper/decorator ⇒ 叠加装饰",
             "apply 前须先 import 目标模块进 sys.modules"]),
           ("from-import 传播  patch.py",
            ["遍历 sys.modules，换掉所有 cur is orig",
             "否则 fm.SinusoidalPositionalEncoding 仍是旧类",
             "跳过 torch/builtins；remove_patches() 可回滚",
             "v0.2 登记 30 个目标（6+6+1+17）"]),
           ("金丝雀 fail-fast  #149 / __init__.py:38-51",
            ["assert ae.SinusoidalPositionalEncoding is NPU_",
             "assert fm.SinusoidalPositionalEncoding is NPU_",
             "探针选小叶子：DiT 是方法包装，绑 orig 也「能跑」",
             "退化症状：逐 op 现算链复活 ≈ +30 ms/轮 host"])],
          None)

y += 22
y = panel(y, "② 权重期", "ops/evo_lpu.py + npu/*._load_lpu()",
          "#fff6e8", "#b9770e", "#9c5f00",
          [("权重上设备：懒 post-hook",
            ["NPU_*._load_lpu()   qwen3.py:127/171/210/326",
             "NPULinear._lpu()     action.py:96-131",
             "rep_np [2,*] die 轴复刻 / to_lpu split0|1|2",
             "主链 fp16（evo_lpu.py:38）：bf16 无 split 互转"]),
           ("常量预计算（第 06 章「常量张量」）",
            ["build_timestep_encoder :646（H2D 一次）",
             "timestep_encoder(ts) 按 (dev,d,ts) 只算一次",
             "build_adalayernorm :745（恒等 lnw/lnb 防 NaN）",
             "configure(ns,buckets) 预热 ⇒ 首轮起零 pe kernel"]),
           ("t 只有 4 个取值 ⇒ 命中率 100%",
            ["num_inference_timesteps = 4（§6.5）",
             "temb / pe / AdaLN scale_shift 三个桶各 4 项常驻",
             "#173 省 9.4 MB/调用；pe 热路径零 kernel 发射",
             "_temb_l 按 id(temb) 缓存，免逐 block d2h + 重传"]),
           ("与封装层的摩擦点（03/04 章 × 09 章）",
            ["策略层加载后整模型 .to(device)，",
             "把我们钉上 LPU 的 future_tokens 又拽回 CPU",
             "它不认识「设备常驻」这个概念",
             "修法：进 cat 前再钉一次  action.py:1257-1263"])],
          ["权重侧结论：权重全部已常驻（首 forward 懒缓存，功能等价 vllm 的 load 时统一预处理）。"
           "真正 per-forward 的 host 大头不在权重，而在激活布局与 k/v 重排：",
           "to_lpu 1919 次 · rep_from 606 · rep_np 211 · page_kv(host)+to_np 各 214　—— "
           "docs/reference/gr00t_npu_weight_preproc_map.md §2"])

y += 22
y = panel(y, "③ 前向期", "npu/{qwen3,siglip,eagle,attention,action}.py → torch_evo.*",
          "#eef8f0", "#1e8449", "#16693a",
          [("backbone 落到哪个 kernel",
            ["SigLIP: conv2d_patch_embed / LN / MLP",
             "Qwen3 : gemm_norm_rope（QKV+qk_norm+RoPE 一次）",
             "        + qwen3_attention · gemm_bias(o_proj)",
             "Eagle : mlp1(1152→2048) → NPULinear"]),
           ("action head 落到哪个 kernel",
            ["Encoder: gemm_bias(W1) + mlp_silu    :1196",
             "Decoder/state：CategorySpecificMLP → mlp_relu",
             "DiT block: torch_evo.dit_block_fused  :773",
             "尾调制 : dit_tail_modulate            :934"]),
           ("守卫即契约  action.py:703-736",
            ["pos_embed is None / 无 residual_connection",
             "to_k.in == to_v.in == enc_dim；头维必须匹配",
             "n % 16 == 0；FFN 必须 plain-GELU（不是 GEGLU）",
             "任一不满足 ⇒ _r(原因) 静默回退分段路径"]),
           ("A/B 开关面板（27 个 GROOT_NPU_*）",
            ["DIT_BLOCK_FUSED / FUSED_FFN_NORM / AE_MLP",
             "AD_MLP_RELU / GETACTION_O1 / SKIP_NOOP_*",
             "ATTN_SINGLE / ATTN_CPU / PE_FFI / LN_BUF …",
             "默认全 1；=0 就是「这一段退回非融合」的对照组"])],
          ["粒度跃迁与发射账见 dit_fusion_granularity.svg（图 9-4）；"
           "kernel 的硬形状校验写在 dit_block_ffi.cc:205-225（错即 TVM_FFI_THROW，不静默算错）。"])

y += 22
s.strip(X0, y, CW, 96, [
    "三条红线（任一违反都不报错，只是慢或算错）：",
    "① 时序：install() 必须早于 from_pretrained()——晚一步类被换了、实例没换，静默走原生路径。",
    "② dtype：主链 fp16 不是审美选择——bf16 下设备端 split0↔split1 互转不可用（evo_lpu.py:12-15），bf16 只在最外边界与模型对齐。",
    "③ 布局：单 Die 必须用 INIT_SINGLE_DIE 初始化，否则 plain 布局直接出垃圾（docs/records/gr00t_npu_precision_localization.md §20）。",
], "#fdecea", "#c0392b", "#8e2b21", 12.5, False)

for ln in []:
    pass

y += 96
H = y + 28

out = os.path.join(os.path.dirname(__file__), "..", "images", "ch09", "npu_three_stage.svg")
with open(out, "w", encoding="utf-8") as f:
    f.write(s.svg())
print("written:", os.path.normpath(out))
print("overflow warnings:", len(WARN))
for w in WARN:
    print("  ", w)
