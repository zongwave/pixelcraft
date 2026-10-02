#!/usr/bin/env python3
"""分相账本 + 有效算力阶梯：把一份 chrome trace 拆成"每个相忙多久 / 每发 kernel 值多少 TFLOPS"。

用法：python3 tools/trace_phase_ledger.py [chrome_trace.json]
默认 = logs/trace_current_clean/n15_sd3_chrome_trace.json（N1.5 单轮热轮、三相机、E100 单 Die）。

两个口径必须分清（本章 §10.5 的"头号货币是发射"就靠这两个量的差说话）：
  busy  = 该相 kernel 时长之和（设备真在算，profiler 顶层能看见）
  span  = 该相第一个 kernel 起点 → 最后一个 kernel 终点（墙钟，含 host 气泡）
  bubble = 1 − busy/span —— 这部分**不在任何 kernel 里**，只能在墙钟里找到。

TFLOPS 的 FLOPs 由**权重形状 × 实测序列长度**算出（形状出处：09 章 §7 第 4 行、06 章 §6.8.6
维度账本、GR00T-N1.5-3B/config.json 与 safetensors 头），µs 全部取自 trace。
所以它是"有效算力"，不是厂商标称峰值；同一份脚本换 trace 即可复跑。
"""
import collections
import json
import os
import re
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAND = [os.path.join(ROOT, "logs", "trace_current_clean", "n15_sd3_chrome_trace.json"),
        "/home/ft/wzong/workspace/embodied/logs/trace_current_clean/n15_sd3_chrome_trace.json"]

# ---------------- 形状常量（全部有出处，见文件头） ----------------
S_CAM, N_TOK, D_VIS, I_VIS, HV, DHV = 3, 256, 1152, 4304, 16, 72     # SigLIP 27L：vision_config
S_LLM, D_LLM, I_LLM, HQ, DHQ, HKV = 296, 2048, 6144, 16, 128, 8      # Qwen3：backbone 296 token
S_VL, D_VL, I_VL, HVV, DHV2 = 296, 2048, 8192, 32, 64                # vl_self_attention 4L
M_D, D_D, I_D, HD, DHD = 49, 1536, 6144, 32, 48                      # DiT 16 块（1+32+16=49）
D_BC = 2048                                                           # backbone 输出维 = cross 的 K/V 输入维


def mm(m, k, n):
    return 2.0 * m * k * n


def ah(q, kv, heads, hd, causal=False):
    return 2.0 * heads * q * kv * hd * 2.0 / (2.0 if causal else 1.0)


# (相, kernel 名) → (人类标签, 每发 FLOPs)
def flops_table():
    vis_g = mm(S_CAM * N_TOK, D_VIS, D_VIS)
    return {
        ("SigLIP", "gemm_bias_run_die"): ("SigLIP q/k/v/out_proj", vis_g),
        ("SigLIP", "unified_mha_run_die_batch_emb"): ("SigLIP attention（3 相机合批）",
                                                      S_CAM * ah(N_TOK, N_TOK, HV, DHV)),
        ("SigLIP", "mlp_gelu_run_die"): ("SigLIP MLP fc1+fc2（gelu-tanh）",
                                         2 * mm(S_CAM * N_TOK, D_VIS, I_VIS)),
        ("Qwen3", "gemm_norm_rope_run_die_single"): ("Qwen3 QKV+RMSNorm+RoPE 一合",
                                                     mm(S_LLM, D_LLM, D_LLM)
                                                     + 2 * mm(S_LLM, D_LLM, D_LLM // 2)),
        ("Qwen3", "qwen3_attention_run_die"): ("Qwen3 attention（causal + GQA 16Q/8KV）",
                                               ah(S_LLM, S_LLM, HQ, DHQ, True)),
        ("Qwen3", "gemm_bias_run_die"): ("Qwen3 o_proj", mm(S_LLM, D_LLM, D_LLM)),
        ("Qwen3", "mlp_swiglu_run_die_single_v1_fast"): ("Qwen3 MLP swiglu（gate/up/down）",
                                                         3 * mm(S_LLM, D_LLM, I_LLM)),
        ("vl", "linear_qkv_run_fused"): ("vl 塔 QKV 一合（2048→3×2048）", mm(S_VL, D_VL, 3 * D_VL)),
        ("vl", "fused_mha_out_run_die_single_vraw_qraw"): ("vl 塔 attention + to_out",
                                                           ah(S_VL, S_VL, HVV, DHV2)
                                                           + mm(S_VL, D_VL, D_VL)),
        ("vl", "mlp_gelu_run_die"): ("vl 塔 FF（2048→8192→2048）", 2 * mm(S_VL, D_VL, I_VL)),
        ("DiT", "adaln_qkv_cross_miss"): ("DiT cross QKV（KV MISS：q + 296×K + 296×V）",
                                          mm(M_D, D_D, D_D) + 2 * mm(S_LLM, D_BC, D_D)),
        ("DiT", "adaln_qkv_cross_hit"): ("DiT cross QKV（KV HIT：只剩 q）", mm(M_D, D_D, D_D)),
        ("DiT", "adaln_qkv_self"): ("DiT self QKV（adaLN 调制后 ×3）", 3 * mm(M_D, D_D, D_D)),
        ("DiT", "mha_cross"): ("DiT cross attention + to_out",
                               ah(M_D, S_LLM, HD, DHD) + mm(M_D, D_D, D_D)),
        ("DiT", "mha_self"): ("DiT self attention + to_out",
                              ah(M_D, M_D, HD, DHD) + mm(M_D, D_D, D_D)),
        ("DiT", "mlp_gelu_norm_run_die"): ("DiT FFN（1536→6144→1536，LN 融合）",
                                           2 * mm(M_D, D_D, I_D)),
    }


def load(path):
    d = json.load(open(path))
    ev = [e for e in d["traceEvents"] if e.get("ph") == "X" and e.get("cat") == "kernel"]
    ev.sort(key=lambda e: e["ts"])
    for e in ev:
        e["k"] = re.sub(r"\(.*", "", e["name"])
    return d, ev


def phases(ev):
    idx = {}
    for i, e in enumerate(ev):
        idx.setdefault(e["k"], []).append(i)
    i_conv = idx["conv2d_patch_embed_run_die"][0]
    i_q3 = idx["qwen3_attention_run_die"][0]
    i_vl = idx["linear_qkv_run_fused"][0]
    ad = idx["adaln_qkv_run_die"]
    groups = [ad[i:i + 16] for i in range(0, len(ad), 16)]
    ph = [("SigLIP", "视觉塔 27 层", i_conv, i_q3),
          ("Qwen3", "语言塔 12 层（select_layer=12 截断）", i_q3, i_vl),
          ("vl", "vl_self_attention 4 层 + head 前置", i_vl, groups[0][0])]
    for gi, g in enumerate(groups):
        end = groups[gi + 1][0] if gi + 1 < len(groups) else len(ev)
        ph.append(("DiT", "去噪步 %d（16 块）" % (gi + 1), g[0], end))
    return ph, groups


def dit_slot(ev, i, groups):
    """DiT 相内：把 adaln_qkv / fused_mha_out 按 self|cross×(MISS|HIT) 分桶。"""
    e = ev[i]
    if e["k"] == "adaln_qkv_run_die":
        # 三簇（09 章 §7 第 1 行的 ON/OFF 对账同一个形状）：
        #   466 µs = cross 且 KV MISS（要算 296 行的 K/V）；185 µs = cross 且 KV HIT（只剩 q）；
        #   250 µs = self（49 行 ×3 投影，与 KV 缓存无关，四个步都一样）
        if e["dur"] > 400:
            return "adaln_qkv_cross_miss"
        return "adaln_qkv_self" if e["dur"] > 195 else "adaln_qkv_cross_hit"
    if e["k"] == "fused_mha_out_run_die_single_vraw_qraw":
        return "mha_cross" if e["dur"] > 68 else "mha_self"
    return e["k"]


def main(path):
    d, ev = load(path)
    ph, groups = phases(ev)
    ft = flops_table()
    print("== ① 分相账本 ==\n%-5s %-34s %6s %9s %9s %8s" % ("相", "说明", "kernel", "busy/ms", "span/ms", "bubble"))
    tb = tsp = tf = 0.0
    per = {}
    for tag, desc, a, b in ph:
        seg = ev[a:b]
        busy = sum(x["dur"] for x in seg)
        span = seg[-1]["ts"] + seg[-1]["dur"] - seg[0]["ts"]
        fl = 0.0
        for e in seg:
            key = (dit_slot(ev, a + seg.index(e), groups) if tag == "DiT" else e["k"])
            ent = ft.get((tag, key))
            if ent:
                fl += ent[1]
        tb += busy
        tsp += span
        tf += fl
        per[tag] = dict(n=len(seg), busy=busy, span=span, fl=fl)
        print("%-5s %-34s %6d %9.2f %9.2f %7.1f%%" % (tag, desc, len(seg), busy / 1e3, span / 1e3,
                                                      100 * (1 - busy / span)))
    print("%-5s %-34s %6d %9.2f %9.2f %7.1f%%" % ("合计", "相内（相外空隙另计）", len(ev), tb / 1e3, tsp / 1e3,
                                                  100 * (1 - tb / tsp)))
    print("\n== ② 相外空隙（host 侧，不属于任何 kernel）==")
    tg = 0.0
    for (t1, d1, a1, b1), (t2, d2, a2, b2) in zip(ph, ph[1:]):
        g = ev[a2]["ts"] - (ev[b1 - 1]["ts"] + ev[b1 - 1]["dur"])
        tg += g
        print("   %-6s -> %-6s %8.2f ms" % (t1, t2, g / 1e3))
    print("   合计 %.2f ms；整轮 ≈ busy %.1f + 相内气泡 %.1f + 相外 %.1f = %.1f ms"
          % (tg / 1e3, tb / 1e3, (tsp - tb) / 1e3, tg / 1e3, (tsp + tg) / 1e3))
    print("\n== ③ 有效算力阶梯（FLOPs 形状算 / µs trace 测）==")
    rows = []
    for (tag, key), (lab, fl) in ft.items():
        if tag != "DiT":
            dur = [e["dur"] for e in ev if e["k"] == key]
            # 同名核跨相出现（gemm_bias/mlp_gelu/fused_mha_out）：按相切窗取中位
            a, b = [(x[2], x[3]) for x in ph if x[0] == tag][0]
            dur = [e["dur"] for e in ev[a:b] if e["k"] == key]
        else:
            a = min(x[2] for x in ph if x[0] == "DiT")
            dur = [e["dur"] for i, e in enumerate(ev[a:], start=a) if dit_slot(ev, i, groups) == key]
        if not dur:
            continue
        u = st.median(dur)
        rows.append((lab, len(dur), fl / 1e9, u, fl / (u * 1e-6) / 1e12))
    rows.sort(key=lambda r: -r[4])
    print("%-40s %5s %9s %8s %8s" % ("计算（每发）", "次数", "GFLOP/发", "µs/发", "TFLOPS"))
    for lab, n, gf, u, tfz in rows:
        print("%-40s %5d %9.2f %8.0f %8.1f" % (lab, n, gf, u, tfz))
    top = rows[0][4]
    print("\n阶梯顶 %.1f TFLOPS（%s）÷ 阶梯底 %.1f TFLOPS（%s）= **%.0f 倍**"
          % (top, rows[0][1], rows[-1][4], rows[-1][0], top / rows[-1][4]))
    print("全轮有用算力 %.0f GFLOP ⇒ 纯算力下限 %.1f ms（按阶梯顶）；实测 busy %.1f ms ⇒ "
          "busy 里只有 %.0f%% 不是空转，整轮（含气泡）有效算力 %.1f TFLOPS = 阶梯顶的 %.0f%%"
          % (tf / 1e9, tf / top / 1e9, tb / 1e3, 100 * (tf / top / 1e9) / (tb / 1e3),
             tf / ((tsp + tg) * 1e-6) / 1e12, 100 * (tf / ((tsp + tg) * 1e-6) / 1e12) / top))
    cnt = collections.Counter(e["k"] for e in ev)
    lk = [e for e in d["traceEvents"] if e.get("ph") == "X" and "LaunchKernel" in e.get("name", "")]
    print("\n== ④ 碎片度 ==\nkernel 总数 %d、种类 %d、中位 %.0f µs、<50µs %d 发、<20µs %d 发；"
          "evLaunchKernel %d 次 / %.1f ms（均 %.1f µs，占整轮墙钟 %.0f%%）"
          % (len(ev), len(cnt), st.median([e["dur"] for e in ev]),
             sum(1 for e in ev if e["dur"] < 50), sum(1 for e in ev if e["dur"] < 20),
             len(lk), sum(e["dur"] for e in lk) / 1e3, st.mean([e["dur"] for e in lk]),
             100 * sum(e["dur"] for e in lk) / (tsp + tg)))
    print("\n== ⑤ KV 缓存 A/B 的时域形状（adaln_qkv 双峰）==")
    aq = [e["dur"] for e in ev if e["k"] == "adaln_qkv_run_die"]
    print("   全 64 发取值簇：", sorted({round(x / 5) * 5 for x in aq}))
    print("   step1 均值 %.0f µs（8 发 cross = MISS 466）；step2-4 均值 %.0f µs（cross 已 HIT 185）"
          % (st.mean(aq[:16]), st.mean(aq[16:])))
    print("   ⇒ 8 发 ×(466−185) = %.1f ms/步、三份 = %.1f ms 已被 KV 缓存吃掉" %
          (8 * (466 - 185) / 1e3, 3 * 8 * (466 - 185) / 1e3))


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else next((c for c in CAND if os.path.exists(c)), CAND[0])
    print("trace:", p, "\n")
    main(p)
