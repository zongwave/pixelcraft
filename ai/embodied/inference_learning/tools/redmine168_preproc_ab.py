#!/usr/bin/env python
"""Redmine #168 闭环实验：同一帧只差一步图像几何，喂同一个 server，比五层指标。

背景：真机倒茶失败（N1.5）疑似"新 client 丢了 pad_and_resize、server 端非等比拉伸补尺寸"，
      但两段录像同属新 client，**不构成对照实验**。本脚本用官方 demo 数据离线复现该机理：
      只改变"从 16:10 原生帧构造模型输入的那一步"，其余（权重/状态/指令/随机种子）全不变。

四个变体（A 是训练契约，B 是事故做法，C 是"没补尺寸"的原始宽幅）：
    native     : 从官方 256x256 训练帧裁出内容区（256x160，宽高比 1.6），当作"相机原生帧"
    A_pad      : native -> 等比 + 居中补黑边 -> 256x256   （= 训练时 pad_and_resize）
    B_stretch  : native -> 非等比拉伸       -> 256x256   （= 事故做法）
    C_raw      : native 原样 256x160                     （应被 check_input 拒收）
    A2_official: 官方训练帧原样（参照，用来证明 A 就是训练契约）

五层指标：
  1 形状闸门  VideoToTensor.check_input 放行 / 拒收 + 拒收原文
  2 形状层    Eagle 真实吃进去的 pixel_values / image_sizes / 图像 token 数 / seq_len
  3 特征层    backbone 输出特征余弦（A vs B）
  4 动作层    对**数据集真值动作**的 MSE（分维：臂 vs 手）+ A-B 差异 + 多随机种子噪声地板
  5 时序层    跨 chunk 重叠段分歧度（真机"悬停 + 横扫 + 末态失稳"的量化代理）
  + 形变系数扫描：1.0(=A) .. 1.6(=B) 的单调性，比单点 A/B 更能说明机理

用法（gr00t 环境，需 GPU；本机 huggingface.co 不可达，走本地缓存）：
    conda activate gr00t
    MP=$(ls -d ~/.cache/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/*/ | head -1)
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=3 \
      python tools/redmine168_preproc_ab.py --model-path "$MP"
"""

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch

REPO = "/home/ft/wzong/workspace/embodied/gr00t/Isaac-GR00T"
CLIENT_UTILS = "/home/ft/wzong/workspace/embodied/gr00t/tao/vla_infer"


# ----------------------------------------------------------------------------- 预处理
def pad_and_resize(img, target_h, target_w, pad_color=(0, 0, 0)):
    """训练/老 client 的做法：等比 + 居中补黑边（tao/vla_infer/client/utils/misc.py 同款）。"""
    import cv2

    h, w = img.shape[:2]
    tr, cr = target_w / target_h, w / h
    if cr > tr:  # 宽幅 -> 上下补
        nh = int(w / tr)
        d = nh - h
        top, bot = d // 2, d - d // 2
        padded = cv2.copyMakeBorder(img, top, bot, 0, 0, cv2.BORDER_CONSTANT, value=pad_color)
    else:  # 高幅 -> 左右补
        nw = int(h * tr)
        d = nw - w
        left, right = d // 2, d - d // 2
        padded = cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_CONSTANT, value=pad_color)
    return cv2.resize(padded, (target_w, target_h), interpolation=cv2.INTER_AREA)


def stretch(img, target_h, target_w):
    """事故做法：直接 resize 到目标尺寸（非等比，形变系数 != 1）。"""
    import cv2

    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)


def distort(img, target_h, target_w, k):
    """形变系数扫描：k=1.0 等比补边(A 契约)，k=native_ar(=1.6) 等价于非等比拉伸(B 事故)。

    实现：先把内容横向压缩 1/k，再按 A 的方式补边 + 缩放到目标尺寸。
    实测等价性（episode_000000 首帧）：k=1.0 与 A 逐像素 MSE=0.0；
    k=1.6 与 B_stretch 的逐像素 MSE=78（只差重采样顺序），而 B 与 A 相差 13546
    ⇒ 扫描的两个端点确实就是 A 和 B，中间点是"部分拉伸"。
    """
    import cv2

    h, w = img.shape[:2]
    if abs(k - 1.0) < 1e-6:
        return pad_and_resize(img, target_h, target_w)
    # 内容先横向缩到 w/k，再按 A 的方式补边 -> 画面被横向压扁 k 倍
    small = cv2.resize(img, (int(round(w / k)), h), interpolation=cv2.INTER_AREA)
    return pad_and_resize(small, target_h, target_w)


def row_profile(img):
    """逐行平均亮度，兼容 (H,W,C) 与 (T,H,W,C)。"""
    a = np.asarray(img).astype(np.float32)
    return a.mean(axis=tuple(range(1, a.ndim))) if a.ndim in (3, 4) else a


def black_band_pct(img, thr=2.0):
    row = row_profile(img)
    h = np.asarray(img).shape[-3]
    nz = np.where(row > thr)[0]
    if len(nz) == 0:
        return 100.0, 100.0, 0, h - 1
    return 100 * nz[0] / h, 100 * (h - 1 - nz[-1]) / h, int(nz[0]), int(h - 1 - nz[-1])


def mean_pairwise_mse(arrays):
    """同一变体、多个随机种子 -> 所有配对 MSE 的均值（噪声地板）。"""
    n = len(arrays)
    if n < 2:
        return float("nan")
    acc, cnt = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            acc += float(((arrays[i] - arrays[j]) ** 2).mean())
            cnt += 1
    return acc / cnt


# ----------------------------------------------------------------------------- 模型侧探针
class Probe:
    """挂在 Eagle 视觉入口(mlp1)、backbone 出口上，抓真实图像 token 数与特征。"""

    def __init__(self, policy):
        self.policy = policy
        self.vision_tokens = None
        self.vision_in_rows = None
        self.features = None
        self._handles = []

    def __enter__(self):
        eagle = self.policy.model.backbone.eagle_model

        def vis_pre(mod, args):
            try:
                self.vision_in_rows = int(args[0].shape[0])  # patch 行数
            except Exception:
                pass
            return None

        def vis_post(mod, args, out):
            try:
                self.vision_tokens = int(np.prod(out.shape[:-1]))  # 送进 LLM 的图像 token 数
            except Exception:
                pass
            return None

        try:
            self._handles.append(eagle.mlp1.register_forward_pre_hook(vis_pre))
            self._handles.append(eagle.mlp1.register_forward_hook(vis_post))
        except Exception:
            pass

        def post(mod, args, out):
            try:
                f = out["backbone_features"]
                m = out.get("backbone_attention_mask", None)
                f = f.detach().float()
                if m is not None:
                    m = m.detach().to(f.device).clamp(min=0).unsqueeze(-1).to(f.dtype)
                    f = (f * m).sum(1) / m.sum(1).clamp(min=1)
                else:
                    f = f.mean(1)
                self.features = f.cpu().numpy()[0]
            except Exception:
                pass
            return None

        self._handles.append(self.policy.model.backbone.register_forward_hook(post))
        return self

    def __exit__(self, *a):
        for h in self._handles:
            h.remove()
        self._handles = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="nvidia/GR00T-N1.5-3B")
    ap.add_argument("--dataset-path", default="demo_data/robot_sim.PickNPlace")
    ap.add_argument("--data-config", default="fourier_gr1_arms_only")
    ap.add_argument("--embodiment-tag", default="gr1")
    ap.add_argument("--video-key", default="video.ego_view")
    ap.add_argument("--denoising-steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5, help="噪声地板用的种子数")
    ap.add_argument("--gt-frames", dest="gt_frames", type=int, default=3,
                    help="配对真值误差用的帧数（在整集上均匀取）")
    ap.add_argument("--gap", type=int, default=2, help="跨 chunk 分歧度用的帧间隔")
    ap.add_argument("--sweep", default="1.0,1.15,1.3,1.45,1.6", help="形变系数扫描")
    ap.add_argument("--sweep-seeds", type=int, default=3)
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--out", default="/tmp/preproc_ab.json")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)  # 后台跑时能实时看日志
    except Exception:
        pass

    os.chdir(REPO)
    sys.path.insert(0, REPO)
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.experiment.data_config import load_data_config
    from gr00t.model.policy import Gr00tPolicy, unsqueeze_dict_values
    from gr00t.model.backbone.eagle_backbone import DEFAULT_EAGLE_PATH
    from gr00t.model.transforms import build_eagle_processor

    data_config = load_data_config(args.data_config)
    policy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag=EmbodimentTag(args.embodiment_tag),
        denoising_steps=args.denoising_steps,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    processor = build_eagle_processor(DEFAULT_EAGLE_PATH)
    ds = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=policy.get_modality_config(),
        video_backend="decord",
        video_backend_kwargs=None,
        transforms=None,
        embodiment_tag=EmbodimentTag(args.embodiment_tag),
    )

    vkeys = list(ds.metadata.modalities.video)
    vkey_raw = args.video_key.replace("video.", "")
    vkey_raw = vkey_raw if vkey_raw in vkeys else vkeys[0]
    vm = ds.metadata.modalities.video[vkey_raw]
    W, H = int(vm.resolution[0]), int(vm.resolution[1])
    vkey = "video." + vkey_raw

    def base_obs(idx, with_action=False):
        o = ds.get_step_data(0, idx)
        out = {}
        for k, v in o.items():
            if not with_action and k.startswith("action."):
                continue
            out[k] = v.copy() if isinstance(v, np.ndarray) else v
        return out

    def frame(idx):
        v = np.asarray(base_obs(idx)[vkey])
        return v.reshape(-1, *v.shape[-3:])[0]

    def gt_action(idx):
        o = base_obs(idx, with_action=True)
        keys = sorted(k for k in o if k.startswith("action."))
        return keys, np.concatenate(
            [np.asarray(o[k]).reshape(np.asarray(o[k]).shape[0], -1) for k in keys], axis=-1
        )

    f0 = frame(0)
    top_b, bot_b, r0, r1 = black_band_pct(f0)
    rows = np.where(row_profile(f0) > 2.0)[0]
    native_src = f0[rows[0] : rows[-1] + 1]  # 内容区 = "相机原生帧"
    native_ar = native_src.shape[1] / native_src.shape[0]
    print(f"[frame] 官方训练帧 {f0.shape[1]}x{f0.shape[0]} 黑边 top/bot {top_b:.2f}%/{bot_b:.2f}%"
          f" -> 内容区(原生) {native_src.shape[1]}x{native_src.shape[0]} ar={native_ar:.3f}")
    print(f"[contract] checkpoint 期望分辨率 (W,H) = ({W},{H})  数据配置={args.data_config}"
          f"  动作 keys={gt_action(0)[0]}")

    variants = {
        "A_pad": pad_and_resize(native_src, H, W),
        "B_stretch": stretch(native_src, H, W),
        "C_raw": native_src,
        "A2_official": f0,
    }
    mse_ap_a2 = float(((variants["A_pad"].astype(np.float32) - f0.astype(np.float32)) ** 2).mean())
    print(f"[sanity] A_pad vs 官方原帧 像素 MSE = {mse_ap_a2:.3f}"
          f"  -> {'A 确实复现了训练契约' if mse_ap_a2 < 200 else 'A 与训练帧差距偏大，结论需谨慎'}")

    def infer(idx, img, seed):
        """把 obs 的图像换成 img，其余（状态/指令/权重/种子）不变，跑一次推理。"""
        obs = base_obs(idx)
        T = np.asarray(obs[vkey]).shape[0]
        obs[vkey] = np.broadcast_to(img[None], (T, img.shape[0], img.shape[1], 3)).astype(np.uint8)

        # ---- 形状层：走一遍变换图 + Eagle 处理器，看模型真正吃进去什么
        prep = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
        norm = policy.apply_transforms(unsqueeze_dict_values(prep))
        # 本仓库版本的变换图直接产出 eagle_* 张量（Eagle 处理器已在其中被调用）
        shape = {"keys": sorted(norm.keys())}

        def tget(k):
            v = norm.get("eagle_" + k, None)
            if v is None:
                ec = norm.get("eagle_content", None)
                if ec is not None and processor is not None:
                    ei = processor(text=ec["text_list"], images=ec["image_inputs"],
                                   return_tensors="pt", padding=True)
                    v = ei.get(k, None)
            return v

        pv, isz, ids, am = tget("pixel_values"), tget("image_sizes"), tget("input_ids"), tget("attention_mask")
        def as_list(v):
            if v is None:
                return None
            if hasattr(v, "tolist"):
                return v.tolist()
            if isinstance(v, (list, tuple)):
                return [list(map(int, x)) if isinstance(x, (list, tuple)) else int(x) for x in v]
            return None

        shape.update({
            "pixel_values_shape": (list(pv.shape) if hasattr(pv, "shape") else None),
            "image_sizes": as_list(isz),
            "seq_len": (int(ids.shape[-1]) if hasattr(ids, "shape") else None),
            "attn_len": (int(am.shape[-1]) if hasattr(am, "shape") else None),
        })

        torch.manual_seed(seed)
        np.random.seed(seed)
        p = Probe(policy)
        with p:
            act = policy.get_action(obs)
        keys = sorted(k for k, v in act.items())
        cat = np.concatenate([np.asarray(act[k]).reshape(np.asarray(act[k]).shape[0], -1)
                             for k in keys], axis=-1)
        shape.update({"patch_rows": p.vision_in_rows, "vision_tokens": p.vision_tokens})
        return {"shape": shape, "feats": copy.deepcopy(p.features),
                "action": cat, "keys": keys, "obs": obs}

    def clean_shape(s):
        return {k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in s.items()}

    def dump():
        """阶段性落盘，防止进程被杀时什么都留不下。"""
        snapshot = {}
        for k, v in results.items():
            if isinstance(v, dict):
                snapshot[k] = {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
            else:
                snapshot[k] = v
        snapshot["_summary"] = summary_holder.get("d", {})
        json.dump(snapshot, open(args.out, "w"), ensure_ascii=False, indent=1,
                  default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else None)

    # ---------------------------------------------------------------- 1~3 层：四个变体
    summary_holder = {"d": {}}
    results = {"_meta": {
        "model_path": args.model_path, "data_config": args.data_config,
        "dataset": args.dataset_path, "video_key": vkey,
        "expected_resolution_WxH": [W, H],
        "official_frame_black_band_pct": [top_b, bot_b],
        "native_content_WxH": [int(native_src.shape[1]), int(native_src.shape[0])],
        "native_content_ar": native_ar,
        "A_pad_vs_official_pixel_mse": mse_ap_a2,
        "denoising_steps": args.denoising_steps,
        "action_keys": gt_action(0)[0],
        "note": "本机 huggingface.co 不可达，权重为本地缓存；base 模型 + PickNPlace 演示数据，"
                "并非事故现场的倒茶微调权重",
    }}
    print("\n%-12s %-8s %s" % ("variant", "闸门", "形状层"))
    for name, img in variants.items():
        try:
            r = infer(0, img, args.seed)
            tb, bb, _, _ = black_band_pct(img)
            r["black_band_pct"] = [tb, bb]
            r["input_shape"] = list(img.shape)
            results[name] = {"shape": clean_shape(r["shape"]), "black_band_pct": [tb, bb],
                             "input_shape": list(img.shape)}
            print("%-12s %-8s %s" % (name, "PASS", json.dumps(clean_shape(r["shape"]),
                                                                ensure_ascii=False)))
            results[name]["_feats"] = r["feats"]
            results[name]["_action"] = r["action"]
        except (AssertionError, ValueError) as e:
            msg = str(e).replace("\n", " ")
            print("%-12s %-8s %s" % (name, "REJECT", msg[:200]))
            results[name] = {"rejected": msg[:600]}

    dump()
    # ---------------------------------------------------------------- 4 层：真值动作 + 噪声地板
    summary = {}
    summary_holder["d"] = summary
    if "A_pad" in results and "B_stretch" in results:
        A, B = results["A_pad"], results["B_stretch"]
        ca = A["_feats"] / np.linalg.norm(A["_feats"])
        cb = B["_feats"] / np.linalg.norm(B["_feats"])
        cos = float((ca * cb).sum())
        mse = float(((A["_action"] - B["_action"]) ** 2).mean())
        n5 = min(5, A["_action"].shape[0])
        d1, d2 = A["_action"][:n5].reshape(-1), B["_action"][:n5].reshape(-1)
        ang = float(np.degrees(np.arccos(np.clip(
            d1 @ d2 / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-12), -1, 1))))
        print("\n[形状层] A/B 图像 token 数 = %s vs %s  seq_len = %s vs %s  -> 是否完全相同: %s"
              % (A["shape"]["vision_tokens"], B["shape"]["vision_tokens"],
                 A["shape"]["seq_len"], B["shape"]["seq_len"],
                 A["shape"]["vision_tokens"] == B["shape"]["vision_tokens"]))
        print("[特征层] backbone 特征余弦 A vs B = %.4f  -> 特征几乎看不出差别" % cos)
        print("[动作层] A vs B action chunk MSE = %.6f | 前 %d 步方向夹角 = %.1f°" % (mse, n5, ang))
        summary.update({"cos_A_B": cos, "mse_A_B": mse, "angle_deg": ang,
                       "tokens_A": A["shape"]["vision_tokens"],
                       "tokens_B": B["shape"]["vision_tokens"],
                       "seq_len_A": A["shape"]["seq_len"], "seq_len_B": B["shape"]["seq_len"]})

        # 多随机种子：每个变体跑 seeds 次 -> 噪声地板 + 对真值动作的误差
        print("\n[噪声地板] 每变体 %d 个随机种子，配对 MSE 均值（flow-matching 本身是随机的）"
              % args.seeds)
        seeds = [args.seed + 101 * i for i in range(args.seeds)]
        per_variant, gt_err = {}, {}
        for name in ("A_pad", "B_stretch", "A2_official"):
            arrs = []
            for s in seeds:
                if s == args.seed:
                    arrs.append(results[name]["_action"])
                else:
                    arrs.append(infer(0, variants[name], s)["action"])
            floor = mean_pairwise_mse(arrs)
            per_variant[name] = {"seed_mse": floor, "n_seeds": args.seeds}
            print("  %-12s 噪声地板 MSE = %.6f (std of seed-mean %.6f)"
                  % (name, floor, float(np.std([float((a ** 2).mean()) for a in arrs]))))
            gkeys, gtv = gt_action(0)
            errs = [float(((a - gtv) ** 2).mean()) for a in arrs]
            gt_err[name] = {"mean": float(np.mean(errs)), "std": float(np.std(errs))}
            print("  %-12s 对真值动作 MSE = %.6f ± %.6f" % (name, np.mean(errs), np.std(errs)))
        dump()
        summary["noise_floor"] = per_variant
        summary["mse_to_ground_truth"] = gt_err

        gkeys, gtv = gt_action(0)
        # 分维：按 action key 拆开（臂 vs 手），逐维看几何影响落在哪
        print("\n[分维] 各动作分组的 MSE(A vs B) / 相对噪声地板 / 对真值误差(A|B)")
        # 拼接顺序与 infer 里 sorted(keys) 一致，维度宽度从数据集取
        base = infer(0, variants["A_pad"], args.seed)  # keys 已在 base["keys"]
        step0 = ds.get_step_data(0, 0)
        widths = {k: int(np.asarray(step0[k]).shape[-1]) for k in base["keys"]}
        dims = {}
        off = 0
        floors = per_variant["A_pad"]["seed_mse"]
        for k in base["keys"]:
            w = widths[k]
            a_sl = results["A_pad"]["_action"][:, off : off + w]
            b_sl = results["B_stretch"]["_action"][:, off : off + w]
            g_sl = gtv[:, off : off + w]
            m_ab = float(((a_sl - b_sl) ** 2).mean())
            m_ag = float(((a_sl - g_sl) ** 2).mean())
            m_bg = float(((b_sl - g_sl) ** 2).mean())
            scale = float((g_sl ** 2).mean()) + 1e-12
            print("  %-22s A-B MSE %.6f | 相对噪声地板 %.2fx | 对真值 A %.6f / B %.6f"
                  " | 真值功率 %.4f | 更偏离真值: %s"
                  % (k, m_ab, m_ab / max(floors, 1e-12), m_ag, m_bg, scale,
                     "B" if m_bg > m_ag else "A"))
            dims[k] = {"mse_A_B": m_ab, "geom_over_noise": m_ab / max(floors, 1e-12),
                       "mse_A_gt": m_ag, "mse_B_gt": m_bg, "gt_power": scale}
            off += w
        dump()
        summary["per_key"] = dims

        # B 换种子一致性：证明 A/B 差异来自图像而不是运气
        B_b = infer(0, variants["B_stretch"], seeds[-1])
        print("\n[一致性] B 换种子 MSE = %.6f，与噪声地板同量级 -> A/B 差异不是运气"
              % float(((B["_action"] - B_b["action"]) ** 2).mean()))

        # ---- 配对统计：A/B 用同一批 (帧, 种子) 组合，配对差才不受采样噪声支配
        gt_seeds = seeds[: min(3, args.seeds)]
        n_f = max(1, args.gt_frames)
        # 只取轨迹 0 的长度（len(ds) 是全集步数，直接用它会让 get_step_data(0, i) 越界）
        n0 = max([i for (t, i) in ds.all_steps if t == 0] or [100]) + 1
        frames = [int(round(x)) for x in np.linspace(0, max(n0 - 17, 1), n_f)]
        print("\n[配对] A vs B 对真值误差：帧 %s × 种子 %d 组，逐对做差"
              % (frames, len(gt_seeds)))
        pa, pb = [], []
        for fi in frames:
            _, gtv_f = gt_action(fi)
            for s_ in gt_seeds:
                pa.append(float(((infer(fi, variants["A_pad"], s_)["action"] - gtv_f) ** 2).mean()))
                pb.append(float(((infer(fi, variants["B_stretch"], s_)["action"] - gtv_f) ** 2).mean()))
        d = np.asarray(pb) - np.asarray(pa)
        print("  A 平均对真值 MSE = %.6f | B 平均 = %.6f" % (np.mean(pa), np.mean(pb)))
        print("  配对差(B-A) = %+.6f ± %.6f | 相对 = %+.2f%% | B 更差的比例 = %d/%d"
              % (d.mean(), d.std(), 100 * d.mean() / (np.mean(pa) + 1e-12),
                 int((d > 0).sum()), len(d)))
        print("  配对差的 t 比 = %.2f（>=2 才算得住；<2 说明在这份 base 权重 + 这一集上"
              "几何效应小于配对可分辨下限）" % (abs(d.mean()) / (d.std() / np.sqrt(len(d)) + 1e-12)))
        summary["paired_gt"] = {"frames": frames, "n_pairs": len(d),
                                "mean_A": float(np.mean(pa)), "mean_B": float(np.mean(pb)),
                                "diff_mean": float(d.mean()), "diff_std": float(d.std()),
                                "diff_rel_pct": float(100 * d.mean() / (np.mean(pa) + 1e-12)),
                                "frac_B_worse": float((d > 0).mean()),
                                "t_ratio": float(d.mean() / (d.std() / np.sqrt(len(d)) + 1e-12))}
        dump()

        # ------------------------------------------------------------ 形变系数单调性扫描
        if not args.skip_sweep:
            ks = [float(x) for x in args.sweep.split(",")]
            print("\n[扫描] 形变系数 k（1.0=等比补边=A，越大越接近横向压扁） 每点 %d 种子，"
                  "帧 %s，与 k=1.0 逐对做差" % (args.sweep_seeds, frames[:1]))
            ref = pad_and_resize(native_src, H, W)
            gkeys, gtv = gt_action(0)
            sweep, base_err = [], None
            for k in ks:
                img = distort(native_src, H, W, k)
                errs = [float(((infer(0, img, args.seed + 101 * i)["action"] - gtv) ** 2).mean())
                        for i in range(args.sweep_seeds)]
                gt_m = float(np.mean(errs))
                pix = float(((img.astype(np.float32) - ref.astype(np.float32)) ** 2).mean())
                if base_err is None:
                    base_err = np.asarray(errs)
                    dd = np.zeros(len(errs))
                else:
                    dd = np.asarray(errs) - base_err
                print("  k=%.2f  对真值动作 MSE = %.6f | 与 A 的像素 MSE %8.1f | "
                      "配对差(k 减 1.0) = %+.6f ± %.6f | 更差 %d/%d"
                      % (k, gt_m, pix, dd.mean(), dd.std(), int((dd > 0).sum()), len(dd)))
                sweep.append({"k": k, "mse_to_gt": gt_m, "pixel_mse_vs_A": pix,
                              "paired_diff_vs_k1": float(dd.mean()),
                              "paired_diff_std": float(dd.std()),
                              "frac_worse": float((dd > 0).mean())})
                summary["distortion_sweep"] = sweep
                dump()
            if len(sweep) >= 2:
                ys = [x["mse_to_gt"] for x in sweep]
                ds_ = np.asarray([x["paired_diff_vs_k1"] for x in sweep])
                rho = float(np.corrcoef([x["k"] for x in sweep], ys)[0, 1])
                print("  -> 单调性: 首点 %.6f 末点 %.6f 末/首 = %.2f | k 与误差的相关系数 r = %.3f"
                      % (ys[0], ys[-1], ys[-1] / (ys[0] + 1e-12), rho))
                summary["sweep_last_over_first"] = ys[-1] / (ys[0] + 1e-12)
                summary["sweep_corr_k_vs_err"] = rho

        # ------------------------------------------------------------ 5 层：跨 chunk 分歧
        gap = args.gap
        print("\n[时序层] 跨 chunk 分歧度（帧 0 与帧 %d 两次推理，重叠段逐步比较）" % gap)
        for name in ("A_pad", "B_stretch"):
            r1 = infer(gap, variants[name], args.seed)
            a0, a1 = results[name]["_action"], r1["action"]
            ov0, ov1 = a0[gap:], a1[: a1.shape[0] - gap]
            n = min(ov0.shape[0], ov1.shape[0])
            if n <= 0:
                continue
            ov0, ov1 = ov0[:n], ov1[:n]
            step = float(np.linalg.norm(np.diff(a0, axis=0), axis=1).mean())
            disp = float(np.linalg.norm(ov0 - ov1, axis=1).mean())
            print("  %-10s 重叠段平均分歧 %.5f | 单步平均位移 %.5f | 分歧/位移 = %.2f"
                  "  <- 越大越【前后矛盾】" % (name, disp, step, disp / (step + 1e-12)))
            results[name]["cross_chunk"] = {"disp": disp, "step": step,
                                            "ratio": disp / (step + 1e-12)}
            summary.setdefault("cross_chunk", {})[name] = results[name]["cross_chunk"]

    results["_summary"] = summary
    dump()
    for v in results.values():
        if isinstance(v, dict):
            v.pop("_feats", None)
            v.pop("_action", None)
            v.pop("_dict", None)
    json.dump(results, open(args.out, "w"), ensure_ascii=False, indent=1,
              default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else None)
    print("\n结果已写入", args.out)


if __name__ == "__main__":
    main()
