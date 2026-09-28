#!/usr/bin/env python
"""N1.5 端到端推理性能基准（L20 实测）：
- Gr00tPolicy 加载耗时
- get_action 端到端延迟分布（mean/median/p50/p90/p95/min/max/std）
- 内部分解：数据变换 / backbone(Eagle) / action head(DiT flow-matching)
- 有效动作频率（action_horizon=16 摊薄）
用法（Isaac-GR00T 目录下）：
  CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 python <this> [--iters 20] [--denoising-steps 4] [--sweep 4,8]
"""
import argparse, copy, json, os, statistics, time

import numpy as np
import torch

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy

MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/"
    "869830fc749c35f34771aa5209f923ac57e4564e")

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--denoising-steps", type=int, default=4)
ap.add_argument("--sweep", type=str, default="", help="e.g. 4,8 对比去噪步数")
args = ap.parse_args()


def stats(ts):
    ts = list(ts)
    q = np.percentile(ts, [50, 90, 95])
    return dict(n=len(ts), mean=sum(ts)/len(ts), median=q[0], p50=q[0],
                p90=q[1], p95=q[2], min=min(ts), max=max(ts),
                std=statistics.pstdev(ts) if len(ts) > 1 else 0.0)


def run_once(policy, dp, iters, warmup, bundle=None):
    """bundle: dict of CUDA-event lists for module breakdown (already patched)."""
    e2e = []
    for i in range(iters + warmup):
        t0 = time.perf_counter()
        out = policy.get_action(copy.deepcopy(dp))
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if i >= warmup:
            e2e.append(dt)
    return e2e, out


def bench(denoising_steps):
    data_config = load_data_config("fourier_gr1_arms_only")
    modality = data_config.modality_config()

    t0 = time.perf_counter()
    policy = Gr00tPolicy(model_path=MODEL, modality_config=modality,
                         modality_transform=data_config.transform(),
                         embodiment_tag="gr1",
                         denoising_steps=denoising_steps, device="cuda")
    load_s = time.perf_counter() - t0

    # ---- CUDA-event 分解：backbone / action head / 数据变换 ----
    model = policy.model
    seg = {k: [] for k in ("backbone", "action_head", "transform")}
    ev = {k: [] for k in ("backbone", "action_head", "transform")}
    bb_fwd, ah_fwd = model.backbone.forward, model.action_head.get_action
    def wrap(key, fn):
        def w(*a, **kw):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record(); r = fn(*a, **kw); e1.record()
            ev[key].append((e0, e1))
            return r
        return w
    model.backbone.forward = wrap("backbone", bb_fwd)
    model.action_head.get_action = wrap("action_head", ah_fwd)

    dataset = LeRobotSingleDataset(
        dataset_path="demo_data/robot_sim.PickNPlace",
        modality_configs=modality, video_backend="decord",
        video_backend_kwargs=None, transforms=None, embodiment_tag="gr1")
    raw = dataset.get_step_data(0, 0)
    dp = {}   # 拆组: {"video": {"ego_view": x}} -> {"video.ego_view": x}
    for k, v in raw.items():
        if isinstance(v, dict):
            for sub, val in v.items():
                dp[f"{k}.{sub}"] = val
        else:
            dp[k] = v

    t0 = time.perf_counter()
    _ = policy.get_action(copy.deepcopy(dp))  # warmup 1: 触发 dataset->policy 输入路径
    torch.cuda.synchronize()
    data_s = time.perf_counter() - t0

    # 数据变换（归一化/resize/tokenize）单独计时
    t0 = time.perf_counter()
    _ = policy.apply_transforms(copy.deepcopy(dp)); torch.cuda.synchronize()
    seg["transform"].append(time.perf_counter() - t0)

    e2e, out = run_once(policy, dp, args.iters, args.warmup)
    # 收敛分段事件耗时（丢弃 warmup 段事件，只取最后 iters 次）
    for k in ("backbone", "action_head"):
        pairs = ev[k][-args.iters:] if len(ev[k]) >= args.iters else ev[k]
        seg[k] = [e0.elapsed_time(e1) / 1e3 for e0, e1 in pairs]

    horizon = out["action.left_arm"].shape[-2]
    res = dict(denoising_steps=denoising_steps, load_s=round(load_s, 2),
               e2e=stats(e2e),
               backbone=stats(seg["backbone"]) if seg["backbone"] else None,
               action_head=stats(seg["action_head"]) if seg["action_head"] else None,
               transform_s=round(seg["transform"][0], 4) if seg["transform"] else None,
               horizon=horizon,
               per_cmd_ms=round(1e3 * (sum(e2e)/len(e2e)) / horizon))
    print(json.dumps(res, indent=2, default=float))
    return res


steps = [int(s) for s in args.sweep.split(",")] if args.sweep else [args.denoising_steps]
results = [bench(s) for s in steps]
if len(results) > 1:
    d = results[0]["e2e"]["mean"] - results[1]["e2e"]["mean"] if len(results) == 2 else None
    print("sweep compare:", [(r["denoising_steps"], round(r["e2e"]["mean"], 3)) for r in results])
