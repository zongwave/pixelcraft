#!/usr/bin/env python
"""TRT vs PyTorch e2e 对比基准（同一 obs、同一 20+3 次协议）。
sys.path 需含 deployment_scripts（setup_tensorrt_engines 在那）。"""
import argparse, copy, json, os, statistics, sys, time
import numpy as np
import torch

REPO = sys.argv.pop(1)  # 位置参数：Isaac-GR00T 根目录
sys.path.insert(0, os.path.join(REPO, "deployment_scripts"))
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy
from trt_model_forward import setup_tensorrt_engines

MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/"
    "869830fc749c35f34771aa5209f923ac57e4564e")

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--denoising-steps", type=int, default=4)
ap.add_argument("--engine-path", default=os.path.join(REPO, "gr00t_engine"))
ap.add_argument("--dtypes", default="fp16,fp16,fp16", help="vit,llm,dit")
ap.add_argument("--mode", choices=["pytorch", "trt"], required=True)
args = ap.parse_args()

vit_d, llm_d, dit_d = args.dtypes.split(",")
cfg = load_data_config("fourier_gr1_arms_only")
t0 = time.perf_counter()
policy = Gr00tPolicy(model_path=MODEL, modality_config=cfg.modality_config(),
                     modality_transform=cfg.transform(), embodiment_tag="gr1",
                     denoising_steps=args.denoising_steps, device="cuda")
if args.mode == "trt":
    setup_tensorrt_engines(policy, args.engine_path, vit_d, llm_d, dit_d)
load_s = time.perf_counter() - t0

ds = LeRobotSingleDataset(dataset_path=os.path.join(REPO, "demo_data/robot_sim.PickNPlace"),
                          modality_configs=cfg.modality_config(), video_backend="decord",
                          video_backend_kwargs=None, transforms=None, embodiment_tag="gr1")
dp = ds.get_step_data(0, 0)

first = policy.get_action(copy.deepcopy(dp))   # 额外一次：数值一致性标本
ts = []
for i in range(args.iters + args.warmup):
    t0 = time.perf_counter()
    out = policy.get_action(copy.deepcopy(dp))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    if i >= args.warmup:
        ts.append(dt)
q = np.percentile(ts, [50, 90, 95])
keys = sorted(k for k in out if k.startswith("action."))
res = dict(mode=args.mode, load_s=round(load_s, 1), n=len(ts),
           mean=round(sum(ts)/len(ts), 4), p50=round(q[0], 4), p90=round(q[1], 4),
           p95=round(q[2], 4), min=round(min(ts), 4), max=round(max(ts), 4),
           std=round(statistics.pstdev(ts), 4),
           per_cmd_ms=round(1e3*sum(ts)/len(ts)/16, 1), actions=keys)
print("RESULT " + json.dumps(res))
np.savez(f"/tmp/n15_actions_{args.mode}.npz", **{k: out[k] for k in keys})
