#!/usr/bin/env python
"""同进程、同 obs、固定随机种子：PyTorch → (patch TRT) → TensorRT 数值一致性 + 双路径计时。"""
import copy, os, statistics, sys, time
import numpy as np
import torch

REPO = sys.argv.pop(1)
sys.path.insert(0, os.path.join(REPO, "deployment_scripts"))
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy
from trt_model_forward import setup_tensorrt_engines

MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/"
    "869830fc749c35f34771aa5209f923ac57e4564e")
SEED, ITERS, WARMUP = 42, 20, 3

cfg = load_data_config("fourier_gr1_arms_only")
policy = Gr00tPolicy(model_path=MODEL, modality_config=cfg.modality_config(),
                     modality_transform=cfg.transform(), embodiment_tag="gr1",
                     denoising_steps=4, device="cuda")
ds = LeRobotSingleDataset(dataset_path=os.path.join(REPO, "demo_data/robot_sim.PickNPlace"),
                          modality_configs=cfg.modality_config(), video_backend="decord",
                          video_backend_kwargs=None, transforms=None, embodiment_tag="gr1")
dp = ds.get_step_data(0, 0)

def timed(mode):
    # 先固定种子取一次“标本”（一致性用）
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    ref = policy.get_action(copy.deepcopy(dp))
    ts = []
    for i in range(ITERS + WARMUP):
        t0 = time.perf_counter()
        policy.get_action(copy.deepcopy(dp))
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if i >= WARMUP:
            ts.append(dt)
    q = np.percentile(ts, [50, 90])
    print("TIMING %s mean=%.4f p50=%.4f p90=%.4f min=%.4f" %
          (mode, sum(ts)/len(ts), q[0], q[1], min(ts)))
    return ref

ref_torch = timed("pytorch")
setup_tensorrt_engines(policy, os.path.join(REPO, "gr00t_engine"), "fp16", "fp16", "fp16")
_ = policy.get_action(copy.deepcopy(dp))          # TRT 预热
ref_trt = timed("tensorrt")

print("\n=== 一致性（同种子同初始噪声）===")
print(f"{'key':22s} {'max|Δ|':>10s} {'MSE':>12s} {'cosine':>9s}")
for k in ref_torch:
    a = torch.from_numpy(ref_torch[k]).float(); b = torch.from_numpy(ref_trt[k]).float()
    d = a - b
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    print(f"{k:22s} {d.abs().max().item():10.4f} {(d**2).mean().item():12.6f} {cos:9.5f}")
