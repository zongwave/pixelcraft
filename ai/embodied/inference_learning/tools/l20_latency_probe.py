#!/usr/bin/env python
"""第 13 章延迟探针：N1.5 单次 get_action 延迟（进程内 / ZMQ 两条路径）。

用法（在 Isaac-GR00T/ 目录、env gr00t 下）：
  CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 python tools/l20_latency_probe.py inprocess
  # 先另开终端起服务：python scripts/inference_service.py --server --model-path <快照>
  python tools/l20_latency_probe.py zmq
"""
import sys, time
import numpy as np, torch
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import load_data_config

MODEL = "/home/ft/.cache/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e"
mode = sys.argv[1]  # inprocess | zmq

data_config = load_data_config("fourier_gr1_arms_only")
modality = data_config.modality_config()

if mode == "inprocess":
    from gr00t.model.policy import Gr00tPolicy
    policy = Gr00tPolicy(model_path=MODEL, modality_config=modality,
                         modality_transform=data_config.transform(),
                         embodiment_tag="gr1", denoising_steps=4, device="cuda")
else:
    from gr00t.eval.robot import RobotInferenceClient
    policy = RobotInferenceClient(host="localhost", port=5555)
    modality = policy.get_modality_config()

dataset = LeRobotSingleDataset(dataset_path="demo_data/robot_sim.PickNPlace",
                               modality_configs=modality, video_backend="decord",
                               video_backend_kwargs=None, transforms=None,
                               embodiment_tag="gr1")
dp = dataset.get_step_data(0, 0)
ts = []
for i in range(6):
    t0 = time.perf_counter()
    out = policy.get_action(dp)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ts.append(time.perf_counter() - t0)
print("mode:", mode)
print("get_action times (s):", [round(t, 3) for t in ts])
print("avg excl. warmup: %.3f s  (%.1f Hz)" % (sum(ts[1:]) / (len(ts) - 1), (len(ts) - 1) / sum(ts[1:])))
k = [x for x in out if x.startswith("action.")]
print("action keys:", k)
