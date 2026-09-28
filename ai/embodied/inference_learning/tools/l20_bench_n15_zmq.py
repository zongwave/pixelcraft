#!/usr/bin/env python
"""ZMQ 服务化路径端到端延迟（客户端视角，含序列化+回环传输）。"""
import argparse, statistics, time
import numpy as np
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.eval.robot import RobotInferenceClient

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--warmup", type=int, default=3)
args = ap.parse_args()

policy = RobotInferenceClient(host="localhost", port=5555)
print("ping:", policy.call_endpoint("ping", requires_input=False))
modality = policy.get_modality_config()   # 契约由服务端定义（第07章）
from gr00t.experiment.data_config import load_data_config
ds = LeRobotSingleDataset(dataset_path="demo_data/robot_sim.PickNPlace",
                          modality_configs=modality, video_backend="decord",
                          video_backend_kwargs=None, transforms=None, embodiment_tag="gr1")
dp = ds.get_step_data(0, 0)
ts = []
for i in range(args.iters + args.warmup):
    t0 = time.perf_counter()
    out = policy.get_action(dp)
    dt = time.perf_counter() - t0
    if i >= args.warmup:
        ts.append(dt)
q = np.percentile(ts, [50, 90, 95])
print("e2e(zmq) n=%d mean=%.3f p50=%.3f p90=%.3f p95=%.3f min=%.3f max=%.3f std=%.3f" %
      (len(ts), sum(ts)/len(ts), q[0], q[1], q[2], min(ts), max(ts), statistics.pstdev(ts)))
print("action keys:", [k for k in out if k.startswith("action.")])
