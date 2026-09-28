#!/usr/bin/env python
"""Python-API 版 trtexec（本机 pip TensorRT 10.9 无 trtexec 二进制的替代）。
复刻 build_engine.sh 的 --stronglyTyped + 动态 shape profile（batch 上限可配）。
用法: CUDA_VISIBLE_DEVICES=4 python build_engine_py.py <onnx_dir> <engine_dir> [--max-batch 8] [--video-views 1]
"""
import argparse, os, sys, time
import tensorrt as trt

MIN_LEN, OPT_LEN, MAX_LEN = 80, 296, 300

def build(onnx, engine, profiles, logger=trt.Logger(trt.Logger.WARNING), strong=True):
    """profiles: list of (tensor_name, min, opt, max) shape tuples (same order/names as inputs)."""
    tb = trt.Builder(logger)
    net_flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED) if strong else 0
    network = tb.create_network(net_flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx):
        for i in range(parser.num_errors):
            print("ONNX parse error:", parser.get_error(i), file=sys.stderr)
        raise RuntimeError(f"failed to parse {onnx}")
    config = tb.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    prof = tb.create_optimization_profile()
    for name, mn, op, mx in profiles:
        if hasattr(prof, "append"):
            prof.append(name, mn, op, mx)
        else:
            prof.set_shape(name, mn, op, mx)
    config.add_optimization_profile(prof)
    t0 = time.time()
    print(f"[build] {os.path.basename(onnx)} ...", flush=True)
    serialized = tb.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"build failed for {onnx}")
    buf = bytes(serialized)
    with open(engine, "wb") as f:
        f.write(buf)
    print(f"[done ] {engine}  ({time.time()-t0:.0f}s, {len(buf)/1e6:.0f} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx_dir"); ap.add_argument("engine_dir")
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--only", type=str, default="", help="comma substrings to filter jobs")
    args = ap.parse_args()
    os.makedirs(args.engine_dir, exist_ok=True)
    B = args.max_batch
    o, e = args.onnx_dir, args.engine_dir
    jobs = [
        # (onnx, engine, profiles)
        (f"{o}/action_head/vlln_vl_self_attention.onnx", f"{e}/vlln_vl_self_attention.engine",
         [("backbone_features", (1, MIN_LEN, 2048), (1, OPT_LEN, 2048), (B, MAX_LEN, 2048))]),
        (f"{o}/action_head/DiT_fp16.onnx", f"{e}/DiT_fp16.engine",
         [("sa_embs", (1, 49, 1536), (1, 49, 1536), (B, 49, 1536)),
          ("vl_embs", (1, MIN_LEN, 2048), (1, OPT_LEN, 2048), (B, MAX_LEN, 2048)),
          ("timesteps_tensor", (1,), (1,), (B,))]),
        (f"{o}/action_head/state_encoder.onnx", f"{e}/state_encoder.engine",
         [("state", (1, 1, 64), (1, 1, 64), (B, 1, 64)),
          ("embodiment_id", (1,), (1,), (B,))]),
        (f"{o}/action_head/action_encoder.onnx", f"{e}/action_encoder.engine",
         [("actions", (1, 16, 32), (1, 16, 32), (B, 16, 32)),
          ("timesteps_tensor", (1,), (1,), (B,)),
          ("embodiment_id", (1,), (1,), (B,))]),
        (f"{o}/action_head/action_decoder.onnx", f"{e}/action_decoder.engine",
         [("model_output", (1, 49, 1024), (1, 49, 1024), (B, 49, 1024)),
          ("embodiment_id", (1,), (1,), (B,))]),
        (f"{o}/eagle2/vit_fp16.onnx", f"{e}/vit_fp16.engine",
         [("pixel_values", (1, 3, 224, 224), (1, 3, 224, 224), (B, 3, 224, 224)),
          ("position_ids", (1, 256), (1, 256), (B, 256))]),
        (f"{o}/eagle2/llm_fp16.onnx", f"{e}/llm_fp16.engine",
         [("inputs_embeds", (1, MIN_LEN, 2048), (1, OPT_LEN, 2048), (B, MAX_LEN, 2048)),
          ("attention_mask", (1, MIN_LEN), (1, OPT_LEN), (B, MAX_LEN))]),
    ]
    only = [s for s in args.only.split(",") if s]
    for onnx, engine, prof in jobs:
        if only and not any(s in os.path.basename(onnx) for s in only):
            continue
        build(onnx, engine, prof)
    print("ALL BUILDS COMPLETE")

if __name__ == "__main__":
    main()
