#!/usr/bin/env python3
"""抓 GR00T-N1.5-3B 真实 SigLIP 视觉塔的注意力分布 alpha（图 14-5 的数据来源）。

跑法（CPU 即可，单帧 224×224、27 层，约十几秒）：
    /home/ft/miniconda3/envs/gr00t/bin/python3 tools/cap_ch14_siglip_alpha.py

产出 /tmp/siglip_alpha.npz：
    attn  [27,16,256,256] float16   每层每头 softmax 之后的真实 alpha
    frame [224,224,3]      float32   归一化前（0~1）的真实 demo 帧
    qidx  [1]             int64     图里高亮的 query patch（行优先展平后的下标）

来源都是真的：
  * 权重：/home/ft/wzong/workspace/images/GR00T-N1_5-3B（剥前缀
    backbone.eagle_model.vision_model.vision_model. 后正好是 transformers 的
    SiglipVisionModel，strict=False 但 missing 为空）。
  * 帧：Isaac-GR00T/demo_data/cube_to_bowl_5 前相机第 0 帧（ffmpeg 抽帧），
    按推理管线的做法 resize 到 224×224（squish，不保长宽比）。
  * alpha：eager attention + output_attentions=True，27 层全都要。
"""
import json
import os
import subprocess

import numpy as np
import torch

CKPT = "/home/ft/wzong/workspace/images/GR00T-N1_5-3B"
EAGLE_CFG = ("/home/ft/wzong/workspace/embodied/gr00t/Isaac-GR00T/gr00t/model/"
             "backbone/eagle2_hg_model/config.json")
VIDEO = ("/home/ft/wzong/workspace/embodied/gr00t/Isaac-GR00T/demo_data/"
         "cube_to_bowl_5/videos/chunk-000/observation.images.front/episode_000000.mp4")
RAW_FRAME = "/tmp/frame0_full.png"
OUT = "/tmp/siglip_alpha.npz"
# transformers 的 SiglipVisionModel 内部还套了一层 vision_model，所以只剥到这一级
PREFIX = "backbone.eagle_model.vision_model."
QIDX = 7 * 16 + 8  # 第 7 行、第 8 列那块（图里画红框的 query patch）


def load_frame():
    from PIL import Image
    if not os.path.exists(RAW_FRAME):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", VIDEO,
                        "-frames:v", "1", RAW_FRAME], check=True)
    im = Image.open(RAW_FRAME).convert("RGB").resize((224, 224), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def load_vision_tower():
    from safetensors.torch import safe_open
    from transformers import SiglipVisionConfig, SiglipVisionModel

    vcfg = json.load(open(EAGLE_CFG))["vision_config"]
    kw = {k: v for k, v in vcfg.items() if not k.startswith("_")}
    cfg = SiglipVisionConfig(attn_implementation="eager", **kw)
    model = SiglipVisionModel(cfg)
    model.config._attn_implementation = "eager"
    sd = {}
    for f in sorted(os.listdir(CKPT)):
        if not f.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(CKPT, f), framework="pt") as h:
            for k in h.keys():
                if k.startswith(PREFIX):
                    sd[k[len(PREFIX):]] = h.get_tensor(k).float()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("loaded %d tensors, missing=%d unexpected=%d" % (len(sd), len(missing), len(unexpected)))
    bad = [k for k in missing if not k.startswith("vision_model.head")]  # head 是 SigLIP 分类头，gr00t 不用
    assert not bad, bad
    return model.eval().to(torch.float32)


def main():
    frame = load_frame()
    pv = torch.from_numpy(((frame - 0.5) / 0.5).transpose(2, 0, 1)[None])
    model = load_vision_tower()
    with torch.no_grad():
        out = model(pixel_values=pv, output_attentions=True)
    attn = torch.stack(out.attentions, dim=0).float()          # [27,B,16,256,256]
    attn = attn.squeeze(1).numpy().astype(np.float16)          # 单图批维去掉
    print("attn", attn.shape, "max|row_sum-1| =",
          np.abs(attn.astype(np.float32).sum(-1) - 1).max())
    np.savez_compressed(OUT, attn=attn, frame=frame.astype(np.float32), qidx=np.int64(QIDX))
    print("saved", OUT, round(os.path.getsize(OUT) / 1e6, 1), "MB")

    a = attn.astype(np.float32)
    ent = -(a * np.log(a + 1e-9)).sum(-1)  # [27,16,256]
    print("该 query 行上每层每头的 top-1 alpha / 熵（均匀分布熵 = ln256 = 5.545）:")
    for L in (0, 6, 12, 13, 20, 26):
        row = a[L, :, QIDX, :]
        top1 = row.max(-1)
        e = ent[L, :, QIDX]
        print("  L%-2d top1 max=%.3f med=%.3f | entropy min=%.2f med=%.2f max=%.2f"
              % (L, top1.max(), np.median(top1), e.min(), np.median(e), e.max()))


if __name__ == "__main__":
    main()
