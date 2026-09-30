#!/usr/bin/env python3
"""第 05 章配图/配表复跑：Eagle 视觉塔 token 账（像素 → tile → 图像 token → seq_len）。

复现 05 §4.5 的实测表；同时兜住 02 §7.3 的 tile 推算表与 06/07/09 引用的 kv_len=296。
用法（需要能 import transformers+torch 的 gr00t 环境）：
    conda activate gr00t
    python3 tools/token_ledger_eagle.py [--repo <Isaac-GR00T路径>]

口径：vendored `gr00t/model/backbone/eagle2_hg_model/`（n1.5-release 与 main 该目录零 diff，2026-09-29 核验）。
自证：脚本内置断言——256×256 单图必得 1 tile / 256 图像 token；模板（空指令）文本侧 = 26。
"""
import argparse, os, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get("GR00T_REPO",
                    "../gr00t/Isaac-GR00T"), help="Isaac-GR00T 仓库路径")
    a = ap.parse_args()
    proc_path = os.path.join(a.repo, "gr00t/model/backbone/eagle2_hg_model")
    if not os.path.isdir(proc_path):
        sys.exit(f"找不到 {proc_path}（用 --repo 指定 Isaac-GR00T 路径）")

    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(proc_path, trust_remote_code=True)
    IMG_TOK = 151669  # config.image_token_index

    def measure(size, nimg=1, instr="pick up the red cube"):
        img = Image.fromarray(np.full((*size, 3), 128, np.uint8))
        conv = [{"role": "user",
                 "content": [{"type": "image", "image": img} for _ in range(nimg)]
                            + [{"type": "text", "text": instr}]}]
        text = proc.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        imgs, _ = proc.process_vision_info(conv)
        f = proc(text=[text], images=imgs, return_tensors="pt")
        ids = f.input_ids[0]
        n_img = int((ids == IMG_TOK).sum())
        tiles = tuple(f.pixel_values.shape)[0]
        return len(ids), n_img, len(ids) - n_img, tiles, tuple(f.pixel_values.shape[2:])

    print(f"{'输入':>18} | tiles | 图像token | 文本token | seq_len")
    cases = [((256, 256), 1, "pick up the red cube"),
             ((256, 256), 1, ""),
             ((256, 256), 3, "pick up the red cube"),
             ((640, 640), 1, ""),
             ((640, 400), 1, "")]
    ledger = {}
    for size, n, ins in cases:
        seq, ni, nt, tiles, hw = measure(size, n, ins)
        ledger[(size, n, ins)] = (seq, ni, nt, tiles)
        print(f"{str(size)+' x'+str(n)+f' instr={len(ins)}':>18} | {tiles:5d} | {ni:9d} | {nt:9d} | {seq}")

    # 自证断言（口径变了会先在这里炸，而不是让 05 章表格悄悄过期）
    seq1, ni1, nt1, t1 = ledger[((256, 256), 1, "pick up the red cube")]
    _, _, nt0, _ = ledger[((256, 256), 1, "")]
    assert t1 == 1 and ni1 == 256, "256×256 应为 1 tile / 256 图像 token"
    assert nt0 == 26, f"空指令模板应为 26 文本 token，实为 {nt0}"
    seq640, ni640, _, t640 = ledger[((640, 640), 1, "")]
    assert t640 == 10 and ni640 == 2560, "640×640 应为 3×3+缩略图 = 10 tiles / 2560 token"
    print("\n断言通过：seq_len = 256×tiles + 模板26 + tokenize(指令)；"
          "\n02 章 demo 的 seq_len 296 ⇔ 指令占 14 token（256+26+14）。")

if __name__ == "__main__":
    main()
