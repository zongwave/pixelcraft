#!/usr/bin/env python
"""第 02 章 §7.7 动图：真机倒茶成功演示，加速压缩为 GIF。

ffmpeg 两遍调色板法（palettegen/paletteuse），默认 4× 提速、宽 280、8 fps。
用法：
    python3 tools/mk_gif_n15_pour.py [--src /path/n15_robot.mp4]
                                     [--speed 4] [--width 280] [--fps 8]
"""
import argparse, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/home/ft/wzong/workspace/docs/n15_robot.mp4")
ap.add_argument("--out", default=os.path.join(ROOT, "images", "ch02", "n15_pour_demo.gif"))
ap.add_argument("--speed", type=float, default=4.0)
ap.add_argument("--width", type=int, default=280)
ap.add_argument("--fps", type=int, default=8)
a = ap.parse_args()

vf = f"fps={a.fps},setpts=PTS/{a.speed},scale={a.width}:-1:flags=lanczos"
pal = "/tmp/_n15pal.png"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a.src, "-vf", f"{vf},palettegen", "-frames:v", "1", pal], check=True)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a.src, "-i", pal,
                "-lavfi", f"{vf}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4", a.out], check=True)
mb = os.path.getsize(a.out) / 1e6
print(f"saved: {a.out}  ({mb:.1f} MB, {a.speed:g}x speed, {a.width}px/{a.fps}fps)")
if mb > 12:
    sys.exit("WARN: >12MB，考虑 --width 240 或 --fps 6")
