#!/usr/bin/env python3
"""第 9 章配图（图 9-8）：Perfetto 原生 UI 截图（ui.perfetto.dev 无头抓取）。

原理：本地起带 CORS 的 HTTP 服务托管 trace，用 chromium(--headless, playwright) 打开
  https://ui.perfetto.dev/?url=http://localhost:PORT/xxx.json
Perfetto 会自动加载并把视口 fit 到整条 trace ⇒ "局部特写"的做法=先把时间窗裁成子 trace
（ts 平移到 0），UI 自然只画那一窗。

产物（images/ch09/）：
  perfetto_old_overview.png   OLD 全程 5.2 s：947 小簇 + HtoD 海洋 + ≈40 ms 同步 gap
  perfetto_old_iter_edge.png  OLD 4400–4560 ms：38 ms CloneTranspose 迭代边界特写
  perfetto_cur_overview.png   CUR 热轮全程 153 ms：prologue + 4 个去噪步
  perfetto_cur_step.png       CUR 63.5–77.5 ms：单去噪步逐 op（16 block 指纹 + launch gap）
  perfetto_new_overview.png   NEW 冷轮全程 1.39 s：同批 kernel 被 host 首帧路径摊开

依赖：pip install playwright；系统 chromium；可访问 ui.perfetto.dev。
用法：python3 tools/mk_fig_ch09_perfetto_snapshots.py
"""
import http.server
import json
import os
import socketserver
import sys
import threading
import urllib.parse

HOME = os.path.expanduser("~")
TRACES = {
    "OLD": f"{HOME}/wzong/workspace/embodied/logs/trace_3cam_2iter_ae_fused_k67/wzong_sd3_2iter_chrome_trace.json",
    "CUR": f"{HOME}/wzong/workspace/embodied/logs/trace_current_clean/n15_sd3_chrome_trace.json",
    "NEW": f"{HOME}/wzong/workspace/embodied/logs/wz_prof_current_20260930_145846/trace/n15_sd3_chrome_trace.json",
}
OUTDIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "images", "ch09"))
TMPDIR = "/tmp/perfetto_shots"

# (out_name, trace_key, lo_ms, hi_ms)
UI_DIR = os.path.expanduser("~/.cache/perfetto-ui")
UI_PORT = int(os.environ.get("PERFETTO_UI_PORT", "8080"))

SHOTS = [
    ("perfetto_old_overview.png",  "OLD", None,  None),
    ("perfetto_old_iter_edge.png", "OLD", 4400, 4560),
    ("perfetto_cur_overview.png",  "CUR", None,  None),
    ("perfetto_cur_step.png",      "CUR", 63.4, 77.6),
    ("perfetto_new_overview.png",  "NEW", None,  None),
]


def load(p):
    d = json.load(open(p))
    return d["traceEvents"] if isinstance(d, dict) else d


def emit(path, events):
    json.dump({"traceEvents": events, "displayTimeUnit": "ms"}, open(path, "w"))


def prepare():
    os.makedirs(TMPDIR, exist_ok=True)
    full = {k: load(v) for k, v in TRACES.items()}
    for i, (out, key, lo, hi) in enumerate(SHOTS):
        ev = full[key]
        src = os.path.join(TMPDIR, f"trace_{i}.json")
        dev = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy")]
        t0 = min(e["ts"] for e in dev)          # 设备时钟为基准（runtime 时钟域可能有偏移）
        if lo is None:
            # 全程图：以设备事件时间窗为准裁掉"时间戳离群"的杂散事件
            #（clock 域不同的 counter/async 事件会把 Perfetto 视口拉到几十天，tracks 全空）
            lo_us, hi_us = t0, max(e["ts"] + e["dur"] for e in dev) + 5000
        else:
            lo_us, hi_us = t0 + lo * 1000, t0 + hi * 1000
        kept = []
        for e in ev:
            if e.get("ph") == "M":               # 线程/进程名等元数据保留
                kept.append(e)
            elif e.get("ph") in ("X",) and e["ts"] + e.get("dur", 0) >= lo_us and e["ts"] <= hi_us:
                e2 = dict(e)
                if lo is not None:               # 局部特写：整体平移到 0，Perfetto 自动 fit 到该窗
                    e2["ts"] = max(e["ts"], lo_us) - lo_us
                    if e["ts"] < lo_us:
                        e2["dur"] = e.get("dur", 0) - (lo_us - e["ts"])
                kept.append(e2)
        emit(src, kept)
        print(f"  prepared {out}: {len(kept)} events (window={[lo, hi]})")


class Handler(http.server.SimpleHTTPRequestHandler):
    """同源托管：/traces/* -> TMPDIR，其余 -> 本地 Perfetto UI（v58.2 release 包）。"""

    def translate_path(self, path):
        root = TMPDIR if path.startswith("/traces/") else UI_DIR
        rel = path[len("/traces"):] if path.startswith("/traces/") else path
        return os.path.join(root, rel.lstrip("/").split("?")[0].split("#")[0])

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")   # 避开 service worker 缓存旧版
        super().end_headers()

    def log_message(self, *a):
        pass


def autocrop(path, side_frac=0.17, pad=30):
    """裁掉轨道行下方的整片空白：左侧 10% 是侧栏(永远有文字)，只看右侧绘图区，
    从底往上找到最后一行非白像素，再留 pad 的边距。"""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    w, h = im.size
    px = im.load()
    sx, step = int(w * side_frac), max(1, w // 900)
    bottom_skip = 140            # 底部状态栏/角标属于 UI chrome，跳过后再找轨道内容
    last = 0
    for y in range(h - 1 - bottom_skip, -1, -1):
        if any(px[x, y][0] < 235 for x in range(sx, w - 40, step)):
            last = y
            break
    if last and last < h - pad:
        im.crop((0, 0, w, min(h, last + pad))).save(path)
        print(f"  cropped {os.path.basename(path)}: {w}x{h} -> {w}x{min(h, last + pad)}")


def snap(shots):
    from playwright.sync_api import sync_playwright
    exe = os.environ.get("CHROME_BIN") or os.path.expanduser(
        "~/.cache/ms-playwright/cft-153/chrome")
    if not os.path.exists(exe):
        exe = None
    threading.Thread(
        target=lambda: socketserver.TCPServer.allow_reuse_address or None, daemon=True).start()
    socketserver.TCPServer.allow_reuse_address = True
    srv = socketserver.TCPServer(("127.0.0.1", UI_PORT), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe, headless=True,
                                    args=["--no-sandbox", "--disable-gpu"])
        for i, (out, key, lo, hi) in enumerate(shots):
            page = browser.new_page(viewport={"width": 1720, "height": 980},
                                    device_scale_factor=2)
            url = (f"http://localhost:{UI_PORT}/#!/?url=" +
                   urllib.parse.quote(f"http://localhost:{UI_PORT}/traces/trace_{i}.json",
                                      safe="") +
                   "&title=" + urllib.parse.quote(
                       f"{key} {'full' if lo is None else f'{lo}-{hi} ms'}"))
            print(f"  opening {out}")
            for attempt in range(3):
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
                try:
                    page.wait_for_selector("canvas", timeout=45000)
                    break
                except Exception:
                    print(f"  retry {attempt+1}")
            try:
                page.wait_for_function(
                    "()=>!document.body.innerText.includes('Loading trace')", timeout=90000)
            except Exception:
                pass
            page.wait_for_timeout(4000)
            page.evaluate("""() => {              // 本地 UI 也会带 GA cookie 横幅
                for (const b of document.querySelectorAll('button'))
                    if (b.textContent.trim() === 'OK') b.click();
            }""")
            page.keyboard.press("Escape")
            page.wait_for_timeout(800)
            shot = os.path.join(OUTDIR, out)
            page.screenshot(path=shot)
            autocrop(shot)
            print(f"  saved {out}")
            page.close()
        browser.close()
    srv.shutdown()


if __name__ == "__main__" and "--crop-only" in sys.argv:
    for out, *_ in SHOTS:
        f = os.path.join(OUTDIR, out)
        if os.path.exists(f):
            from PIL import Image
            h0 = Image.open(f).size[1]
            autocrop(f)   # 注意：已裁过的再跑一次只会更小一点或不变
            print(f"  {out}: h {h0} -> {Image.open(f).size[1]}")
elif __name__ == "__main__":
    print("preparing traces ...")
    prepare()
    print("snapshotting via ui.perfetto.dev ...")
    snap(SHOTS)
    print("done ->", OUTDIR)
