#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KV-cache ON/OFF 两份 pytorch(LPU) perf trace 的对账脚本（§6.7 落地实况 / 第 09 章 §7）。

用法:
    python3 tools/trace_kv_cache_diff.py <trace_ON 目录> <trace_OFF 目录>
目录可以是 scp 下来的外层目录（trace_kvcache_ON/），脚本自动找里面的 *_pt 子目录。

输出五张表，对应 §6.7 的四笔账：
  [1] step 级   —— Step_time / Computation / Free（省下的算力有没有变成墙钟）
  [2] kernel 级 —— 哪些 kernel 变短、launch 数有没有变少（本节核心论断）
  [3] API 级    —— 次数字节是否一致（证「省不掉发射/往返」）
  [4] adaln_qkv 逐次分类 —— cross K/V「计算 / HIT」计数，即坑 6 要的 HIT 探针
  [5] gap 归因  —— 设备空闲长在哪一类 kernel 之后
"""
import csv, glob, os, sys, collections

def pt_dir(d):
    if os.path.isfile(os.path.join(d, 'step_trace_time.csv')):
        return d
    for p in sorted(glob.glob(os.path.join(d, '*', 'step_trace_time.csv'))):
        return os.path.dirname(p)
    sys.exit(f'[!] {d} 下找不到 step_trace_time.csv')

def read(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))

def t1(d, tag):
    rows = read(os.path.join(d, 'step_trace_time.csv'))
    for r in rows:
        print(f'  {tag} step{r["Step"]}: step={float(r["Step_time(us)"])/1000:7.1f}ms '
              f'comp={float(r["Computation"])/1000:6.1f}ms({float(r["ComputationRatio"])*100:5.1f}%) '
              f'free={float(r["Free"])/1000:6.1f}ms({float(r["FreeRatio"])*100:5.1f}%) '
              f'host={float(r["Host_duration(us)"])/1000:7.1f}ms')

def t2(d, tag):
    agg = collections.defaultdict(lambda: [0, 0.0])
    for r in read(os.path.join(d, 'kernel_details.csv')):
        k = r['Kernel_function_name'] or r['Type']
        agg[k][0] += 1
        agg[k][1] += float(r['Kernel_dur(us)'])
    return tag, agg

def t3(d, tag):
    agg = collections.defaultdict(lambda: [0, 0.0])
    for r in read(os.path.join(d, 'runtime_api.csv')):
        agg[r['Name']][0] += 1
        agg[r['Name']][1] += float(r['Duration(us)'])
    return agg

def t4(d, tag, kern='adaln_qkv_run_die', full_lo=350, hit_hi=220):
    """按单次时长把融合 kernel 分成 cross-全算 / cross-HIT / self 三类。"""
    seq = collections.defaultdict(list)
    for r in read(os.path.join(d, 'kernel_details.csv')):
        if r['Kernel_function_name'] == kern:
            seq[int(r['Step_id'])].append((float(r['Kernel_start(s)']), float(r['Kernel_dur(us)'])))
    for s in sorted(seq):
        v = [x for _, x in sorted(seq[s])]
        full = [x for x in v if x >= full_lo]
        hit  = [x for x in v if x < hit_hi]
        slf  = [x for x in v if hit_hi <= x < full_lo]
        av = lambda a: sum(a)/len(a) if a else 0.0
        print(f'  {tag} step{s}: K/V计算={len(full):2d}x{av(full):5.0f}us  '
              f'HIT跳投影={len(hit):2d}x{av(hit):5.0f}us  self={len(slf):2d}x{av(slf):5.0f}us  '
              f'合计={sum(v)/1000:5.2f}ms')

def t5(d, tag):
    ks = collections.defaultdict(list)
    for r in read(os.path.join(d, 'kernel_details.csv')):
        ks[int(r['Step_id'])].append((float(r['Kernel_start(s)'])*1e6, float(r['Kernel_dur(us)']),
                                      r['Kernel_function_name'] or r['Type']))
    att = collections.defaultdict(lambda: [0, 0.0])
    for s, v in ks.items():
        v.sort()
        for a, b in zip(v, v[1:]):
            g = b[0] - (a[0] + a[1])
            if g > 0:
                att[a[2]][0] += 1
                att[a[2]][1] += g
    return att

def main():
    global A, B
    A, B = pt_dir(sys.argv[1]), pt_dir(sys.argv[2])
    print('[1] step 级（us→ms）')
    for tag, d in (('ON ', A), ('OFF', B)):
        t1(d, tag)
    print('\n[2] kernel 级差表（两份合计；只看总时长 top / 或次数变化）')
    (_, a), (_, b) = t2(A, 'ON'), t2(B, 'OFF')
    print(f'    kernel 总数 ON={sum(v[0] for v in a.values())}  OFF={sum(v[0] for v in b.values())}')
    print(f'    kernel 总时长 ON={sum(v[1] for v in a.values())/1000:.2f}ms  '
          f'OFF={sum(v[1] for v in b.values())/1000:.2f}ms')
    keys = sorted(set(a) | set(b), key=lambda k: -abs(a.get(k, [0, 0.])[1] - b.get(k, [0, 0.])[1]))
    print(f'    {"kernel":40s} {"ON_n":>6s} {"ON_ms":>8s} {"OFF_n":>6s} {"OFF_ms":>8s} {"d_ms":>7s}')
    for k in keys[:8]:
        x, y = a.get(k, [0, 0.]), b.get(k, [0, 0.])
        print(f'    {k[:40]:40s} {x[0]:6d} {x[1]/1000:8.2f} {y[0]:6d} {y[1]/1000:8.2f} {(x[1]-y[1])/1000:+7.2f}')
    print('\n[3] runtime API 次数（次数一致 = 发射/往返一条没省）')
    xa, xb = t3(A, 'ON'), t3(B, 'OFF')
    for k in sorted(set(xa) | set(xb)):
        x, y = xa.get(k, [0, 0.]), xb.get(k, [0, 0.])
        print(f'    {k[:40]:40s} ON {x[0]:5d}x {x[1]/1000:8.2f}ms | OFF {y[0]:5d}x {y[1]/1000:8.2f}ms')
    print('\n[4] adaln_qkv_run_die 逐次分类 = 坑 6 要的 HIT 计数探针')
    for tag, d in (('ON ', A), ('OFF', B)):
        t4(d, tag)
    print('\n[5] 设备空闲(gap)归因 top6 —— 按 gap 前一个 kernel 归类')
    ga, gb = t5(A, 'ON'), t5(B, 'OFF')
    for k in sorted(set(ga) | set(gb), key=lambda k: -(ga.get(k, [0, 0.])[1] + gb.get(k, [0, 0.])[1]))[:6]:
        x, y = ga.get(k, [0, 0.]), gb.get(k, [0, 0.])
        print(f'    {k[:40]:40s} ON {x[0]:4d}x {x[1]/1000:7.2f}ms | OFF {y[0]:4d}x {y[1]/1000:7.2f}ms'
              f'  d={(x[1]-y[1])/1000:+7.2f}ms')

def t6(d, tag):
    """host 侧 Python 帧总账（用于「本次改动不可能影响的路径是否也变慢了」的对照）。"""
    import json, re
    js = sorted(glob.glob(os.path.join(d, '*.aligned.pt.trace.json')))
    if not js:
        return {}
    with open(js[0]) as fp:
        doc = json.load(fp)
    evs = doc['traceEvents'] if isinstance(doc, dict) else doc
    agg = collections.defaultdict(lambda: [0, 0.0])
    for e in evs:
        if e.get('cat') in ('python_function', 'cpu_op'):
            k = re.sub(r' at 0x[0-9a-f]+', '', str(e.get('name')))
            agg[k][0] += 1
            agg[k][1] += e.get('dur', 0) / 1000
    return agg

def main6(A, B):
    print('\n[6] host 侧帧总账 top 差值（gross，含子调用）—— 用来判断差异能否归因给本次改动')
    ga, gb = t6(A, 'ON'), t6(B, 'OFF')
    if not ga or not gb:
        print('    (缺 *.aligned.pt.trace.json，跳过)')
        return
    for k in sorted(set(ga) | set(gb), key=lambda k: -abs(ga.get(k, [0, 0.])[1] - gb.get(k, [0, 0.])[1]))[:10]:
        x, y = ga.get(k, [0, 0.]), gb.get(k, [0, 0.])
        print(f'    {k[:66]:66s} ON {x[0]:5d}x {x[1]:7.1f}ms | OFF {y[0]:5d}x {y[1]:7.1f}ms  d={x[1]-y[1]:+7.1f}ms')


main()
main6(A, B)
