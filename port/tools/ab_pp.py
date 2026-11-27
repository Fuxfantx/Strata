"""Interleaved A/B with ABBA ordering and a null control, for decode as well as prefill.

    python port/tools/ab_pp.py <pack> --mode decode --pairs 5 --b "GGML_VK_FORCE_MMVQ=1"
    python port/tools/ab_pp.py <pack> --mode decode --pairs 3 --control

Each run reloads the pack, so a pair takes about three minutes; the control is what says whether a difference is real.
"""
import argparse
import os
import re
import statistics
import subprocess

EXE = r'build\bin\strata-vk.exe'
KNOBS = ('GGML_VK_DISABLE_MMQ', 'GGML_VK_DISABLE_MMVQ', 'GGML_VK_FORCE_MMVQ', 'GGML_VK_DISABLE_COOPMAT',
         'GGML_VK_DISABLE_COOPMAT2', 'STRATA_VK_FORCE_MMVQ')


def parse_setting(text):
    text = (text or '').strip()
    if not text:
        return {}
    out = {}
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        key, _, value = part.partition('=')
        out[key.strip()] = value.strip() or '1'
    return out


def run(pack, mode, extra):
    env = dict(os.environ)
    for key in KNOBS:
        env.pop(key, None)
    env.update(extra)
    args = [EXE, '--native', pack]
    if mode == 'decode':
        args += ['--tokens', '9707', '--n', '8']
        pattern = r'steady ([0-9.]+)'
    else:
        args += ['--bench-pp', str(mode)]
        pattern = r'= ([0-9.]+) tok/s'
    out = subprocess.run(args, capture_output=True, text=True, env=env, timeout=1800).stdout
    m = re.search(pattern, out)
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pack')
    ap.add_argument('--mode', default='64', help="decode, or a prefill token count")
    ap.add_argument('--pairs', type=int, default=3)
    ap.add_argument('--a', default='')
    ap.add_argument('--b', default='')
    ap.add_argument('--control', action='store_true')
    args = ap.parse_args()

    a_env = parse_setting(args.a)
    b_env = a_env if args.control else parse_setting(args.b)
    label_a = args.a or 'default'
    label_b = 'default again' if args.control else (args.b or 'default')

    a, b = [], []
    for i in range(args.pairs):
        a_first = (i % 2 == 0)
        if a_first:
            a.append(run(args.pack, args.mode, a_env))
            b.append(run(args.pack, args.mode, b_env))
        else:
            b.append(run(args.pack, args.mode, b_env))
            a.append(run(args.pack, args.mode, a_env))
        print(f'pair {i + 1} ({"AB" if a_first else "BA"}): A {a[-1]:.2f}  B {b[-1]:.2f}', flush=True)

    ma, mb = statistics.median(a), statistics.median(b)
    wins = sum(1 for x, y in zip(a, b) if y > x)
    # Under drift the median across a whole session is not the statistic to trust: each pair is measured a minute apart,
    # so the median of the per-pair ratios is immune to the machine slowing down over the session.
    ratios = [y / x - 1 for x, y in zip(a, b) if x and y]
    print(f'A ({label_a}) median {ma:.2f}   B ({label_b}) median {mb:.2f}')
    print(f'delta {100 * (mb / ma - 1):+.2f}%   B faster in {wins}/{args.pairs} pairs')
    if ratios:
        print(f'paired median delta {100 * statistics.median(ratios):+.2f}%  (the statistic to decide on)')
    if args.control:
        print('this is the noise floor for this session; any A/B smaller than it is not a result')


if __name__ == '__main__':
    main()
