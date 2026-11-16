"""Interleaved A/B with ABBA ordering, plus a null control that measures this session's noise floor.

Two lessons are baked in here.  Comparing a configuration against itself (--control) is the only way to know how large a
difference has to be before it means anything: an earlier prefill comparison against a GGML_VK_DISABLE_MMQ that turned out
not to exist in the backend still produced a clean-looking +3%, which was position bias plus drift, not a result.  And
because the second run of a pair tends to be warmer or colder than the first, the order alternates per pair (ABBA).

    python port/tools/ab_pp.py <pack> --pp 64 --pairs 4 --b "GGML_VK_DISABLE_MMVQ=1"
    python port/tools/ab_pp.py <pack> --pp 64 --pairs 3 --control
"""
import argparse
import os
import re
import statistics
import subprocess

EXE = r'build\bin\strata-vk.exe'
KNOBS = ('GGML_VK_DISABLE_MMQ', 'GGML_VK_DISABLE_MMVQ', 'GGML_VK_DISABLE_COOPMAT', 'GGML_VK_DISABLE_COOPMAT2')


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


def run(pack, pp, extra):
    env = dict(os.environ)
    for key in KNOBS:
        env.pop(key, None)
    env.update(extra)
    out = subprocess.run([EXE, '--native', pack, '--bench-pp', str(pp)], capture_output=True, text=True, env=env,
                         timeout=1800).stdout
    m = re.search(r'= ([0-9.]+) tok/s', out)
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pack')
    ap.add_argument('--pp', type=int, default=64)
    ap.add_argument('--pairs', type=int, default=4)
    ap.add_argument('--a', default='')
    ap.add_argument('--b', default='')
    ap.add_argument('--control', action='store_true', help='run the same setting twice to measure the noise floor')
    args = ap.parse_args()

    a_env = parse_setting(args.a)
    b_env = a_env if args.control else parse_setting(args.b)
    label_a, label_b = (args.a or 'default'), ('default again' if args.control else (args.b or 'default'))

    a, b = [], []
    for i in range(args.pairs):
        a_first = (i % 2 == 0)
        if a_first:
            a.append(run(args.pack, args.pp, a_env))
            b.append(run(args.pack, args.pp, b_env))
        else:
            b.append(run(args.pack, args.pp, b_env))
            a.append(run(args.pack, args.pp, a_env))
        print(f'pair {i + 1} ({"AB" if a_first else "BA"}): A {a[-1]:.2f}  B {b[-1]:.2f}', flush=True)

    ma, mb = statistics.median(a), statistics.median(b)
    wins = sum(1 for x, y in zip(a, b) if y > x)
    print(f'A ({label_a}) median {ma:.2f}   B ({label_b}) median {mb:.2f}')
    print(f'delta {100 * (mb / ma - 1):+.2f}%   B faster in {wins}/{args.pairs} pairs')
    if args.control:
        print('this is the noise floor for this session; any A/B smaller than it is not a result')


if __name__ == '__main__':
    main()
