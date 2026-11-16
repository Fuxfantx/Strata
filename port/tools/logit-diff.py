"""Compare two logit dumps: the quality proxy for a companion pack.

The engine writes one float32 logit vector with `--dump-logits <file>`, so two runs of the same prompt - one on the pack,
one on the pack plus a companion - give the numbers a quantisation change has to be judged by: top-1 agreement, top-k
overlap, and the symmetric KL divergence between the two distributions.  A companion that keeps top-1 and keeps KL small
is a candidate; one that does not is not, whatever it does to the rate.

    python port/tools/logit-diff.py base.bin compat.bin [--k 10]
"""
import argparse
import math
import struct
import sys


def load(path):
    data = pathlib_bytes(path)
    if len(data) % 4 != 0:
        sys.exit(f'{path}: {len(data)} bytes is not a multiple of 4')
    return list(struct.unpack(f'<{len(data) // 4}f', data))


def pathlib_bytes(path):
    with open(path, 'rb') as f:
        return f.read()


def softmax(xs):
    m = max(xs)
    ex = [pow(2.718281828459045, x - m) for x in xs]
    s = sum(ex)
    return [e / s for e in ex]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('base')
    ap.add_argument('compat')
    ap.add_argument('-k', type=int, default=10)
    args = ap.parse_args()

    a = load(args.base)
    b = load(args.compat)
    if len(a) != len(b):
        sys.exit(f'logit vectors differ in length: {len(a)} against {len(b)}')

    pa, pb = softmax(a), softmax(b)
    kl_ab = sum(p * math.log(p / q) for p, q in zip(pa, pb) if p > 0)
    kl_ba = sum(q * math.log(q / p) for p, q in zip(pa, pb) if q > 0)
    sym = 0.5 * (kl_ab + kl_ba)

    order_a = sorted(range(len(a)), key=lambda i: -a[i])
    order_b = sorted(range(len(b)), key=lambda i: -b[i])
    top1_same = order_a[0] == order_b[0]
    k = min(args.k, len(a))
    overlap = len(set(order_a[:k]) & set(order_b[:k])) / k
    max_delta = max(abs(x - y) for x, y in zip(a, b))

    print(f'vocab           {len(a)}')
    print(f'top-1           {"same" if top1_same else "DIFFERENT"}  ({order_a[0]} against {order_b[0]})')
    print(f'top-{k} overlap   {overlap * 100:.1f}%')
    print(f'symmetric KL    {sym:.6f} nats')
    print(f'max |delta|     {max_delta:.4f} logits')
    print()
    print('reading: KL under ~0.001 and the same top-1 is a mild change; a different top-1 on a factual prompt is a')
    print('visible change, and a KL in the tenths is a different model.')


if __name__ == '__main__':
    main()
