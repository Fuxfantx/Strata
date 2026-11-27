"""How fast can the pack's experts be streamed per token?

The 40.8 GiB blob cannot be resident in device-visible memory on this machine (see LATENT-MOE.md), so the design that
remains is streaming the selected experts from the file each token.  This measures exactly that access pattern: forty-
eight layers times ten experts of 1.63 MB, read at the offsets a token would need.
"""
import pathlib
import random
import time

PACK = pathlib.Path(r'C:\Users\Fuxfantx\A-Works\DSH Works\.scratch\ms-145b-q3k')
BLOB = 1_630_720
LAYERS, EXPERTS, TOPK = 48, 512, 10


def main() -> int:
    path = PACK / 'experts.bin'
    size = path.stat().st_size
    rng = random.Random(7)
    rounds = 3
    total = 0
    started = time.perf_counter()
    with path.open('rb', buffering=0) as f:
        for _ in range(rounds):
            for layer in range(LAYERS):
                base = layer * EXPERTS * BLOB
                for e in rng.sample(range(EXPERTS), TOPK):
                    f.seek(base + e * BLOB)
                    buf = f.read(BLOB)
                    total += len(buf)
    elapsed = time.perf_counter() - started
    per_token = total / rounds
    print('read %.2f GiB in %.2f s for %d token-mixes' % (total / 1073741824.0, elapsed, rounds))
    print('per token: %.0f MiB in %.1f ms -> %.2f s of a 21.7 s token budget'
          % (per_token / 1048576.0, 1000.0 * elapsed / rounds, elapsed / rounds))
    print('file size %.2f GiB, whole-pack scan would take %.1f s'
          % (size / 1073741824.0, elapsed * size / total))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
