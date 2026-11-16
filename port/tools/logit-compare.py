"""Compare two dumped logit vectors: max difference and the decision-relevant figures."""
import struct
import sys


def load(path):
    data = open(path, "rb").read()
    return struct.unpack("<%df" % (len(data) // 4), data)


for i in range(1, len(sys.argv), 3):
    label, pa, pb = sys.argv[i], sys.argv[i + 1], sys.argv[i + 2]
    a, b = load(pa), load(pb)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    mx = max(abs(a[j] - b[j]) for j in range(n))
    ia = max(range(n), key=lambda j: a[j])
    ib = max(range(n), key=lambda j: b[j])
    print("%-6s max|diff|=%.4f  cpu_top1=%d (%.4f)  gpu_top1=%d (%.4f)  diff@cpu=%0.4f  diff@gpu=%.4f" % (
        label, mx, ia, a[ia], ib, b[ib], abs(a[ia] - b[ia]), abs(a[ib] - b[ib])))
