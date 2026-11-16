"""Compare the prefill logits of the original 27B with a quantised pack (the quality instrument)."""
import math
import pathlib
import struct
import subprocess
import sys

EXE = r"build\bin\strata-vk.exe"
ORIG = r"C:\Users\Fuxfantx\C-Res\Qwen\3.8-27B-Q4\Qwen3.8-27B-Q4_0.gguf"
PROMPT = "The capital of France is"


def dump(model, out):
    r = subprocess.run([EXE, "--native", model, "--prompt", PROMPT, "--n", "1", "--dump-logits", out],
                       capture_output=True, text=True)
    ok = pathlib.Path(out).exists() and pathlib.Path(out).stat().st_size > 0
    if not ok:
        print(f"dump failed for {model}:\n{r.stderr[-800:]}")
        sys.exit(1)
    return out


def compare(a_path, b_path, label):
    a = pathlib.Path(a_path).read_bytes()
    b = pathlib.Path(b_path).read_bytes()
    n = min(len(a), len(b)) // 4
    la = struct.unpack(f"<{n}f", a[:4 * n])
    lb = struct.unpack(f"<{n}f", b[:4 * n])

    m = max(la)
    ea = [math.exp(v - m) for v in la]
    sa = sum(ea)
    m2 = max(lb)
    eb = [math.exp(v - m2) for v in lb]
    sb = sum(eb)
    pa = [v / sa for v in ea]
    pb = [v / sb for v in eb]
    kl = sum(pa[i] * math.log(pa[i] / pb[i]) for i in range(n) if pa[i] > 1e-12)

    top_a = sorted(range(n), key=lambda i: -la[i])[:5]
    top_b = sorted(range(n), key=lambda i: -lb[i])[:5]
    ma, mb = sum(la) / n, sum(lb) / n
    cov = sum((la[i] - ma) * (lb[i] - mb) for i in range(n))
    va = math.sqrt(sum((v - ma) ** 2 for v in la))
    vb = math.sqrt(sum((v - mb) ** 2 for v in lb))
    print(f"{label}: top1_agree={top_a[0] == top_b[0]} top5_overlap={len(set(top_a) & set(top_b))}/5 "
          f"KL={kl:.4f} corr={cov / (va * vb):.4f}  (orig top1={top_a[0]}, quant top1={top_b[0]})")


if __name__ == "__main__":
    orig = dump(ORIG, r"build\quant\logits_orig.bin")
    for pack, label in [(a, b) for a, b in zip(sys.argv[1::2], sys.argv[2::2])]:
        compare(orig, dump(pack, pathlib.Path(pack).with_suffix(".logits.bin")), label)
