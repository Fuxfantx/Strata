"""Does a change alter the text a user would see?  The gate the product actually cares about.

Every other check in this port compares numbers (logits correlation, argmax, a token stream against the
CPU backend), and those are stricter than the requirement: a re-quantised tensor may pass or fail them
while the visible answer stays the same.  This tool asks the product's question instead - run a few
prompts greedily on one engine configuration, decode the tokens back to text with the engine's own
vocabulary, and compare the text of two configurations.

    python port/tools/textgate.py --model Q2.gguf --tag base
    python port/tools/textgate.py --model Q2.gguf --tag q8 --compat companion.gguf
    python port/tools/textgate.py --compare base q8

The prompts are continuations, not chat: a 3-bit pack answers a chat turn with a stop token often
enough that a gate built on chat prompts measures one token of nothing.
"""
import argparse
import os
import re
import subprocess
import sys

PROMPTS = [
    "The capital of France is",
    "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n\nprint(fib(10))  #",
    "The rain had stopped by morning, and the street smelled of wet stone and diesel. She stood at the "
    "corner for a while, watching the",
    "Question: A train travels 120 km in 1.5 hours, then 180 km in 2.5 hours. What is its average speed "
    "for the whole trip?\nAnswer: The total distance is",
]
N_NEW = 48


def engine(root):
    for cand in (os.path.join(root, "build", "bin", "strata-vk.exe"),
                 os.path.join(root, "build", "bin", "strata-vk")):
        if os.path.exists(cand):
            return cand
    sys.exit("no strata-vk under build/bin")


def kill(name):
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/IM", name], capture_output=True)


def one_shot(exe, model, prompt, extra, env):
    cmd = [exe, "--native", model, "--prompt", prompt, "--n", str(N_NEW)] + extra
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
    ids = []
    for line in p.stdout.splitlines():
        line = line.strip()
        if line and re.fullmatch(r"[0-9,]+", line):
            ids += [int(x) for x in line.split(",") if x]
    return ids, p.stdout + p.stderr


def detok(exe, model, ids):
    if not ids:
        return ""
    p = subprocess.run([exe, "--native", model, "--detok", "--tokens", ",".join(str(i) for i in ids)],
                       capture_output=True, text=True, timeout=600)
    return p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag")
    ap.add_argument("--out", default=os.path.join("build", "textgate"))
    ap.add_argument("--compat", default="")
    ap.add_argument("--companion-dir", default="",
                    help="STRATA_VK_COMPANION_DIR for the run: the engine's automatic BF16 -> F16 companion "
                         "changes the text, so an A/B has to keep it on the same side for both runs")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    exe = engine(root)
    os.makedirs(a.out, exist_ok=True)

    if a.compare:
        same = 0
        for i in range(1, len(PROMPTS) + 1):
            t = [os.path.join(a.out, f"{tag}-p{i}.txt") for tag in a.compare]
            if not all(os.path.exists(p) for p in t):
                print(f"p{i}: missing a run")
                continue
            s = [open(p, encoding="utf-8", newline="").read() for p in t]
            same += s[0] == s[1]
            print(f"p{i}: text {'SAME' if s[0] == s[1] else 'DIFFERS'}")
            if s[0] != s[1]:
                k = next((k for k, (x, y) in enumerate(zip(*s)) if x != y), min(len(s[0]), len(s[1])))
                print(f"    at char {k}: {s[0][max(0, k - 30):k + 30]!r}")
                print(f"              {s[1][max(0, k - 30):k + 30]!r}")
        print(f"\n{same}/{len(PROMPTS)} prompts with identical text")
        return 0 if same == len(PROMPTS) else 1

    if not a.tag:
        sys.exit("--tag or --compare")
    env = dict(os.environ)
    if a.companion_dir:
        env["STRATA_VK_COMPANION_DIR"] = os.path.abspath(a.companion_dir)
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v
    extra = ["--compat", os.path.abspath(a.compat)] if a.compat else []
    for i, prompt in enumerate(PROMPTS, 1):
        kill(os.path.basename(exe))
        ids, log = one_shot(exe, a.model, prompt, extra, env)
        text = detok(exe, a.model, ids)
        open(os.path.join(a.out, f"{a.tag}-p{i}.txt"), "w", encoding="utf-8", newline="").write(text)
        open(os.path.join(a.out, f"{a.tag}-p{i}.ids"), "w", encoding="utf-8").write(",".join(map(str, ids)))
        print(f"== {a.tag} p{i}: {len(ids)} tokens")
        print(text[:400])
        sys.stdout.flush()


if __name__ == "__main__":
    sys.exit(main())
