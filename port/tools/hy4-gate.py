#!/usr/bin/env python3
"""hy4-gate.py - ask a Hy4 pack one question with a knowable answer, and say whether it got it.

Every measurement in the port compares one reading of a pack against another, which says which is better but not whether either is
good. This asks something a person can check in a second: "1 + 1 =" and the model should continue with "2". It runs the engine twice -
once to generate, once to detokenise with the pack's own vocabulary - so the text is the model's, not a script's byte table.

    python port/tools/hy4-gate.py --engine build/bin/strata-vk.exe --model Hy4-flash-Uncensored-Compact-00001-of-00002.gguf

Exit code 0 when the answer starts with a 2, 1 when it does not, 2 when the run itself failed. Extra engine arguments can be passed
after `--` so a switch can be tried against the same gate:

    python port/tools/hy4-gate.py ... -- --gpu cpu
"""
import argparse
import os
import subprocess
import sys

# the turn: start marker, "user", middle marker, "1+1=", end marker, start marker, "assistant", middle marker
PROMPT = "120000,3717,120001,16,10,16,28,120025,120000,611,10372,120001"
N_GEN = 6


def run(engine, model, tokens, n, extra):
    cmd = [engine, "--native", model, "--tokens", tokens, "--n", str(n), "--max-context", "2048", "--gpu", "cpu"] + extra
    out = subprocess.run(cmd, capture_output=True)
    if out.returncode != 0:
        sys.stderr.write(out.stderr.decode("utf-8", "replace")[-2000:])
        return None
    return out.stdout.decode("utf-8", "replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=N_GEN)
    args, extra = ap.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]

    gen = run(args.engine, args.model, PROMPT, args.n, extra)
    if gen is None:
        print("the run failed")
        return 2

    ids = None
    for line in gen.splitlines():
        if line.strip().endswith(",") and line.strip().replace(",", "").isdigit():
            ids = line.strip().rstrip(",")
    if not ids:
        print("no token stream came back")
        return 2

    detok = run(args.engine, args.model, ids, 1, ["--detok"])
    if detok is None:
        print("the token stream could not be read back")
        return 2

    text = detok.strip()
    ok = text.lstrip().startswith("2")
    print("tokens: " + ids)
    print("text  : " + text)
    print("gate  : " + ("pass" if ok else "fail") + " (1+1 should continue with 2)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
