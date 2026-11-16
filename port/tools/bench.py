#!/usr/bin/env python3
"""Interleaved A/B benchmark for the Strata engine.

Single runs on this part vary by +-20%, so every configuration is run once per round and the rounds are interleaved:
thermal drift and cache state then hit all configurations equally, and the reported number is the median per config.

    python port/tools/bench.py --model C:/.../Tiny-Qwen.gguf --rounds 3 --tg 64 \
        --config "vulkan||" --config "opencl|STRATA_VK_GPU=opencl|"

Each --config is `name|ENV=VAL;ENV=VAL|extra args`.  `--pp N` measures prompt processing instead of decode, and
`--metric` picks which parsed line to report (default: the one the mode produces).
"""
from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_config(spec: str):
    parts = spec.split("|")
    name = parts[0]
    env = {}
    if len(parts) > 1 and parts[1].strip():
        for item in parts[1].split(";"):
            item = item.strip()
            if not item or "=" not in item:
                continue
            k, v = item.split("=", 1)
            env[k.strip()] = v.strip()
    args = parts[2].split() if len(parts) > 2 and parts[2].strip() else []
    return name, env, args


PP_RE = re.compile(r"pp (\d+) tokens in ([\d.]+) ms = ([\d.]+) tok/s")
TG_RE = re.compile(r"tg (\d+) tokens in ([\d.]+) ms = ([\d.]+) tok/s")


def run(exe, model, env_extra, extra_args, mode, n, timeout):
    env = dict(os.environ)
    env.update(env_extra)
    cmd = [exe, "--native", model]
    if mode == "pp":
        cmd += ["--bench-pp", str(n)]
    else:
        cmd += ["--bench-tg", str(n)]
    cmd += extra_args
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    out = proc.stdout + proc.stderr
    m = (PP_RE if mode == "pp" else TG_RE).search(proc.stdout)
    if m is None:
        return None, out, time.time() - t0
    return float(m.group(3)), out, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=os.path.join(ROOT, "build", "bin", "strata-vk.exe"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--pp", type=int, default=0, help="measure prompt processing over N tokens")
    ap.add_argument("--tg", type=int, default=0, help="measure decode over N tokens")
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    mode = "pp" if a.pp else "tg"
    n = a.pp or a.tg or 64
    configs = [parse_config(c) for c in a.config] or [("default", {}, [])]

    results = {name: [] for name, _, _ in configs}
    failures = {name: "" for name, _, _ in configs}
    for r in range(a.rounds):
        for name, env, args in configs:
            rate, out, secs = run(a.exe, a.model, env, args, mode, n, a.timeout)
            if rate is None:
                failures[name] = out.strip().splitlines()[-1] if out.strip() else "no output"
                print(f"  round {r + 1} {name:<22} FAILED  {failures[name][:110]}")
                continue
            results[name].append(rate)
            extra = ""
            if a.verbose:
                keep = [ln for ln in out.splitlines() if "INFO" in ln or "offload" in ln or "load_tensors" in ln]
                extra = " | " + keep[-1][:100] if keep else ""
            print(f"  round {r + 1} {name:<22} {rate:8.2f} tok/s  ({secs:5.1f}s){extra}")

    print(f"\n{mode} ({n} tokens), medians over {a.rounds} interleaved rounds:")
    base = None
    rows = []
    for name, _, _ in configs:
        vals = results[name]
        if not vals:
            rows.append((name, None, None))
            continue
        med = statistics.median(vals)
        if base is None:
            base = med
        rows.append((name, med, med / base * 100.0 if base else None))
    width = max(len(n) for n, _, _ in rows)
    for name, med, rel in rows:
        if med is None:
            print(f"  {name:<{width}}  failed")
        else:
            print(f"  {name:<{width}}  {med:8.2f} tok/s   {rel:6.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
