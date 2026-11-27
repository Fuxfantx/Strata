#!/usr/bin/env python3
"""Price one engine benchmark in the platform's energy counters.

    python port/tools/energy-ab.py build/bin/strata-vk.exe MODEL.gguf -- \
        --batch-slots 1 --bench-tg 512 + --batch-slots 32 --bench-tg 16

The `--` is argparse's: everything after it is an engine argument, including
the `+` that separates the two runs.

The engine prints `bench window <t0> <t1>` for a `--batch-slots` run, and
port/tools/energy-sample.ps1 writes `epoch,pwr_sum,samples` once a second. This
runs both at the same time, cuts the counter samples to the engine's own window
and reports the mean and the energy per 1000 generated tokens.

The counters are uncalibrated: `pwr_sum` is a sum over the Energy Meter
instances, not a watt. Only the ratio between two runs of this same probe is
meaningful, so both runs are given on one command line and compared directly.
The runs must generate the same number of tokens, which is what makes their
ratio a J/token ratio rather than a rate difference.
"""
from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time

WINDOW = re.compile(r"bench window ([0-9.]+) ([0-9.]+)")
SLOTS = re.compile(r"slots (\d+): (\d+) tokens in ([0-9.]+) ms")

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLER = os.path.join(HERE, "energy-sample.ps1")


def read_csv(path: str) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    with open(path, "r", encoding="ascii", errors="replace") as fh:
        for line in fh:
            cols = line.strip().split(",")
            if len(cols) == 3 and cols[0] != "epoch":
                rows.append((float(cols[0]), float(cols[1])))
    return rows


def sample_once(exe: str, model: str, args: list[str], seconds: int, csv: str) -> dict:
    """Run the engine once under the sampler and return the window and its samples."""
    sampler = subprocess.Popen(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", SAMPLER,
         "-Seconds", str(seconds), "-Out", csv],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5)   # some idle samples before the process starts
    try:
        p = subprocess.run([exe, "--native", model, *args], capture_output=True, text=True, errors="replace")
    finally:
        sampler.kill()
        sampler.wait(timeout=10)
    log = (p.stdout or "") + (p.stderr or "")
    w = WINDOW.search(log)
    s = SLOTS.search(log)
    if not w or not s:
        raise SystemExit("no bench window in the engine log:\n" + log[-2000:])
    t0, t1 = float(w.group(1)), float(w.group(2))
    inside = [v for (t, v) in read_csv(csv) if t0 <= t <= t1]
    return {"slots": int(s.group(1)), "tokens": int(s.group(2)), "ms": float(s.group(3)),
            "window_s": t1 - t0, "samples": len(inside),
            "mean": statistics.fmean(inside) if inside else 0.0}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("exe")
    ap.add_argument("model")
    ap.add_argument("--timeout", type=int, default=1200, help="sampler lifetime, must exceed one whole run")
    ap.add_argument("runs", nargs="+", help="engine args for the first run, then `+`, then the engine args for the next")
    a = ap.parse_args(argv)

    groups: list[list[str]] = [[]]
    for tok in a.runs:
        (groups.append([]) if tok == "+" else groups[-1].append(tok))
    groups = [g for g in groups if g]

    results = []
    with tempfile.TemporaryDirectory() as d:
        for i, g in enumerate(groups):
            r = sample_once(a.exe, a.model, g, a.timeout, os.path.join(d, f"run{i}.csv"))
            results.append(r)
            print(f"run {i + 1}: slots {r['slots']}, {r['tokens']} tokens, {r['ms'] / 1000.0:.2f} s, "
                  f"{r['samples']} power samples in the window, mean {r['mean']:.4f}")

    if len(results) == 2:
        one, two = results
        # Energy is mean x seconds, so per 1000 tokens it is mean x seconds / tokens x 1000.
        e1 = one["mean"] * one["window_s"] / one["tokens"] * 1000.0
        e2 = two["mean"] * two["window_s"] / two["tokens"] * 1000.0
        print(f"energy per 1000 tokens: {e1:.4f} -> {e2:.4f} = {e1 / e2:.2f}x less")
        print(f"mean power in the window: {one['mean']:.4f} -> {two['mean']:.4f} = {two['mean'] / one['mean']:.2f}x")
        print(f"aggregate throughput: {one['tokens'] * 1000.0 / one['ms']:.2f} -> "
              f"{two['tokens'] * 1000.0 / two['ms']:.2f} tok/s = "
              f"{(two['tokens'] / two['ms']) / (one['tokens'] / one['ms']):.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
