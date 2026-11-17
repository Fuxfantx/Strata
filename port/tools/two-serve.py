#!/usr/bin/env python3
"""Measure one or two `--serve` peers on the same pack, with the generations synchronised.

`serve/replicas.py` starts one engine per replica as `[exe, "--serve", *args]`, and a CPU replica is that same line with
`--gpu cpu`.  This script drives those command lines directly, so the aggregate of a Vulkan peer and a CPU peer can be
measured without the Python server (which needs a tokenizer pack this checkout does not carry).

The engines are waited on separately but their GENs are released together: the GPU engine takes a minute to load, so
sending each GEN as soon as its own peer says READY would leave the two never running at the same time and would measure
nothing.  The aggregate below is therefore tokens of both peers over the window in which both were generating.

    python port/tools/two-serve.py build/bin/strata-vk.exe MODEL.gguf --n 32 --engine-arg --load-mode --engine-arg mmap
    python port/tools/two-serve.py build/bin/strata-vk.exe MODEL.gguf --n 32 --solo cpu --engine-arg --load-mode --engine-arg mmap

Exit code 0 only when every peer answered the token count it was given.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time


class Peer:
    """One engine process, from spawn to QUIT."""

    def __init__(self, exe: str, model: str, args: list[str], label: str, release: threading.Event) -> None:
        self.label = label
        self.cmd = [exe, "--serve", "--native", model, *args]
        self.p: subprocess.Popen | None = None
        self.ready = threading.Event()
        self.gen_release = release
        self.done_at = 0.0
        self.send_at = 0.0
        self.seen = 0
        self.line = ""
        self.error = ""

    def start(self) -> None:
        self.p = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def reader(self, want_n: int) -> None:
        assert self.p is not None
        try:
            while True:
                line = (self.p.stdout.readline() or "").strip()
                if not line:
                    self.error = "the engine stopped before READY" if not self.ready.is_set() else "the engine stopped"
                    break
                if line.startswith("READY"):
                    self.ready.set()
                    continue
                if not self.ready.is_set():
                    continue
                if line.startswith("T "):
                    self.seen += 1
                    continue
                if line.startswith("DONE") or line.startswith("ERR"):
                    self.done_at = time.time()
                    self.line = line
                    break
        finally:
            self.ready.set()
            self.gen_release.set()

    def generate(self, want_n: int, ids: str) -> None:
        self.gen_release.wait()
        if self.p is None or self.p.poll() is not None:
            return
        self.send_at = time.time()
        try:
            self.p.stdin.write(f"GEN {want_n} {ids}\n")
            self.p.stdin.flush()
        except OSError as exc:
            self.error = f"could not send GEN: {exc}"

    def stop(self) -> None:
        if self.p is None:
            return
        try:
            if self.p.poll() is None:
                self.p.stdin.write("QUIT\n")
                self.p.stdin.flush()
                self.p.wait(timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            pass
        if self.p.poll() is None:
            self.p.kill()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exe")
    ap.add_argument("model")
    ap.add_argument("--tokens", default="9707", help="the input token ids for every engine (default 9707)")
    ap.add_argument("--n", type=int, default=32, help="tokens to generate from each engine (default 32)")
    ap.add_argument("--json", action="store_true", help="machine-readable result line")
    ap.add_argument("--engine-arg", action="append", default=[], metavar="ARG",
                    help="extra argument for every engine; repeat for each word (--engine-arg=--load-mode ...)")
    ap.add_argument("--vulkan-arg", action="append", default=[], metavar="ARG",
                    help="extra argument for the Vulkan peer only; repeat for each word")
    ap.add_argument("--cpu-arg", action="append", default=[], metavar="ARG",
                    help="extra argument for the CPU peer only; repeat for each word")
    # --solo picks one peer, so a single-engine rate can be measured through the same protocol and the same window.
    ap.add_argument("--solo", choices=["vulkan", "cpu"], help="run only this peer")
    ap.add_argument("--max-context", default="2048")
    a = ap.parse_args(argv)

    common = ["--max-context", a.max_context, *a.engine_arg]
    release = threading.Event()
    peers = [Peer(a.exe, a.model, [*common, *a.cpu_arg, "--gpu", "cpu"], "cpu", release),
             Peer(a.exe, a.model, [*common, *a.vulkan_arg], "vulkan", release)]
    if a.solo:
        peers = [p for p in peers if p.label == a.solo]

    readers = []
    for p in peers:
        try:
            p.start()
        except OSError as exc:
            print(f"[fail] {p.label}: could not start: {exc}", file=sys.stderr)
            return 2
        t = threading.Thread(target=p.reader, args=(a.n,), daemon=True)
        t.start()
        readers.append(t)

    # Wait for every peer to be ready before releasing the GENs, so the two windows overlap.
    deadline = time.time() + 900
    for p in peers:
        if not p.ready.wait(timeout=max(1.0, deadline - time.time())):
            print(f"[fail] {p.label}: no READY within the load budget", file=sys.stderr)
            for q in peers:
                q.stop()
            return 2
        if p.error:
            print(f"[fail] {p.label}: {p.error}", file=sys.stderr)
            for q in peers:
                q.stop()
            return 2

    gens = [threading.Thread(target=p.generate, args=(a.n, a.tokens), daemon=True) for p in peers]
    t_release = time.time()
    for t in gens:
        t.start()
    # One event releases every GEN, so they leave together and the measured window really is shared.
    release.set()
    for t in readers:
        t.join(timeout=1800)
    for t in gens:
        t.join(timeout=60)

    ok = True
    rates = {}
    for p in peers:
        if not p.line.startswith("DONE"):
            print(f"[fail] {p.label}: no DONE ({p.line or p.error or 'stopped'})", file=sys.stderr)
            ok = False
            continue
        if p.seen != a.n:
            print(f"[fail] {p.label}: {p.seen} T lines for GEN {a.n}", file=sys.stderr)
            ok = False
            continue
        dt = p.done_at - p.send_at
        rates[p.label] = a.n / dt if dt > 0 else 0.0
        print(f"[{p.label}] {p.line}  ({dt * 1000:.0f} ms for {a.n} = {rates[p.label]:.2f} tok/s)")

    for p in peers:
        p.stop()

    if not ok:
        return 1

    end = max((p.done_at for p in peers), default=0.0)
    window = end - t_release
    tokens = a.n * len(peers)
    # Aggregate throughput is the SUM of the peers' rates: two engines serving continuously each add their own stream, and
    # a fixed-size request pair makes the faster one idle once it is done.  Dividing by the shared window instead would
    # report the slower peer's rate and hide the gain (see features/HYBRID.md for the same convention).
    agg = sum(rates.values())
    if len(peers) == 1:
        print(f"solo {peers[0].label}: {rates[peers[0].label]:.2f} tok/s")
    else:
        print(f"concurrent: {', '.join(f'{k} {v:.2f}' for k, v in sorted(rates.items()))} "
              f"-> {agg:.2f} tok/s aggregate (both finished in {window:.2f} s for {tokens} tokens)")
    if a.json:
        import json
        # Absolute epochs let an energy sampler started alongside align its samples with exactly this window.
        print(json.dumps({"rates": rates, "aggregate": agg, "tokens": tokens, "window_s": window,
                          "release_epoch": t_release, "end_epoch": end,
                          "done_epoch": {p.label: p.done_at for p in peers}}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
