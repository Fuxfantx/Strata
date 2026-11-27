#!/usr/bin/env python3
"""Drive the serve protocol: one solo GEN, then N BGEN slots in the same process.

    python port/tools/batch-serve.py build/bin/strata-vk.exe MODEL.gguf --slots 32 --tokens 64

One engine launch, one copy of the weights, two measurements taken on either
side of the same load: `solo` sends a single `GEN` and times it, `batch` sends
`--slots` `BGEN` lines and times the interleaved stream until the last `BDONE`.
The pair is therefore the same weights, the same process and the same prompt
shape, which is what `--parallel` has to be judged on.

The prompts are synthetic token ids: this measures the engine, not the answer.
"""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time

PROMPT = 48
DUMP = False      # --dump: print every protocol line, for reading a burst by hand


def ids_for(slot: int, n: int = PROMPT) -> list[int]:
    """A deterministic prompt of plain ids, different per slot so the experts differ."""
    return [(1000 + slot * 37 + i * 11) % 20000 + 100 for i in range(n)]


def csv(ids: list[int]) -> str:
    return ",".join(str(i) for i in ids)


class Peer:
    def __init__(self, exe: str, model: str, engine: list[str]) -> None:
        self.cmd = [exe, "--serve", "--native", model, *engine]
        self.p = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self.p.stdout:
            self.q.put(line.strip())
        self.q.put(None)

    def send(self, line: str) -> None:
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def wait_ready(self, timeout: float = 300.0) -> dict:
        info: dict = {}
        end = time.time() + timeout
        while time.time() < end:
            try:
                line = self.q.get(timeout=1.0)
            except queue.Empty:
                continue
            if line is None:
                raise SystemExit("the engine ended before READY")
            if line.startswith("INFO "):
                for word in line.split()[1:]:
                    if "=" in word:
                        k, v = word.split("=", 1)
                        info[k] = v
            if line.startswith("READY"):
                info["ready"] = line
                return info
        raise SystemExit("no READY within the timeout")

    def close(self) -> None:
        try:
            self.send("QUIT")
            self.p.wait(timeout=30)
        except Exception:
            self.p.kill()


def solo(peer: Peer, tokens: int) -> tuple[list[int], float, float]:
    """Returns the tokens, the wall time, and the mean gap between two tokens (prefill excluded)."""
    peer.send(f"GEN {tokens} {csv(ids_for(0))}")
    toks: list[int] = []
    stamps: list[float] = []
    t0, elapsed = time.time(), 0.0
    while True:
        line = peer.q.get()
        if line is None:
            raise SystemExit("the engine ended during GEN")
        if DUMP:
            print("RAW", line)
        if line.startswith("T "):
            toks.append(int(line.split()[1]))
            stamps.append(time.time())
        elif line.startswith("DONE "):
            elapsed = time.time() - t0
            break
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    return toks, elapsed, (sum(gaps) / len(gaps) if gaps else 0.0)


def batch(peer: Peer, slots: int, tokens: int) -> tuple[int, float, list[int], float, list[int]]:
    t0 = time.time()
    for s in range(slots):
        peer.send(f"BGEN {s} {tokens} {csv(ids_for(s))}")
    per_slot = [0] * slots
    admitted, done = 0, 0
    first: dict[int, int] = {}        # the admission's token, per slot
    stream0: list[int] = []           # slot 0's `BT` tokens: the same request the solo run made
    stamps: list[float] = []          # every `BT` of slot 0: one per step, so their gaps are the step time
    while done < slots:
        line = peer.q.get()
        if line is None:
            raise SystemExit("the engine ended during BGEN")
        if DUMP:
            print("RAW", line)
        if line.startswith("T "):
            first.setdefault(admitted, int(line.split()[1]))   # admissions answer in send order
        elif line.startswith("BT "):
            slot = int(line.split()[1])
            per_slot[slot] += 1
            if slot == 0:
                stream0.append(int(line.split()[2]))
                stamps.append(time.time())
        elif line.startswith("BADM "):
            admitted += 1
        elif line.startswith("BDONE "):
            done += 1
        elif line.startswith("ERR "):
            raise SystemExit("the engine refused a BGEN: " + line)
    elapsed = time.time() - t0
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    stream0 = ([first[0]] if 0 in first else []) + stream0
    return sum(per_slot) + admitted, elapsed, per_slot, (sum(gaps) / len(gaps) if gaps else 0.0), stream0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("exe")
    ap.add_argument("model")
    ap.add_argument("--slots", type=int, default=32)
    ap.add_argument("--dump", action="store_true", help="print every protocol line")
    ap.add_argument("--tokens", type=int, default=64, help="tokens per request")
    ap.add_argument("--engine", nargs=argparse.REMAINDER, default=[],
                    help="engine arguments, after `--engine`")
    a = ap.parse_args(argv)
    global DUMP
    DUMP = a.dump

    peer = Peer(a.exe, a.model, a.engine)
    try:
        info = peer.wait_ready()
        print(f"engine says batch_slots={info.get('batch_slots', '0')}")
        if int(info.get("batch_slots", "0")) < a.slots:
            raise SystemExit(f"the engine advertised {info.get('batch_slots', '0')} slots, {a.slots} asked for")
        solo_toks, t1, g1 = solo(peer, a.tokens)
        rate_s = 1.0 / g1
        n1 = len(solo_toks)
        print(f"solo:  {n1} tokens in {t1:.2f} s = {n1 / t1:.2f} tok/s wall, {rate_s:.2f} tok/s per step"
              f" ({g1 * 1000.0:.0f} ms/token)")
        nb, tb, per, gb, stream0 = batch(peer, a.slots, a.tokens)
        print(f"batch: {nb} tokens in {tb:.2f} s = {nb / tb:.2f} tok/s wall, over {a.slots} slots"
              f" ({min(per)}-{max(per)} per slot)")
        rate_b = a.slots / gb
        print(f"per step: {gb * 1000.0:.0f} ms for {a.slots} tokens = {rate_b:.2f} tok/s aggregate, "
              f"{rate_b / rate_s:.2f}x the solo rate")
        # The same request ran alone and then in slot 0 beside 31 others: greedy decoding makes the two streams
        # comparable, so this is the cheapest check that a slot's state is its own.
        same = sum(1 for x, y in zip(solo_toks, stream0) if x == y)
        print(f"slot 0 against the solo run: {same}/{min(len(solo_toks), len(stream0))} tokens identical")
        bad = next((i for i, (x, y) in enumerate(zip(solo_toks, stream0)) if x != y), None)
        if bad is not None:
            print(f"  first difference at {bad}: solo {solo_toks[bad]} against batch {stream0[bad]}")
    finally:
        peer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
