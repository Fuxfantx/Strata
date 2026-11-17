#!/usr/bin/env python3
"""End to end: one server, two replicas, one request each, at the same time.

This is the last unverified layer of the data-parallel replica path (port/docs/PERFORMANCE.md 9.4 and 15.2-15.4): the config
reaches the engine's argv (a unit test proves that), the engine speaks the line protocol under `--gpu cpu` (`two-serve.py`
proves that), and what is left is the server starting both and routing a request to each.

It needs a tokenizer directory (`--tokenizer`, produce one with `gguf-tokenizer.py`) because `serve/server.py` refuses to
start the strata engine without `<tokenizer>/vocab.json`, and it needs the engine and a pack:

    python port/tools/gguf-tokenizer.py MODEL.gguf --out pack/full/tokenizer
    python port/tools/e2e-replicas.py --exe build/bin/strata-vk.exe \
        --model tiny.gguf --tokenizer pack/full/tokenizer --out-json tmp-replicas.json

`--out-json` writes the server config it used, so the run can be repeated by hand.  The exit code is 0 only when both
requests answered and the log shows two replicas with one on `cpu`.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


def wait_ready(base: str, proc: subprocess.Popen, timeout_s: float, log: list) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        for path in ("/health", "/v1/models", "/"):
            try:
                with urllib.request.urlopen(base + path, timeout=2) as r:
                    if r.status < 500:
                        return True
            except (urllib.error.URLError, OSError):
                pass
        time.sleep(1.0)
    return False


def ask(base: str, prompt: str, n: int, out: dict, key: str) -> None:
    body = json.dumps({"model": "m", "max_tokens": n, "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            payload = json.loads(r.read())
        out[key] = {"s": time.monotonic() - t0, "text": payload["choices"][0]["message"]["content"],
                    "usage": payload.get("usage")}
    except Exception as e:                                   # noqa: BLE001 - reported, not raised
        out[key] = {"s": time.monotonic() - t0, "error": f"{type(e).__name__}: {e}"}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True, help="the engine program (strata-vk.exe)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default="pack/full/tokenizer")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--n", type=int, default=16, help="tokens per request")
    ap.add_argument("--args", default="", help="extra engine arguments, space separated")
    ap.add_argument("--startup-s", type=float, default=600.0, help="how long the two engines may take to load")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--cpu-via-cli", action="store_true",
                    help="name the CPU worker with the server's --replicas-cpu flag instead of the config's \"replicas\" list, "
                         "which is the form a person types (both must reach the same two engines)")
    a = ap.parse_args(argv)

    exe, model = os.path.abspath(a.exe), os.path.abspath(a.model)
    tok = os.path.abspath(a.tokenizer)
    if not os.path.exists(os.path.join(tok, "vocab.json")):
        print(f"{tok} has no vocab.json: run port/tools/gguf-tokenizer.py {a.model} --out {a.tokenizer}", file=sys.stderr)
        return 2
    cfg = {"exe": exe, "model_name": os.path.basename(model), "tokenizer": tok,
           "args": ["--serve", "--native", model, "--batch", "8"] + [x for x in a.args.split(" ") if x],
           "log": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "e2e-engine.log")}
    if a.cpu_via_cli:
        cfg["gpu"] = 0                                  # the single card; --replicas-cpu adds the CPU worker beside it
    else:
        cfg["replicas"] = [{"gpus": [0]}, {"backend": "cpu"}]
    if a.out_json:
        with open(a.out_json, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=1)
    base = f"http://127.0.0.1:{a.port}"
    cmd = [sys.executable, "-u", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "serve", "server.py"),
           "--engine", "strata", "--config", a.out_json or "-", "--port", str(a.port), "--no-api-key"]
    if a.cpu_via_cli:
        cmd += ["--replicas-cpu", "1"]
    if not a.out_json:
        print("--out-json is required: the server reads the config from a file", file=sys.stderr)
        return 2
    print(f"starting: {' '.join(cmd)}", flush=True)
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace")
    log: list[str] = []
    threading.Thread(target=lambda: [log.append(l) for l in proc.stdout], daemon=True).start()
    try:
        if not wait_ready(base, proc, a.startup_s, log):
            print("the server never became ready; its output:", file=sys.stderr)
            print("".join(log[-40:]), file=sys.stderr)
            return 1
        print(f"ready in {time.monotonic() - t0:.1f} s", flush=True)
        out: dict = {}
        ts = [threading.Thread(target=ask, args=(base, p, a.n, out, k)) for k, p in
              (("a", "Write one sentence about the sea."), ("b", "Write one sentence about a mountain."))]
        for t in ts:
            t.start()
        t1 = time.monotonic()
        for t in ts:
            t.join()
        wall = time.monotonic() - t1
        ok = all("error" not in v for v in out.values())
        for k, v in sorted(out.items()):
            if "error" in v:
                print(f"  request {k}: FAILED after {v['s']:.2f} s: {v['error']}")
            else:
                print(f"  request {k}: {v['s']:.2f} s, {(v['text'] or '')[:60]!r}, usage {v['usage']}")
        joined = "".join(log)
        cpu_replica = 'cpu' in joined and "replicas on the cards" in joined
        print(f"both requests in {wall:.2f} s; two replicas with a cpu one: {cpu_replica}")
        for line in log:
            if "replicas on the cards" in line:
                print("  " + line.strip())
        return 0 if ok and cpu_replica else 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
