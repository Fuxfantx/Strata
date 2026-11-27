"""Measure the CPU+GPU+NPU hybrid placements the engine supports, on prefill and decode.

UMA means the devices share one DRAM, so a hybrid cannot add bandwidth; it can only add compute to work that runs in
parallel.  The engine's placement modes are the only shapes llama.cpp's scheduler can build (`--npu head` puts the output
head on the HTP, `--npu split --npu-layers N` the last N blocks plus the head, `dense`/`all` everything), so this matrix
is the honest answer to "is there a hybrid that beats GPU-only here".
"""
import os
import re
import subprocess

EXE = r"build\bin\strata-vk.exe"
MODELS = {
    "27b": r"C:\Users\Fuxfantx\C-Res\Qwen\3.8-27B-Q4\Qwen3.8-27B-Q4_0.gguf",
    "moe": r"C:\Users\Fuxfantx\C-Res\Qwen\3.6-35A3\Qwen3.6-35B-A3B-Q4_0.gguf",
}
CONFIGS = [
    ("base", []),
    ("head", ["--npu", "head"]),
    ("split4", ["--npu", "split", "--npu-layers", "4"]),
    ("split8", ["--npu", "split", "--npu-layers", "8"]),
    ("split16", ["--npu", "split", "--npu-layers", "16"]),
    ("dense", ["--npu", "dense"]),
]

env = dict(os.environ)
env["STRATA_VK_NPU_FORCE"] = "1"
env["GGML_HEXAGON_RT_REPACK"] = "0"

for model, path in MODELS.items():
    print(f"== {model}", flush=True)
    for name, extra in CONFIGS:
        row = [name]
        for flag, n in (("--bench-pp", 512), ("--bench-tg", 32 if model == "moe" else 16)):
            try:
                r = subprocess.run([EXE, "--native", path, flag, str(n)] + extra,
                                   capture_output=True, text=True, env=env, timeout=240)
                m = re.search(r"(pp|tg) \d+ tokens in [\d.]+ ms = ([\d.]+) tok/s", r.stdout)
                row.append(m.group(2) if m else "n/a")
                if not m:
                    line = [l for l in (r.stdout + r.stderr).splitlines() if "NPU" in l or "error" in l]
                    row.append((line[-1][:90] if line else "no output"))
            except subprocess.TimeoutExpired:
                row.append("timeout")
                row.append("")
        print("   %-8s pp %8s   tg %8s   %s" % (row[0], row[1], row[2], row[3] if len(row) > 3 else ""), flush=True)
