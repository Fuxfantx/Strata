"""Run the engine's NPU placement under a watchdog for a matrix of Hexagon backend knobs.

Every run also sets STRATA_VK_VERBOSE (the engine's own gate) and GGML_HEXAGON_VERBOSE, so the log shows the last
ggml-hex line before the hang - which is what says whether the DSP never took the packet or took it and stalled.
"""
import os
import pathlib
import subprocess
import sys
import time

EXE = r"build\bin\strata-vk.exe"
MODEL = r"C:\Users\Fuxfantx\C-Res\Qwen\3.6-35A3\Qwen3.6-35B-A3B-Q4_0.gguf"

BASE = {"STRATA_VK_VERBOSE": "1", "GGML_HEXAGON_VERBOSE": "1", "STRATA_VK_NPU_FORCE": "1"}

CONFIGS = [
    ("baseline",        {}),
    ("poll",            {"GGML_HEXAGON_OPPOLL": "1"}),
    ("no_fusion",       {"GGML_HEXAGON_OPFUSION": "0"}),
    ("no_repack",       {"GGML_HEXAGON_RT_REPACK": "0"}),
    ("no_fuse_no_repack", {"GGML_HEXAGON_OPFUSION": "0", "GGML_HEXAGON_RT_REPACK": "0"}),
    ("opqueue1",        {"GGML_HEXAGON_OPQUEUE": "1"}),
    ("arch79",          {"GGML_HEXAGON_ARCH": "79"}),
    ("arch85",          {"GGML_HEXAGON_ARCH": "85"}),
    ("one_hvx",         {"GGML_HEXAGON_NHVX": "1"}),
    ("no_hmx",          {"GGML_HEXAGON_NHMX": "0"}),
    ("trace",           {"GGML_HEXAGON_OPTRACE": "256"}),
]


def run(name, env_extra, timeout=150):
    env = dict(os.environ)
    env.update(BASE)
    env.update(env_extra)
    log = pathlib.Path(os.environ["TEMP"]) / f"npu_{name}.log"
    with open(log, "wb") as f:
        p = subprocess.Popen([EXE, "--native", MODEL, "--prompt", "capital", "--n", "1", "--npu", "head"],
                             stdout=f, stderr=subprocess.STDOUT, env=env)
        try:
            p.wait(timeout=timeout)
            hung = False
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
            hung = True
    text = log.read_text(encoding="utf-8", errors="replace").splitlines()
    hexlines = [l for l in text if "ggml-hex" in l or "opqueue" in l or "queue-opbatch" in l or "dsp" in l.lower()]
    status = "HUNG" if hung else "exited"
    print(f"=== {name}: {status} ===")
    for l in hexlines[-4:]:
        print("   " + l[:170])
    if status == "exited":
        tail = [l for l in text if l.strip()][-3:]
        for l in tail:
            print("   > " + l[:170])


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    for name, env_extra in CONFIGS:
        if only and only != name:
            continue
        run(name, env_extra)
