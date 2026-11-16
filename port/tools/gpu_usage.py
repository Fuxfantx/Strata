"""Sample the Windows GPU engine counters while a GPU-only decode runs, so "GPU 3D 15%" can be put next to what the
decode actually asks of the device: tokens/s, bytes/token, and the implied DRAM rate.

Windows exposes per-engine utilization through the GPU Engine performance counter set (one instance per process and
engine, e.g. pid_1234_luid_..._phys_0_eng_0_engtype_3D). The counter reports the share of the sampling interval during
which the engine had work, which for a bandwidth-bound MoE decode is low even when DRAM is saturated.
"""
import os
import pathlib
import subprocess
import sys
import time

EXE = r"build\bin\strata-vk.exe"
MODEL = r"C:\Users\Fuxfantx\C-Res\Qwen\3.6-35A3\Qwen3.6-35B-A3B-Q4_0.gguf"


def sample(seconds_warmup, seconds_sample, n_tokens):
    log = pathlib.Path(os.environ["TEMP"]) / "gpuusage.log"
    with open(log, "wb") as f:
        p = subprocess.Popen([EXE, "--native", MODEL, "--prompt", "The quick brown fox", "--n", str(n_tokens)],
                             stdout=f, stderr=subprocess.STDOUT)
        time.sleep(seconds_warmup)
        samples = []
        t_end = time.time() + seconds_sample
        while time.time() < t_end:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-Counter '\\GPU Engine(*)\\Utilization Percentage' -ErrorAction SilentlyContinue).CounterSamples | "
                 "Where-Object { $_.InstanceName -match 'engtype_(3D|Compute)' } | "
                 "Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum"],
                capture_output=True, text=True)
            try:
                samples.append(float(out.stdout.strip() or 0.0))
            except ValueError:
                pass
            time.sleep(0.5)
        p.wait(timeout=60)
    text = log.read_text(encoding="utf-8", errors="replace")
    tokline = [l for l in text.splitlines() if "tok/s decode" in l]
    print(f"gpu engine utilization (3D+Compute, summed over engines), {len(samples)} samples:")
    if samples:
        samples.sort()
        print(f"  median {samples[len(samples)//2]:.1f}%  max {max(samples):.1f}%  min {min(samples):.1f}%")
    for l in tokline:
        print("  " + l.strip())


if __name__ == "__main__":
    sample(45, 20, 128)
