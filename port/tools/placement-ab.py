"""The engine's three implemented-but-never-measured placement knobs, and whether they add bandwidth.

Every measurement so far ran everything on the GPU and left the CPU and the Hexagon NPU idle.  The engine has flags for all three
(--npu, --n-cpu-moe, --split-cpu) and none of them has been measured with the current build, so "use the whole machine" is the one
implementation gap left.  On a UMA machine the question is not whether the units share DRAM - they do - but whether the GPU's
kernels leave bandwidth unused, in which case a second engine working concurrently fills it.
"""
import os
import re
import subprocess

EXE = r'build\bin\strata-vk.exe'
PACK = r'C:\Users\Fuxfantx\C-Res\Qwen\3.8-Flash-Next-GS\Gyro-S-TQ1_0.gguf'

CASES = [
    ('A default (GPU only)      ', []),
    ('B --npu dense             ', ['--npu', 'dense']),
    ('C --npu split, 8 layers   ', ['--npu', 'split', '--npu-layers', '8']),
    ('D --n-cpu-moe 12          ', ['--n-cpu-moe', '12']),
    ('E --split-cpu 0.3         ', ['--split-cpu', '0.3']),
]


def run(extra):
    out = subprocess.run([EXE, '--native', PACK, '--tokens', '9707', '--n', '8'] + extra,
                         capture_output=True, text=True, env=dict(os.environ))
    steady = re.search(r'steady ([\d.]+) over (\d+)', out.stdout)
    placement = [l.strip() for l in out.stderr.splitlines()
                 if any(k in l for k in ('NPU:', 'expert policy', 'split', 'placed'))]
    return (float(steady.group(1)) if steady else float('nan')), placement[:3], out


for label, extra in CASES:
    rate, placement, out = run(extra)
    print(f'{label}: steady {rate:5.2f} tok/s')
    for line in placement:
        print('     ', line[:130])
    if rate != rate:  # nan
        tail = [l for l in out.stderr.splitlines() if 'error' in l.lower() or 'fail' in l.lower()][:2]
        for line in tail:
            print('      !', line.strip()[:130])
