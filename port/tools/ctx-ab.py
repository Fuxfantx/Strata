"""Is the engine's context/KV configuration pushing the mapped pack out of RAM?

The engine creates a 32k context by default with an f16 KV cache, and this pack is 54.5 GiB mapped into a machine with about 40 GiB
free - the loader said the resident set (54.5 minus the 26.8 GiB table) fits, but that was before the KV cache existed.  If the KV
allocation evicts the dense weights, every token re-reads them from the SSD and the 39 GB/s effective bandwidth is a page-fault
number.  A smaller context, or a quantised KV, would show it immediately.
"""
import os
import re
import subprocess

EXE = r'build\bin\strata-vk.exe'
PACK = r'C:\Users\Fuxfantx\C-Res\Qwen\3.8-Flash-Next-GS\Gyro-S-TQ1_0.gguf'

CASES = [
    ('A default 32k f16', []),
    ('B 4k f16          ', ['-c', '4096']),
    ('C 32k q8_0        ', ['--kv', 'q8_0']),
    ('D 32k q4_0        ', ['--kv', 'q4_0']),
    ('E 4k q8_0         ', ['-c', '4096', '--kv', 'q8_0']),
]


def run(extra):
    out = subprocess.run([EXE, '--native', PACK, '--tokens', '9707', '--n', '16'] + extra,
                         capture_output=True, text=True, env=dict(os.environ))
    steady = re.search(r'steady ([\d.]+) over (\d+)', out.stdout)
    ctx = re.search(r'creating a (\d+)-token context \(batch (\d+), (\d+) threads, kv (\w+)\)', out.stderr)
    return (float(steady.group(1)) if steady else float('nan')), (ctx.groups() if ctx else None), out


for label, extra in CASES:
    rate, ctx, out = run(extra)
    print(f'{label}: steady {rate:5.2f} tok/s   context {ctx}')
