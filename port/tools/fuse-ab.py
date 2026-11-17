"""Run the two fusion switches against the default, interleaved, and report the steady rates.
Run the two fusion switches against the default, interleaved, and report the steady rates.  NOTE: the switches
themselves are four lines at the two fusion sites in the backend (see GYRO-S-DECODE.md section 11.5); they are not
left in a default build, so re-add them to run this instrument.
STRATA_VK_NO_MULTI_ADD=1 turns off the multi-add fusion, STRATA_VK_NO_HC=1 the hyper-connection gate fusion.  Turning one
off and timing the whole step is the only honest way to price a fusion whose row in the per-op table cannot be trusted.
"""
import os
import re
import subprocess
import sys

EXE = r'build\bin\strata-vk.exe'
PACK = r'C:\Users\Fuxfantx\C-Res\Qwen\3.8-Flash-Next-GS\Gyro-S-TQ1_0.gguf'
N = sys.argv[1] if len(sys.argv) > 1 else '8'


def run(env_extra):
    env = dict(os.environ)
    env.update(env_extra)
    out = subprocess.run([EXE, '--native', PACK, '--tokens', '9707', '--n', N],
                         capture_output=True, text=True, env=env)
    m = re.search(r'steady ([\d.]+) over (\d+)', out.stdout)
    return float(m.group(1)) if m else float('nan')


cases = [
    ('A default        ', {}),
    ('B no multi-add   ', {'STRATA_VK_NO_MULTI_ADD': '1'}),
    ('C no hc fusion   ', {'STRATA_VK_NO_HC': '1'}),
    ('B no multi-add   ', {'STRATA_VK_NO_MULTI_ADD': '1'}),
    ('C no hc fusion   ', {'STRATA_VK_NO_HC': '1'}),
    ('A default        ', {}),
]

results = {}
for name, env in cases:
    r = run(env)
    results.setdefault(name.strip(), []).append(r)
    print(f'{name} {r:6.2f} tok/s')

print()
base = sum(results['A default']) / len(results['A default'])
for key in ('B no multi-add', 'C no hc fusion'):
    vals = results[key]
    mean = sum(vals) / len(vals)
    print(f'{key}: {mean:6.2f} vs default {base:6.2f}  -> {100 * (mean / base - 1):+.1f}%')
