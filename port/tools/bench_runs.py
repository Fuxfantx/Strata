"""Run the engine N times and print the steady-state decode rate of each run plus the median.

The machine's run-to-run spread is about 4%, so a change has to be judged on the median of several runs rather than on
a single number (port/docs/OPTIMIZATION-BACKLOG.md, phase 4 round 5).
"""
import re
import statistics
import subprocess
import sys

exe = sys.argv[1]
pack = sys.argv[2]
runs = int(sys.argv[3]) if len(sys.argv) > 3 else 3
tokens = sys.argv[4] if len(sys.argv) > 4 else '9707'
n = sys.argv[5] if len(sys.argv) > 5 else '32'

rates = []
for i in range(runs):
    out = subprocess.run([exe, '--native', pack, '--tokens', tokens, '--n', n],
                         capture_output=True, text=True, timeout=900).stdout
    m = re.search(r'steady ([0-9.]+)', out)
    if not m:
        print('run', i, 'produced no rate')
        continue
    rates.append(float(m.group(1)))
    print(f'run {i + 1}: {rates[-1]:.2f} tok/s')

if rates:
    print(f'median: {statistics.median(rates):.2f} tok/s over {len(rates)} runs')
