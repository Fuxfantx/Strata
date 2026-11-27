"""Diagnose speculative decoding on a real long prompt: acceptance rate and where the cost goes.

The earlier lookup drafter lost 8% and the reason was never established: either the acceptance rate is too low to pay for the
verification pass, or the rollback after a rejection costs more than the accepted tokens save.  STRATA_VK_PROFILE prints one
line per draft attempt - the verification pass, the rollback, and how many drafts were kept - which answers both.
"""
import os
import re
import subprocess

EXE = r'build\bin\strata-vk.exe'
PACK = r'C:\Users\Fuxfantx\C-Res\Qwen\3.8-Flash-Next-GS\Gyro-S-TQ1_0.gguf'
PROMPT = open(r'build\prompt-code.txt', encoding='utf-8').read()


def run(lookup, profile=False, n=32):
    env = dict(os.environ)
    if profile:
        env['STRATA_VK_PROFILE'] = '1'
    out = subprocess.run([EXE, '--native', PACK, '--prompt', PROMPT, '--n', str(n), '--lookup', lookup],
                         capture_output=True, text=True, env=env)
    steady = re.search(r'steady ([\d.]+) over (\d+)', out.stdout)
    drafts = re.findall(r'draft: pass (\d+) tok = ([\d.]+) ms, rollback ([\d.]+) ms, kept (\d+)/(\d+)', out.stderr)
    return (float(steady.group(1)) if steady else float('nan')), drafts, out


for label, lookup in (('A lookup off', 'off'), ('B lookup on', 'on')):
    rate, drafts, _ = run(lookup, profile=True)
    print(f'{label}: steady {rate:5.2f} tok/s, {len(drafts)} draft attempts')
    if drafts:
        kept = sum(int(d[3]) for d in drafts)
        total = sum(int(d[4]) for d in drafts)
        pass_ms = sum(float(d[1]) for d in drafts)
        roll_ms = sum(float(d[2]) for d in drafts)
        print(f'   kept {kept}/{total} = {100 * kept / max(total, 1):.1f}% accepted')
        print(f'   verification {pass_ms / len(drafts):.1f} ms/attempt, rollback {roll_ms / len(drafts):.1f} ms/attempt')
        print(f'   drafts per attempt {total / len(drafts):.1f}')
