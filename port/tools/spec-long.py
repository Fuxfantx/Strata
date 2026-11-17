"""Measure speculative acceptance on a long repetitive token prompt, which is what the suffix drafter is for.

A raw code string went through the chat template and hit EOG after three tokens, so the drafter had nothing to copy and no
draft was ever attempted.  A long token prompt built by repeating the known eight-token pattern gives the drafter a history
with exactly the repetition it looks for, and keeps the run on the deterministic path.
"""
import os
import re
import subprocess

EXE = r'build\bin\strata-vk.exe'
PACK = r'C:\Users\Fuxfantx\C-Res\Qwen\3.8-Flash-Next-GS\Gyro-S-TQ1_0.gguf'
PATTERN = '1156,4439,883,279,1379,10919,18255,303'
TOKENS = ','.join([PATTERN] * 32)          # 256 prompt tokens

env = dict(os.environ)
env['STRATA_VK_PROFILE'] = '1'
out = subprocess.run([EXE, '--native', PACK, '--tokens', TOKENS, '--n', '64', '--lookup', 'on'],
                     capture_output=True, text=True, env=env)

steady = re.search(r'steady ([\d.]+) over (\d+)', out.stdout)
drafts = re.findall(r'draft: pass (\d+) tok = ([\d.]+) ms, rollback ([\d.]+) ms, kept (\d+)/(\d+)', out.stderr)
print('steady', steady.group(1) if steady else '?', 'tok/s   draft attempts', len(drafts))
if drafts:
    kept = sum(int(d[3]) for d in drafts)
    total = sum(int(d[4]) for d in drafts)
    print(f'accepted {kept}/{total} = {100 * kept / total:.1f}%')
    print(f'verification {sum(float(d[1]) for d in drafts) / len(drafts):.1f} ms, rollback {sum(float(d[2]) for d in drafts) / len(drafts):.1f} ms per attempt')
else:
    print('no draft attempts')
print('--- engine lines ---')
for line in out.stderr.splitlines():
    if 'lookup' in line.lower() or 'draft' in line.lower():
        print(' ', line.strip()[:150])
