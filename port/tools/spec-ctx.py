"""Measure speculative acceptance on a coherent long context, all through the token-id path.

The text prompt path is unusable for this pack, so the context is built the way the token path supports: generate 64 tokens from
the known eight-token seed, then use seed + generated as the prompt of a second run.  That context is the model's own coherent text,
which is what a suffix drafter has to find copies in - and it is exactly the situation where a user would want the feature.
"""
import os
import re
import subprocess

EXE = r'build\bin\strata-vk.exe'
PACK = r'C:\Users\Fuxfantx\C-Res\Qwen\3.8-Flash-Next-GS\Gyro-S-TQ1_0.gguf'
SEED = '1156,4439,883,279,1379,10919,18255,303'


def run(tokens, n, extra_env=None, lookup='off'):
    env = dict(os.environ)
    env['STRATA_VK_PROFILE'] = '1'
    if extra_env:
        env.update(extra_env)
    out = subprocess.run([EXE, '--native', PACK, '--tokens', tokens, '--n', str(n), '--lookup', lookup],
                         capture_output=True, text=True, env=env)
    steady = re.search(r'steady ([\d.]+) over (\d+)', out.stdout)
    first = out.stdout.split('\n')[0].strip()
    return (float(steady.group(1)) if steady else None), first, out


# 1) build a coherent context with the token path
rate, generated, out = run(SEED, 64, lookup='off')
ids = [t for t in generated.split(',') if t.strip().isdigit()]
context = SEED + ',' + ','.join(ids)
print(f'step 1: {rate} tok/s, {len(ids)} generated tokens')
print(f'   first 20: {",".join(ids[:20])}')

# 2) speculate on that context
for label, lookup in (('A lookup off', 'off'), ('B lookup on', 'on'), ('B lookup on', 'on'), ('A lookup off', 'off')):
    rate, _, out = run(context, 32, extra_env={'STRATA_VK_LOOKUP_DEBUG': '1'}, lookup=lookup)
    drafts = re.findall(r'draft: pass (\d+) tok = ([\d.]+) ms, rollback ([\d.]+) ms, kept (\d+)/(\d+)', out.stderr)
    hits = sum(1 for l in out.stderr.splitlines() if '[lookup]' in l and 'miss' not in l)
    misses = sum(1 for l in out.stderr.splitlines() if '[lookup]' in l and 'miss' in l)
    if drafts:
        kept = sum(int(d[3]) for d in drafts)
        total = sum(int(d[4]) for d in drafts)
        print(f'{label}: {rate:5.2f} tok/s | {len(drafts)} attempts | kept {kept}/{total} = {100*kept/total:.0f}%'
              f' | pass {sum(float(d[1]) for d in drafts)/len(drafts):.0f} ms, rollback {sum(float(d[2]) for d in drafts)/len(drafts):.0f} ms'
              f' | lookup hits {hits} misses {misses}')
    else:
        print(f'{label}: {rate:5.2f} tok/s | no attempts | lookup hits {hits} misses {misses}')
