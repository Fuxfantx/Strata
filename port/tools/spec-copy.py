"""Measure speculative acceptance where a suffix drafter can actually win: a prompt that asks the model to reproduce text.

The repetitive token prompt degenerated to token zero, so it measured nothing.  This prompt asks the model to repeat a block
verbatim, which is the shape suffix drafting is built for - the continuation is a copy of the input - and it is also the shape
where a user notices if the feature works.
"""
import os
import re
import subprocess

EXE = r'build\bin\strata-vk.exe'
PACK = r'C:\Users\Fuxfantx\C-Res\Qwen\3.8-Flash-Next-GS\Gyro-S-TQ1_0.gguf'

BLOCK = '\n'.join([
    'The palm tree grows beside the harbour wall.',
    'Every morning the ferry leaves at seven sharp.',
    'A grey cat sleeps on the third mooring post.',
    'The lighthouse keeper writes the wind speed down.',
    'Nobody remembers who painted the blue door.',
])
PROMPT = f'Repeat the following five lines exactly, with no introduction and no commentary:\n{BLOCK}\n'

env = dict(os.environ)
env['STRATA_VK_PROFILE'] = '1'
env['STRATA_VK_LOOKUP_DEBUG'] = '1'
out = subprocess.run([EXE, '--native', PACK, '--prompt', PROMPT, '--n', '64', '--lookup', 'on'],
                     capture_output=True, text=True, env=env)

steady = re.search(r'steady ([\d.]+) over (\d+)', out.stdout)
hits = sum(1 for l in out.stderr.splitlines() if '[lookup]' in l and 'miss' not in l)
miss = sum(1 for l in out.stderr.splitlines() if '[lookup]' in l and 'miss' in l)
drafts = re.findall(r'draft: pass (\d+) tok = ([\d.]+) ms, rollback ([\d.]+) ms, kept (\d+)/(\d+)', out.stderr)
print('steady', steady.group(1) if steady else '?', 'tok/s')
print(f'lookup hits {hits}, misses {miss}')
print('draft attempts', len(drafts))
if drafts:
    kept = sum(int(d[3]) for d in drafts)
    total = sum(int(d[4]) for d in drafts)
    print(f'accepted {kept}/{total} = {100 * kept / max(total, 1):.1f}%')
    print(f'verification {sum(float(d[1]) for d in drafts) / len(drafts):.1f} ms, rollback {sum(float(d[2]) for d in drafts) / len(drafts):.1f} ms')
print('--- output ---')
print(out.stdout.split('\n')[0][:200])
