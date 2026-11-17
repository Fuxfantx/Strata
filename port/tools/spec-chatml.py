"""Drive the model the way a user does: through its chat template, so the continuation is coherent text.

The one-shot path tokenizes --prompt raw, with no chat template, so an instruct model given a bare instruction string
degenerates - which is what the token-zero runs were.  Wrapping the request in ChatML is what the served path does, and it is
the only way to measure speculative acceptance on text a user would actually get.
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
])
PROMPT = (f'<|im_start|>user\nRepeat these three lines exactly, no commentary:\n{BLOCK}<|im_end|>\n'
          f'<|im_start|>assistant\n')

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
print(f'lookup hits {hits}, misses {miss}, draft attempts {len(drafts)}')
if drafts:
    kept = sum(int(d[3]) for d in drafts)
    total = sum(int(d[4]) for d in drafts)
    print(f'accepted {kept}/{total} = {100 * kept / max(total, 1):.1f}%')
    print(f'verification {sum(float(d[1]) for d in drafts) / len(drafts):.1f} ms, rollback {sum(float(d[2]) for d in drafts) / len(drafts):.1f} ms')
print('output tokens:', out.stdout.split('\n')[0][:160])
