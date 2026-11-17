"""Why no draft attempts: print what the engine says about its speculation state."""
import os
import re
import subprocess

EXE = r'build\bin\strata-vk.exe'
PACK = r'C:\Users\Fuxfantx\C-Res\Qwen\3.8-Flash-Next-GS\Gyro-S-TQ1_0.gguf'
PROMPT = open(r'build\prompt-code.txt', encoding='utf-8').read()

env = dict(os.environ)
env['STRATA_VK_PROFILE'] = '1'
out = subprocess.run([EXE, '--native', PACK, '--prompt', PROMPT, '--n', '64', '--lookup', 'on'],
                     capture_output=True, text=True, env=env)
print('--- stdout tail ---')
for line in out.stdout.splitlines()[-6:]:
    print(' ', line[:120])
print('--- spec / lookup related stderr ---')
for line in out.stderr.splitlines():
    low = line.lower()
    if any(k in low for k in ('lookup', 'spec', 'draft', 'cache refused', 'greedy')):
        print(' ', line.strip()[:150])
print('--- all engine lines ---')
for line in out.stderr.splitlines():
    if line.startswith('[strata-vk]'):
        print(' ', line.strip()[:150])
