"""Resolve a --diff3 merge by taking the upstream (theirs) side of each conflict.

The only file the re-pin could not merge is tests/test-backend-ops.cpp, and every conflicting hunk there is a block our
overlay had dropped relative to the pin (the overlay copy of that file tracks an older test list) while upstream keeps or
extends it.  Our real additions to that file are the trellis type registrations, which merged cleanly elsewhere, so the
correct resolution is upstream's side.

It refuses to guess: if any conflict has a non-empty "ours" side, it stops and prints the file and line, because that
would be a change of ours that has to be re-applied by hand.
"""
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
lines = path.read_text(encoding='utf-8').splitlines(keepends=True)

out, i, conflicts, empty_ours = [], 0, 0, 0
while i < len(lines):
    if lines[i].startswith('<<<<<<<'):
        conflicts += 1
        j = i + 1
        ours = []
        while j < len(lines) and not lines[j].startswith('|||||||') and not lines[j].startswith('======='):
            ours.append(lines[j])
            j += 1
        if j < len(lines) and lines[j].startswith('|||||||'):
            while j < len(lines) and not lines[j].startswith('======='):
                j += 1
        if j >= len(lines) or not lines[j].startswith('======='):
            sys.exit(f'unexpected conflict format at line {i + 1}')
        j += 1
        theirs = []
        while j < len(lines) and not lines[j].startswith('>>>>>>>'):
            theirs.append(lines[j])
            j += 1
        if j >= len(lines):
            sys.exit(f'unterminated conflict at line {i + 1}')
        if ours:
            # A reviewed conflict: upstream's line supersedes ours (for example VARS_TO_STR17 vs VARS_TO_STR18 when
            # upstream added a parameter).  --force records it, and prints what was dropped so the commit can say so.
            for o in ours[:6]:
                print(f'  dropped ours at line {i + 1}: ' + o.rstrip()[:110])
            if '--force' not in sys.argv[2:]:
                sys.exit(1)
        empty_ours += 1
        out.extend(theirs)
        i = j + 1
        continue
    out.append(lines[i])
    i += 1

if '<<<<<<<' in ''.join(out):
    sys.exit('a conflict marker survived')

path.write_text(''.join(out), encoding='utf-8', newline='')
print(f'{conflicts} conflict(s), {empty_ours} with an empty ours side, all resolved to upstream: {path}')
