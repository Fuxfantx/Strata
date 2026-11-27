#!/usr/bin/env python3
"""Re-apply this port's overlay onto a new upstream pin, one file at a time.

The vendored tree is regenerated from vanilla plus `port/llama-overlay-vanilla`, and the overlay stores whole files.  When
the pin moves, copying those whole files over the new upstream throws away whatever upstream changed in them - the first
attempt at b12722 failed exactly that way: the overlay's `ggml/include/ggml.h` predated an op that the new upstream's own
`ggml-alloc.c` names.

So each overlay file is merged three ways:

    base   = vanilla at the old pin      (what the overlay was written against)
    ours   = the overlay file            (this port's version)
    theirs = vanilla at the new pin      (what upstream has now)

`git merge-file` does the merge; a file with no conflicts is written back to the overlay, and a file with conflicts is
left alone and reported, with the merged text kept beside it for inspection.

    python port/tools/repin-merge.py --old <vanilla-old> --new <vanilla-new> [--apply] [--only pattern]
"""
import argparse
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
OVERLAY = ROOT / 'port' / 'llama-overlay-vanilla'
SCRATCH = ROOT.parent / '.scratch' / 'repin'


def merge_one(rel: pathlib.Path, old: pathlib.Path, new: pathlib.Path) -> str:
    ours = OVERLAY / rel
    base_p, ours_p, theirs_p = old / rel, ours, new / rel

    if not base_p.exists():
        # the overlay adds a file upstream does not have: nothing to merge, but upstream may have added it meanwhile
        return 'added' if not theirs_p.exists() else 'both-added'
    if not theirs_p.exists():
        return 'removed-upstream'

    tmp = SCRATCH / 'merged' / rel
    tmp.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ours_p, tmp)
    r = subprocess.run(['git', 'merge-file', '-p', str(base_p), str(ours_p), str(theirs_p)],
                       capture_output=True, text=True, errors='surrogateescape')
    if r.returncode == 0:
        tmp.write_text(r.stdout, encoding='utf-8', errors='surrogateescape', newline='')
        return 'clean'
    tmp.write_text(r.stdout, encoding='utf-8', errors='surrogateescape', newline='')
    return 'conflict'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--old', required=True, help='vanilla tree at the previous pin')
    ap.add_argument('--new', required=True, help='vanilla tree at the new pin')
    ap.add_argument('--apply', action='store_true', help='write the clean merges back into the overlay')
    ap.add_argument('--only', help='merge only paths matching this substring')
    args = ap.parse_args()

    old, new = pathlib.Path(args.old), pathlib.Path(args.new)
    SCRATCH.mkdir(parents=True, exist_ok=True)

    files = sorted(p.relative_to(OVERLAY) for p in OVERLAY.rglob('*') if p.is_file())
    tally = {}
    problems = []
    for rel in files:
        if args.only and args.only not in str(rel).replace('\\', '/'):
            continue
        state = merge_one(rel, old, new)
        tally[state] = tally.get(state, 0) + 1
        if state in ('conflict', 'both-added', 'removed-upstream'):
            problems.append((state, str(rel).replace('\\', '/')))
        elif state == 'clean' and args.apply:
            shutil.copyfile(SCRATCH / 'merged' / rel, OVERLAY / rel)

    print('overlay files: %d' % len(files))
    for k in sorted(tally):
        print('  %-16s %d' % (k, tally[k]))
    if problems:
        print('needs a decision:')
        for state, rel in problems:
            print('  %-16s %s' % (state, rel))
    if args.apply:
        print('clean merges written back to the overlay')
    else:
        print('dry run: nothing written (pass --apply to write the clean merges)')
    return 0 if not problems else 1


if __name__ == '__main__':
    sys.exit(main())
