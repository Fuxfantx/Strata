#!/usr/bin/env python3
"""Resolve the mechanical conflicts a repin produces, and say which ones need a person.

`repin-merge.py` leaves conflicting files with merge markers.  Many of those hunks are not disagreements: our side is
empty because upstream added lines in a place the overlay never touched, and the right resolution is simply upstream's
text.  This takes that case automatically, leaves the rest with their markers intact, and reports them.

    python port/tools/repin-resolve.py [--apply]
"""
import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
OVERLAY = ROOT / 'port' / 'llama-overlay-vanilla'
MERGED = ROOT.parent / '.scratch' / 'repin' / 'merged'

HUNK = re.compile(r'<<<<<<< [^\n]*\n(.*?)=======\n(.*?)>>>>>>> [^\n]*\n', re.S)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--apply', action='store_true', help='write files whose conflicts all resolve mechanically')
    args = ap.parse_args()

    taken = 0
    written = []
    manual: list[tuple[str, int, list[str]]] = []

    for path in sorted(MERGED.rglob('*')):
        if not path.is_file():
            continue
        text = path.read_text(encoding='utf-8', errors='surrogateescape')
        if '<<<<<<<' not in text:
            continue
        rel = str(path.relative_to(MERGED)).replace('\\', '/')
        local_manual = []

        def resolve(m: re.Match, rel=rel, local_manual=local_manual) -> str:
            nonlocal taken
            ours, theirs = m.group(1), m.group(2)
            if ours.strip() == '':
                taken += 1
                return theirs
            local_manual.append(len(local_manual))
            return m.group(0)

        out = HUNK.sub(resolve, text)
        path.write_text(out, encoding='utf-8', errors='surrogateescape', newline='')
        if '<<<<<<<' not in out:
            if args.apply:
                dst = OVERLAY / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(out, encoding='utf-8', errors='surrogateescape', newline='')
                written.append(rel)
        else:
            manual.append((rel, out.count('<<<<<<<'), []))

    print('hunks taken from upstream (our side empty): %d' % taken)
    if args.apply:
        print('files written back: %d' % len(written))
        for rel in written:
            print('  wrote', rel)
    if manual:
        print('files still needing a decision: %d' % len(manual))
        for rel, n, _ in manual:
            print('  %-56s %d hunk(s)' % (rel, n))
    return 0


if __name__ == '__main__':
    sys.exit(main())
