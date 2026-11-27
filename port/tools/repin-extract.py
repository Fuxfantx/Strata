#!/usr/bin/env python3
"""Extract this port's own change to each overlay file, so it can be re-applied onto a new pin.

A three-way merge (`repin-merge.py`) is the fastest way through a repin, but for the files where upstream rewrote the
same region it can silently lose our side.  The reliable representation of "our change" is the diff from the old vanilla
file to the overlay file: that diff is exactly what the port means, independent of what upstream did afterwards.

    python port/tools/repin-extract.py --old <vanilla-old> --out <dir>

The result is one `*.patch` per file under `<dir>`, which can be applied to the new vanilla tree with
`git apply --3way` or `patch --merge` to see precisely which hunks upstream moved.
"""
import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
OVERLAY = ROOT / 'port' / 'llama-overlay-vanilla'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--old', required=True, help='vanilla tree at the previous pin')
    ap.add_argument('--out', required=True, help='where to write the per-file patches')
    args = ap.parse_args()

    old = pathlib.Path(args.old)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = 0
    lines = 0
    added = []
    for p in sorted(OVERLAY.rglob('*')):
        if not p.is_file():
            continue
        rel = p.relative_to(OVERLAY)
        base = old / rel
        if not base.exists():
            added.append(str(rel).replace('\\', '/'))
            continue
        r = subprocess.run(['git', 'diff', '--no-index', '--unified=3', str(base), str(p)],
                           capture_output=True, text=True, errors='surrogateescape')
        if not r.stdout.strip():
            continue
        dst = out / (str(rel).replace('\\', '/').replace('/', '__') + '.patch')
        dst.write_text(r.stdout, encoding='utf-8', errors='surrogateescape', newline='')
        files += 1
        lines += len(r.stdout.splitlines())

    print('patches written: %d (%d lines) in %s' % (files, lines, out))
    print('files the overlay adds outright: %d' % len(added))
    for rel in added[:10]:
        print('  added', rel)
    return 0


if __name__ == '__main__':
    sys.exit(main())
