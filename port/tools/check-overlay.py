#!/usr/bin/env python3
"""Check that the overlay and the vendored tree agree.

`port/llama-overlay-vanilla/` is supposed to be the truth: `fetch-llama.ps1` regenerates `third_party/llama.cpp` from
vanilla plus this overlay, so a file that differs between the two means the next fetch will not reproduce the tree that
was built and measured.  This check exists because that had silently happened: before the repin to b12722, 74 of the 113
overlay files differed from the tree that had been built, including the ones that define the port's quantisation types -
so a fresh fetch produced a tree that could not compile.

    python port/tools/check-overlay.py [--tree DIR] [--fix]

`--fix` copies the tree's version of every differing file back into the overlay, which is the right direction when the
tree is the one that built and ran.
"""
import argparse
import hashlib
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
OVERLAY = ROOT / 'port' / 'llama-overlay-vanilla'
DEFAULT_TREE = ROOT / 'third_party' / 'llama.cpp'


def digest(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--tree', default=str(DEFAULT_TREE))
    ap.add_argument('--fix', action='store_true', help='copy the tree version back into the overlay')
    args = ap.parse_args()

    tree = pathlib.Path(args.tree)
    if not tree.exists():
        print('no tree at %s' % tree)
        return 2

    differ = []
    only_overlay = []
    for p in sorted(OVERLAY.rglob('*')):
        if not p.is_file():
            continue
        rel = p.relative_to(OVERLAY)
        q = tree / rel
        if not q.exists():
            only_overlay.append(str(rel).replace('\\', '/'))
            continue
        if digest(p) != digest(q):
            differ.append(rel)

    print('overlay files: %d' % sum(1 for p in OVERLAY.rglob('*') if p.is_file()))
    print('differ from the tree: %d' % len(differ))
    print('in the overlay but not in the tree: %d' % len(only_overlay))
    for rel in differ[:20]:
        print('  differs', str(rel).replace('\\', '/'))
    if len(differ) > 20:
        print('  ... and %d more' % (len(differ) - 20))

    if args.fix and differ:
        for rel in differ:
            dst = OVERLAY / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(tree / rel, dst)
        print('copied the tree version of %d file(s) back into the overlay' % len(differ))
    elif differ:
        print('the overlay and the tree disagree; --fix copies the tree back, which is right when the tree is what built')
    return 0 if not differ else 1


if __name__ == '__main__':
    sys.exit(main())
