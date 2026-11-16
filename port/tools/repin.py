"""Three-way merge every overlay file onto a newer llama.cpp pin.

The overlay ships whole files, so copying it onto a new pin would silently revert whatever upstream changed in those
files.  For each overlay file this fetches the pristine old-pin version (the merge base) and the new-pin version (theirs),
merges ours (old pin + patch) in between, and writes the result to build/repin/merged/.  Files upstream did not touch
between the pins are copied as they are: for those the merge is a no-op by construction.

    python port/tools/repin.py <old-pin> <new-pin> [--fetch-only]
"""
import pathlib
import subprocess
import sys
import urllib.request

R = pathlib.Path(__file__).resolve().parents[2]
OVL = R / 'port' / 'llama-overlay-vanilla'
WORK = R / 'build' / 'repin'
RAW = 'https://raw.githubusercontent.com/ggml-org/llama.cpp/{rev}/{path}'


def fetch(rev: str, path: str) -> bytes | None:
    try:
        req = urllib.request.Request(RAW.format(rev=rev, path=path), headers={'User-Agent': 'strata-vk-repin'})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()
    except Exception:
        return None


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    old, new = sys.argv[1], sys.argv[2]
    merged_dir = WORK / 'merged'
    merged_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in sorted(x for x in OVL.rglob('*') if x.is_file()):
        rel = p.relative_to(OVL).as_posix()
        ours = p.read_bytes()
        base = fetch(old, rel)
        theirs = fetch(new, rel)
        if theirs is None:
            # upstream does not have the file at the new pin: either it was removed, or the port added it
            rows.append((rel, 'upstream lost it' if base is not None else 'added by the port', 0))
            (merged_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            (merged_dir / rel).write_bytes(ours)
            continue
        if base is None:
            rows.append((rel, 'new in both (port added, upstream now has it)', 0))
            (merged_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            (merged_dir / rel).write_bytes(theirs)
            continue
        if base == theirs:
            rows.append((rel, 'upstream unchanged', 0))
            (merged_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            (merged_dir / rel).write_bytes(ours)
            continue
        if ours == base:
            rows.append((rel, 'port unchanged upstream moved (take theirs)', 0))
            (merged_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            (merged_dir / rel).write_bytes(theirs)
            continue
        # real three-way merge
        tmp = WORK / 'tmp'
        tmp.mkdir(exist_ok=True)
        (tmp / 'ours').write_bytes(ours)
        (tmp / 'base').write_bytes(base)
        (tmp / 'theirs').write_bytes(theirs)
        r = subprocess.run(['git', 'merge-file', '-p', '--diff3',
                            str(tmp / 'ours'), str(tmp / 'base'), str(tmp / 'theirs')],
                           capture_output=True)
        out = r.stdout
        conflicts = out.count(b'<<<<<<<')
        (merged_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (merged_dir / rel).write_bytes(out)
        rows.append((rel, f'merged, {conflicts} conflict region(s)' if conflicts else 'merged cleanly', conflicts))

    for rel, state, conflicts in rows:
        if state.startswith('upstream unchanged'):
            continue
        print(f'  {state:52s} {rel}')
    print()
    print('summary:')
    for state in sorted({s for _, s, _ in rows}):
        print(f'  {sum(1 for _, s, _ in rows if s == state):3d}  {state}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
