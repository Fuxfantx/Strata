"""Normalise the overlay to upstream's conventions: UTF-8 without BOM, LF line endings.

The overlay is copied verbatim onto a vanilla llama.cpp tree, so a file with CRLF or a BOM differs from upstream's in
every line.  That is not harmless: a three-way merge against a newer pin then reports the whole file as a conflict (the
re-pin on 2026-11-17 hit exactly that on llama-quant.cpp), and a diff of the overlay against upstream is unreadable.
"""
import pathlib
import sys

OVL = pathlib.Path(__file__).resolve().parents[1] / 'llama-overlay-vanilla'

changed = []
for p in sorted(x for x in OVL.rglob('*') if x.is_file()):
    if p.suffix in ('.bin',):
        continue
    raw = p.read_bytes()
    original = raw
    if raw.startswith(b'\xef\xbb\xbf'):
        raw = raw[3:]
    # normalise CRLF and lone CR to LF
    raw = raw.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
    if raw != original:
        p.write_bytes(raw)
        changed.append(p.relative_to(OVL).as_posix())

print(f'{len(changed)} file(s) normalised')
for c in changed:
    print('  ' + c)
sys.exit(0)
