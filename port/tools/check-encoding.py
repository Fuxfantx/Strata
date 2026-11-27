#!/usr/bin/env python3
"""Encoding gate: no tracked text file carries invalid UTF-8 or the mojibake a bad editor write leaves behind.

`audit-docs.py` checks the documents; this checks the source, because a write that decodes a UTF-8 em dash as GBK still
compiles and only shows up as garbage in a comment. A replacement character U+FFFD is not flagged: serve/ uses it on
purpose, to detect that a decode produced one. Binary files (a NUL byte in the first 8 KiB) are skipped.

    python port/tools/check-encoding.py [repo-root]
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

# Sequences a UTF-8 em dash / arrow / punctuation mark turns into when its bytes are read as GBK. Written as escapes so
# this file does not contain the damage it looks for.
MOJIBAKE = ("\u9225", "\u9229", "\u951b", "\u9428", "\u9286", "\u6d60", "\u5a34", "\u59af", "\u923b", "\u9239", "\u951f")

BINARY = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2", ".ttf", ".otf", ".gguf", ".bin",
          ".zip", ".gz", ".so", ".dll", ".exe", ".cat", ".comp", ".lockb", ".mp4", ".webm"}


def main() -> int:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[2])
    files = subprocess.run(["git", "-C", str(root), "ls-files"], capture_output=True, text=True, check=True).stdout.split()
    bad: list[tuple[str, str]] = []
    for rel in files:
        if pathlib.Path(rel).suffix.lower() in BINARY:
            continue
        try:
            raw = (root / rel).read_bytes()
        except OSError:
            continue
        if b"\x00" in raw[:8192]:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            bad.append((rel, f"invalid UTF-8 at byte {e.start}"))
            continue
        hits = sorted({m for m in MOJIBAKE if m in text})
        if hits:
            bad.append((rel, "mojibake: " + " ".join(hits)))
    for rel, why in bad:
        print(f"  {rel}: {why}")
    print(f"{len(files)} files checked, {len(bad)} with encoding damage")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
