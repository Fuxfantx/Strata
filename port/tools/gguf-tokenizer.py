#!/usr/bin/env python3
"""Write the tokenizer directory `serve/server.py` wants, out of a GGUF pack.

`serve/server.py --engine strata` refuses to start without `<tokenizer>/vocab.json`, and that directory is produced by the
release setup rather than kept in the source tree, so a source checkout cannot run the server at all.  Everything it needs is
in the pack's own metadata, so this writes it:

    vocab.json        {token: id}                  from tokenizer.ggml.tokens
    merges.txt        one "left right" per line    from tokenizer.ggml.merges
    token_type.json   [type per id]                from tokenizer.ggml.token_type
    tokenizer.json    {"pre": ..., "special_ids": {...}}   from tokenizer.ggml.pre and the special ids

and then proves the result loads and round-trips, using `tools/strata_tokenizer.py` itself - not a copy of its rules.  A
tokenizer that loads but encodes differently would change every prompt silently, which is why the check is a round-trip and
not just a file-exists check.

    python port/tools/gguf-tokenizer.py MODEL.gguf --out pack/full/tokenizer [--check "Hello, world."]
"""
from __future__ import annotations

import json
import os
import struct
import sys

SPECIAL_KEYS = {
    "bos": "tokenizer.ggml.bos_token_id",
    "eos": "tokenizer.ggml.eos_token_id",
    "pad": "tokenizer.ggml.padding_token_id",
    "unk": "tokenizer.ggml.unknown_token_id",
    "eot": "tokenizer.ggml.eot_token_id",
    "sep": "tokenizer.ggml.separator_token_id",
    "mask": "tokenizer.ggml.mask_token_id",
}
WANT = ("tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.token_type",
        "tokenizer.ggml.pre") + tuple(SPECIAL_KEYS.values())

SCALAR = {0: 1, 1: 1, 7: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 10: 8, 11: 8, 12: 8}
FMT = {4: "<I", 5: "<i", 6: "<f", 10: "<Q", 11: "<q", 12: "<d"}


def read_metadata(path: str) -> dict:
    """Every scalar and every array this tool wants, read from the header only (no tensor data)."""
    out = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version = struct.unpack("<I", f.read(4))[0]
        if version < 2:
            raise ValueError(f"unsupported GGUF version {version}")
        n_tensors, n_kv = struct.unpack("<Q", f.read(8))[0], struct.unpack("<Q", f.read(8))[0]

        def rd_str() -> str:
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n).decode("utf-8", "replace")

        def skip(t: int) -> None:
            if t in SCALAR:
                f.read(SCALAR[t])
            elif t == 8:
                rd_str()
            elif t == 9:
                elem = struct.unpack("<I", f.read(4))[0]
                n = struct.unpack("<Q", f.read(8))[0]
                if elem in SCALAR:
                    f.read(SCALAR[elem] * n)
                elif elem == 8:
                    for _ in range(n):
                        rd_str()
                else:
                    raise ValueError(f"nested array elem {elem}")
            else:
                raise ValueError(f"bad kv type {t}")

        for _ in range(n_kv):
            key = rd_str()
            vt = struct.unpack("<I", f.read(4))[0]
            if key not in WANT:
                if vt == 8:
                    rd_str()
                elif vt == 9:
                    skip(vt)
                elif vt in (4, 10):
                    f.read(SCALAR[vt])
                else:
                    skip(vt)
                continue
            if vt == 9:
                elem = struct.unpack("<I", f.read(4))[0]
                n = struct.unpack("<Q", f.read(8))[0]
                if elem == 8:
                    out[key] = [rd_str() for _ in range(n)]
                elif elem in FMT and n <= 1 << 22:
                    raw = f.read(SCALAR[elem] * n)
                    out[key] = list(struct.unpack("<" + FMT[elem][1] * n, raw))
                else:
                    f.read(SCALAR[elem] * n)
            elif vt == 8:
                out[key] = rd_str()
            elif vt in FMT:
                out[key] = struct.unpack(FMT[vt], f.read(SCALAR[vt]))[0]
            else:
                skip(vt)
        del n_tensors
    return out


def write_tokenizer(meta: dict, out_dir: str) -> dict:
    tokens = meta.get("tokenizer.ggml.tokens")
    if not tokens:
        raise SystemExit("this pack has no tokenizer.ggml.tokens: nothing to write")
    merges = meta.get("tokenizer.ggml.merges") or []
    types = meta.get("tokenizer.ggml.token_type") or []
    pre = meta.get("tokenizer.ggml.pre") or ""
    vocab = {}
    for i, t in enumerate(tokens):
        if t in vocab:                       # the tokenizer refuses duplicates; say which one here instead
            raise SystemExit(f"duplicate token {t!r} at ids {vocab[t]} and {i}")
        vocab[t] = i
    special = {}
    for name, key in SPECIAL_KEYS.items():
        if isinstance(meta.get(key), int) and meta[key] >= 0:
            special[name] = meta[key]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(merges))
    with open(os.path.join(out_dir, "token_type.json"), "w", encoding="utf-8") as f:
        json.dump(types, f)
    with open(os.path.join(out_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump({"pre": pre, "special_ids": special}, f)
    return {"tokens": len(tokens), "merges": len(merges), "types": len(types), "pre": pre, "special": special}


def check(out_dir: str, text: str) -> list[int]:
    """Load the directory the way the server does and encode `text`; the round-trip must return the same string."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
    import strata_tokenizer as ST
    vocab = json.loads(open(os.path.join(out_dir, "vocab.json"), encoding="utf-8").read())
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = open(os.path.join(out_dir, "merges.txt"), encoding="utf-8").read().split("\n")
    types = json.loads(open(os.path.join(out_dir, "token_type.json"), encoding="utf-8").read())
    tcfg = json.loads(open(os.path.join(out_dir, "tokenizer.json"), encoding="utf-8").read())
    tok = ST.Tokenizer(tokens, merges, types, tcfg.get("pre") or "qwen35", tcfg.get("special_ids") or None)
    ids = tok.encode(text)
    back = tok.decode(ids)
    if back != text:
        raise SystemExit(f"round-trip changed the text: {text!r} -> {back!r}")
    # a second, longer string exercises the merge ranks past the first few hundred rules
    text2 = "def f(x):\n    return x * 2  # a comment, with punctuation: \"quotes\", (parens) and 1234.\n"
    if tok.decode(tok.encode(text2)) != text2:
        raise SystemExit(f"round-trip changed the second text: {text2!r}")
    return ids


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    model = argv[0]
    out = "pack/full/tokenizer"
    if "--out" in argv:
        out = argv[argv.index("--out") + 1]
    text = "Hello, world. The quick brown fox jumps over the lazy dog."
    if "--check" in argv:
        text = argv[argv.index("--check") + 1]
    meta = read_metadata(model)
    info = write_tokenizer(meta, out)
    print(f"{out}: {info['tokens']} tokens, {info['merges']} merges, {info['types']} types, pre {info['pre']!r}, "
          f"special {info['special']}")
    ids = check(out, text)
    print(f"round-trip ok: {len(ids)} ids for {len(text)} chars: {ids[:12]}{'...' if len(ids) > 12 else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
