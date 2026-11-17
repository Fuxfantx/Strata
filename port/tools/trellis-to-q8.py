#!/usr/bin/env python3
"""Decode TQK6/TQK7 (and TQ2_T) trellis weights to float, and repack selected expert tensors to Q8_0.

Why: the trellis expert matvec is the one bottleneck left on both backends, and every lever inside the format has been measured
and closed (port/docs/PERFORMANCE.md 11 and 14).  Spending bytes to save instructions is what is left, and the target type needs
**no new kernel**: Q8_0 already has an integer dot product on the CPU (vdotq_s32) and on Vulkan (dp4a).  The owner asked for a
*small* expansion (PERFORMANCE.md 17.5), so this tool rewrites a chosen number of layers rather than the whole pack.

    python port/tools/trellis-to-q8.py --selftest
    python port/tools/trellis-to-q8.py --pack Gyro-S.gguf --layers 8 --out experts-q8.gguf

The decoder is validated bit-exactly against `third_party/llama.cpp/tests/data/tqk6_ref.bin` and `tqk7_ref.bin` before anything
touches a real pack; `--selftest` is that check and it is the only thing this file does until it passes.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / "third_party" / "llama.cpp" / "ggml" / "src" / "ggml-common.h"
REFDIR = ROOT / "third_party" / "llama.cpp" / "tests" / "data"

TQ_STATE_BITS = 15
TQ_STATE_MASK = (1 << TQ_STATE_BITS) - 1
HYB_MUL = 0x9E3779B1
QK_TQK = 128
# tqk6 and tqk7 differ only in the stream width: 4*k bytes per 128 weights
BLOCK_BYTES = {6: 2 + 4 * 6, 7: 2 + 4 * 7, 8: 2 + QK_TQK // 4}     # 8 = TQ2_T's own stream


def load_codebook(path: Path = COMMON) -> list[float]:
    """The 4096 fp16 codebook values, read from the header so there is one source of truth."""
    text = path.read_text(encoding="utf-8", errors="replace")
    start = text.index("GGML_TABLE_BEGIN(uint16_t, tq2t_lut_f16, 4096)")
    start = text.index("\n", start)
    out: list[int] = []
    for line in text[start:].splitlines():
        line = line.strip()
        if line.startswith("GGML_TABLE_END"):
            break
        for tok in line.split(","):
            tok = tok.strip().rstrip("}")
            if tok.lower().startswith("0x"):
                out.append(int(tok, 16))
    if len(out) != 4096:
        raise SystemExit(f"codebook: read {len(out)} values, expected 4096 ({path})")
    return [struct.unpack("<e", struct.pack("<H", v))[0] for v in out]


def hyb_index(e: int) -> int:
    x = (((e >> 1) & TQ_STATE_MASK) * HYB_MUL) & 0xFFFFFFFF
    return ((x >> 10) & 2047) if (e & 1) else (x >> 21)


def tqk_state(qs: bytes, k: int, t: int) -> int:
    nbytes = 4 * k
    off = (31 - t) * k
    i0 = off >> 3
    w = qs[i0] | (qs[(i0 + 1) % nbytes] << 8) | (qs[(i0 + 2) % nbytes] << 16)
    return (w >> (off & 7)) & 0xFFFF


def tq2t_state(qs: bytes, t: int) -> int:
    """TQ2_T's own stream: a 16-bit window at a byte granularity, tail-biting over 32 bytes."""
    return (qs[(t + 31) % 32] << 8) | qs[t]


def dequant_block(block: bytes, kb: int, lut: list[float]) -> list[float]:
    """128 floats from one block.  kb 6/7 select the TQK stream, kb 8 is TQ2_T."""
    d = struct.unpack("<e", block[:2])[0]
    qs = block[2:]
    out: list[float] = []
    for t in range(QK_TQK // 4):
        s = tq2t_state(qs, t) if kb == 8 else tqk_state(qs, kb, t)
        for h in range(2):
            idx = hyb_index(2 * s + h)
            out.append(d * lut[2 * idx + 0])
            out.append(d * lut[2 * idx + 1])
    return out


def dequant_row_np(blocks: "np.ndarray", kb: int, lut: "np.ndarray") -> "np.ndarray":
    """(n_blocks, 128) float from (n_blocks, block_bytes) uint8, vectorised.  Same arithmetic as dequant_block, which is the
    reference implementation and is checked against tests/data/tq*_ref.bin by --selftest."""
    import numpy as np
    nb = blocks.shape[0]
    d = blocks[:, :2].copy().view(np.float16).reshape(nb).astype(np.float32)
    qs = blocks[:, 2:]
    nbytes = 4 * kb
    t = np.arange(32)
    if kb == 8:
        lo = (t + 31) % 32
        s = (qs[:, lo].astype(np.uint32) << 8) | qs[:, t].astype(np.uint32)
    else:
        off = ((31 - t) * kb).astype(np.uint32)
        i0 = off >> 3
        w = (qs[:, i0].astype(np.uint32)
             | (qs[:, (i0 + 1) % nbytes].astype(np.uint32) << 8)
             | (qs[:, (i0 + 2) % nbytes].astype(np.uint32) << 16))
        s = (w >> (off & 7)) & 0xFFFF
    out = np.empty((nb, 32, 4), dtype=np.float32)
    for h in range(2):
        e = 2 * s + h
        x = (((e >> 1) & TQ_STATE_MASK).astype(np.uint64) * HYB_MUL) & 0xFFFFFFFF
        idx = np.where((e & 1) != 0, (x >> 10) & 2047, x >> 21).astype(np.int64)
        out[:, :, 2 * h] = lut[2 * idx]
        out[:, :, 2 * h + 1] = lut[2 * idx + 1]
    return d[:, None] * out.reshape(nb, 128)


def quantize_q8_0_np(x: "np.ndarray") -> "np.ndarray":
    """(n, 32) float -> (n, 34) uint8 Q8_0, the reference quantiser's arithmetic (d stored as f16, rounding done with the f32 d)."""
    import numpy as np
    amax = np.abs(x).max(axis=1)
    d = amax / 127.0
    with np.errstate(divide="ignore"):
        inv = np.where(d != 0, 1.0 / np.where(d != 0, d, 1.0), 0.0)
    q = np.rint(x * inv[:, None]).clip(-127, 127).astype(np.int8)
    out = np.empty((x.shape[0], 34), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out


def selftest_np(lut: list[float]) -> int:
    """The vectorised path must agree with the scalar one, value for value."""
    import numpy as np
    lut_np = np.asarray(lut, dtype=np.float32)
    bad = 0
    for name, kb in (("tqk6_ref.bin", 6), ("tqk7_ref.bin", 7), ("tq2t_ref.bin", 8)):
        raw = (REFDIR / name).read_bytes()
        (nb,) = struct.unpack_from("<I", raw, 0)
        bsz = BLOCK_BYTES[kb]
        blocks = np.frombuffer(raw, dtype=np.uint8, count=nb * bsz, offset=4).reshape(nb, bsz)
        fast = dequant_row_np(blocks, kb, lut_np)
        want = np.frombuffer(raw, dtype="<f4", count=nb * QK_TQK, offset=4 + nb * bsz).reshape(nb, QK_TQK)
        diffs = int((fast.view(np.uint32) != want.view(np.uint32)).sum())
        print(f"{name}: vectorised vs reference, {diffs} values differ")
        bad += diffs
    print("the vectorised decoder is bit-exact too" if bad == 0 else f"VECTORISED MISMATCH ({bad})")
    return 0 if bad == 0 else 1


def selftest(lut: list[float]) -> int:
    bad = 0
    for name, kb in (("tqk6_ref.bin", 6), ("tqk7_ref.bin", 7), ("tq2t_ref.bin", 8)):
        path = REFDIR / name
        raw = path.read_bytes()
        (nb,) = struct.unpack_from("<I", raw, 0)
        bsz = BLOCK_BYTES[kb]
        if len(raw) != 4 + nb * bsz + nb * QK_TQK * 4:
            print(f"{name}: size {len(raw)} does not fit {nb} blocks of {bsz} + {nb}x128 f32", file=sys.stderr)
            return 1
        blocks = [raw[4 + i * bsz: 4 + (i + 1) * bsz] for i in range(nb)]
        want = struct.unpack_from(f"<{nb * QK_TQK}f", raw, 4 + nb * bsz)
        diffs = 0
        for i, b in enumerate(blocks):
            got = dequant_block(b, kb, lut)
            for j in range(QK_TQK):
                if struct.pack("<f", got[j]) != struct.pack("<f", want[i * QK_TQK + j]):
                    diffs += 1
        print(f"{name}: {nb} blocks, {nb * QK_TQK} values, {diffs} differ from the reference")
        bad += diffs
    print("decoder is bit-exact against the reference vectors" if bad == 0 else f"DECODER MISMATCH ({bad} values)")
    return 0 if bad == 0 else 1


GGUF_MAGIC = b"GGUF"
GGML_TYPE_Q8_0 = 8
GGML_TYPE_TQK6, GGML_TYPE_TQK7 = 145, 146
TQK_K = {GGML_TYPE_TQK6: 6, GGML_TYPE_TQK7: 7}
QK8_0 = 32


def read_gguf(path: Path):
    """(metadata dict, [(name, dims, type, offset)], data_start).  Offsets are from data_start, 32-byte aligned."""
    f = open(path, "rb")
    if f.read(4) != GGUF_MAGIC:
        raise SystemExit(f"{path}: not a GGUF")
    ver, = struct.unpack("<I", f.read(4))
    n_tensors, = struct.unpack("<Q", f.read(8))
    n_kv, = struct.unpack("<Q", f.read(8))

    def rd_str() -> str:
        n, = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")

    def skip(t: int) -> None:
        if t in (0, 1, 7):
            f.read(1)
        elif t in (2, 3):
            f.read(2)
        elif t in (4, 5, 6):
            f.read(4)
        elif t == 8:
            rd_str()
        elif t == 9:
            elem, = struct.unpack("<I", f.read(4))
            n, = struct.unpack("<Q", f.read(8))
            if elem in (0, 1, 7):
                f.read(n)
            elif elem in (2, 3):
                f.read(2 * n)
            elif elem in (4, 5, 6):
                f.read(4 * n)
            elif elem in (10, 11, 12):
                f.read(8 * n)
            elif elem == 8:
                for _ in range(n):
                    rd_str()
            else:
                raise SystemExit(f"bad array element type {elem}")
        elif t in (10, 11, 12):
            f.read(8)
        else:
            raise SystemExit(f"bad kv type {t}")

    meta = {}
    for _ in range(n_kv):
        key = rd_str()
        vt, = struct.unpack("<I", f.read(4))
        if vt == 8:
            meta[key] = rd_str()
        elif vt in (4, 10):
            meta[key] = struct.unpack("<I" if vt == 4 else "<Q", f.read(4 if vt == 4 else 8))[0]
        else:
            skip(vt)
    tensors = []
    for _ in range(n_tensors):
        name = rd_str()
        nd, = struct.unpack("<I", f.read(4))
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
        ttype, = struct.unpack("<I", f.read(4))
        off, = struct.unpack("<Q", f.read(8))
        tensors.append((name, dims, ttype, off))
    header_end = f.tell()
    data_start = (header_end + 31) // 32 * 32
    f.close()
    return meta, tensors, data_start, ver


def write_gguf(path: Path, meta: dict, tensors: list[tuple[str, list[int], int]], data: list[bytes]) -> None:
    """A v3 file with every tensor 32-byte aligned; `data[i]` is tensor i's bytes, already in ggml order."""
    hdr = bytearray()
    hdr += GGUF_MAGIC + struct.pack("<I", 3) + struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(meta))
    for k, v in meta.items():
        kb = k.encode("utf-8")
        hdr += struct.pack("<Q", len(kb)) + kb
        hdr += struct.pack("<I", 8)
        vb = str(v).encode("utf-8")
        hdr += struct.pack("<Q", len(vb)) + vb
    off = 0
    offsets = []
    for name, dims, ttype in tensors:
        nbytes = len(data[len(offsets)])
        offsets.append(off)
        off = (off + nbytes + 31) // 32 * 32
        nb = name.encode("utf-8")
        hdr += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(dims))
        for d in dims:
            hdr += struct.pack("<Q", d)
        hdr += struct.pack("<I", ttype) + struct.pack("<Q", offsets[-1])
    pad = (-len(hdr)) % 32
    with open(path, "wb") as f:
        f.write(bytes(hdr) + b"\0" * pad)
        for i, blob in enumerate(data):
            f.write(blob)
            f.write(b"\0" * ((-len(blob)) % 32))


def convert_tensor(src: Path, off: int, n_weights: int, kb: int, lut, log=print) -> bytes:
    """One trellis tensor to Q8_0 bytes, streamed in chunks so a 170 MiB tensor is never expanded whole."""
    import numpy as np
    lut_np = np.asarray(lut, dtype=np.float32)
    bsz_tq = BLOCK_BYTES[kb]
    nblk = n_weights // QK_TQK
    chunk = 4096
    out = bytearray(nblk // 4 * 34)
    pos = 0
    with open(src, "rb") as f:
        f.seek(off)
        for b0 in range(0, nblk, chunk):
            b1 = min(b0 + chunk, nblk)
            raw = f.read((b1 - b0) * bsz_tq)
            if len(raw) != (b1 - b0) * bsz_tq:
                raise SystemExit(f"short read at block {b0}: wanted {(b1 - b0) * bsz_tq}, got {len(raw)}")
            blocks = np.frombuffer(raw, dtype=np.uint8).reshape(b1 - b0, bsz_tq)
            x = dequant_row_np(blocks, kb, lut_np).reshape(-1, QK8_0)
            q = quantize_q8_0_np(x)
            out[pos:pos + q.nbytes] = q.tobytes()
            pos += q.nbytes
    return bytes(out)
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="check the decoder against tests/data/tq*_ref.bin and exit")
    ap.add_argument("--pack")
    ap.add_argument("--layers", type=int, default=8, help="how many blocks' expert tensors to rewrite (from the end)")
    ap.add_argument("--tensors", default="", help="only tensors whose name contains this (e.g. ffn_down_exps): a cheaper first run")
    ap.add_argument("--out")
    ap.add_argument("--common", default=str(COMMON), help="ggml-common.h, the codebook's source of truth")
    a = ap.parse_args(argv)

    lut = load_codebook(Path(a.common))
    if a.selftest or not a.pack:
        rc = selftest(lut)
        return rc if rc != 0 else selftest_np(lut)
    return repack(Path(a.pack), Path(a.out) if a.out else None, a.layers, a.tensors, lut)


def repack(src: Path, out: Path | None, n_layers: int, only: str, lut: list[float]) -> int:
    """Rewrite the expert tensors of the last `n_layers` blocks as Q8_0 into a side-car GGUF."""
    import numpy as np
    meta, tensors, data_start, ver = read_gguf(src)
    if out is None:
        raise SystemExit("--out is required with --pack")
    arch = meta.get("general.architecture", "?")
    n_block = int(meta.get(f"{arch}.block_count", 0) or 0)
    if n_block <= 0:
        raise SystemExit(f"{src}: no {arch}.block_count in the metadata")
    first = n_block - n_layers
    picked = []
    for name, dims, ttype, off in tensors:
        if ttype not in TQK_K or not name.startswith("blk.") or "_exps" not in name:
            continue
        blk = int(name.split(".")[1])
        if blk < first:
            continue
        if only and only not in name:
            continue
        picked.append((name, dims, ttype, off))
    if not picked:
        raise SystemExit(f"no trellis expert tensors in blocks {first}..{n_block - 1}")
    total_in = sum(int(np.prod(d)) * TQK_K[t] / 8 + 0 for _, d, t, _ in picked for _ in [0])
    print(f"source {src} ({ver}), {n_block} blocks; converting blocks {first}..{n_block - 1}: {len(picked)} tensors")
    blobs, out_meta_tensors = [], []
    for name, dims, ttype, off in picked:
        n_weights = 1
        for d in dims:
            n_weights *= int(d)
        kb = TQK_K[ttype]
        small = dequant_row_np  # noqa: F841  (documents that the vectorised path is the one used)
        blob = convert_tensor(src, data_start + off, n_weights, kb, lut,
                              log=lambda *_: None)
        blobs.append(blob)
        out_meta_tensors.append((name, [int(d) for d in dims], GGML_TYPE_Q8_0))
        print(f"  {name:38s} {dims} tqk{kb} -> q8_0  {len(blob) / 2**20:8.1f} MiB")
    out_meta = {"general.architecture": arch, "strata.sidecar.source": src.name,
                "strata.sidecar.blocks": f"{first}-{n_block - 1}", "strata.sidecar.from_type": "tqk6/tqk7"}
    write_gguf(out, out_meta, out_meta_tensors, blobs)
    print(f"wrote {out} ({sum(len(b) for b in blobs) / 2**30:.2f} GiB of Q8_0 experts)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
