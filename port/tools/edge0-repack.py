#!/usr/bin/env python3
"""Repack an Edge0 latent pack into a companion GGUF the existing loader can read.

Why a repack and not a new kernel: the latent block can be written with ops the Vulkan backend already has
(MUL_MAT, MUL_MAT_ID, SWIGLU, GET_ROWS), if the group-selected pieces are ordinary expert-shaped tensors.  Every
piece of the pack is already stored in exactly the byte order those tensors want, so the repack is a split of byte
ranges, not a requantisation:

    blk.L.latent_d.weight        [n_embd, ELL*G]   Q8_0 or Q4_0   <- the D section, row per (group, latent)
    blk.L.latent_u.weight        [ELL, n_embd, G]  Q8_0 or Q4_0   <- the U section, row per (group, embd)
    blk.L.latent_labels          [n_expert]        I32            <- labels.bin, layer L
    blk.L.ffn_gate_exps.weight   [ELL, n_ff, E]    the pack's gu type
    blk.L.ffn_up_exps.weight     [ELL, n_ff, E]    the pack's gu type
    blk.L.ffn_down_exps.weight   [n_ff, ELL, E]    the pack's down type

The tool never dequantises and never reorders within a row; it copies the quantised rows as they are and only writes
new tensor headers.  Use --dry-run to see the table and the total size without writing anything.

    python port/tools/edge0-repack.py --pack DIR --out latent.gguf --dry-run
    python port/tools/edge0-repack.py --pack DIR --out latent.gguf            # writes about 45 GB for 145B Q3_K
"""
import argparse
import json
import pathlib
import struct
import sys

GGUF_MAGIC = b'GGUF'
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8
GGML_TYPE_I32 = 26
# This tree's ggml_type numbering.  It is *not* the same as upstream's, and counting entries in the enum header gets it
# wrong (removed entries and explicit values); the numbers below were read from the built library with
# ggml_type_name()/ggml_blck_size(), which is the only authoritative source:
#     q8_0 = 8, q2_K = 10, q3_K = 11, iq4_nl = 20, i32 = 26
# The packs' `native_experts.txt` happens to use the same numbers for the expert formats it names.
GGML_TYPE_Q3_K = 11
GGML_TYPE_IQ4_NL = 20

PACK_TO_GGML = {11: GGML_TYPE_Q3_K, 20: GGML_TYPE_IQ4_NL, 2: GGML_TYPE_Q4_0,
                8: GGML_TYPE_Q8_0, 0: GGML_TYPE_F32, 1: GGML_TYPE_F16}

TYPE_BLOCK = {GGML_TYPE_Q3_K: (256, 110), GGML_TYPE_IQ4_NL: (32, 18), GGML_TYPE_Q8_0: (32, 34),
              GGML_TYPE_Q4_0: (32, 18), GGML_TYPE_I32: (1, 4), GGML_TYPE_F32: (1, 4), GGML_TYPE_F16: (1, 2)}


def tensor_bytes(ggml_type, dims):
    block, size = TYPE_BLOCK[ggml_type]
    n = 1
    for d in dims:
        n *= d
    if n % block:
        raise SystemExit('shape %s is not a whole number of %d-value blocks for type %d' % (dims, block, ggml_type))
    return n // block * size


def read_manifest(path):
    lines = [l.strip() for l in path.read_text(encoding='utf-8', errors='replace').splitlines() if l.strip()]
    ver = int(lines[0].split(' v', 1)[1].split(':', 1)[0])
    rows = []
    for line in lines[1:]:
        if line.startswith('#'):
            continue
        p = line.split()
        rows.append({'layer': int(p[0]), 'gu': int(p[1]), 'dn': int(p[2]), 'off': int(p[3]), 'blob': int(p[4])})
    rows.sort(key=lambda r: r['layer'])
    return ver, rows


def read_gguf_table(path):
    """[(name, dims, type, offset)] plus the data start, from a GGUF header."""
    with open(path, 'rb') as f:
        head = f.read(24)
        if head[:4] != b'GGUF':
            raise SystemExit('%s: not a GGUF file' % path)
        n_tensors = struct.unpack_from('<Q', head, 8)[0]
        n_kv = struct.unpack_from('<Q', head, 16)[0]

        def rstr():
            n = struct.unpack('<Q', f.read(8))[0]
            return f.read(n).decode('utf-8', 'replace')

        for _ in range(n_kv):
            rstr()
            t = struct.unpack('<I', f.read(4))[0]
            if t == 8:
                rstr()
            elif t == 9:
                at = struct.unpack('<I', f.read(4))[0]
                n = struct.unpack('<Q', f.read(8))[0]
                for _i in range(n):
                    if at == 8:
                        rstr()
                    elif at in (0, 1, 7):
                        f.read(1)
                    elif at in (2, 3):
                        f.read(2)
                    elif at in (4, 5):
                        f.read(4)
                    elif at in (10, 11):
                        f.read(8)
                    else:
                        raise SystemExit('array element type %d not handled' % at)
            elif t in (0, 1, 7):
                f.read(1)
            elif t in (2, 3):
                f.read(2)
            elif t in (4, 5, 6):
                f.read(4)
            elif t in (10, 11, 12):
                f.read(8)
            else:
                raise SystemExit('metadata type %d not handled' % t)
        table = []
        for _ in range(n_tensors):
            name = rstr()
            nd = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
            ttype = struct.unpack('<I', f.read(4))[0]
            off = struct.unpack('<Q', f.read(8))[0]
            table.append((name, dims, ttype, off))
        data_start = (f.tell() + 31) // 32 * 32
    return table, data_start


def verify(pack, gguf, du, n_embd, groups):
    """Compare every tensor in the written file with the pack range it came from."""
    root = pathlib.Path(pack)
    meta = json.loads((root / 'model.json').read_text(encoding='utf-8'))
    _, rows = read_manifest(root / 'native_experts.txt')
    n_expert = int(meta.get('experts') or 512)
    n_ff = 0
    head = (root / 'native_experts.txt').read_text(encoding='utf-8', errors='replace').splitlines()[0]
    if ' expert_ff ' in head:
        n_ff = int(head.split(' expert_ff ', 1)[1].split(')', 1)[0].split()[0])
    ell = int(meta.get('latent_dim') or 0)
    du_name = 'latent_dense_q8.bin' if du == 'q8' else 'latent_dense_q4.bin'
    row_b = 34 if du == 'q8' else 18
    d_row = (n_embd // 32) * row_b
    u_row = (ell // 32) * row_b
    sources = {}
    blobs = {}
    for r in rows:
        blobs[r['layer']] = r['blob']
        L = r['layer']
        gu_row = tensor_bytes(PACK_TO_GGML.get(r['gu'], r['gu']), [ell])
        dn_row = tensor_bytes(PACK_TO_GGML.get(r['dn'], r['dn']), [n_ff])
        base_du = L * (groups * ell * d_row + groups * n_embd * u_row)
        sources[f'blk.{L}.latent_d.weight'] = (du_name, base_du, groups * ell * d_row, 0, 0)
        sources[f'blk.{L}.latent_u.weight'] = (du_name, base_du + groups * ell * d_row, groups * n_embd * u_row, 0, 0)
        sources[f'blk.{L}.latent_labels'] = ('labels.bin', L * n_expert * 4, n_expert * 4, 0, 0)
        sources[f'blk.{L}.ffn_gate_exps.weight'] = ('experts.bin', r['off'], n_expert * n_ff * gu_row, n_ff * gu_row,
                                                     r['blob'])
        sources[f'blk.{L}.ffn_up_exps.weight'] = ('experts.bin', r['off'] + n_ff * gu_row, n_expert * n_ff * gu_row,
                                                  n_ff * gu_row, r['blob'])
        sources[f'blk.{L}.ffn_down_exps.weight'] = ('experts.bin', r['off'] + 2 * n_ff * gu_row,
                                                    n_expert * ell * dn_row, ell * dn_row, r['blob'])
    table, data_start = read_gguf_table(gguf)
    bad = []
    handles = {}
    with open(gguf, 'rb') as f:
        for name, dims, ttype, off in table:
            if name not in sources:
                bad.append((name, 'not expected'))
                continue
            src, base, count, per_expert, blob = sources[name]
            n = tensor_bytes(ttype, dims)
            if n != count:
                bad.append((name, 'size %d != source %d' % (n, count)))
                continue
            if src not in handles:
                handles[src] = open(root / src, 'rb')
            fh = handles[src]
            f.seek(data_start + off)
            got = f.read(n)
            if blob:
                want = b''
                for e in range(n_expert):
                    fh.seek(base + e * blob)
                    want += fh.read(per_expert)
            else:
                fh.seek(base)
                want = fh.read(n)
            if got != want:
                bad.append((name, 'bytes differ'))
    for fh in handles.values():
        fh.close()
    print('verify %s: %d tensors, %d problems' % (gguf, len(table), len(bad)))
    for b in bad[:12]:
        print('  !', b[0], '-', b[1])
    return 0 if not bad else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--pack', required=True, help='the Edge0 model folder')
    ap.add_argument('--out', help='the companion GGUF to write')
    ap.add_argument('--du', choices=('q8', 'q4'), default='q8')
    ap.add_argument('--layers', type=int, default=0, help='only the first N layers (for a fixture or a smoke test)')
    ap.add_argument('--dry-run', action='store_true', help='print the tensor table and the size, write nothing')
    ap.add_argument('--embedding-length', type=int, default=0,
                    help='model width; derived from the D/U file size when omitted')
    ap.add_argument('--verify', action='store_true', help='after writing, read the file back and compare')
    ap.add_argument('--groups', type=int, default=8)
    args = ap.parse_args()

    root = pathlib.Path(args.pack)
    meta = json.loads((root / 'model.json').read_text(encoding='utf-8'))
    n_layers = int(meta.get('layers') or 48)
    n_expert = int(meta.get('experts') or 512)
    ell = int(meta.get('latent_dim') or 0)
    if not ell:
        raise SystemExit('model.json has no latent_dim')
    n_embd = int(meta.get('embedding_length') or 0)
    if not n_embd:
        n_embd = int(getattr(args, 'embedding_length', 0) or 0)
    if not n_embd:
        # solve it from the D/U file: one layer holds groups*ell rows of n_embd plus groups*n_embd rows of ell, and both
        # row sizes are (values/32)*row_b, so per_layer = 2*groups*ell*n_embd*row_b/32
        du_probe = 'latent_dense_q8.bin' if args.du == 'q8' else 'latent_dense_q4.bin'
        row_b_probe = 34 if args.du == 'q8' else 18
        du_size = (root / du_probe).stat().st_size if (root / du_probe).exists() else 0
        if du_size and ell and n_layers:
            per_layer = du_size // n_layers
            n_embd = per_layer * 32 // (2 * int(meta.get('latent_groups') or args.groups) * ell * row_b_probe)
            print('derived embedding_length %d from %s (%d bytes, %d layers)'
                  % (n_embd, du_probe, du_size, n_layers))
    if not n_embd:
        raise SystemExit('cannot determine the model width: pass --embedding-length')
    groups = int(meta.get('latent_groups') or args.groups)
    args_groups = groups
    ver, rows = read_manifest(root / 'native_experts.txt')
    gu_type = PACK_TO_GGML.get(rows[0]['gu'], rows[0]['gu'])
    dn_type = PACK_TO_GGML.get(rows[0]['dn'], rows[0]['dn'])
    n_ff = 0

    # the manifest header names the expert widths; trust it when it is there
    head = (root / 'native_experts.txt').read_text(encoding='utf-8', errors='replace').splitlines()[0]
    for token, key in ((' expert_embd ', 'expert_embd'), (' expert_ff ', 'expert_ff')):
        if token in head:
            want = int(head.split(token, 1)[1].split(')', 1)[0].split()[0])
            if key == 'expert_embd':
                ell = want
            else:
                n_ff = want
    if not n_ff:
        raise SystemExit('native_experts.txt does not state the expert width (expert_ff)')
    args.groups = groups

    du_name = 'latent_dense_q8.bin' if args.du == 'q8' else 'latent_dense_q4.bin'
    du_type = GGML_TYPE_Q8_0 if args.du == 'q8' else GGML_TYPE_Q4_0
    row_b = 34 if args.du == 'q8' else 18
    d_row = (n_embd // 32) * row_b
    u_row = (ell // 32) * row_b

    layers = args.layers or n_layers
    table = []
    for r in rows[:layers]:
        L = r['layer']
        gu_row = tensor_bytes(gu_type, [ell]) // 1  # bytes of one row of ell values
        dn_row = tensor_bytes(dn_type, [n_ff])
        if n_ff * gu_row * 2 + ell * dn_row != r['blob']:
            raise SystemExit('layer %d: the manifest blob (%d) does not equal the arithmetic size (%d)'
                             % (L, r['blob'], n_ff * gu_row * 2 + ell * dn_row))
        base = r['off']
        table.append((f'blk.{L}.latent_d.weight', [n_embd, ell * args.groups], du_type,
                      ('du', L * (args.groups * ell * d_row + args.groups * n_embd * u_row), args.groups * ell * d_row)))
        table.append((f'blk.{L}.latent_u.weight', [ell, n_embd, args.groups], du_type,
                      ('du', L * (args.groups * ell * d_row + args.groups * n_embd * u_row) + args.groups * ell * d_row,
                       args.groups * n_embd * u_row)))
        table.append((f'blk.{L}.latent_labels', [n_expert], GGML_TYPE_I32, ('labels', L * n_expert * 4, n_expert * 4)))
        table.append((f'blk.{L}.ffn_gate_exps.weight', [ell, n_ff, n_expert], gu_type,
                      ('exp', base, n_expert * n_ff * gu_row, n_ff * gu_row, r['blob'])))
        table.append((f'blk.{L}.ffn_up_exps.weight', [ell, n_ff, n_expert], gu_type,
                      ('exp', base + n_ff * gu_row, n_expert * n_ff * gu_row, n_ff * gu_row, r['blob'])))
        table.append((f'blk.{L}.ffn_down_exps.weight', [n_ff, ell, n_expert], dn_type,
                      ('exp', base + 2 * n_ff * gu_row, n_expert * ell * dn_row, ell * dn_row, r['blob'])))

    total_data = 0
    for name, dims, ttype, src in table:
        n = tensor_bytes(ttype, dims)
        total_data += (n + 31) // 32 * 32

    if args.dry_run:
        print('%d layers, %d experts, ELL %d, n_ff %d, gu %s, down %s, D/U %s'
              % (layers, n_expert, ell, n_ff, gu_type, dn_type, du_name))
        for name, dims, ttype, src in table[:12]:
            print('  %-32s %-18s type %d  %d bytes' % (name, dims, ttype, tensor_bytes(ttype, dims)))
        print('  ... %d tensors in total' % len(table))
        print('data bytes: %d (%.2f GiB), with 32-byte alignment' % (total_data, total_data / 2 ** 30))
        return 0

    if not args.out:
        raise SystemExit('--out is required unless --dry-run is given')

    # write the file: header, tensor infos, then each tensor's bytes straight from its source range
    out = pathlib.Path(args.out)
    names = [t[0] for t in table]
    hdr = bytearray()
    hdr += GGUF_MAGIC + struct.pack('<I', 3) + struct.pack('<Q', len(table)) + struct.pack('<Q', 4)
    for k, v in (('general.architecture', 'latentcompanion'), ('latent.pack', meta.get('model_name', '')),
                 ('latent.du_format', args.du), ('latent.groups', str(args.groups))):
        kb, vb = k.encode(), str(v).encode()
        hdr += struct.pack('<Q', len(kb)) + kb + struct.pack('<I', 8) + struct.pack('<Q', len(vb)) + vb
    offsets = []
    off = 0
    for name, dims, ttype, src in table:
        n = tensor_bytes(ttype, dims)
        offsets.append(off)
        off = (off + n + 31) // 32 * 32
        nb = name.encode()
        hdr += struct.pack('<Q', len(nb)) + nb + struct.pack('<I', len(dims))
        for d in dims:
            hdr += struct.pack('<Q', d)
        hdr += struct.pack('<I', ttype) + struct.pack('<Q', offsets[-1])
    while len(hdr) % 32:
        hdr += b'\x00'

    handlers = {}
    for src in {t[3][0] for t in table}:
        if src == 'du':
            handlers['du'] = (root / du_name).open('rb')
        elif src == 'labels':
            handlers['labels'] = (root / 'labels.bin').open('rb')
        else:
            handlers['exp'] = (root / 'experts.bin').open('rb')

    written = 0
    with out.open('wb') as f:
        f.write(hdr)
        for i, (name, dims, ttype, src) in enumerate(table):
            n = tensor_bytes(ttype, dims)
            kind = src[0]
            if kind in ('du', 'labels'):
                _, base, count = src
                fh = handlers[kind]
                fh.seek(base)
                chunk = 1 << 20
                left = count
                while left:
                    b = fh.read(min(chunk, left))
                    if not b:
                        raise SystemExit('short read in ' + kind)
                    f.write(b)
                    left -= len(b)
            else:
                _, base, count, per_expert, blob = src
                fh = handlers['exp']
                for e in range(n_expert):
                    fh.seek(base + e * blob)
                    left = per_expert
                    while left:
                        b = fh.read(min(1 << 20, left))
                        if not b:
                            raise SystemExit('short read in experts.bin')
                        f.write(b)
                        left -= len(b)
            written += n
            pad = (offsets[i] + n) - (offsets[i] + n)
            if i + 1 < len(offsets):
                pad = offsets[i + 1] - (offsets[i] + n)
            if pad > 0:
                f.write(b'\x00' * pad)
        f.flush()
    for fh in handlers.values():
        fh.close()
    print('wrote %s: %d tensors, %d bytes of tensor data (%.2f GiB)'
          % (out, len(table), written, written / 2 ** 30))
    if args.verify:
        return verify(args.pack, out, args.du, n_embd, args.groups)
    return 0


if __name__ == '__main__':
    sys.exit(main())
