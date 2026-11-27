"""Write a companion GGUF: a few tensors copied verbatim out of a base pack, plus the compose marker.

The engine's `--compat FILE` composes such a file over the base pack: a tensor in the companion replaces the base
pack's tensor of the same name (llama-model-loader.cpp, the two registration sites).  The marker is the KV
`strata.companion = true`, which is what distinguishes a companion from a shard of a split model -- a duplicate name
inside the base pack is still an error.

This exists because `llama-quantize --only-tensors <one tensor>` walks every tensor of the pack before writing the
one, which on a 54 GiB pack costs tens of minutes for a few hundred kilobytes of output.  Copying bytes is O(the
tensors asked for).

    python port/tools/gguf-companion.py BASE.gguf OUT.gguf blk.0.ssm_norm [more names...]

Selection is by exact tensor name.  Values are copied unchanged, so a companion built this way must leave the
model's output bit-identical -- that is the point of the first smoke test.  Requantising is a later step and needs a
quantiser, not this tool.
"""
import struct
import sys

GGUF_MAGIC = 0x46554747
# gguf metadata value types
T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32, T_FLOAT32, T_BOOL, T_STRING, T_ARRAY, \
    T_UINT64, T_INT64, T_FLOAT64 = range(13)
FIXED = {T_UINT8: 1, T_INT8: 1, T_UINT16: 2, T_INT16: 2, T_UINT32: 4, T_INT32: 4, T_FLOAT32: 4,
         T_BOOL: 1, T_UINT64: 8, T_INT64: 8, T_FLOAT64: 8}


def read_str(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', 'replace')


def write_str(b, s):
    e = s.encode('utf-8')
    b += struct.pack('<Q', len(e)) + e


def skip_value(f, t):
    if t in FIXED:
        f.read(FIXED[t])
    elif t == T_STRING:
        read_str(f)
    elif t == T_ARRAY:
        at, n = struct.unpack('<IQ', f.read(12))
        for _ in range(n):
            skip_value(f, at)
    else:
        raise ValueError(f'unknown metadata type {t}')


def parse(path):
    """Header only: metadata, tensor infos, and the data-section offset."""
    with open(path, 'rb') as f:
        magic, version, n_tensors, n_kv = struct.unpack('<IIQQ', f.read(24))
        if magic != GGUF_MAGIC:
            sys.exit(f'{path}: not a gguf')
        for _ in range(n_kv):
            read_str(f)
            t, = struct.unpack('<I', f.read(4))
            skip_value(f, t)
        infos = []
        for _ in range(n_tensors):
            name = read_str(f)
            nd, = struct.unpack('<I', f.read(4))
            dims = list(struct.unpack('<' + 'Q' * nd, f.read(8 * nd)))
            tt, = struct.unpack('<I', f.read(4))
            off, = struct.unpack('<Q', f.read(8))
            infos.append({'name': name, 'dims': dims, 'type': tt, 'off': off})
        data_off = f.tell()
        align, = struct.unpack('<Q', b'\0' * 8)
        # the data section starts after the alignment padding
        while data_off % 32:
            data_off += 1
    infos.sort(key=lambda r: r['off'])
    for i, r in enumerate(infos):
        r['size'] = (infos[i + 1]['off'] - r['off']) if i + 1 < len(infos) else None
    return version, infos, data_off


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    base, out, names = sys.argv[1], sys.argv[2], sys.argv[3:]
    version, infos, data_off = parse(base)
    by_name = {r['name']: r for r in infos}

    missing = [n for n in names if n not in by_name]
    if missing:
        sys.exit('not in the base pack: ' + ', '.join(missing))
    picked = [by_name[n] for n in names]
    for r in picked:
        if r['size'] is None:
            sys.exit(f'{r["name"]}: it is the last tensor in the base pack, so its size cannot be derived from the '
                     'next offset; this tool does not guess it')

    with open(base, 'rb') as f:
        blobs = []
        for r in picked:
            f.seek(data_off + r['off'])
            blobs.append(f.read(r['size']))

    hdr = bytearray()
    hdr += struct.pack('<IIQQ', GGUF_MAGIC, version, len(picked), 2)
    # metadata: the arch is read with required=false, but give the loader something familiar, and the marker
    write_str(hdr, 'general.architecture')
    hdr += struct.pack('<I', T_STRING)
    write_str(hdr, 'strata-companion')
    write_str(hdr, 'strata.companion')
    hdr += struct.pack('<I', T_BOOL) + struct.pack('<B', 1)
    # tensor infos, packed from offset 0 in the order given
    cursor = 0
    align = 32
    for r, blob in zip(picked, blobs):
        write_str(hdr, r['name'])
        hdr += struct.pack('<I', len(r['dims']))
        for d in r['dims']:
            hdr += struct.pack('<Q', d)
        hdr += struct.pack('<I', r['type'])
        hdr += struct.pack('<Q', cursor)
        cursor += (len(blob) + align - 1) // align * align

    while len(hdr) % align:
        hdr += b'\0'

    with open(out, 'wb') as f:
        f.write(hdr)
        pos = 0
        for blob in blobs:
            f.write(blob)
            pos += len(blob)
            pad = (align - pos % align) % align
            if pad:
                f.write(b'\0' * pad)
                pos += pad
    print(f'{out}: {len(picked)} tensor(s), {sum(len(b) for b in blobs)} bytes')


main()
