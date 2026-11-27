"""Decode a few real blocks of the Hy4 pack with a reference implementation written from the pack's own encoder.

This is the ground truth the C/Vulkan dequantisers will be checked against: it reads the codebooks out of the GGUF
key-values and applies the layout documented in port/docs/features/FUTURE-MODELS.md.
"""
import pathlib
import struct
import numpy as np

PACK = pathlib.Path(r'C:\Users\Fuxfantx\C-Res\Qwen\Hy4-Flash-Compact\Hy4-flash-Uncensored-Compact-00001-of-00002.gguf')

GGUF_MAGIC = 0x46554747


class Reader:
    def __init__(self, path):
        self.f = open(path, 'rb')
        magic, version, n_tensors, n_kv = struct.unpack('<IIQQ', self.f.read(24))
        assert magic == GGUF_MAGIC, hex(magic)
        self.version, self.n_tensors, self.n_kv = version, n_tensors, n_kv
        self.kv = {}
        for _ in range(n_kv):
            key, vtype = self._str(), struct.unpack('<I', self.f.read(4))[0]
            self.kv[key] = self._value(vtype)
        self.tensors = {}
        for _ in range(n_tensors):
            name = self._str()
            n_dims = struct.unpack('<I', self.f.read(4))[0]
            dims = struct.unpack('<%dQ' % n_dims, self.f.read(8 * n_dims))
            ttype = struct.unpack('<I', self.f.read(4))[0]
            offset = struct.unpack('<Q', self.f.read(8))[0]
            self.tensors[name] = (dims, ttype, offset)
        align = 32
        pos = self.f.tell()
        self.data_offset = (pos + align - 1) // align * align

    def _str(self):
        n = struct.unpack('<Q', self.f.read(8))[0]
        return self.f.read(n).decode('utf-8', errors='replace')

    def _value(self, vtype):
        if vtype == 8:
            return self._str()
        if vtype == 9:
            subtype = struct.unpack('<I', self.f.read(4))[0]
            n = struct.unpack('<Q', self.f.read(8))[0]
            return [self._value(subtype) for _ in range(n)]
        fmt = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<B', 10: '<Q', 11: '<q', 12: '<d'}[vtype]
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.f.read(size))[0]

    def tensor_bytes(self, name, nbytes):
        _, _, offset = self.tensors[name]
        self.f.seek(self.data_offset + offset)
        return self.f.read(nbytes)


# layouts: (block elements, block bytes, index bytes, bits, kind, codebook key)
LAYOUT = {
    'CSQ8':     (512, 196, 192, 3, 'scalar', 'prism.csq8.codebook'),
    'CSQ4':     (512, 132, 128, 2, 'scalar', 'prism.csq4.codebook'),
    'VQ4_128':  (512, 116, 112, 7, 'vector', 'prism.vq4_128.codebook'),
    'VQ4_64':   (512, 100,  96, 6, 'vector', 'prism.vq4_64.codebook'),
}
TYPE_NAMES = {144: 'CSQ8', 145: 'CSQ4', 146: 'VQ4_128', 147: 'VQ4_64'}


def unpack_bits(buf, n, bits):
    """LSB-first packing into little-endian 32-bit words, as PackBits does."""
    out = []
    acc = 0
    nb = 0
    pos = 0
    mask = (1 << bits) - 1
    for _ in range(n):
        while nb < bits:
            acc |= int.from_bytes(buf[pos:pos + 4], 'little') << nb
            pos += 4
            nb += 32
        out.append(acc & mask)
        acc >>= bits
        nb -= bits
    return out


def dequant_block(kind, block, codebook):
    bel, bbytes, ib, bits, _, _ = LAYOUT[kind]
    payload = block[:ib]
    scale = struct.unpack('<f', block[ib:ib + 4])[0]
    if kind == 'CSQ4':
        idx = []
        for byte in payload:
            idx += [byte & 3, (byte >> 2) & 3, (byte >> 4) & 3, (byte >> 6) & 3]
    elif kind == 'CSQ8':
        idx = unpack_bits(payload, bel, bits)
    else:
        idx = unpack_bits(payload, bel // 4, bits)
    if kind in ('CSQ8', 'CSQ4'):
        return [scale * codebook[i] for i in idx]
    out = []
    for i in idx:
        grp = codebook[i * 4:i * 4 + 4]
        out += [scale * v for v in grp]
    return out




def dump(outdir):
    """Write one real block, its codebook and the expected values for each type, for the C probe to compare against."""
    import os
    os.makedirs(outdir, exist_ok=True)
    r = Reader(PACK)
    want = [('CSQ8', 'token_embd.weight'),
            ('CSQ4', 'blk.1.ffn_gate_exps_hot.weight'),
            ('VQ4_128', 'blk.1.ffn_gate_exps_warm.weight'),
            ('VQ4_64', 'blk.1.ffn_gate_exps_cold.weight')]
    for kind, name in want:
        bel, bbytes, ib, bits, _, cbkey = LAYOUT[kind]
        codebook = r.kv[cbkey]
        raw = r.tensor_bytes(name, bbytes)
        vals = dequant_block(kind, raw, codebook)
        base = os.path.join(outdir, kind.lower())
        with open(base + '-block.bin', 'wb') as f:
            f.write(raw)
        with open(base + '-cb.bin', 'wb') as f:
            f.write(struct.pack('<%df' % len(codebook), *codebook))
        with open(base + '-expected.txt', 'w', encoding='utf-8') as f:
            f.write('%d\n' % len(vals))
            for v in vals:
                f.write('%.9g\n' % v)
        print('dumped', kind, name, len(vals), 'values')


def main():
    r = Reader(PACK)
    print('tensors:', r.n_tensors, 'kvs:', r.n_kv)

    for name in ('blk.1.ffn_gate_exps_hot.weight', 'blk.1.ffn_gate_exps_warm.weight',
                 'blk.1.ffn_gate_exps_cold.weight', 'token_embd.weight'):
        dims, ttype, _ = r.tensors[name]
        kind = TYPE_NAMES.get(ttype)
        if kind is None:
            print('%-40s type %d (skipped)' % (name, ttype))
            continue
        bel, bbytes, _, _, _, cbkey = LAYOUT[kind]
        codebook = r.kv[cbkey]
        row_len = dims[0]
        nbytes = row_len // bel * bbytes
        raw = r.tensor_bytes(name, nbytes)
        vals = dequant_block(kind, raw[:bbytes], codebook)
        print('%-40s %-8s dims=%s scale=%.6g first=%s' % (
            name, kind, dims, struct.unpack('<f', raw[bbytes - 4:bbytes])[0],
            ', '.join('%.5f' % v for v in vals[:4])))


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == '--dump':
        dump(sys.argv[2])
    else:
        main()
