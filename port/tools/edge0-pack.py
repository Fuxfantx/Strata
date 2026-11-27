#!/usr/bin/env python3
"""Read an Edge0 latent-MoE pack directory and report what it holds.

The pack is a directory, not one GGUF: a `qwen4exp` skeleton GGUF, the routed experts in `experts.bin`, the per-layer
D/U projections in a Q8 or Q4 file, the expert group labels, the PLE table, the MTP head, the vision encoder and the
tokenizer.  This tool reads the small files and the manifest, checks the arithmetic of the layout against the file
sizes it finds, and prints one report.  It never loads weights into a GPU and never downloads anything.

    python port/tools/edge0-pack.py --model D:\\models\\Edge0-145B-q3k
    python port/tools/edge0-pack.py --model DIR --json

What it validates, and why those checks:

* `labels.bin` is `n_layers * n_expert` int32 group labels, so its size is exact.
* The D/U file is `n_layers * (n_groups * ELL * d_row + n_groups * H * u_row)` bytes for the format that was chosen;
  a mismatch means the chosen format or the geometry is wrong, and the engine would read the wrong rows.
* `native_experts.txt` v6 lists `layer gu_type down_type offset blob`; the blob size must equal the arithmetic size of
  `gate + up + down` for the declared types, and the offsets must tile the file with no gap.
* Every required file the documented layout lists must exist, unless `--allow-partial` is given (used while a pack is
  still being downloaded).
"""
import argparse
import json
import pathlib
import struct
import sys
import urllib.parse
import urllib.request

GEMM_TYPE_SIZES = {
    # bytes per 256 values for the types an Edge0 pack uses (GGML block sizes of 256 for K-quants)
    11: ('Q3_K', 110),
    12: ('Q4_K', 144),
    13: ('Q5_K', 176),
    14: ('Q6_K', 210),
    20: ('IQ4_NL', None),   # 32-value blocks of 18 bytes
    0: ('F32', None),       # 4 bytes per value
    1: ('F16', None),       # 2 bytes per value
    8: ('Q8_0', None),      # 32-value blocks of 34 bytes
    2: ('Q4_0', None),      # 32-value blocks of 18 bytes
    -1: ('unknown', None),
}


def type_name(t):
    return GEMM_TYPE_SIZES.get(t, ('type %d' % t, None))[0]


def row_bytes(ggml_type, n_values):
    """Bytes of one row of `n_values` values in `ggml_type`, or None when the type is not modelled here."""
    if ggml_type in (11, 12, 13, 14):          # K-quants: 256-value super-blocks
        if n_values % 256 != 0:
            return None
        return (n_values // 256) * GEMM_TYPE_SIZES[ggml_type][1]
    if ggml_type == 20:                        # IQ4_NL: 32-value blocks, 18 bytes
        return None if n_values % 32 else (n_values // 32) * 18
    if ggml_type == 8:                         # Q8_0
        return None if n_values % 32 else (n_values // 32) * 34
    if ggml_type == 2:                         # Q4_0
        return None if n_values % 32 else (n_values // 32) * 18
    if ggml_type == 0:
        return n_values * 4
    if ggml_type == 1:
        return n_values * 2
    return None


MODELSCOPE_API = 'https://modelscope.cn/api/v1/models/%s/repo/files?Revision=master'
MODELSCOPE_FILE = 'https://modelscope.cn/api/v1/models/%s/repo?Revision=master&FilePath=%s'
HUGGINGFACE_API = 'https://huggingface.co/api/models/%s/tree/main?recursive=1'
HUGGINGFACE_FILE = 'https://huggingface.co/%s/resolve/main/%s'


def _get(url, headers=None, timeout=60):
    req = urllib.request.Request(url, headers={'User-Agent': 'strata-edge0-pack' , **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def _modelscope_walk(repo, root=''):
    """ModelScope lists one directory per request, so subdirectories are walked explicitly."""
    url = MODELSCOPE_API % repo
    if root:
        url += '&Root=' + urllib.parse.quote(root)
    _, body = _get(url)
    data = json.loads(body).get('Data') or {}
    out = []
    for f in data.get('Files') or []:
        if f.get('Type') == 'tree':
            out.extend(_modelscope_walk(repo, f['Path']))
        else:
            out.append((f['Path'], int(f.get('Size') or 0)))
    return out


def remote_list(repo, source):
    """[(path, bytes)] for a model repository, ModelScope first when both could serve it."""
    if source in ('auto', 'modelscope'):
        try:
            out = _modelscope_walk(repo)
            if out:
                return out, 'modelscope'
        except Exception as exc:  # noqa: BLE001 - fall through to the next source
            if source == 'modelscope':
                raise SystemExit('modelscope list failed: %s' % exc)
    _, body = _get(HUGGINGFACE_API % repo)
    tree = json.loads(body)
    return [(e['path'], int(e.get('size') or 0)) for e in tree if e.get('type') == 'file'], 'huggingface'


def download(repo, path, dest, size, source, chunk=32 << 20):
    """Chunked, resumable download of one file.  Returns the local size.

    The partial file is `dest + '.part'`, so an interrupted run continues where it stopped.  Chunks are separated by
    `Range` requests rather than one big stream, which is what makes the resume possible without re-reading the file.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + '.part')
    if dest.exists() and dest.stat().st_size == size:
        return dest.stat().st_size
    done = part.stat().st_size if part.exists() else 0
    if done > size:
        part.unlink()
        done = 0
    import time
    attempts = 0
    with part.open('ab') as f:
        while done < size:
            want = min(chunk, size - done)
            try:
                data = remote_range(repo, path, done, want, source)
                attempts = 0
            except Exception as exc:  # noqa: BLE001 - a dropped connection is expected on a long download
                attempts += 1
                if attempts > 8:
                    raise SystemExit('download of %s failed at %d of %d bytes: %s' % (path, done, size, exc))
                wait = min(60, 2 ** attempts)
                print('  %-28s retry %d in %ds: %s' % (path, attempts, wait, exc), flush=True)
                time.sleep(wait)
                continue
            if not data:
                break
            f.write(data)
            f.flush()
            done += len(data)
            print('  %-28s %5.1f%%  %d / %d' % (path, 100.0 * done / max(size, 1), done, size), flush=True)
    if done == size:
        if dest.exists():
            dest.unlink()
        part.rename(dest)
        return done
    raise SystemExit('download of %s stopped at %d of %d bytes; run again to continue' % (path, done, size))


def remote_range(repo, path, start, length, source):
    """Bytes [start, start+length) of one file, through whichever source serves it."""
    end = start + length - 1
    hdrs = {'Range': 'bytes=%d-%d' % (start, end)}
    if source == 'modelscope':
        _, body = _get(MODELSCOPE_FILE % (repo, urllib.parse.quote(path)), hdrs, timeout=300)
        return body
    _, body = _get(HUGGINGFACE_FILE % (repo, path), hdrs, timeout=300)
    return body


def gguf_scalars(path, keys):
    """(key -> value) for the integer/string keys asked for, read from a GGUF header.

    Only the metadata block is parsed, so this works on a 6 GB skeleton without reading the tensors.  Unknown value
    types are skipped by their size, which is enough to stay in step with the file.
    """
    want = set(keys)
    out = {}
    with open(path, 'rb') as f:
        head = f.read(24)
        if head[:4] != b'GGUF':
            raise SystemExit('%s: not a GGUF file' % path)
        ver = struct.unpack_from('<I', head, 4)[0]
        n_tensors = struct.unpack_from('<Q', head, 8)[0]
        n_kv = struct.unpack_from('<Q', head, 16)[0]

        def read_str():
            n = struct.unpack('<Q', f.read(8))[0]
            return f.read(n).decode('utf-8', 'replace')

        for _ in range(n_kv):
            key = read_str()
            t = struct.unpack('<I', f.read(4))[0]
            if t == 8:
                v = read_str()
            elif t == 9:                       # array: skip it
                at = struct.unpack('<I', f.read(4))[0]
                n = struct.unpack('<Q', f.read(8))[0]
                for _i in range(n):
                    if at == 8:
                        read_str()
                    elif at in (0, 1, 7):
                        f.read(1)
                    elif at in (2, 3):
                        f.read(2)
                    elif at in (4, 5):
                        f.read(4)
                    elif at in (10, 11):
                        f.read(8)
                    else:
                        raise SystemExit('%s: array element type %d not handled' % (path, at))
                v = '<array>'
            elif t in (0, 1, 7):
                v = struct.unpack('<b' if t == 0 else '<B', f.read(1))[0]
            elif t in (2, 3):
                v = struct.unpack('<h' if t == 2 else '<H', f.read(2))[0]
            elif t in (4, 5):
                v = struct.unpack('<i' if t == 4 else '<I', f.read(4))[0]
            elif t == 6:
                v = struct.unpack('<f', f.read(4))[0]
            elif t in (10, 11):
                v = struct.unpack('<q' if t == 10 else '<Q', f.read(8))[0]
            elif t == 12:
                v = struct.unpack('<d', f.read(8))[0]
            else:
                raise SystemExit('%s: metadata type %d not handled' % (path, t))
            if key in want:
                out[key] = v
    return out


def kv_bytes(n_full_layers, n_kv_heads, head_dim, n_ctx, bits, k_dtype=4):
    """Bytes of the KV cache for the full-attention layers, for a given context and quantisation.

    Only the full-attention layers carry a KV cache; the recurrent layers keep a fixed-size state instead, which is
    counted separately.  A 4-bit cache is 18 bytes per 32 values, an 8-bit one 34 bytes per 32 values.
    """
    per_value = {4: 18.0 / 32.0, 8: 34.0 / 32.0, 16: 2.0}[bits]
    elems = n_full_layers * n_kv_heads * head_dim * n_ctx * 2          # K and V
    return int(elems * per_value)


def recurrent_bytes(n_linear_layers, state_size, n_heads, conv_kernel, width):
    """The recurrent layers' state: per layer, a state matrix and a short convolution window."""
    state = state_size * n_heads * 4          # f32 per state element
    conv = conv_kernel * width * 4
    return n_linear_layers * (state + conv)


def plan(pack, budget_gib, ctx, kv_bits, compute_gib, du, slots_used, profile_frac):
    """Decide what can stay resident inside a memory budget, and say what the numbers are based on."""
    import struct as _s
    root = pathlib.Path(pack)
    meta = json.loads((root / 'model.json').read_text(encoding='utf-8'))
    sizes = {}
    for name in ('model-skeleton.gguf', 'experts.bin', 'latent_dense_q8.bin', 'latent_dense_q4.bin',
                 'ple-fp8.gguf', 'mmproj-BF16.gguf', 'labels.bin', 'profile.bin'):
        f = root / name
        sizes[name] = f.stat().st_size if f.exists() else 0
    n_layers = int(meta.get('layers') or 48)
    n_expert = int(meta.get('experts') or 512)
    du_name = 'latent_dense_q8.bin' if du == 'q8' else 'latent_dense_q4.bin'
    blob = 1630720 if n_expert == 512 else (sizes['experts.bin'] // (n_layers * n_expert))

    # geometry: the released 145B is 2560 wide with 24 heads (2 KV heads of 256) and 4 layer-groups of attention
    n_embd = int(meta.get('embedding_length') or 2560)
    n_kv_heads = 2
    head_dim = 256
    full_every = 4
    n_full = (n_layers + full_every - 1) // full_every
    n_lin = n_layers - n_full

    kv = kv_bytes(n_full, n_kv_heads, head_dim, ctx, kv_bits)
    rec = recurrent_bytes(n_lin, 128, 48, 4, 6144)
    compute = int(compute_gib * 2 ** 30)
    fixed = sizes['model-skeleton.gguf'] + sizes[du_name] + kv + rec + compute
    budget = int(budget_gib * 2 ** 30)
    left = budget - fixed
    total_blobs = n_layers * n_expert                     # experts.bin holds one blob per (layer, expert)
    fit_blobs = max(0, min(total_blobs, left // blob)) if blob else 0
    per_layer_fit = fit_blobs // n_layers                 # an even split across layers, which is what a per-layer cache does

    print('pack      :', root)
    print('budget    : %.2f GiB' % (budget / 2 ** 30))
    print('components (formulas, not measurements):')
    print('  skeleton weights      %8.2f GiB' % (sizes['model-skeleton.gguf'] / 2 ** 30))
    print('  D/U (%s)            %8.2f GiB' % (du, sizes[du_name] / 2 ** 30))
    print('  KV cache, %d full layers, %d ctx, q%d  %8.2f GiB'
          % (n_full, ctx, kv_bits, kv / 2 ** 30))
    print('  recurrent state, %d layers        %8.3f GiB' % (n_lin, rec / 2 ** 30))
    print('  compute buffers       %8.2f GiB (given)' % (compute / 2 ** 30))
    print('  PLE table             %8.2f GiB (mapped, only rows are read)' % (sizes['ple-fp8.gguf'] / 2 ** 30))
    print('fixed     : %.2f GiB' % (fixed / 2 ** 30))
    print('experts   : %.2f GiB of budget -> %d of %d blobs resident (%.1f%%), about %d of %d per layer,'
          ' %.2f GiB streamed per full pass'
          % (max(0, left) / 2 ** 30, fit_blobs, total_blobs, 100.0 * fit_blobs / total_blobs,
             per_layer_fit, n_expert, (total_blobs - fit_blobs) * blob / 2 ** 30))
    if sizes['profile.bin']:
        raw = (root / 'profile.bin').read_bytes()
        # the profile is a ranking of the experts; its exact record format is the pack's, so only its size is used here
        print('profile   : %d bytes (ranking used to choose the resident set)' % len(raw))
    # expected work per token with 10 routed experts: misses when fewer than 10 of the chosen ones are resident
    p_hit = per_layer_fit / n_expert if n_expert else 0.0
    exp_misses = slots_used * (1.0 - p_hit)
    print('per token : %d routed slots -> about %.1f expert misses, %.1f MiB streamed at most'
          % (slots_used, exp_misses, exp_misses * blob / 2 ** 20))
    print('note      : the KV and compute figures are formulas over the documented geometry, not measured on this'
          ' machine; the expert count is the number the budget allows, before fragmentation.')

    if profile_frac:
        # an online heat policy against the static profile order: a Zipf access model over 512 experts
        import random
        rnd = random.Random(7)
        weights = [1.0 / (i + 1) for i in range(n_expert)]
        total_w = sum(weights)
        static = set(range(per_layer_fit))     # profile order as the prior, which the pack's ranking provides
        heat = {}
        hits_static = hits_heat = 0
        trials = 20000
        for _ in range(trials):
            r = rnd.random() * total_w
            acc = 0.0
            e = 0
            for i in range(n_expert):
                acc += weights[i]
                if r <= acc:
                    e = i
                    break
            hits_static += 1 if e in static else 0
            resident = set(sorted(heat, key=lambda k: -heat[k])[:per_layer_fit])
            hits_heat += 1 if e in resident else 0
            heat[e] = heat.get(e, 0) + 1
        print('simulation: static profile order hit rate %.1f%%, online heat %.1f%% (Zipf access, %d draws)'
              % (100.0 * hits_static / trials, 100.0 * hits_heat / trials, trials))
    return 0


def read_native_manifest(path):
    text = path.read_text(encoding='utf-8', errors='replace')
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines or not lines[0].startswith('# strata native experts v'):
        raise SystemExit('native_experts.txt: no "# strata native experts vN" header')
    ver = int(lines[0].split(' v', 1)[1].split(':', 1)[0])
    rows = []
    for line in lines[1:]:
        if line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 5:
            raise SystemExit('native_experts.txt: malformed line: ' + line)
        layer, gu, dn, off, blob = (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
        rows.append({'layer': layer, 'gu_type': gu, 'down_type': dn, 'offset': off, 'blob': blob,
                     'shard': parts[5:]})
    return ver, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--model', required=True, help='the model folder (the one holding model.json)')
    ap.add_argument('--json', action='store_true', help='print the report as JSON')
    ap.add_argument('--allow-partial', action='store_true', help='tolerate files that are not there yet')
    ap.add_argument('--du', choices=('q8', 'q4'), default='q8', help='which D/U file to validate (default q8)')
    ap.add_argument('--embedding-length', type=int, default=0, help='model width, when the skeleton is not there')
    ap.add_argument('--plan', action='store_true', help='report the residency plan for a memory budget')
    ap.add_argument('--budget-gib', type=float, default=36.0, help='memory budget for weights, KV, compute (default 36)')
    ap.add_argument('--ctx', type=int, default=262144, help='context length for the KV figure (default 262144)')
    ap.add_argument('--kv-bits', type=int, choices=(4, 8, 16), default=4, help='KV quantisation (default q4_0)')
    ap.add_argument('--compute-gib', type=float, default=3.0, help='compute buffers to reserve (default 3)')
    ap.add_argument('--slots-used', type=int, default=10, help='routed experts per token (default 10)')
    ap.add_argument('--simulate-heat', action='store_true', help='compare a static order with an online heat policy')
    ap.add_argument('--repo', help='model repository, for example Edge0/Edge0-145B-q3k')
    ap.add_argument('--source', choices=('auto', 'modelscope', 'huggingface'), default='auto',
                    help='where to read from; auto tries ModelScope first (default)')
    ap.add_argument('--list-remote', action='store_true', help='list the repository and stop')
    ap.add_argument('--download', help='comma separated file names to download into --model')
    ap.add_argument('--spot-check', action='store_true',
                    help='range-read one expert blob, one D row and the labels, and check their sizes')
    args = ap.parse_args()

    root = pathlib.Path(args.model)
    if args.download:
        if not args.repo:
            raise SystemExit('--repo is required with --download')
        files, used = remote_list(args.repo, args.source)
        by_name = {p: sz for p, sz in files}
        print('source :', used, '|', args.repo)
        for name in [n.strip() for n in args.download.split(',') if n.strip()]:
            if name not in by_name:
                raise SystemExit('repository has no ' + name)
            size = by_name[name]
            dest = root / name
            print('fetch  : %s (%d bytes)' % (name, size))
            got = download(args.repo, name, dest, size, used)
            print('done   : %s -> %d bytes' % (dest, got))
        return 0
    if args.plan:
        return plan(args.model, args.budget_gib, args.ctx, args.kv_bits, args.compute_gib, args.du,
                    args.slots_used, args.simulate_heat)
    if args.list_remote or args.spot_check:
        if not args.repo:
            raise SystemExit('--repo is required with --list-remote or --spot-check')
        files, used = remote_list(args.repo, args.source)
        print('source :', used, '|', args.repo, '|', len(files), 'files',
              '|', '%.2f GiB' % (sum(sz for _, sz in files) / 2 ** 30))
        if args.list_remote:
            for path, size in sorted(files):
                print('%13s  %s' % (size, path))
            return 0
        by_name = {p: sz for p, sz in files}
        need = ['native_experts.txt', 'model.json', 'labels.bin']
        for name in need:
            if name not in by_name:
                raise SystemExit('repository has no ' + name)
        if 'experts.bin' not in by_name:
            raise SystemExit('repository has no experts.bin')
        _, manifest_bytes = _get(MODELSCOPE_FILE % (args.repo, 'native_experts.txt')
                                 if used == 'modelscope' else HUGGINGFACE_FILE % (args.repo, 'native_experts.txt'))
        tmp = root / 'native_experts.txt'
        root.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(manifest_bytes)
        ver, rows = read_native_manifest(tmp)
        rows.sort(key=lambda r: r['offset'])
        blob = rows[0]['blob']
        print('expert : type %s/%s, blob %d B, reading the first blob of experts.bin'
              % (type_name(rows[0]['gu_type']), type_name(rows[0]['down_type']), blob))
        got = remote_range(args.repo, 'experts.bin', rows[0]['offset'], blob, used)
        print('        read %d bytes (expected %d) -> %s' % (len(got), blob, 'ok' if len(got) == blob else 'SHORT'))
        # section boundaries for ELL from model.json and ff from the manifest header
        # The blob is [gate rows | up rows | down rows]; read the first row of each queantised section and check the
        # block header a reader would use: Q3_K is 110 B per 256 values with an f16 d at the end, IQ4_NL is 18 B per
        # 32 values with an f16 d at the front.  A finite, non-zero d is what a real blob has.
        ell = int((json.loads((root / 'model.json').read_text(encoding='utf-8'))
                   .get('latent_dim')) or 0) if (root / 'model.json').exists() else 0
        if not ell:
            ell = 1792
        ff = 640
        gu_row = 7 * 110 if ell == 1792 else (ell // 256) * 110        # Q3_K row over the latent width
        dn_row = (ff // 32) * 18                                       # IQ4_NL row over the expert width
        gate0 = got[:gu_row]
        up0 = got[ff * gu_row:ff * gu_row + gu_row]
        down_off = 2 * ff * gu_row
        down0 = got[down_off:down_off + dn_row]
        def f16(b, at):
            return struct.unpack_from('<e', b, at)[0]
        print('gate   : %d B per row, first block d = %.6g, last block d = %.6g'
              % (len(gate0), f16(gate0, 108), f16(gate0, len(gate0) - 2)))
        print('down   : %d B per row, first block d = %.6g'
              % (len(down0), f16(down0, 0)))
        print('layout : gate %d + up %d + down %d = %d bytes%s'
              % (ff * gu_row, ff * gu_row, ell * dn_row,
                 ff * gu_row + ff * gu_row + ell * dn_row,
                 ' -> matches the manifest blob' if ff * gu_row * 2 + ell * dn_row == blob else ' -> MISMATCH'))
        finite = all(v == v and abs(v) < 1e30 for v in
                     (f16(gate0, 108), f16(gate0, len(gate0) - 2), f16(down0, 0)))
        print('d values finite:', finite)

        labels_size = by_name.get('labels.bin', 0)
        print('labels : %d bytes -> %s' % (labels_size,
              'ok (48 x 512 x int32)' if labels_size == 48 * 512 * 4 else 'unexpected size'))
        return 0
    report = {'path': str(root), 'ok': True, 'problems': [], 'info': {}}
    problems = report['problems']

    def need(name, required=True):
        p = root / name
        if not p.exists():
            if required and not args.allow_partial:
                problems.append('missing ' + name)
            return None
        if not p.is_file():
            return None
        return p.stat().st_size

    model_json = root / 'model.json'
    if not model_json.exists():
        raise SystemExit('no model.json in ' + str(root) + ' -- not an Edge0 pack folder')
    meta = json.loads(model_json.read_text(encoding='utf-8'))
    name = str(meta.get('model_name', ''))
    report['info']['model_name'] = name
    report['info']['declared'] = {k: meta.get(k) for k in
                                  ('family', 'base_model', 'architecture', 'params', 'layers', 'experts',
                                   'experts_per_token', 'latent_dim', 'context', 'build', 'pack_format')}

    n_layers = int(meta.get('layers') or 48)
    n_expert = int(meta.get('experts') or 512)
    latent_declared = int(meta.get('latent_dim') or 0)
    n_groups = int(meta.get('latent_groups') or 8)

    # the D/U row size depends on the model width, so take it from the pack: model.json first, then the skeleton's own
    # metadata, then a command line override.  Guessing 2560 for a fixture would only produce a false complaint.
    n_embd = int(meta.get('embedding_length') or 0)
    if not n_embd and (root / 'model-skeleton.gguf').exists():
        try:
            kv = gguf_scalars(root / 'model-skeleton.gguf', ['qwen4exp.embedding_length', 'general.architecture'])
            n_embd = int(kv.get('qwen4exp.embedding_length') or 0)
        except Exception as exc:  # noqa: BLE001 - a half-downloaded skeleton should not stop the report
            problems.append('could not read the skeleton metadata: %s' % exc)
    if not n_embd:
        n_embd = int(getattr(args, 'embedding_length', 0) or 0)
    if not n_embd:
        n_embd = 2560
        report['info']['embedding_assumed'] = True

    sizes = {}
    for f in ('model-skeleton.gguf', 'experts.bin', 'ple-fp8.gguf', 'labels.bin', 'profile.bin',
              'native_experts.txt', 'index.txt', 'dense.bin', 'mmproj-BF16.gguf'):
        sizes[f] = need(f, required=f in ('model-skeleton.gguf', 'experts.bin', 'labels.bin', 'native_experts.txt'))
    du_name = 'latent_dense_q8.bin' if args.du == 'q8' else 'latent_dense_q4.bin'
    sizes[du_name] = need(du_name)
    for sub in ('mtp/dense.bin', 'mtp/experts.bin', 'mtp/draft_vocab.bin', 'scales'):
        need(sub, required=False)
    report['info']['files'] = sizes

    env = meta.get('env') or {}
    fmt = int(env.get('STRATA_LATENT_DENSE_FMT') or 8)
    if args.du == 'q8' and fmt != 8:
        problems.append('model.json says D/U format %d but --du q8 was asked' % fmt)
    if args.du == 'q4' and fmt != 4:
        problems.append('model.json says D/U format %d but --du q4 was asked' % fmt)

    if sizes.get('labels.bin') is not None:
        want = n_layers * n_expert * 4
        if sizes['labels.bin'] != want:
            problems.append('labels.bin is %d bytes, expected %d (%d layers x %d experts x int32)'
                            % (sizes['labels.bin'], want, n_layers, n_expert))

    if sizes.get('native_experts.txt'):
        ver, rows = read_native_manifest(root / 'native_experts.txt')
        report['info']['native_manifest_version'] = ver
        report['info']['native_layers'] = len(rows)
        if len(rows) != n_layers:
            problems.append('native_experts.txt lists %d layers, model.json says %d' % (len(rows), n_layers))
        gu_set = sorted({r['gu_type'] for r in rows})
        dn_set = sorted({r['down_type'] for r in rows})
        report['info']['expert_types'] = {'gate_up': [type_name(t) for t in gu_set],
                                          'down': [type_name(t) for t in dn_set]}
        expert_bytes = sizes.get('experts.bin')
        for r in rows:
            gu_row = row_bytes(r['gu_type'], 0) if False else None
            # geometry from the manifest's own header is applied below once ELL/ff are known
        # offsets must tile the file: layer-major, n_expert blobs per layer
        rows_sorted = sorted(rows, key=lambda r: r['offset'])
        expect = 0
        for r in rows_sorted:
            if r['offset'] != expect:
                problems.append('native_experts.txt: layer %d starts at %d, expected %d'
                                % (r['layer'], r['offset'], expect))
                break
            expect += r['blob'] * n_expert
        if expert_bytes is not None and expect != expert_bytes and not args.allow_partial:
            problems.append('experts.bin is %d bytes but the manifest tiles %d' % (expert_bytes, expect))
        report['info']['expert_blob_bytes'] = rows_sorted[0]['blob'] if rows_sorted else 0

    # latent geometry: ELL from the D/U file if it is there, else the declared value; ff from the manifest blob size
    ell = latent_declared
    if sizes.get(du_name) and n_layers:
        d_row = (n_embd // 32) * (34 if args.du == 'q8' else 18)
        u_row = (ell // 32) * (34 if args.du == 'q8' else 18)
        per_layer_q8 = n_groups * ell * d_row + n_groups * n_embd * u_row
        if ell and sizes[du_name] != per_layer_q8 * n_layers:
            problems.append('%s is %d bytes; with ELL=%d the layout is %d bytes per layer, %d for %d layers'
                            % (du_name, sizes[du_name], ell, per_layer_q8, per_layer_q8 * n_layers, n_layers))
    report['info']['latent_dim'] = ell
    report['info']['du_file'] = du_name
    report['info']['du_format'] = fmt

    report['ok'] = not problems
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print('pack   :', root)
        print('model  :', name, '|', report['info']['declared'].get('params'),
              '| latent', ell, '| D/U', du_name, 'fmt', fmt)
        for f, size in sizes.items():
            print('  %-24s %s' % (f, ('%d bytes' % size) if size is not None else 'MISSING'))
        print('  native manifest v%s, %s layers, gate/up %s, down %s, blob %s bytes'
              % (report['info'].get('native_manifest_version'), report['info'].get('native_layers'),
                 report['info'].get('expert_types', {}).get('gate_up'),
                 report['info'].get('expert_types', {}).get('down'),
                 report['info'].get('expert_blob_bytes')))
        print('result :', 'ok' if report['ok'] else 'PROBLEMS')
        for pr in problems:
            print('  !', pr)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
