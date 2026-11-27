"""Generate the executable checklist: assert every fork marker is present in a tree.

Run after a base sync. Prints one line per marker and exits non-zero if any is missing, so a rebase cannot silently drop
the trellis types or the NextN API.
"""
import io
import os
import sys

TARGET = sys.argv[1] if len(sys.argv) > 1 else r"third_party/llama.cpp"

MARKERS = [
    "GGML_TYPE_TQ2_T", "GGML_TYPE_TQ1_0", "GGML_TYPE_TQ2_0", "GGML_TYPE_PTQ1_0",
    "tq2_t_lut.glsl", "ptq1_0.glsl", "FOR_EACH_TRELLIS_TYPE", "dequant_tq2_t", "mul_mat_vec_tq2_0",
    "LLAMA_CONTEXT_TYPE_MTP", "LLM_GRAPH_TYPE_DECODER_MTP", "llama_set_embeddings_nextn",
    "llama_get_embeddings_nextn_ith", "n_layer_nextn", "graph_mtp", "load_mtp", "mtp_path", "llama-ext.h",
    "embeddings_nextn",
]

EXT = {'.cpp', '.h', '.hpp', '.c', '.comp', '.glsl', '.cmake', '.py'}
found = {}
for dp, dn, fn in os.walk(TARGET):
    dn[:] = [d for d in dn if d not in {'.git', 'build'}]
    for f in fn:
        p = os.path.join(dp, f)
        hay = None
        for m in MARKERS:
            if m in f:
                found.setdefault(m, []).append(os.path.relpath(p, TARGET))
                continue
            if os.path.splitext(f)[1].lower() not in EXT and not f.endswith('CMakeLists.txt'):
                continue
            if hay is None:
                try:
                    hay = open(p, encoding='utf-8', errors='ignore').read()
                except OSError:
                    hay = ''
            if m in hay:
                found.setdefault(m, []).append(os.path.relpath(p, TARGET))

missing = [m for m in MARKERS if m not in found]
for m in MARKERS:
    n = len(found.get(m, []))
    print(f"{'ok  ' if n else 'MISS'} {m:32s} {n} file(s)")
print()
print(f"{len(MARKERS) - len(missing)}/{len(MARKERS)} markers present in {TARGET}")
sys.exit(1 if missing else 0)
