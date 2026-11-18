// requant-tensor - write a companion GGUF holding a few tensors of a base pack, requantised to another type.
//
// The engine's --compat composes such a file over the base pack (a tensor in the companion replaces the base pack's
// tensor of the same name; the marker is strata.companion = true).  This exists because llama-quantize cannot do it:
// its `--only-tensors` does not filter what gets written, and the option is not in the vendored source at all
// (PERFORMANCE.md section 5.0.1).
//
//   requant-tensor BASE.gguf OUT.gguf src=attn_qkv:q4_K src=ssm_out:q4_K ...
//
// Each argument is `<name-substring>:<ggml type name>`; every base tensor whose name contains the substring is
// written, so `attn_q:q4_K` and `attn_qkv:q4_K` are different selections.  The tensors are dequantised with the
// source type's to_float trait, requantised with ggml_quantize_chunk, and the companion is written with ggml's own
// GGUF writer, so the file layout is whatever gguf_write_to_file produces.
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// long is 32-bit on Windows even on ARM64, and a tensor offset in a 54 GiB pack does not fit in it

namespace {

struct pick {
    std::string   name;      // tensor name in the base pack
    enum ggml_type type;     // destination type
};

// A pattern selects a tensor when it matches a whole run of dot-separated components, so "ssm_a" cannot pick up
// "ssm_alpha.weight" while "attn_qkv.weight" still picks up every layer.
bool selects(const char * name, const std::string & pat) {
    const std::string n = name;
    if (n == pat) return true;
    for (size_t pos = n.find(pat); pos != std::string::npos; pos = n.find(pat, pos + 1)) {
        const bool head_ok = pos == 0 || n[pos - 1] == '.';
        const size_t end   = pos + pat.size();
        const bool tail_ok = end == n.size() || n[end] == '.';
        if (head_ok && tail_ok) return true;
    }
    return false;
}

ggml_type type_by_name(const std::string & n) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        if (n == ggml_type_name((ggml_type) t)) return (ggml_type) t;
    }
    return GGML_TYPE_COUNT;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s BASE.gguf OUT.gguf substr:type [substr:type ...]\n", argv[0]);
        return 2;
    }
    const char * base_path = argv[1];
    const char * out_path  = argv[2];

    struct ggml_context * meta_ctx = nullptr;
    struct gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &meta_ctx };
    struct gguf_context * base = gguf_init_from_file(base_path, gp);
    if (base == nullptr) {
        std::fprintf(stderr, "not a gguf: %s\n", base_path);
        return 1;
    }

    // resolve the selection
    std::vector<pick> picks;
    for (int a = 3; a < argc; ++a) {
        std::string spec = argv[a];
        const size_t colon = spec.rfind(':');
        if (colon == std::string::npos) {
            std::fprintf(stderr, "bad selection '%s' (want substr:type)\n", spec.c_str());
            return 2;
        }
        const std::string pat  = spec.substr(0, colon);
        const ggml_type   type = type_by_name(spec.substr(colon + 1));
        if (type == GGML_TYPE_COUNT) {
            std::fprintf(stderr, "unknown type '%s'\n", spec.substr(colon + 1).c_str());
            return 2;
        }
        bool any = false;
        for (int64_t i = 0; i < gguf_get_n_tensors(base); ++i) {
            const char * name = gguf_get_tensor_name(base, i);
            if (!selects(name, pat)) continue;
            picks.push_back({ name, type });
            any = true;
        }
        if (!any) {
            std::fprintf(stderr, "no tensor in %s matches '%s'\n", base_path, pat.c_str());
            return 1;
        }
    }
    std::fprintf(stderr, "%zu tensor(s) selected\n", picks.size());

    // read the base file's tensor bytes directly: the metadata context has no data
    FILE * f = std::fopen(base_path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", base_path); return 1; }
    const size_t data_off = gguf_get_data_offset(base);

    // The output context holds a quantised copy of every selected tensor, so it has to be sized from the
    // destinations; a fixed 1 MiB pool fails on the first multi-megabyte group with "not enough space in the
    // context's memory pool".
    size_t need = (size_t) 32 << 20;
    for (const pick & p : picks) {
        for (struct ggml_tensor * cur = ggml_get_first_tensor(meta_ctx); cur; cur = ggml_get_next_tensor(meta_ctx, cur)) {
            if (p.name != cur->name) continue;
            // the same guard as the writing loop: a narrow tensor (ne0 = 320 or 640) cannot take a 256-block type
            if (cur->ne[0] % ggml_blck_size(p.type) == 0) {
                need += ggml_row_size(p.type, cur->ne[0]) * (size_t) (cur->ne[1] * cur->ne[2] * cur->ne[3]);
            }
            break;
        }
    }
    struct ggml_init_params ip = { /*.mem_size =*/ need, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
    struct ggml_context * out_ctx = ggml_init(ip);
    if (out_ctx == nullptr) { std::fprintf(stderr, "cannot allocate a %zu byte context\n", need); return 1; }

    struct gguf_context * out = gguf_init_empty();
    gguf_set_val_bool(out, "strata.companion", true);
    const int arch_key = gguf_find_key(base, "general.architecture");
    if (arch_key >= 0) {
        gguf_set_val_str(out, "general.architecture", gguf_get_val_str(base, arch_key));
    } else {
        gguf_set_val_str(out, "general.architecture", "strata-companion");
    }

    for (const pick & p : picks) {
        const int idx = gguf_find_tensor(base, p.name.c_str());
        if (idx < 0) { std::fprintf(stderr, "%s vanished\n", p.name.c_str()); return 1; }
        const ggml_type src_type = gguf_get_tensor_type(base, idx);
        const int n_dims = 1;   // filled below from the metadata context
        (void) n_dims;

        // shape and source type come from the metadata context, which holds a tensor for every entry
        struct ggml_tensor * src_meta = nullptr;
        for (struct ggml_tensor * cur = ggml_get_first_tensor(meta_ctx); cur; cur = ggml_get_next_tensor(meta_ctx, cur)) {
            if (p.name == cur->name) { src_meta = cur; break; }
        }
        if (src_meta == nullptr) { std::fprintf(stderr, "%s not in the metadata context\n", p.name.c_str()); return 1; }

        const int64_t ne0 = src_meta->ne[0];
        const int64_t ne1 = src_meta->ne[1];
        const int64_t ne2 = src_meta->ne[2];
        const int64_t ne3 = src_meta->ne[3];
        const int64_t nrows = ne1 * ne2 * ne3;

        if (ne0 % ggml_blck_size(p.type) != 0) {
            std::fprintf(stderr, "%s: ne0 %lld is not a multiple of %s's block size %lld; skipped\n",
                         p.name.c_str(), (long long) ne0, ggml_type_name(p.type),
                         (long long) ggml_blck_size(p.type));
            continue;
        }

        // dequantise the source into f32, row by row
        const size_t src_row = ggml_row_size(src_type, ne0);
        std::vector<unsigned char> src_bytes(src_row * nrows);
        if (_fseeki64(f, (long long) (data_off + gguf_get_tensor_offset(base, idx)), SEEK_SET) != 0 ||
            std::fread(src_bytes.data(), 1, src_bytes.size(), f) != src_bytes.size()) {
            std::fprintf(stderr, "%s: short read\n", p.name.c_str());
            return 1;
        }
        std::vector<float> f32((size_t) ne0 * nrows);
        // F32 and F16 have no to_float trait (nothing to dequantise); everything else does
        const ggml_to_float_t to_float = ggml_get_type_traits(src_type)->to_float;
        if (src_type == GGML_TYPE_F32) {
            std::memcpy(f32.data(), src_bytes.data(), (size_t) ne0 * nrows * sizeof(float));
        } else if (src_type == GGML_TYPE_F16) {
            for (int64_t r = 0; r < nrows; ++r) {
                ggml_fp16_to_fp32_row((const ggml_fp16_t *) (src_bytes.data() + r * src_row), f32.data() + r * ne0, ne0);
            }
        } else if (to_float != nullptr) {
            for (int64_t r = 0; r < nrows; ++r) {
                to_float(src_bytes.data() + r * src_row, f32.data() + r * ne0, ne0);
            }
        } else {
            std::fprintf(stderr, "%s: no way to dequantise %s\n", p.name.c_str(), ggml_type_name(src_type));
            return 1;
        }

        struct ggml_tensor * dst = ggml_new_tensor_4d(out_ctx, p.type, ne0, ne1, ne2, ne3);
        ggml_set_name(dst, p.name.c_str());
        const size_t dst_row = ggml_row_size(p.type, ne0);
        size_t n = 0;
        // ggml_quantize_chunk only implements the quantised targets; F32 and F16 have to be written directly, and
        // skipping that leaves dst->data as uninitialised arena memory (which is how the first version produced a
        // companion full of noise that turned the model's logits into nan).
        if (p.type == GGML_TYPE_F32) {
            std::memcpy(dst->data, f32.data(), (size_t) ne0 * nrows * sizeof(float));
            n = (size_t) ne0 * nrows * sizeof(float);
        } else if (p.type == GGML_TYPE_F16) {
            for (int64_t r = 0; r < nrows; ++r) {
                ggml_fp32_to_fp16_row(f32.data() + r * ne0, (ggml_fp16_t *) ((char *) dst->data + r * dst_row), ne0);
            }
            n = dst_row * nrows;
        } else {
            n = ggml_quantize_chunk(p.type, f32.data(), (char *) dst->data, 0, nrows, ne0, nullptr);
        }
        if (n != dst_row * nrows) {
            std::fprintf(stderr, "%s: wrote %zu, expected %zu\n", p.name.c_str(), n, dst_row * nrows);
            return 1;
        }
        gguf_add_tensor(out, dst);
        std::fprintf(stderr, "  %-40s %-8s -> %-8s %lld rows\n", p.name.c_str(), ggml_type_name(src_type),
                     ggml_type_name(p.type), (long long) nrows);
    }
    std::fclose(f);

    if (!gguf_write_to_file(out, out_path, false)) {
        std::fprintf(stderr, "failed to write %s\n", out_path);
        return 1;
    }
    std::fprintf(stderr, "wrote %s\n", out_path);
    return 0;
}
