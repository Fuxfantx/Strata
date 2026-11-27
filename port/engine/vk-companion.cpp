// vk-companion.cpp - build a companion GGUF (see vk-companion.h).
//
// This is the code that used to live inside port/probe/requant-tensor.cpp, moved here so the engine's automatic
// BF16 -> F16 pass and the measurement tool write the same file with the same rules.  The engine has written
// files before (the picture reader's .sve), but this is the first time it writes a *model* artifact, so the rules
// are explicit: never overwrite an existing file, never write without a name the caller chose, and report every
// skip.
#include "vk-companion.h"

#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <filesystem>

namespace strata {

namespace {
std::string base_name_of(const std::string & path) {
    const size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}
} // namespace

// Is this companion the one built from `base`?  A companion without the markers (an older one, or one built by
// `requant-tensor` before this check existed) is given the benefit of the doubt and rebuilt only when the engine
// generated it itself - see the call site.
std::vector<CompanionTensorInfo> companion_tensor_list(const std::string & path) {
    std::vector<CompanionTensorInfo> out;
    struct ggml_context * meta_ctx = nullptr;
    struct gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &meta_ctx };
    struct gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (g == nullptr) return out;
    for (struct ggml_tensor * cur = meta_ctx != nullptr ? ggml_get_first_tensor(meta_ctx) : nullptr;
         cur != nullptr; cur = ggml_get_next_tensor(meta_ctx, cur)) {
        out.push_back({ cur->name, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3] });
    }
    gguf_free(g);
    if (meta_ctx != nullptr) ggml_free(meta_ctx);
    return out;
}

bool companion_tensor_rows(const std::string & path, const std::string & name, int64_t & ne0, int64_t & ne1) {
    struct ggml_context * meta_ctx = nullptr;
    struct gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &meta_ctx };
    struct gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (g == nullptr) return false;
    bool found = false;
    for (struct ggml_tensor * cur = meta_ctx != nullptr ? ggml_get_first_tensor(meta_ctx) : nullptr;
         cur != nullptr; cur = ggml_get_next_tensor(meta_ctx, cur)) {
        if (name == cur->name) { ne0 = cur->ne[0]; ne1 = cur->ne[1]; found = true; break; }
    }
    gguf_free(g);
    if (meta_ctx != nullptr) ggml_free(meta_ctx);
    return found;
}

bool companion_matches(const std::string & companion, const std::string & base) {
    struct ggml_context * meta_ctx = nullptr;
    struct gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &meta_ctx };
    struct gguf_context * g = gguf_init_from_file(companion.c_str(), gp);
    if (g == nullptr) return false;
    bool ok = true;
    const int name_key = gguf_find_key(g, "strata.companion.base_file");
    const int size_key = gguf_find_key(g, "strata.companion.base_size");
    if (name_key >= 0 && gguf_get_val_str(g, name_key) != nullptr) {
        ok = ok && base_name_of(base) == gguf_get_val_str(g, name_key);
    }
    if (size_key >= 0) {
        std::error_code ec;
        const auto sz = std::filesystem::file_size(base, ec);
        ok = ok && !ec && (uint64_t) sz == gguf_get_val_u64(g, size_key);
    }
    gguf_free(g);
    if (meta_ctx != nullptr) ggml_free(meta_ctx);
    return ok;
}

std::vector<std::string> companion_tensor_names(const std::string & base) {
    std::vector<std::string> names;
    struct ggml_context * meta_ctx = nullptr;
    struct gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &meta_ctx };
    struct gguf_context * g = gguf_init_from_file(base.c_str(), gp);
    if (g == nullptr) return names;
    const int64_t n = gguf_get_n_tensors(g);
    for (int64_t i = 0; i < n; ++i) {
        const char * name = gguf_get_tensor_name(g, i);
        if (name != nullptr) names.emplace_back(name);
    }
    gguf_free(g);
    if (meta_ctx != nullptr) ggml_free(meta_ctx);
    return names;
}

std::vector<CompanionPick> companion_picks_of_type(const std::string & base, enum ggml_type from, enum ggml_type to) {
    std::vector<CompanionPick> picks;
    struct ggml_context * meta_ctx = nullptr;
    struct gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &meta_ctx };
    struct gguf_context * g = gguf_init_from_file(base.c_str(), gp);
    if (g == nullptr) return picks;
    const int64_t n = gguf_get_n_tensors(g);
    for (int64_t i = 0; i < n; ++i) {
        if (gguf_get_tensor_type(g, i) != from) continue;
        const char * name = gguf_get_tensor_name(g, i);
        if (name != nullptr) picks.push_back({ name, to });
    }
    gguf_free(g);
    if (meta_ctx != nullptr) ggml_free(meta_ctx);
    return picks;
}

bool write_companion(const std::string & base, const std::string & out,
                     const std::vector<CompanionPick> & picks, std::string & err,
                     const std::vector<CompanionSlice> & slices) {
    if (picks.empty() && slices.empty()) { err = "nothing to write"; return false; }

    struct ggml_context * meta_ctx = nullptr;
    struct gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &meta_ctx };
    struct gguf_context * g = gguf_init_from_file(base.c_str(), gp);
    if (g == nullptr) { err = "not a gguf: " + base; return false; }

    FILE * f = std::fopen(base.c_str(), "rb");
    if (f == nullptr) { err = "cannot open " + base; gguf_free(g); ggml_free(meta_ctx); return false; }
    const size_t data_off = gguf_get_data_offset(g);

    // The output context holds a converted copy of every selected tensor, so it has to be sized from the
    // destinations; a fixed pool fails on the first multi-megabyte group.
    size_t need = (size_t) 32 << 20;
    for (const CompanionSlice & sl : slices) {
        const int idx = gguf_find_tensor(g, sl.name.c_str());
        if (idx >= 0) need += (size_t) gguf_get_tensor_size(g, idx);
    }
    for (const CompanionPick & p : picks) {
        for (struct ggml_tensor * cur = ggml_get_first_tensor(meta_ctx); cur != nullptr;
             cur = ggml_get_next_tensor(meta_ctx, cur)) {
            if (p.name != cur->name) continue;
            if (cur->ne[0] % ggml_blck_size(p.type) == 0) {
                need += ggml_row_size(p.type, cur->ne[0]) * (size_t) (cur->ne[1] * cur->ne[2] * cur->ne[3]);
            }
            break;
        }
    }
    struct ggml_init_params ip = { /*.mem_size =*/ need, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
    struct ggml_context * out_ctx = ggml_init(ip);
    if (out_ctx == nullptr) { err = "cannot allocate the conversion context"; std::fclose(f); gguf_free(g); ggml_free(meta_ctx); return false; }

    struct gguf_context * o = gguf_init_empty();
    gguf_set_val_bool(o, "strata.companion", true);
    // What this companion was built from.  The engine names the companion after the pack, so a pack replaced under the
    // same name would otherwise be silently composed with tensors converted out of the old one - a wrong model that
    // reads as bad output.  The size is enough to catch that without hashing 60 GiB.
    gguf_set_val_str(o, "strata.companion.base_file", base_name_of(base).c_str());
    {
        std::error_code sz_ec;
        const auto sz = std::filesystem::file_size(base, sz_ec);
        gguf_set_val_u64(o, "strata.companion.base_size", sz_ec ? 0ull : (uint64_t) sz);
    }
    const int arch_key = gguf_find_key(g, "general.architecture");
    if (arch_key >= 0) {
        gguf_set_val_str(o, "general.architecture", gguf_get_val_str(g, arch_key));
    } else {
        gguf_set_val_str(o, "general.architecture", "strata-companion");
    }

    int written = 0;
    for (const CompanionPick & p : picks) {
        const int idx = gguf_find_tensor(g, p.name.c_str());
        if (idx < 0) { err = p.name + ": vanished"; std::fclose(f); goto fail; }

        struct ggml_tensor * src_meta = nullptr;
        for (struct ggml_tensor * cur = ggml_get_first_tensor(meta_ctx); cur != nullptr;
             cur = ggml_get_next_tensor(meta_ctx, cur)) {
            if (p.name == cur->name) { src_meta = cur; break; }
        }
        if (src_meta == nullptr) { err = p.name + ": not in the metadata context"; std::fclose(f); goto fail; }

        const int64_t ne0 = src_meta->ne[0];
        const int64_t ne1 = src_meta->ne[1];
        const int64_t ne2 = src_meta->ne[2];
        const int64_t ne3 = src_meta->ne[3];
        const int64_t nrows = ne1 * ne2 * ne3;
        const enum ggml_type src_type = gguf_get_tensor_type(g, idx);

        if (src_type == p.type) continue;   // already the destination type: nothing to do
        if (ne0 % ggml_blck_size(p.type) != 0) {
            std::fprintf(stderr, "[strata-vk] companion: %s ne0 %lld is not a multiple of %s's block size; skipped\n",
                         p.name.c_str(), (long long) ne0, ggml_type_name(p.type));
            continue;
        }

        // dequantise the source into f32, row by row
        const size_t src_row = ggml_row_size(src_type, ne0);
        std::vector<unsigned char> src_bytes(src_row * (size_t) nrows);
        if (_fseeki64(f, (long long) (data_off + gguf_get_tensor_offset(g, idx)), SEEK_SET) != 0 ||
            std::fread(src_bytes.data(), 1, src_bytes.size(), f) != src_bytes.size()) {
            err = p.name + ": short read"; std::fclose(f); goto fail;
        }
        std::vector<float> f32((size_t) ne0 * (size_t) nrows);
        const ggml_to_float_t to_float = ggml_get_type_traits(src_type)->to_float;
        if (src_type == GGML_TYPE_F32) {
            std::memcpy(f32.data(), src_bytes.data(), (size_t) ne0 * (size_t) nrows * sizeof(float));
        } else if (src_type == GGML_TYPE_F16) {
            for (int64_t r = 0; r < nrows; ++r) {
                ggml_fp16_to_fp32_row((const ggml_fp16_t *) (src_bytes.data() + r * src_row), f32.data() + r * ne0, ne0);
            }
        } else if (to_float != nullptr) {
            for (int64_t r = 0; r < nrows; ++r) {
                to_float(src_bytes.data() + r * src_row, f32.data() + r * ne0, ne0);
            }
        } else {
            err = p.name + ": no way to dequantise " + ggml_type_name(src_type); std::fclose(f); goto fail;
        }

        struct ggml_tensor * dst = ggml_new_tensor_4d(out_ctx, p.type, ne0, ne1, ne2, ne3);
        ggml_set_name(dst, p.name.c_str());
        const size_t dst_row = ggml_row_size(p.type, ne0);
        size_t n = 0;
        // ggml_quantize_chunk only implements the quantised targets: F32 and F16 have to be written directly, and
        // skipping that leaves dst->data as uninitialised arena memory.
        if (p.type == GGML_TYPE_F32) {
            std::memcpy(dst->data, f32.data(), (size_t) ne0 * (size_t) nrows * sizeof(float));
            n = (size_t) ne0 * (size_t) nrows * sizeof(float);
        } else if (p.type == GGML_TYPE_F16) {
            for (int64_t r = 0; r < nrows; ++r) {
                ggml_fp32_to_fp16_row(f32.data() + r * ne0, (ggml_fp16_t *) ((char *) dst->data + r * dst_row), ne0);
            }
            n = dst_row * (size_t) nrows;
        } else {
            n = ggml_quantize_chunk(p.type, f32.data(), (char *) dst->data, 0, nrows, ne0, nullptr);
        }
        if (n != dst_row * (size_t) nrows) {
            char buf[160];
            std::snprintf(buf, sizeof(buf), "%s: wrote %zu of %zu bytes", p.name.c_str(), n, dst_row * (size_t) nrows);
            err = buf;
            std::fclose(f);
            goto fail;
        }
        gguf_add_tensor(o, dst);
        ++written;
    }
    // One tensor in, two out: the high rows keep the name, the low rows become `low_name`.  Rows of a row-major
    // quantised tensor are whole blocks, so the bytes move unchanged and the two halves keep the source type.
    for (const CompanionSlice & sl : slices) {
        const int idx = gguf_find_tensor(g, sl.name.c_str());
        if (idx < 0) { err = sl.name + ": vanished"; std::fclose(f); goto fail; }
        struct ggml_tensor * src_meta = nullptr;
        for (struct ggml_tensor * cur = ggml_get_first_tensor(meta_ctx); cur != nullptr;
             cur = ggml_get_next_tensor(meta_ctx, cur)) {
            if (sl.name == cur->name) { src_meta = cur; break; }
        }
        if (src_meta == nullptr) { err = sl.name + ": not in the metadata context"; std::fclose(f); goto fail; }
        const int64_t ne0 = src_meta->ne[0];
        const int64_t ne1 = src_meta->ne[1];
        const int64_t ne2 = src_meta->ne[2];
        const enum ggml_type src_type = gguf_get_tensor_type(g, idx);
        if (ne0 % ggml_blck_size(src_type) != 0) {
            err = sl.name + ": ne0 is not whole blocks";
            std::fclose(f);
            goto fail;
        }
        // Rows of a row-major quantised tensor are whole blocks, and the expert dimension of a stack is a whole number
        // of those 2-D slabs, so both splits are a byte range of the source and never a requantisation.
        const int64_t along = sl.dim == 2 ? ne2 : ne1;
        if (sl.dim != 1 && sl.dim != 2) {
            err = sl.name + ": a slice runs along rows or experts only";
            std::fclose(f);
            goto fail;
        }
        if (sl.keep_rows <= 0 || sl.keep_rows >= along || src_meta->ne[3] != 1 ||
            (sl.dim == 1 && ne2 != 1)) {
            err = sl.name + ": slice bounds do not fit the tensor";
            std::fclose(f);
            goto fail;
        }
        const size_t row     = ggml_row_size(src_type, ne0);   // one row of one expert
        const size_t slab    = row * (size_t) ne1;             // one expert of a stack
        const size_t unit    = sl.dim == 2 ? slab : row;
        const size_t per_out = sl.dim == 2 ? ne1 : 1;
        for (int part = 0; part < 2; ++part) {
            const int64_t r0 = part == 0 ? 0 : sl.keep_rows;
            const int64_t rn = part == 0 ? sl.keep_rows : along - sl.keep_rows;
            struct ggml_tensor * dst = sl.dim == 2
                ? ggml_new_tensor_3d(out_ctx, src_type, ne0, ne1, rn)
                : ggml_new_tensor_2d(out_ctx, src_type, ne0, rn);
            ggml_set_name(dst, (part == 0 ? sl.name : sl.low_name).c_str());
            const size_t want = unit * (size_t) rn;
            if (_fseeki64(f, (long long) (data_off + gguf_get_tensor_offset(g, idx) + (uint64_t) r0 * unit), SEEK_SET) != 0 ||
                std::fread(dst->data, 1, want, f) != want) {
                err = sl.name + ": short read"; std::fclose(f); goto fail;
            }
            (void) per_out;
            gguf_add_tensor(o, dst);
            ++written;
        }
    }
    std::fclose(f);
    f = nullptr;

    if (written == 0) { err = "nothing converted and nothing sliced"; goto fail; }
    if (!gguf_write_to_file(o, out.c_str(), false)) { err = "failed to write " + out; goto fail; }
    gguf_free(o);
    gguf_free(g);
    ggml_free(meta_ctx);
    ggml_free(out_ctx);
    return true;

fail:
    if (f != nullptr) std::fclose(f);
    gguf_free(o);
    gguf_free(g);
    ggml_free(meta_ctx);
    ggml_free(out_ctx);
    return false;
}

} // namespace strata
