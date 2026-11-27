// engram-probe.cpp: read one row of one Engram tier table from the companion pack and dequantise it, so the path the
// graph will use can be compared against the reference decoder.
//
//   engram-probe --pack <companion.gguf> --tensor engram_warm.weight --row <n>
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

extern "C" void ggml_hy4_set_engram_range(float lo, float hi);

int main(int argc, char ** argv) {
    std::string pack;
    std::string tensor = "engram_warm.weight";
    long row = 0;
    long index_row = -1;
    long tk[3] = { 0, 0, 0 };
    bool have_tokens = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--pack" && i + 1 < argc)        { pack = argv[++i]; }
        else if (a == "--tensor" && i + 1 < argc) { tensor = argv[++i]; }
        else if (a == "--row" && i + 1 < argc)    { row = std::atol(argv[++i]); }
        else if (a == "--index" && i + 1 < argc)   { index_row = std::atol(argv[++i]); }
        else if (a == "--tokens" && i + 3 < argc)  { tk[0] = std::atol(argv[++i]); tk[1] = std::atol(argv[++i]); tk[2] = std::atol(argv[++i]); have_tokens = true; }
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (pack.empty()) {
        std::fprintf(stderr, "usage: engram-probe --pack <file> [--tensor name] [--row n]\n");
        return 2;
    }

    gguf_init_params params;
    params.no_alloc = true;
    params.ctx      = nullptr;

    gguf_context * ctx = gguf_init_from_file(pack.c_str(), params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "cannot open %s\n", pack.c_str());
        return 1;
    }

    // the scale range the Engram block types use, from the pack's own key-values
    float lo = 0.0f, hi = 0.0f;
    const int klo = gguf_find_key(ctx, "hy4f.engram.scale_lo");
    const int khi = gguf_find_key(ctx, "hy4f.engram.scale_hi");
    if (klo >= 0) { lo = gguf_get_val_f32(ctx, klo); }
    if (khi >= 0) { hi = gguf_get_val_f32(ctx, khi); }
    ggml_hy4_set_engram_range(lo, hi);


    // --index: the whole lookup the graph performs, one step earlier - index entry, tier, local row, table row
    if (index_row >= 0) {
        const int64_t iid = gguf_find_tensor(ctx, "engram_index");
        if (iid < 0) {
            std::fprintf(stderr, "engram_index not found\n");
            gguf_free(ctx);
            return 1;
        }
        const int64_t * ine = gguf_get_tensor_ne(ctx, iid);
        if (index_row >= ine[0]) {
            std::fprintf(stderr, "index row %ld out of range (%lld)\n", index_row, (long long) ine[0]);
            gguf_free(ctx);
            return 1;
        }

        const size_t ioff = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, iid) + (size_t) index_row*4;
        FILE * fi = std::fopen(pack.c_str(), "rb");
        if (fi == nullptr) { gguf_free(ctx); return 1; }
#if defined(_WIN32)
        _fseeki64(fi, (long long) ioff, SEEK_SET);
#else
        std::fseek(fi, (long) ioff, SEEK_SET);
#endif
        uint32_t entry = 0;
        const size_t got_i = std::fread(&entry, 1, 4, fi);
        std::fclose(fi);
        if (got_i != 4) {
            std::fprintf(stderr, "short read at index %ld\n", index_row);
            gguf_free(ctx);
            return 1;
        }

        const int tier = (int) (entry >> 30);
        const long lrow = (long) (entry & 0x3FFFFFFFu);

        static const char * names[4] = { "engram_low.weight", "engram_high.weight", "engram_hot.weight", "engram_warm.weight" };
        printf("index %ld: entry 0x%08x -> tier %d, row %ld (%s)\n", index_row, entry, tier, lrow,
               tier < 4 ? names[tier] : "?");

        gguf_free(ctx);
        return 0;
    }


    // --tokens: the hash the model uses, for the four Engram layers and their sixteen columns each
    if (have_tokens) {
        std::vector<long> mult, offs, voc;
        const auto arr = [&](const char * key, std::vector<long> & out) {
            const int kid = gguf_find_key(ctx, key);
            if (kid < 0) { return false; }
            const uint64_t n = gguf_get_arr_n(ctx, kid);
            const enum gguf_type t = gguf_get_arr_type(ctx, kid);
            const void * data = gguf_get_arr_data(ctx, kid);
            out.resize((size_t) n);
            for (uint64_t i = 0; i < n; ++i) {
                switch (t) {
                    case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: out[(size_t) i] = ((const int32_t *) data)[i]; break;
                    case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: out[(size_t) i] = (long) ((const int64_t *) data)[i]; break;
                    default: return false;
                }
            }
            return true;
        };

        if (!arr("hy_v4_flash.ple.layer_multipliers", mult) ||
            !arr("hy_v4_flash.ple.head_offsets",      offs) ||
            !arr("hy_v4_flash.ple.head_vocab_sizes",  voc)) {
            std::fprintf(stderr, "the ple arrays are missing\n");
            gguf_free(ctx);
            return 1;
        }

        const int64_t iid = gguf_find_tensor(ctx, "engram_index");
        if (iid < 0) { gguf_free(ctx); return 1; }
        const size_t ibase = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, iid);

        FILE * fi = std::fopen(pack.c_str(), "rb");
        if (fi == nullptr) { gguf_free(ctx); return 1; }

        const int n_gram = 3, n_head = 8, n_cols = 16, n_layers = 4;
        uint64_t rolling = (uint64_t) tk[0]*mult[0];
        for (int o = 1; o < n_gram; ++o) {
            rolling ^= (uint64_t) tk[o]*mult[o];
            for (int h = 0; h < n_head; ++h) {
                const int col = (o - 1)*n_head + h;
                printf("layer 0 col %2d -> index row %llu\n", col,
                       (unsigned long long) (rolling % (uint64_t) voc[col] + (uint64_t) offs[col]));
            }
        }

        // read the sixteen index entries this triple selects, then the 96 values of each - the row the block feeds to
        // engram_kv, whose input width is 16 x 96 = 1536
        std::vector<double> acc((size_t) n_cols * 96, 0.0);
        double sq = 0.0;
        int    used = 0;

        static const char * tname[4] = { "engram_low.weight", "engram_high.weight", "engram_warm.weight", "engram_hot.weight" };
        int64_t tids[4] = { -1, -1, -1, -1 };
        size_t  toff[4] = { 0, 0, 0, 0 };
        size_t  trow[4] = { 0, 0, 0, 0 };

        for (int t = 0; t < 4; ++t) {
            tids[t] = gguf_find_tensor(ctx, tname[t]);
            if (tids[t] >= 0) {
                toff[t] = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, tids[t]);
                const ggml_type tt = gguf_get_tensor_type(ctx, tids[t]);
                trow[t] = ggml_row_size(tt, 96);
            }
        }

        uint64_t rolling2 = (uint64_t) tk[0]*mult[0];
        for (int o = 1; o < n_gram; ++o) {
            rolling2 ^= (uint64_t) tk[o]*mult[o];
            for (int h = 0; h < n_head; ++h) {
                const int col = (o - 1)*n_head + h;
                const uint64_t rowi = rolling2 % (uint64_t) voc[col] + (uint64_t) offs[col];

                uint32_t entry = 0;
#if defined(_WIN32)
                _fseeki64(fi, (long long) (ibase + rowi*4), SEEK_SET);
#else
                std::fseek(fi, (long) (ibase + rowi*4), SEEK_SET);
#endif
                if (std::fread(&entry, 1, 4, fi) != 4) {
                    continue;
                }
                const int tier = (int) (entry >> 30);
                const uint64_t lrow = entry & 0x3FFFFFFFu;
                if (tier > 3 || tids[tier] < 0) {
                    continue;
                }

                std::vector<uint8_t> raw(trow[tier]);
#if defined(_WIN32)
                _fseeki64(fi, (long long) (toff[tier] + lrow*trow[tier]), SEEK_SET);
#else
                std::fseek(fi, (long) (toff[tier] + lrow*trow[tier]), SEEK_SET);
#endif
                if (std::fread(raw.data(), 1, raw.size(), fi) != raw.size()) {
                    continue;
                }

                std::vector<float> v(96);
                const ggml_type_traits * tr2 = ggml_get_type_traits(gguf_get_tensor_type(ctx, tids[tier]));
                if (tr2 == nullptr || tr2->to_float == nullptr) {
                    continue;
                }
                tr2->to_float(raw.data(), v.data(), 96);

                for (int k = 0; k < 96; ++k) {
                    acc[(size_t) col*96 + k] = v[(size_t) k];
                    sq += (double) v[(size_t) k] * (double) v[(size_t) k];
                }
                used++;
            }
        }

        printf("gathered %d of %d columns, %zu values, norm %.6f, first %.6f %.6f %.6f\n",
               used, n_cols, acc.size(), std::sqrt(sq), acc[0], acc[1], acc[2]);

        std::fclose(fi);
        gguf_free(ctx);
        return 0;
    }

    const int64_t tid = gguf_find_tensor(ctx, tensor.c_str());
    if (tid < 0) {
        std::fprintf(stderr, "tensor %s not found\n", tensor.c_str());
        gguf_free(ctx);
        return 1;
    }

    const ggml_type type   = gguf_get_tensor_type(ctx, tid);
    const int64_t * ne     = gguf_get_tensor_ne(ctx, tid);
    const int64_t   n_cols = ne[0];
    const int64_t   n_rows = ne[1];
    const size_t    row_bytes = ggml_row_size(type, n_cols);

    printf("tensor %s type %s rows %lld row_bytes %zu scale %.6f..%.6f\n", tensor.c_str(),
           ggml_type_name(type), (long long) n_rows, row_bytes, lo, hi);

    const size_t off = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, tid) + (size_t) row*row_bytes;
    printf("data offset %zu, tensor offset %zu, row offset %zu\n", gguf_get_data_offset(ctx),
           gguf_get_tensor_offset(ctx, tid), off);

    FILE * f = std::fopen(pack.c_str(), "rb");
    if (f == nullptr) { gguf_free(ctx); return 1; }
#if defined(_WIN32)
    // a row offset passes 2^31 for the larger tables, and a 32-bit long would truncate it and read the wrong bytes
    const int seek_res = _fseeki64(f, (long long) off, SEEK_SET);
#else
    const int seek_res = std::fseek(f, (long) off, SEEK_SET);
#endif
    if (seek_res != 0) {
        std::fprintf(stderr, "cannot seek to %zu\n", off);
        std::fclose(f);
        gguf_free(ctx);
        return 1;
    }

    std::vector<uint8_t> raw(row_bytes);
    const size_t got = std::fread(raw.data(), 1, row_bytes, f);
    std::fclose(f);
    if (got != row_bytes) {
        std::fprintf(stderr, "short read: %zu of %zu\n", got, row_bytes);
        gguf_free(ctx);
        return 1;
    }

    std::vector<float> vals((size_t) n_cols);
    const ggml_type_traits * tr = ggml_get_type_traits(type);
    if (tr == nullptr || tr->to_float == nullptr) {
        std::fprintf(stderr, "no converter for %s\n", ggml_type_name(type));
        gguf_free(ctx);
        return 1;
    }
    tr->to_float(raw.data(), vals.data(), n_cols);

    // the same row decoded by hand here, so a disagreement can be pinned to the converter or to its scale table
    if (n_cols == 96 && row_bytes == 80) {
        const float scale = expf(lo + raw[72]*(hi - lo)/255.0f);
        printf("manual: scale code %u -> %.6f; first four ", raw[72], scale);
        uint64_t lo64 = 0;
        std::memcpy(&lo64, raw.data(), 8);
        const uint8_t hi8 = raw[8];
        for (int j = 0; j < 4; ++j) {
            const uint64_t v = j < 10 ? (lo64 >> (6*j)) : ((lo64 >> 60) | ((uint64_t) hi8 << 4));
            const int q = (int) (v & 63);
            printf("%.6f ", scale * (q - 31.5f) / 31.5f);
        }
        printf("\n");
    }

    for (int64_t i = 0; i < n_cols; ++i) {
        printf("%.9g\n", vals[(size_t) i]);
    }

    gguf_free(ctx);
    return 0;
}
