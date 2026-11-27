// trellis-encode: the same block through the CPU encoder and through the GPU set_rows path, byte for byte.
//
// test-backend-ops already says whether the two agree, but only as a single squared error over the whole tensor.
// When they disagree this program prints both 32-byte paths and the first chunk that differs, which is what turns
// "ERR = 0.07" into a line of code.  The block is fixed (an LCG), so two builds are comparable.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>

static uint32_t lcg = 12345;
static float next_f32() {
    lcg = lcg * 1664525u + 1013904223u;
    return ((float) (lcg >> 8) / (float) (1u << 24)) * 2.0f - 1.0f;
}

int main(int argc, char ** argv) {
    const char * backend_name = argc > 1 ? argv[1] : "Vulkan0";

    ggml_backend_load_all();
    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (strcmp(ggml_backend_dev_name(d), backend_name) == 0) { dev = d; break; }
    }
    if (dev == nullptr) { printf("no device %s\n", backend_name); return 1; }

    const int64_t ne0 = 128;
    const int64_t nrows = 64;

    std::vector<float> src((size_t) ne0 * nrows);
    for (auto & v : src) v = next_f32() * 0.3f;

    // CPU: the encoder test-backend-ops compares against
    std::vector<uint8_t> cpu(ggml_row_size(GGML_TYPE_TQ2_T, ne0) * nrows);
    ggml_quantize_chunk(GGML_TYPE_TQ2_T, src.data(), cpu.data(), 0, nrows, ne0, nullptr);

    // GPU: set_rows into a TQ2_T tensor
    ggml_init_params ip = { 2 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, nrows);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, nrows);
    ggml_tensor * d = ggml_new_tensor_2d(ctx, GGML_TYPE_TQ2_T, ne0, nrows);
    ggml_tensor * op = ggml_set_rows(ctx, d, a, b);

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) { printf("alloc failed\n"); return 1; }

    ggml_backend_tensor_set(a, src.data(), 0, src.size() * sizeof(float));
    std::vector<int64_t> rows(nrows);
    for (int64_t i = 0; i < nrows; ++i) rows[i] = i;
    ggml_backend_tensor_set(b, rows.data(), 0, rows.size() * sizeof(int64_t));

    if (!ggml_backend_supports_op(backend, op)) { printf("set_rows not supported here\n"); return 1; }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, op);
    ggml_backend_graph_compute(backend, gf);

    const size_t row_size = ggml_row_size(GGML_TYPE_TQ2_T, ne0);
    std::vector<uint8_t> gpu(row_size * nrows);
    ggml_backend_tensor_get(d, gpu.data(), 0, gpu.size());

    int bad = 0;
    for (int64_t r = 0; r < nrows; ++r) {
        const uint8_t * c = cpu.data() + r * row_size;
        const uint8_t * g = gpu.data() + r * row_size;
        if (memcmp(c, g, row_size) == 0) continue;
        ++bad;
        printf("row %lld differs\n  cpu:", (long long) r);
        for (size_t i = 0; i < row_size; ++i) printf(" %02x", c[i]);
        printf("\n  gpu:");
        for (size_t i = 0; i < row_size; ++i) printf(" %02x", g[i]);
        printf("\n");
        for (int t = 0; t < 32; ++t) {
            const uint8_t cv = c[2 + t];
            const uint8_t gv = g[2 + t];
            if (cv != gv) { printf("  first differing chunk %d: cpu %u gpu %u\n", t, cv, gv); break; }
        }
    }
    // Distortion of each side's chosen path, decoded through the CPU dequantiser.  Equal numbers mean the byte
    // difference is a tie the two sides broke differently; a lower number on one side means that side searched
    // better and the other one has a defect.
    auto row_dist = [&](const uint8_t * row) {
        std::vector<float> dec((size_t) ne0 * nrows);
        ggml_get_type_traits(GGML_TYPE_TQ2_T)->to_float(row, dec.data(), ne0);
        double e = 0.0;
        for (int64_t j = 0; j < ne0; ++j) {
            const double dv = (double) src[(size_t) j] - (double) dec[(size_t) j];
            e += dv * dv;
        }
        return e;
    };
    for (int64_t r = 0; r < nrows; ++r) {
        const uint8_t * c = cpu.data() + r * row_size;
        const uint8_t * g = gpu.data() + r * row_size;
        printf("row %lld distortion: cpu %.6f gpu %.6f\n", (long long) r, row_dist(c), row_dist(g));
    }
    printf("%d of %lld rows differ\n", bad, (long long) nrows);

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return bad == 0 ? 0 : 1;
}
