// codec-probe - measure a quantisation codec against the f32 truth: quantise one model-like row, dequantise it, report
// the relative L2 error and the cosine similarity.  The backend-vs-backend oracle cannot see a codec that is
// self-consistent but wrong, which is how the trellis types passed every oracle while producing unusable packs.
//
//   codec-probe                 every coded type, a synthetic model-like row
//   codec-probe q2_k tq2_t      just those
//   codec-probe --row FILE      use a raw f32 row (port/tools/extract_weight_row.py writes one)
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr size_t kRow = 5120;   // a real weight row: 5120 floats, a multiple of every block size here

std::vector<float> model_like_row(size_t n) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        const double block = 0.008 + 0.010 * (0.5 + 0.5 * std::sin((double) (i / 32) * 0.61));
        const double x = std::sin((double) i * 0.37) * 0.7 + std::cos((double) i * 1.13) * 0.3;
        v[i] = (float) (x * block);
    }
    return v;
}

bool read_row(const char * path, std::vector<float> & out) {
    std::FILE * f = std::fopen(path, "rb");
    if (f == nullptr) {
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    const long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (bytes <= 0 || bytes % 4 != 0) {
        std::fclose(f);
        return false;
    }
    out.resize((size_t) bytes / 4);
    const bool ok = std::fread(out.data(), 4, out.size(), f) == out.size();
    std::fclose(f);
    return ok;
}

bool dot_check = false;
ggml_backend_t cpu = nullptr;

void check(enum ggml_type type, const std::vector<float> & src) {
    const size_t n = src.size();
    std::vector<uint8_t> q(ggml_row_size(type, (int64_t) n));
    ggml_quantize_chunk(type, src.data(), q.data(), 0, 1, (int64_t) n, nullptr);
    std::vector<float> back(n);
    ggml_get_type_traits(type)->to_float(q.data(), back.data(), (int64_t) n);

    double se = 0.0, sn = 0.0, dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = (double) back[i] - (double) src[i];
        se += d * d;
        sn += (double) src[i] * (double) src[i];
        dot += (double) back[i] * (double) src[i];
        na += (double) back[i] * (double) back[i];
        nb += (double) src[i] * (double) src[i];
    }
    std::printf("%-8s bpw %5.3f  relative L2 %.4f  cosine %.6f\n", ggml_type_name(type),
                ggml_type_size(type) * 8.0 / (double) ggml_blck_size(type), std::sqrt(se / sn), dot / std::sqrt(na * nb));

    if (dot_check) {
        // The q8_0 quantiser produces the B operand every quantised vec_dot expects.  Compare the kernel's answer with
        // the same dot computed from the dequantised rows: if the two differ, the type works for the loader and breaks
        // for the matmul.
        std::vector<uint8_t> bq(ggml_row_size(GGML_TYPE_Q8_0, (int64_t) n));
        ggml_quantize_chunk(GGML_TYPE_Q8_0, src.data(), bq.data(), 0, 1, (int64_t) n, nullptr);
        std::vector<float> bf(n);
        ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(bq.data(), bf.data(), (int64_t) n);

        double ref = 0.0;
        for (size_t i = 0; i < n; ++i) {
            ref += (double) back[i] * (double) bf[i];
        }

        // The kernel through the public API: a one-row MUL_MAT on the CPU backend, A = the quantised row, B = the same
        // q8_0 row.  This is the path a model takes; the reference above is the dequantiser, so a difference means the
        // dot-product kernel and the dequantiser disagree about the format.
        ggml_init_params ip = { /*.mem_size =*/ 4u * 1024 * 1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
        ggml_context * gctx = ggml_init(ip);
        ggml_tensor * A = ggml_new_tensor_2d(gctx, type, (int64_t) n, 1);
        ggml_tensor * B = ggml_new_tensor_2d(gctx, GGML_TYPE_Q8_0, (int64_t) n, 1);
        ggml_tensor * C = ggml_mul_mat(gctx, A, B);
        ggml_backend_buffer_t abuf = ggml_backend_alloc_ctx_tensors(gctx, cpu);
        ggml_backend_tensor_set(A, q.data(), 0, ggml_nbytes(A));
        ggml_backend_tensor_set(B, bq.data(), 0, ggml_nbytes(B));
        ggml_cgraph * gf = ggml_new_graph(gctx);
        ggml_build_forward_expand(gf, C);
        ggml_backend_graph_compute(cpu, gf);
        float kernel = 0.0f;
        ggml_backend_tensor_get(C, &kernel, 0, sizeof(float));
        ggml_backend_buffer_free(abuf);
        ggml_free(gctx);
        const double rel = std::fabs((double) kernel - ref) / std::max(std::fabs(ref), 1e-9);
        std::printf("         dot: kernel %.6f vs dequantised %.6f  rel %.2e %s\n", (double) kernel, ref, rel,
                    rel > 1e-4 ? "MISMATCH" : "ok");
    }
    std::fflush(stdout);
}

} // namespace

int main(int argc, char ** argv) {
    llama_backend_init();

    std::vector<float> src;
    std::vector<std::string> only;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--dot-check") == 0) {
            dot_check = true;
        } else if (std::strcmp(argv[i], "--row") == 0 && i + 1 < argc) {
            if (!read_row(argv[++i], src)) {
                std::fprintf(stderr, "cannot read row %s\n", argv[i]);
                return 2;
            }
        } else {
            only.push_back(argv[i]);
        }
    }
    std::printf("row: %zu values (%s)\n", src.size(), src.empty() ? "synthetic" : "from --row");
    if (dot_check) {
        cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (cpu == nullptr) {
            std::fprintf(stderr, "no CPU backend for --dot-check\n");
            return 2;
        }
    }
    if (src.empty()) {
        src = model_like_row(kRow);
    }

    static const enum ggml_type all[] = {
        GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, GGML_TYPE_Q2_0, GGML_TYPE_Q1_0,
        GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0, GGML_TYPE_Q2_K, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K,
        GGML_TYPE_TQ2_T, GGML_TYPE_PTQ1_0,
    };
    for (enum ggml_type t : all) {
        if (!only.empty()) {
            bool want = false;
            for (const auto & name : only) {
                want = want || std::strcmp(name.c_str(), ggml_type_name(t)) == 0;
            }
            if (!want) {
                continue;
            }
        }
        check(t, src);
    }

    llama_backend_free();
    return 0;
}
