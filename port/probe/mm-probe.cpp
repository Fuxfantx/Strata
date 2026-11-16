// mm-probe.cpp - time one ggml_mul_mat (or mul_mat_id) shape on one backend, in isolation.
//
// The per-op table the Vulkan backend prints brackets more than the kernel (port/docs/ADRENO.md section "Where the time
// actually goes"), so every kernel decision in this port is made against this instrument instead: one dispatch, one
// shape, wall clock around the compute, repeated until it is stable.
//
//   mm-probe -b Vulkan0 -t q5_0 -m 896 -n 512 -k 896
//   mm-probe -b OpenCL -t q4_K -m 4864 -n 512 -k 896 --reps 20
//   mm-probe -b Vulkan0 --chain 256 -k 2560        # marginal cost of one dispatch
//
// MUL_MAT_ID (the MoE shape) is not covered here: its tensor-rank rules need the surrounding graph, and llama.cpp's
// `test-backend-ops -o MUL_MAT` already exercises it for every type against a CPU reference.
//
// Two traps this tool has fallen into, both of which produce confident, wrong GB/s numbers:
//   * one op per submit charges ~90 us of submit and fence to it, so an op whose work is 10 us measures 10% of its real
//     rate. `--ops K` amortises the submit but then K repeats of one weight slice hit the cache (400-3000 GB/s on a
//     135 GB/s part); distinct slices only help when K x weight bytes exceeds ~100 MB. For ops smaller than that, read
//     the backend's own per-op table or an end-to-end token rate instead of this tool.
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#ifdef _WIN32
#define NOMINMAX   // windows.h's min/max macros break std::max below
#include <windows.h>
#include "renderdoc_app.h"
#endif
#include <string>
#include <vector>

namespace {

double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

ggml_backend_dev_t find_device(const std::string & name) {
    if (!name.empty()) {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (name == ggml_backend_dev_name(d)) return d;
        }
        return nullptr;
    }
    if (ggml_backend_dev_t d = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) return d;
    return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
}

ggml_type type_from_name(const std::string & name) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        if (name == ggml_type_name((ggml_type) t)) return (ggml_type) t;
    }
    return GGML_TYPE_COUNT;
}

void fill_random(ggml_tensor * t, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const size_t n = ggml_nelements(t);
    std::vector<float> f(n);
    for (size_t i = 0; i < n; ++i) f[i] = dist(rng);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, f.data(), 0, n * sizeof(float));
        return;
    }
    // Quantised blocks keep the scale sane: ggml_quantize_chunk maps an f32 matrix of the same shape into the type.
    if (ggml_is_quantized(t->type)) {
        std::vector<char> q(ggml_nbytes(t));
        ggml_quantize_chunk(t->type, f.data(), q.data(), 0, ggml_nrows(t), t->ne[0], nullptr);
        ggml_backend_tensor_set(t, q.data(), 0, q.size());
        return;
    }
    ggml_backend_tensor_set(t, f.data(), 0, n * sizeof(float));
}

} // namespace

int main(int argc, char ** argv) {
    std::string backend_name, type_name = "f16";
    int64_t m = 4096, n = 1, k = 14336;
    int chain = 0;
    int handoff = 0;
    int ops = 1;
    bool scale = false;
    bool verify = false;
    int reps = 0;
    double budget_ms = 400.0;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * { return (i + 1 < argc) ? argv[++i] : ""; };
        if      (a == "-b" || a == "--backend") backend_name = next();
        else if (a == "-t" || a == "--type")    type_name = next();
        else if (a == "-m")                     m = std::atoll(next());
        else if (a == "-n")                     n = std::atoll(next());
        else if (a == "-k")                     k = std::atoll(next());
        else if (a == "--chain")                chain = std::atoi(next());
        else if (a == "--handoff")              handoff = std::atoi(next());
        else if (a == "--ops")                  ops = std::max(1, std::atoi(next()));
        else if (a == "--scale")                scale = true;
        else if (a == "--reps")                 reps = std::atoi(next());
        else if (a == "--budget-ms")            budget_ms = std::atof(next());
        else if (a == "--verify")               verify = true;
        else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }

    const ggml_type type = type_from_name(type_name);
    if (type == GGML_TYPE_COUNT) {
        std::fprintf(stderr, "unknown type %s\n", type_name.c_str());
        return 2;
    }

    ggml_backend_load_all();
    ggml_backend_dev_t dev = find_device(backend_name);
    if (dev == nullptr) {
        std::fprintf(stderr, "no such backend device: %s\n", backend_name.c_str());
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            std::fprintf(stderr, "  %s\n", ggml_backend_dev_name(ggml_backend_dev_get(i)));
        }
        return 1;
    }
    ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend.get());
    if (buft == nullptr) {
        std::fprintf(stderr, "device %s has no default buffer type\n", ggml_backend_dev_name(dev));
        return 1;
    }

    ggml_init_params ip = { /*mem_size*/ 256u * 1024 * 1024, /*mem_buffer*/ nullptr, /*no_alloc*/ true };
    ggml_context_ptr ctx(ggml_init(ip));

    ggml_tensor * src0 = nullptr;
    ggml_tensor * src1 = nullptr;
    ggml_tensor * dst  = nullptr;

    if (chain > 0) {
        // `--chain N`: N chained elementwise ops on a k-element f32 tensor, in one graph and one submit.  The marginal
        // wall time per node is the per-dispatch cost of this driver with the kernels held constant and negligible,
        // which is the number that decides whether cutting the op count is worth more than making one kernel faster.
        ggml_tensor * a = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, k);
        ggml_tensor * b = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, k);
        ggml_set_name(a, "a");
        ggml_set_name(b, "b");
        src0 = a;
        src1 = b;
        dst = a;
        for (int i = 0; i < chain; ++i) dst = ggml_add(ctx.get(), dst, b);
    } else if (handoff > 0) {
        // `--handoff N`: a device-produced N-element f32 tensor that the host reads back and replaces.  It is the cost of
        // one scheduler split boundary - what a CPU-side op costs before it does any work - and it decides whether an op
        // is worth moving to a second backend at all.
        src0 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, handoff);
        src1 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, handoff);
        dst  = ggml_add(ctx.get(), src0, src1);
    } else if (scale) {
        // `--scale`: one scalar SCALE over a [k, m] f32 tensor, the op qwen4exp's per-layer `input_scale` becomes.
        src0 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, m);
        src1 = src0;
        dst  = ggml_scale(ctx.get(), src0, 0.5f);
    } else {
        // `--ops K`: K independent matmuls in one submit, each over a DIFFERENT slice of the weight tensor.  Reusing one
        // slice measured 400-3000 GB/s, which is the L2 answering, not DRAM; distinct slices make the K dispatches read
        // K x the weight bytes, so the marginal rate is the one a model sees when every layer has its own weights.
        ggml_tensor * big = ggml_new_tensor_2d(ctx.get(), type, k, m * ops);
        src0 = big;
        src1 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, n);
        ggml_tensor * first = nullptr;
        for (int i = 0; i < ops; ++i) {
            ggml_tensor * slice = ops == 1 ? big
                : ggml_view_2d(ctx.get(), big, k, m, big->nb[1], (size_t) i * (size_t) m * (size_t) big->nb[1]);
            ggml_tensor * out = ggml_mul_mat(ctx.get(), slice, src1);
            if (first == nullptr) first = out;
        }
        dst = first;
    }

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (buf == nullptr) {
        std::fprintf(stderr, "allocation failed\n");
        return 1;
    }

    if (chain > 0 || scale || handoff > 0) {
        std::vector<float> z((size_t) ggml_nelements(src0), 1.0f);
        ggml_backend_tensor_set(src0, z.data(), 0, z.size() * sizeof(float));
        if (chain > 0 || handoff > 0) ggml_backend_tensor_set(src1, z.data(), 0, z.size() * sizeof(float));
    } else {
        fill_random(src0, 1);
        fill_random(src1, 2);
    }

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, dst);

    auto compute = [&]() {
        ggml_backend_graph_compute(backend.get(), gf);
        // The Vulkan backend records and submits without waiting unless submission is serialized, so a timing loop that
        // does not synchronize measures how fast the CPU can record.  Every number this probe prints is wall clock
        // around a completed device run.
        ggml_backend_synchronize(backend.get());
    };

    compute();   // warm up: pipeline creation, descriptor pools, first-use allocations

    // STRATA_VK_RDC_PROBE=1: run one more compute bracketed by RenderDoc's capture API.  One matmul is one dispatch, so
    // the capture stays tiny - unlike a whole decode step, which references a 542 MiB compute buffer and a 32k KV cache
    // and never finished serialising (port/docs/OPTIMIZATION-BACKLOG.md, phase 4 round 10).  The layer has to be loaded
    // by the launcher (VK_LAYER_PATH + VK_INSTANCE_LAYERS=VK_LAYER_RENDERDOC_Capture).
    if (std::getenv("STRATA_VK_RDC_PROBE") != nullptr) {
#ifdef _WIN32
        if (HMODULE rdc = LoadLibraryA("renderdoc.dll")) {
            auto get_api = (pRENDERDOC_GetAPI) GetProcAddress(rdc, "RENDERDOC_GetAPI");
            void * api = nullptr;
            if (get_api != nullptr && get_api(eRENDERDOC_API_Version_1_6_0, &api) && api != nullptr) {
                auto * rdoc = (RENDERDOC_API_1_6_0 *) api;
                const char * path = std::getenv("STRATA_VK_RDC_PATH");
                rdoc->SetCaptureFilePathTemplate(path != nullptr ? path : "strata_probe");
                std::fprintf(stderr, "[mm-probe] RenderDoc: capturing one compute\n");
                rdoc->StartFrameCapture(nullptr, nullptr);
                compute();
                const bool ok = rdoc->EndFrameCapture(nullptr, nullptr);
                std::fprintf(stderr, "[mm-probe] RenderDoc: capture %s\n", ok ? "written" : "FAILED");
            } else {
                std::fprintf(stderr, "[mm-probe] RenderDoc: no 1.6.0 capture API in renderdoc.dll\n");
            }
        } else {
            std::fprintf(stderr, "[mm-probe] RenderDoc: renderdoc.dll not found\n");
        }
#endif
    }

    if (handoff > 0) {
        const size_t bytes = (size_t) ggml_nbytes(dst);
        std::vector<char> host(bytes);
        auto time = [&](auto && step) {
            const int n = 200;
            const double t0 = now_ms();
            for (int i = 0; i < n; ++i) step();
            return (now_ms() - t0) / n * 1000.0;
        };
        const double base = time([&] { compute(); });
        const double get  = time([&] { compute(); ggml_backend_tensor_get(dst, host.data(), 0, bytes); });
        const double both = time([&] {
            compute();
            ggml_backend_tensor_get(dst, host.data(), 0, bytes);
            ggml_backend_tensor_set(dst, host.data(), 0, bytes);
        });
        std::printf("%s handoff %d floats (%zu B): compute %7.1f us, +readback %7.1f us (+%5.1f), "
                    "+upload %7.1f us (+%5.1f)\n",
                    ggml_backend_dev_name(dev), handoff, bytes, base, get, get - base, both, both - get);
        ggml_backend_free(backend.get());
        backend.release();
        return 0;
    }

    // Time enough repeats that scheduler noise is averaged out, without spending minutes per shape.
    int iters = reps;
    if (iters <= 0) {
        const double t0 = now_ms();
        compute();
        const double one = now_ms() - t0;
        iters = (int) std::max(3.0, std::min(400.0, budget_ms / std::max(one, 0.001)));
    }
    const double t0 = now_ms();
    for (int i = 0; i < iters; ++i) compute();
    const double total = now_ms() - t0;
    const double per = total / iters;

    const double flops = 2.0 * (double) m * (double) n * (double) k * (double) ops;
    const double bytes = chain > 0 ? 3.0 * (double) k * 4.0 * (double) chain
                       : scale    ? 2.0 * (double) ggml_nbytes(dst)
                                   : ((double) ggml_nbytes(src0) + (double) ggml_nbytes(src1) + (double) ggml_nbytes(dst)) * ops;

    if (chain > 0 || scale) {
        std::printf("%s %s k=%lld m=%lld: %8.3f ms  %7.3f us/op  %6.1f GB/s  (%d iters)\n",
                    ggml_backend_dev_name(dev), chain > 0 ? "add-chain" : "scale",
                    (long long) k, (long long) (chain > 0 ? 1 : m),
                    per, per * (chain > 0 ? 1000.0 / chain : 1.0), bytes / per / 1e6, iters);
        ggml_backend_free(backend.get());
        backend.release();
        return 0;
    }
    std::printf("%s %-6s m=%lld n=%lld k=%lld ops=%d: %8.3f ms  %7.3f us/op  %8.1f GFLOP/s  %7.1f GB/s  (%d iters)\n",
                ggml_backend_dev_name(dev), ggml_type_name(type), (long long) m, (long long) n, (long long) k, ops,
                per, per * 1000.0 / ops, flops / per / 1e6, bytes / per / 1e6, iters);

    if (verify) {
        if (ops != 1) {
            std::fprintf(stderr, "--verify supports ops=1 only\n");
            return 2;
        }
        const size_t ne = (size_t) m * (size_t) n;
        std::vector<float> dev_out(ne);
        ggml_backend_tensor_get(dst, dev_out.data(), 0, ne * sizeof(float));

        // the same graph on the CPU backend, from the same deterministic inputs
        ggml_backend_t cpu = ggml_backend_dev_init(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr);
        ggml_context_ptr ctx2(ggml_init(ip));
        ggml_tensor * w2 = ggml_new_tensor_2d(ctx2.get(), type, k, m);
        ggml_tensor * a2 = ggml_new_tensor_2d(ctx2.get(), GGML_TYPE_F32, k, n);
        ggml_tensor * o2 = ggml_mul_mat(ctx2.get(), w2, a2);
        ggml_backend_buffer_ptr buf2(ggml_backend_alloc_ctx_tensors(ctx2.get(), cpu));
        if (buf2 == nullptr) {
            std::fprintf(stderr, "cpu allocation failed\n");
            return 1;
        }
        fill_random(w2, 1);
        fill_random(a2, 2);
        ggml_cgraph * gf2 = ggml_new_graph(ctx2.get());
        ggml_build_forward_expand(gf2, o2);
        ggml_backend_graph_compute(cpu, gf2);
        ggml_backend_synchronize(cpu);
        std::vector<float> cpu_out(ne);
        ggml_backend_tensor_get(o2, cpu_out.data(), 0, ne * sizeof(float));

        size_t bad = 0;
        double max_rel = 0.0;
        double sum_d2 = 0.0;
        double sum_ref2 = 0.0;
        size_t first_bad = SIZE_MAX;
        std::vector<size_t> per_row((size_t) m, 0);
        for (size_t i = 0; i < ne; ++i) {
            const double d = std::fabs((double) dev_out[i] - (double) cpu_out[i]);
            const double s = std::max(std::fabs((double) cpu_out[i]), 1e-3);
            max_rel = std::max(max_rel, d / s);
            sum_d2 += d * d;
            sum_ref2 += (double) cpu_out[i] * (double) cpu_out[i];
            if (d / s > 1e-3) {
                ++bad;
                per_row[i / (size_t) n]++;
                if (first_bad == SIZE_MAX) first_bad = i;
            }
        }
        const double nmse = sum_ref2 > 0.0 ? sum_d2 / sum_ref2 : 0.0;
        std::printf("verify: nmse %.3e (oracle default bound %.1e, ratio %.1f) | %zu of %zu outside 1e-3 rel "
                    "(max rel %.3e)", nmse, 1e-7, nmse / 1e-7, bad, ne, max_rel);
        if (first_bad != SIZE_MAX) {
            const size_t row = first_bad / (size_t) n;
            const size_t col = first_bad % (size_t) n;
            std::printf(", first at row %zu col %zu: dev %.6g cpu %.6g", row, col, dev_out[first_bad], cpu_out[first_bad]);
        }
        std::printf("\nrow pattern (wrong/total per row, first 16):");
        if (type == GGML_TYPE_Q8_0 && k == 32 && n == 1) {
            std::vector<uint8_t> wb(ggml_nbytes(src0));
            ggml_backend_tensor_get(src0, wb.data(), 0, wb.size());
            std::vector<float> xb((size_t) k * (size_t) n);
            ggml_backend_tensor_get(src1, xb.data(), 0, xb.size() * sizeof(float));
            for (size_t r = 0; r < per_row.size() && r < 4; ++r) {
                const uint8_t * b = wb.data() + r * 34;
                const float d = ggml_fp16_to_fp32(*(const ggml_fp16_t *) b);
                std::printf("  blk %zu d=%.8g qs=", r, d);
                for (int i = 0; i < 32; ++i) std::printf("%d,", (int) (int8_t) b[2 + i]);
                std::printf("\n");
            }
            std::printf("  x=");
            for (int i = 0; i < (int) k; ++i) std::printf("%.6g,", xb[i]);
            std::printf("\n");
        }
        for (size_t r = 0; r < per_row.size() && r < 8; ++r) {
            const double dv = dev_out[r * (size_t) n];
            const double cv = cpu_out[r * (size_t) n];
            std::printf("  row %zu: dev %.6g cpu %.6g ratio %+.4f\n", r, dv, cv, cv != 0.0 ? dv / cv : 0.0);
        }
        for (size_t r = 0; r < per_row.size() && r < 16; ++r) std::printf(" %zu/%lld", per_row[r], (long long) n);
        std::printf("\n");
        fflush(stdout);
        ggml_backend_free(cpu);
    }

    ggml_backend_free(backend.get());
    backend.release();
    return 0;
}
