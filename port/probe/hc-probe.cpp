// hc-probe.cpp - run one GGML_OP_DSV4_HC_PRE on two backends and compare element by element.
//
// The op is covered by test-backend-ops for several shapes, but the product model calls it with
// n_embd = 2560 and n_tokens = 1, which no case exercises, and the node-by-node model diff put a factor of 21
// between the CPU and Vulkan values of that node.  This tool reproduces exactly that shape.
//
//   hc-probe --n-embd 2560 --hc 4 --n-tokens 1 [--gated 0|1] [--scale 1.0]
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

struct result {
    std::vector<float> v;
};

bool run(const char * backend_name, int64_t n_embd, int64_t hc, int64_t n_tokens, float scale, bool gated,
         const std::vector<float> & x_data, const std::vector<float> & w_data, result & out) {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(backend_name);
    if (dev == nullptr) {
        std::fprintf(stderr, "no device %s\n", backend_name);
        return false;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (backend == nullptr) {
        std::fprintf(stderr, "cannot init %s\n", backend_name);
        return false;
    }

    ggml_init_params ip = { 64u * 1024u * 1024u, nullptr, true };
    ggml_context_ptr ctx(ggml_init(ip));

    ggml_tensor * x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, n_embd, hc, n_tokens);
    ggml_set_name(x, "x");
    ggml_tensor * w = gated
        ? ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, n_embd, hc, n_tokens)
        : ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hc, n_tokens);
    ggml_set_name(w, "w");
    // The plain form takes no scale (the op carries 1.0), the gated form takes one.
    ggml_tensor * dst = gated
        ? ggml_dsv4_hc_pre_gated(ctx.get(), x, w, scale)
        : ggml_dsv4_hc_pre(ctx.get(), x, w);
    ggml_set_name(dst, "dst");

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, dst);

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (buf == nullptr) {
        std::fprintf(stderr, "%s: allocation failed\n", backend_name);
        ggml_backend_free(backend);
        return false;
    }

    ggml_backend_tensor_set(x, x_data.data(), 0, x_data.size() * sizeof(float));
    ggml_backend_tensor_set(w, w_data.data(), 0, w_data.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: compute failed\n", backend_name);
        ggml_backend_free(backend);
        return false;
    }

    out.v.resize((size_t) n_embd * n_tokens);
    ggml_backend_tensor_get(dst, out.v.data(), 0, out.v.size() * sizeof(float));
    ggml_backend_free(backend);
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    int64_t n_embd = 2560, hc = 4, n_tokens = 1;
    float scale = 1.0f;
    bool gated = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const char * val = (i + 1 < argc) ? argv[++i] : nullptr;
        if (a == "--n-embd" && val)       n_embd = std::atoll(val);
        else if (a == "--hc" && val)      hc = std::atoll(val);
        else if (a == "--n-tokens" && val) n_tokens = std::atoll(val);
        else if (a == "--scale" && val)   scale = (float) std::atof(val);
        else if (a == "--gated" && val)   gated = std::atoi(val) != 0;
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x_data((size_t) n_embd * hc * n_tokens);
    for (auto & v : x_data) v = dist(rng);
    const size_t w_count = gated ? (size_t) n_embd * hc * n_tokens : (size_t) hc * n_tokens;
    std::vector<float> w_data(w_count);
    for (auto & v : w_data) v = dist(rng);

    result cpu, dev;
    if (!run("CPU", n_embd, hc, n_tokens, scale, gated, x_data, w_data, cpu)) return 1;
    if (!run("Vulkan0", n_embd, hc, n_tokens, scale, gated, x_data, w_data, dev)) return 1;

    double max_rel = 0.0;
    size_t worst = 0;
    double abs_sum_cpu = 0.0, abs_sum_dev = 0.0;
    for (size_t i = 0; i < cpu.v.size(); ++i) {
        abs_sum_cpu += std::fabs(cpu.v[i]);
        abs_sum_dev += std::fabs(dev.v[i]);
        const double denom = std::max(std::fabs((double) cpu.v[i]), 1e-6);
        const double rel = std::fabs((double) dev.v[i] - cpu.v[i]) / denom;
        if (rel > max_rel) {
            max_rel = rel;
            worst = i;
        }
    }
    std::printf("n_embd=%lld hc=%lld n_tokens=%lld gated=%d scale=%g\n",
                (long long) n_embd, (long long) hc, (long long) n_tokens, (int) gated, (double) scale);
    std::printf("abs_sum cpu=%.9g dev=%.9g  max_rel=%.3g at %zu (cpu %.9g dev %.9g)\n",
                abs_sum_cpu, abs_sum_dev, max_rel, worst, (double) cpu.v[worst], (double) dev.v[worst]);
    return max_rel > 1e-3 ? 1 : 0;
}
