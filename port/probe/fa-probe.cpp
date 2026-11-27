// fa-probe.cpp - run one flash-attention op with the shape the model uses and compare CPU against a device.
//
// The product path with a q4_0 K/V cache faults during prefill, and the last attention call before the fault has
// hsk=hsv=256, a sparse selection (op_params[4] = n_kv_max > 0) and 24 query tokens, which is a combination
// test-backend-ops does not generate.  This tool builds exactly that call so the failure can be reproduced without the
// model, and so a fix can be checked against the CPU backend's result.
//
//   fa-probe [--hsk 256] [--nh 16] [--tokens 24] [--cells 256] [--kv q4_0] [--sparse 1]
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

struct params {
    int64_t hsk    = 256;
    int64_t nh     = 16;
    int64_t tokens = 24;
    int64_t cells  = 256;
    int64_t n_used = 259;
    std::string kv = "q4_0";
    bool sparse    = true;
    bool kv_view   = true;
    int64_t streams = 1;
    int64_t mask_slack = 0;
};

bool run(const char * backend_name, const params & prm, ggml_type kv_type, std::vector<float> & out) {
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

    // ggml_flash_attn_ext requires q and k to agree on dims 0 and 2 and for the sequence counts to line up, so the
    // query carries one entry per stream exactly as the model builds it.
    ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, prm.hsk, prm.nh, prm.tokens, prm.streams);
    // The model hands attention a view into a cache tensor rather than a freshly allocated one, so the kernel
    // sees a stride that spans the cache; --kv-view 1 is the default for that reason.
    ggml_tensor * k_full = ggml_new_tensor_4d(ctx.get(), kv_type, prm.hsk, prm.cells, 1, prm.streams);
    ggml_tensor * v_full = ggml_new_tensor_4d(ctx.get(), kv_type, prm.hsk, prm.cells, 1, prm.streams);
    ggml_tensor * k = prm.kv_view ? ggml_view_4d(ctx.get(), k_full, prm.hsk, prm.cells, 1, prm.streams,
            k_full->nb[1], k_full->nb[2], k_full->nb[3], 0) : k_full;
    ggml_tensor * v = prm.kv_view ? ggml_view_4d(ctx.get(), v_full, prm.hsk, prm.cells, 1, prm.streams,
            v_full->nb[1], v_full->nb[2], v_full->nb[3], 0) : v_full;
    ggml_tensor * m = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, prm.cells, prm.tokens);
    ggml_set_name(q, "q");
    ggml_set_name(k, "k");
    ggml_set_name(v, "v");
    ggml_set_name(m, "m");

    ggml_tensor * fa = ggml_flash_attn_ext(ctx.get(), q, k, v, m, 1.0f / std::sqrt((float) prm.hsk), 0.0f, 0.0f);
    ggml_set_name(fa, "fa");
    if (prm.sparse) {
        // llama.cpp passes the selection length through op_params[4], which is what switches the shader to the sparse
        // gather; without it the same shapes take the dense route.
        ggml_flash_attn_ext_set_n_kv_max(fa, (int32_t) prm.n_used);
    }

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, fa);

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (buf == nullptr) {
        std::fprintf(stderr, "%s: allocation failed\n", backend_name);
        ggml_backend_free(backend);
        return false;
    }

    std::mt19937 rng(11);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> q_host((size_t) (prm.hsk * prm.nh * prm.tokens * prm.streams));
    for (auto & x : q_host) x = dist(rng);
    ggml_backend_tensor_set(q, q_host.data(), 0, q_host.size() * sizeof(float));

    // K and V are filled through the CPU quantiser so both backends start from identical bytes.
    const size_t row_size = ggml_row_size(kv_type, prm.hsk);
    std::vector<uint8_t> kv_host(row_size * (size_t) prm.cells * (size_t) prm.streams);
    std::vector<float> row((size_t) prm.hsk);
    for (int64_t c = 0; c < prm.cells; ++c) {
        for (auto & x : row) x = dist(rng);
        ggml_quantize_chunk(kv_type, row.data(), kv_host.data() + (size_t) c * row_size, 0, 1, prm.hsk, nullptr);
    }
    ggml_backend_tensor_set(k_full, kv_host.data(), 0, kv_host.size());
    ggml_backend_tensor_set(v_full, kv_host.data(), 0, kv_host.size());

    std::vector<uint16_t> mask_host((size_t) (prm.cells * prm.tokens));
    for (int64_t t = 0; t < prm.tokens; ++t) {
        for (int64_t c = 0; c < prm.cells; ++c) {
            const bool active = c < std::min(prm.cells, prm.n_used);
            // --mask-slack lets an index point past the cache rows, which is what the model does when the
            // selection is sized by n_kv_max rather than by the number of rows the cache holds.
            const float idx_value = (float) (c + prm.mask_slack);
            mask_host[(size_t) (t * prm.cells + c)] = active ? (uint16_t) ggml_fp32_to_fp16(idx_value) : (uint16_t) ggml_fp32_to_fp16(-1.0f);
        }
    }
    ggml_backend_tensor_set(m, mask_host.data(), 0, mask_host.size() * sizeof(uint16_t));

    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: compute failed\n", backend_name);
        ggml_backend_free(backend);
        return false;
    }

    out.assign((size_t) ggml_nelements(fa), 0.0f);
    ggml_backend_tensor_get(fa, out.data(), 0, out.size() * sizeof(float));
    ggml_backend_free(backend);
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    params prm;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const char * val = (i + 1 < argc) ? argv[++i] : nullptr;
        if (!val) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); return 2; }
        if (a == "--hsk")          prm.hsk = std::atoll(val);
        else if (a == "--nh")      prm.nh = std::atoll(val);
        else if (a == "--tokens")  prm.tokens = std::atoll(val);
        else if (a == "--cells")   prm.cells = std::atoll(val);
        else if (a == "--n-used")  prm.n_used = std::atoll(val);
        else if (a == "--kv")      prm.kv = val;
        else if (a == "--sparse")  prm.sparse = std::atoi(val) != 0;
        else if (a == "--kv-view") prm.kv_view = std::atoi(val) != 0;
        else if (a == "--streams") prm.streams = std::atoll(val);
        else if (a == "--mask-slack") prm.mask_slack = std::atoll(val);
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }

    const ggml_type kv_type = prm.kv == "f16"  ? GGML_TYPE_F16
                            : prm.kv == "q8_0" ? GGML_TYPE_Q8_0
                            : GGML_TYPE_Q4_0;

    std::printf("hsk=%lld nh=%lld tokens=%lld cells=%lld kv=%s sparse=%d slack=%lld\n",
                (long long) prm.hsk, (long long) prm.nh, (long long) prm.tokens, (long long) prm.cells,
                prm.kv.c_str(), (int) prm.sparse, (long long) prm.mask_slack);

    std::vector<float> cpu, dev;
    const bool cpu_ok = run("CPU", prm, kv_type, cpu);
    std::printf("cpu: %s\n", cpu_ok ? "ok" : "failed");
    const bool dev_ok = run("Vulkan0", prm, kv_type, dev);
    std::printf("vulkan: %s\n", dev_ok ? "ok" : "failed");
    if (!cpu_ok || !dev_ok) {
        return 1;
    }

    double num = 0.0, den = 0.0, max_abs = 0.0;
    for (size_t i = 0; i < cpu.size(); ++i) {
        const double d = (double) cpu[i] - (double) dev[i];
        num += d * d;
        den += (double) cpu[i] * (double) cpu[i];
        max_abs = std::fmax(max_abs, std::fabs(d));
    }
    std::printf("nmse %.3e, max abs diff %.3e over %zu values\n", den > 0 ? num / den : num, max_abs, cpu.size());
    return 0;
}
