// setrows-probe.cpp - copy rows into a quantised cache the way the model writes K/V, on two backends.
//
// The q4_0 cache path faults during prefill while a single attention call reproduces fine, so the write side is the
// remaining suspect.  This tool writes 24 rows of f16 into a quantised cache through a view (the same shape the model
// uses) and compares the cache bytes against the CPU backend.
//
//   setrows-probe [--row-len 128] [--rows 24] [--cells 256] [--dst q4_0]
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

struct params {
    int64_t row_len = 128;
    int64_t rows    = 24;
    int64_t cells   = 256;
    std::string dst = "q4_0";
};

bool run(const char * backend_name, const params & prm, ggml_type dst_type, std::vector<uint8_t> & out) {
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

    ggml_init_params ip = { 32u * 1024u * 1024u, nullptr, true };
    ggml_context_ptr ctx(ggml_init(ip));

    // the cache is two streams wide, and the write goes through a view of the first one, as it does in the model
    ggml_tensor * cache = ggml_new_tensor_3d(ctx.get(), dst_type, prm.row_len, prm.cells, 2);
    ggml_tensor * dstv  = ggml_view_3d(ctx.get(), cache, prm.row_len, prm.cells, 1,
                                       cache->nb[1], cache->nb[2], 0);
    ggml_tensor * src   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, prm.row_len, prm.rows);
    ggml_tensor * idx   = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, prm.rows);
    ggml_set_name(cache, "cache");
    ggml_set_name(dstv, "cache_view");
    ggml_set_name(src, "src");
    ggml_set_name(idx, "idx");

    ggml_tensor * sr = ggml_set_rows(ctx.get(), dstv, src, idx);
    ggml_set_name(sr, "set_rows");

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, sr);

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (buf == nullptr) {
        std::fprintf(stderr, "%s: allocation failed\n", backend_name);
        ggml_backend_free(backend);
        return false;
    }

    std::mt19937 rng(23);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<uint16_t> src_host((size_t) (prm.row_len * prm.rows));
    for (auto & x : src_host) x = ggml_fp32_to_fp16(dist(rng));
    ggml_backend_tensor_set(src, src_host.data(), 0, src_host.size() * sizeof(uint16_t));

    std::vector<int64_t> idx_host((size_t) prm.rows);
    for (int64_t i = 0; i < prm.rows; ++i) idx_host[(size_t) i] = i;
    ggml_backend_tensor_set(idx, idx_host.data(), 0, idx_host.size() * sizeof(int64_t));

    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: compute failed\n", backend_name);
        ggml_backend_free(backend);
        return false;
    }

    out.assign(ggml_nbytes(cache), 0);
    ggml_backend_tensor_get(cache, out.data(), 0, out.size());
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
        if (a == "--row-len")      prm.row_len = std::atoll(val);
        else if (a == "--rows")    prm.rows = std::atoll(val);
        else if (a == "--cells")   prm.cells = std::atoll(val);
        else if (a == "--dst")     prm.dst = val;
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }

    const ggml_type dst_type = prm.dst == "f16"  ? GGML_TYPE_F16
                             : prm.dst == "q8_0" ? GGML_TYPE_Q8_0
                             : GGML_TYPE_Q4_0;

    std::printf("row_len=%lld rows=%lld cells=%lld dst=%s\n",
                (long long) prm.row_len, (long long) prm.rows, (long long) prm.cells, prm.dst.c_str());

    std::vector<uint8_t> cpu, dev;
    const bool cpu_ok = run("CPU", prm, dst_type, cpu);
    std::printf("cpu: %s\n", cpu_ok ? "ok" : "failed");
    const bool dev_ok = run("Vulkan0", prm, dst_type, dev);
    std::printf("vulkan: %s\n", dev_ok ? "ok" : "failed");
    if (!cpu_ok || !dev_ok) {
        return 1;
    }

    size_t diff = 0;
    size_t first = (size_t) -1, last = 0;
    for (size_t i = 0; i < cpu.size() && i < dev.size(); ++i) {
        if (cpu[i] != dev[i]) {
            ++diff;
            if (first == (size_t) -1) first = i;
            last = i;
        }
    }
    std::printf("differing bytes: %zu of %zu\n", diff, cpu.size());
    if (diff != 0) {
        const size_t row_size = ggml_row_size(dst_type, prm.row_len);
        const size_t stream = row_size * (size_t) prm.cells;
        std::printf("first at %zu, last at %zu (stream size %zu, so first is in stream %s)\n",
                    first, last, stream, first < stream ? "0" : "1");
        // count per cache row so an overrun shows up as rows that were never written
        std::printf("first row %zu, last row %zu, rows written %lld\n",
                    first / row_size, last / row_size, (long long) prm.rows);
    }
    return diff == 0 ? 0 : 1;
}
