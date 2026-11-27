// cpy-probe.cpp - copy f32 to a quantised destination on two backends and compare the raw block bytes.
//
// test-backend-ops reports a relative error of about 2 for CPY into q4_0, but its comparison dequantises, so it cannot
// say what the kernel actually wrote.  This tool reads the destination back as bytes and prints both copies, which
// separates "nothing written", "only the scale written" and "nibbles packed in the wrong order".
//
//   cpy-probe [--n 96] [--dst q4_0] [--blocks 3]
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

bool run(const char * backend_name, int64_t n, ggml_type dst_type, std::vector<uint8_t> & out) {
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

    ggml_tensor * src = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_set_name(src, "src");
    ggml_tensor * dst = ggml_new_tensor_1d(ctx.get(), dst_type, n);
    ggml_set_name(dst, "dst");
    ggml_tensor * cpy = ggml_cpy(ctx.get(), src, dst);
    ggml_set_name(cpy, "cpy");

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, cpy);

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (buf == nullptr) {
        std::fprintf(stderr, "%s: allocation failed\n", backend_name);
        ggml_backend_free(backend);
        return false;
    }

    std::vector<float> f((size_t) n);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto & v : f) v = dist(rng);
    ggml_backend_tensor_set(src, f.data(), 0, f.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: compute failed\n", backend_name);
        ggml_backend_free(backend);
        return false;
    }

    out.assign(ggml_nbytes(dst), 0);
    ggml_backend_tensor_get(dst, out.data(), 0, out.size());
    ggml_backend_free(backend);
    return true;
}

void dump(const char * tag, const std::vector<uint8_t> & bytes) {
    std::printf("%s (%zu bytes):\n", tag, bytes.size());
    for (size_t i = 0; i < bytes.size(); ++i) {
        std::printf("%02x%s", bytes[i], (i % 18 == 17) ? "\n" : " ");
    }
    std::printf("\n");
}

}  // namespace

int main(int argc, char ** argv) {
    int64_t n = 96;
    std::string dst_name = "q4_0";

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const char * val = (i + 1 < argc) ? argv[++i] : nullptr;
        if (a == "--n" && val)               n = std::atoll(val);
        else if (a == "--dst" && val)        dst_name = val;
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }

    ggml_type dst_type = GGML_TYPE_Q4_0;
    if (dst_name == "q4_0")      dst_type = GGML_TYPE_Q4_0;
    else if (dst_name == "q8_0") dst_type = GGML_TYPE_Q8_0;
    else { std::fprintf(stderr, "unsupported dst %s\n", dst_name.c_str()); return 2; }

    std::vector<uint8_t> cpu, dev;
    if (!run("CPU", n, dst_type, cpu)) return 1;
    if (!run("Vulkan0", n, dst_type, dev)) return 1;

    dump("cpu", cpu);
    dump("vulkan", dev);

    size_t diff = 0;
    for (size_t i = 0; i < cpu.size() && i < dev.size(); ++i) {
        if (cpu[i] != dev[i]) ++diff;
    }
    std::printf("differing bytes: %zu of %zu\n", diff, cpu.size());
    return diff == 0 ? 0 : 1;
}
