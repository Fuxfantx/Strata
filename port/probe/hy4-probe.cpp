// hy4-probe.cpp - decode one real block per Hy4 type through ggml's dequantiser and compare with the pack reference.
//
// The expected values come from port/tools/hy4-dequant-ref.py, which was written from the pack's own encoder; this tool
// is the check that the C implementation agrees with it value for value.
//
//   hy4-probe --dir <dump dir produced by hy4-dequant-ref.py --dump>
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" void ggml_hy4_set_codebooks(const float * csq8, const float * csq4, const float * vq4_128, const float * vq4_64);
extern "C" void ggml_vec_dot_csq8   (int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_csq4   (int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_vq4_128(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_vq4_64 (int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);

namespace {

bool read_file(const std::string & path, std::vector<uint8_t> & out) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out.resize((size_t) n);
    const size_t got = std::fread(out.data(), 1, out.size(), f);
    std::fclose(f);
    return got == out.size();
}

bool read_expected(const std::string & path, std::vector<float> & out) {
    FILE * f = std::fopen(path.c_str(), "r");
    if (f == nullptr) {
        return false;
    }
    long n = 0;
    if (std::fscanf(f, "%ld", &n) != 1) {
        std::fclose(f);
        return false;
    }
    out.resize((size_t) n);
    for (long i = 0; i < n; ++i) {
        double v = 0.0;
        if (std::fscanf(f, "%lf", &v) != 1) {
            std::fclose(f);
            return false;
        }
        out[(size_t) i] = (float) v;
    }
    std::fclose(f);
    return true;
}

bool check(const std::string & dir, const char * kind, ggml_type type, const float * cb, size_t cb_floats) {
    std::vector<uint8_t> block;
    std::vector<uint8_t> cb_raw;
    std::vector<float>   expected;
    const std::string base = dir + "\\" + kind;
    if (!read_file(base + "-block.bin", block) || !read_file(base + "-cb.bin", cb_raw) ||
        !read_expected(base + "-expected.txt", expected)) {
        std::fprintf(stderr, "%s: cannot read the dump\n", kind);
        return false;
    }
    if (cb_raw.size() != cb_floats * sizeof(float)) {
        std::fprintf(stderr, "%s: codebook is %zu bytes, expected %zu\n", kind, cb_raw.size(), cb_floats * sizeof(float));
        return false;
    }
    (void) cb;

    const ggml_type_traits * tr = ggml_get_type_traits(type);
    if (tr == nullptr || tr->to_float == nullptr) {
        std::fprintf(stderr, "%s: no dequantiser registered\n", kind);
        return false;
    }
    if ((size_t) tr->type_size != block.size()) {
        std::fprintf(stderr, "%s: type_size %d, block is %zu bytes\n", kind, tr->type_size, block.size());
        return false;
    }

    std::vector<float> got((size_t) tr->blck_size);
    tr->to_float(block.data(), got.data(), tr->blck_size);

    if (got.size() != expected.size()) {
        std::fprintf(stderr, "%s: decoded %zu values, reference has %zu\n", kind, got.size(), expected.size());
        return false;
    }

    double num = 0.0, den = 0.0, max_abs = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = (double) got[i] - (double) expected[i];
        num += d * d;
        den += (double) expected[i] * (double) expected[i];
        max_abs = std::fmax(max_abs, std::fabs(d));
    }
    std::printf("%-8s %5zu values  max abs diff %.3e  nmse %.3e  first %.6f (ref %.6f)\n",
                kind, got.size(), max_abs, den > 0 ? num / den : num, got[0], expected[0]);

    // The dot product must be the same arithmetic as the dequantiser with the row temporary removed, so it is compared with
    // the dequantised row itself rather than with a reference: a disagreement here is the function's fault, not the pack's.
    decltype(ggml_vec_dot_csq8) * dot = nullptr;
    if      (type == GGML_TYPE_CSQ8)    dot = ggml_vec_dot_csq8;
    else if (type == GGML_TYPE_CSQ4)    dot = ggml_vec_dot_csq4;
    else if (type == GGML_TYPE_VQ4_128) dot = ggml_vec_dot_vq4_128;
    else if (type == GGML_TYPE_VQ4_64)  dot = ggml_vec_dot_vq4_64;
    if (dot != nullptr) {
        std::vector<float> y(got.size());
        for (size_t i = 0; i < y.size(); ++i) {
            y[i] = std::sin(0.01f * (float) (i + 1));      // a fixed, non-constant row so a mis-indexed product cannot pass
        }
        double want = 0.0;
        for (size_t i = 0; i < got.size(); ++i) {
            want += (double) got[i] * (double) y[i];
        }
        float got_dot = 0.0f;
        dot((int) got.size(), &got_dot, 0, block.data(), 0, y.data(), 0, 1);
        const double diff = std::fabs((double) got_dot - want);
        std::printf("%-8s dot product  %.6f (dequantised %.6f)  diff %.3e\n", kind, (double) got_dot, want, diff);
        if (diff > 1e-3) {
            return false;
        }
    }

    return max_abs < 1e-6;
}

}  // namespace

int main(int argc, char ** argv) {
    std::string dir = ".";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--dir" && i + 1 < argc) {
            dir = argv[++i];
        } else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }

    // The loader does this in the real flow; the probe has to do it because the codebooks belong to the model.
    std::vector<uint8_t> raw;
    std::vector<float>   csq8(8), csq4(4), vq128(128 * 4), vq64(64 * 4);
    auto load_cb = [&](const char * kind, std::vector<float> & dst) {
        if (!read_file(dir + "\\" + kind + "-cb.bin", raw) || raw.size() != dst.size() * sizeof(float)) {
            std::fprintf(stderr, "%s: cannot read the codebook\n", kind);
            return false;
        }
        std::memcpy(dst.data(), raw.data(), raw.size());
        return true;
    };
    if (!load_cb("csq8", csq8) || !load_cb("csq4", csq4) || !load_cb("vq4_128", vq128) || !load_cb("vq4_64", vq64)) {
        return 1;
    }
    ggml_hy4_set_codebooks(csq8.data(), csq4.data(), vq128.data(), vq64.data());

    bool ok = true;
    ok = check(dir, "csq8",    GGML_TYPE_CSQ8,    csq8.data(), csq8.size()) && ok;
    ok = check(dir, "csq4",    GGML_TYPE_CSQ4,    csq4.data(), csq4.size()) && ok;
    ok = check(dir, "vq4_128", GGML_TYPE_VQ4_128, vq128.data(), vq128.size()) && ok;
    ok = check(dir, "vq4_64",  GGML_TYPE_VQ4_64,  vq64.data(), vq64.size()) && ok;

    std::printf("%s\n", ok ? "all four types match the reference" : "MISMATCH");
    return ok ? 0 : 1;
}
