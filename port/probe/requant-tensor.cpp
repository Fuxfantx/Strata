// requant-tensor - write a companion GGUF holding a few tensors of a base pack, requantised to another type.
//
// The engine's --compat composes such a file over the base pack (a tensor in the companion replaces the base pack's
// tensor of the same name; the marker is strata.companion = true).  This exists because llama-quantize cannot do it:
// its `--only-tensors` does not filter what gets written, and the option is not in the vendored source at all
// (PERFORMANCE.md section 5.0.1).  The file itself is written by port/engine/vk-companion.cpp, which the engine's own
// BF16 -> F16 pass uses too, so the two cannot drift.
//
//   requant-tensor BASE.gguf OUT.gguf name-substring:type [name-substring:type ...]
//
// Each argument is `<name-substring>:<ggml type name>`; every base tensor whose name contains the substring (as a
// whole dot-separated component) is written, so `attn_q:q4_K` and `attn_qkv:q4_K` are different selections.
#include "vk-companion.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

// A pattern selects a tensor when it matches a whole run of dot-separated components, so "ssm_a" cannot pick up
// "ssm_alpha.weight" while "attn_qkv.weight" still picks up every layer.
bool selects(const char * name, const std::string & pat) {
    const std::string n = name;
    if (n == pat) return true;
    for (size_t pos = n.find(pat); pos != std::string::npos; pos = n.find(pat, pos + 1)) {
        const bool head_ok = pos == 0 || n[pos - 1] == '.';
        const size_t end   = pos + pat.size();
        const bool tail_ok = end == n.size() || n[end] == '.';
        if (head_ok && tail_ok) return true;
    }
    return false;
}

ggml_type type_by_name(const std::string & n) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        if (n == ggml_type_name((ggml_type) t)) return (ggml_type) t;
    }
    return GGML_TYPE_COUNT;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s BASE.gguf OUT.gguf substr:type [substr:type ...]\n", argv[0]);
        return 2;
    }
    const std::string base = argv[1];
    const std::string out  = argv[2];

    // Every (tensor, destination type) pair begins from the base pack's own tensor list, read once.
    const std::vector<std::string> names = strata::companion_tensor_names(base);
    std::vector<strata::CompanionPick> picks;
    for (int a = 3; a < argc; ++a) {
        const std::string spec = argv[a];
        const size_t colon = spec.rfind(':');
        if (colon == std::string::npos) {
            std::fprintf(stderr, "bad selection '%s' (want substr:type)\n", spec.c_str());
            return 2;
        }
        const std::string pat = spec.substr(0, colon);
        const ggml_type type = type_by_name(spec.substr(colon + 1));
        if (type == GGML_TYPE_COUNT) {
            std::fprintf(stderr, "unknown type '%s'\n", spec.substr(colon + 1).c_str());
            return 2;
        }
        bool any = false;
        for (const std::string & name : names) {
            if (!selects(name.c_str(), pat)) continue;
            picks.push_back({ name, type });
            any = true;
        }
        if (!any) {
            std::fprintf(stderr, "no tensor in %s matches '%s'\n", base.c_str(), pat.c_str());
            return 1;
        }
    }
    std::fprintf(stderr, "%zu tensor(s) selected\n", picks.size());

    std::string err;
    if (!strata::write_companion(base, out, picks, err)) {
        std::fprintf(stderr, "failed: %s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr, "wrote %s\n", out.c_str());
    return 0;
}
