// vk-options.cpp - arg parsing.  Unknown flags are accepted on purpose: the app passes the released engine's whole
// option set, and an engine that refused a flag it does not need would not be a drop-in component.
#include "vk-options.h"

#include <cstdlib>
#include <cstring>

namespace strata {
namespace {

// Flags that take one value and that this engine does not use.  They are consumed so the value cannot be mistaken
// for the model path.
const char * const kIgnoredValueFlags[] = {
    "--batch", "--slots", "--draft", "--draft-vocab", "--draft-conf", "--draft-max", "--spec", "--resident-budget-gib",
    "--resident-experts", "--vram-reserve", "--pool-workers", "--hybrid-rows", "--batch-groups", "--gpu", "--gpus",
    "--tensor-split", "--defrag-thold", "--tune", "--vision", "--vision-tokens", "--rope-scaling", "--rope-scale",
    "--seed", "--temp", "--top-k", "--top-p", "--min-p", "--repeat-penalty", "--presence-penalty",
    "--frequency-penalty", "--penalty-last-n", "--expert-vram", "--prefill-head", "--serve-log", "--session",
    "--session-dir", "--api-key", "--host", "--port", "--draft-window", "--dflash", "--lookup-max-t",
    "--cache-type-k", "--cache-type-v", "--ubatch-size", "--logits-file", "--threads-batch", "--numa", "--device",
    "--mmproj", "--image-max-tokens", "--chat-template", "--reasoning", "--effort", "--sym", "--chat-template-file",
    "--parallel", "--batch-slots", "--slot", "--split-mode", "--main-gpu", "--session-file", "--dump-logits",
};

bool is_ignored_value_flag(const std::string & a) {
    for (const char * f : kIgnoredValueFlags) {
        if (a == f) return true;
    }
    return false;
}

bool parse_int(const char * s, int & out) {
    if (s == nullptr || *s == '\0') return false;
    char * end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0') return false;
    out = (int) v;
    return true;
}

Tri parse_tri(const char * v) {
    if (std::strcmp(v, "off") == 0)  return Tri::Off;
    if (std::strcmp(v, "auto") == 0) return Tri::Auto;
    return Tri::On;
}

void print_help() {
    std::printf("strata-vk %s - the Strata engine on ggml Vulkan (Windows on ARM)\n"
                "  --serve --native MODEL.gguf [--max-context N] [--prefill N] [--threads N]\\n"
                "  --tokens IDS [--n N] | --bench-pp N | --bench-tg N\n"
                "  --detok                  read --tokens IDS back as text (vocabulary only, no context; the other half of\n"
                "                           `llama-tokenize --ids`)\n"
                "  --lazy-mode on|off|auto  --flash-attn on|off|auto  --lookup on|off (default off)\n"
                "  --dry-run                print the placement and tuning that the flags resolve to, then exit\n"
                "  --load-mode mmap|direct|none  how the weights reach their buffers (default mmap)\n"
                "  --mtp MTP.gguf           draft from a trained nextn head (needs the loader graft, vk-mtp.h)\n"
                "  --compat COMPAT.gguf     use a companion pack for the tensors it carries (port/tools/make-compat.ps1)\n"
                "  --npu off|auto|head|dense|all|split  Hexagon NPU placement: none, find only, output head, dense, whole\n"
                "                           graph, or split (the last --npu-layers layers + head on the HTP)\n"
                "                           (head needs a per-buffer budget above ggml-hexagon's 1 GiB default on a\n" 
                "                            large-vocab model; see docs/NPU-FEASIBILITY.md before raising it)\n"
                "  --gpu vulkan|opencl|cpu  which compute backend runs the graph (default: vulkan when present)\n"
                "  --npu-layers N           --npu split: last N blocks plus the head on the HTP\n"
                "  --npu-mtp                run the MTP draft head on the HTP, not the GPU\n"
                "  --kv f16|q8_0|q4_0       KV cache type; a quantized one needs --flash-attn on\n"
                "  --npu-probe              print what the NPU path found\n"
                "  --geometry               print the devices\n"
                "  --version\n", kEngineVersion);
}

} // namespace

std::vector<llama_token> parse_ids(const std::string & s) {
    std::vector<llama_token> out;
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && (s[i] == ',' || s[i] == ' ')) ++i;
        if (i >= s.size()) break;
        char * end = nullptr;
        const long v = std::strtol(s.c_str() + i, &end, 10);
        if (end == s.c_str() + i) break;
        out.push_back((llama_token) v);
        i = (size_t) (end - s.c_str());
    }
    return out;
}

Options parse_args(int argc, char ** argv) {
    Options o;
    if (const char * cap = std::getenv("STRATA_VK_PERF_CAP")) { o.perf_cap = std::atoi(cap); if (o.perf_cap < 1) o.perf_cap = 1; if (o.perf_cap > 100) o.perf_cap = 100; }
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const char * v = nullptr;
        const auto take = [&]() -> bool {
            if (i + 1 < argc) { v = argv[++i]; return true; }
            return false;
        };

        if (a == "--serve") { o.serve = true; }
        else if (a == "--version" || a == "-v") { o.version = true; }
        else if (a == "--geometry") { o.geometry = true; }
        else if (a == "--detok") { o.detok = true; }
        else if (a == "--npu-probe") { o.npu_probe = true; }
        else if (a == "--native" || a == "--model" || a == "--model-path" || a == "-m") { if (take()) o.model = v; }
        else if (a == "--max-context" || a == "--ctx-size" || a == "-c") { if (take()) parse_int(v, o.n_ctx); }
        // `--prefill` is Strata's prompt-chunk flag and maps to llama.cpp's batch.  `--batch` is NOT that: in Strata's
        // protocol it is the number of concurrent request slots, which this engine answers by running one at a time -
        // reading it as the chunk size would turn a 4-slot run into a 4-token prefill chunk.  It is consumed below with
        // the other flags this engine does not use.
        else if (a == "--prefill") { int n = 0; if (take() && parse_int(v, n) && n > 0) o.n_batch = n; }
        else if (a == "--batch-size" || a == "-b") { if (take()) parse_int(v, o.n_batch); }
        else if (a == "--perf-cap") { if (take()) parse_int(v, o.perf_cap); if (o.perf_cap < 1) o.perf_cap = 1; if (o.perf_cap > 100) o.perf_cap = 100; }
        else if (a == "--threads" || a == "-t") { if (take()) parse_int(v, o.threads); }
        else if (a == "--n" || a == "--max-new" || a == "-n") { if (take()) parse_int(v, o.max_new); }
        else if (a == "--tokens") { if (take()) o.tokens = parse_ids(v); }
        else if (a == "--prompt" || a == "-p") { if (take()) o.prompt = v; }
        else if (a == "--kv" || a == "--kv-type" || a == "--cache-type-k" || a == "--cache-type-v") { if (take()) o.kv = v; }
        else if (a == "--dry-run") { o.dry_run = true; }
        else if (a == "--compat") { if (take()) o.compat = v; }
        else if (a == "--mtp") { if (take()) o.mtp = v; }
        else if (a == "--lazy-mode") { if (take()) o.lazy = parse_tri(v); }
        else if (a == "--load-mode") { if (take()) { o.load_mode = v; o.load_mode_set = true; } }
        else if (a == "--dump-logits") { if (take()) o.dump_logits = v; }
        else if (a == "--split-cpu") { if (take()) o.split_cpu = (float) std::atof(v); }
        else if (a == "--cpu-moe") { o.cpu_moe = true; }
        else if (a == "--n-cpu-moe") { if (take()) parse_int(v, o.n_cpu_moe); }
        else if (a == "--moe-cache-size") { int mib = 0; if (take() && parse_int(v, mib) && mib >= 0) o.moe_cache_mib = (size_t) mib; }
        else if (a == "--moe-cache-cpu-share") { if (take()) o.moe_cache_cpu_share = (float) std::atof(v); }
        else if (a == "--flash-attn" || a == "-fa") { if (take()) { o.flash_attn = parse_tri(v); o.flash_attn_set = true; } }
        else if (a == "--lookup") { if (take()) o.lookup = parse_tri(v); }
        else if (a == "--npu") {
            if (take()) o.npu = std::strcmp(v, "off") == 0   ? NpuMode::Off
                             : std::strcmp(v, "head") == 0  ? NpuMode::Head
                             : std::strcmp(v, "dense") == 0 ? NpuMode::Dense
                             : std::strcmp(v, "all") == 0   ? NpuMode::All
                             : std::strcmp(v, "split") == 0 ? NpuMode::Split : NpuMode::Auto;
        }
        else if (a == "--npu-layers") { int n = 0; if (take() && parse_int(v, n) && n >= 0) o.npu_layers = n; }
        else if (a == "--npu-mtp") { o.npu_mtp = true; }
        else if (a == "--gpu" || a == "--backend") { if (take()) o.gpu = gpu_backend_from(v); }
        else if (a == "--bench-pp" || a == "--bench-tg") {
            const bool pp = a == "--bench-pp";
            o.bench_pp = o.bench_pp || pp;
            o.bench_tg = o.bench_tg || !pp;
            int n = 0;
            if (i + 1 < argc && argv[i + 1][0] != '-' && parse_int(argv[i + 1], n) && n > 0) { o.bench_n = n; ++i; }
        }
        else if (a == "--help" || a == "-h") { print_help(); std::exit(0); }
        else if (is_ignored_value_flag(a)) { take(); }               // accepted, unused
        else if (!a.empty() && a[0] != '-' && o.model.empty()) { o.model = a; }
        // any other flag: ignored on purpose
    }
    if (o.n_ctx <= 0) o.n_ctx = kDefaultCtx;
    if (o.n_batch <= 0) o.n_batch = kDefaultBatch;
    if (o.n_batch > o.n_ctx) o.n_batch = o.n_ctx;
    // STRATA_VK_GPU lets a sweep pick the backend without touching the app's command line.
    if (o.gpu == GpuBackend::Auto) {
        if (const char * e = std::getenv("STRATA_VK_GPU")) o.gpu = gpu_backend_from(e);
    }
    return o;
}

} // namespace strata







