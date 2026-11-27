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
    // not "--gpu": that one is honoured below (it picks the backend).  It sat in this list too, unreachable only because
    // the chain above matched first, which would have turned "--gpu cpu" into a silent no-op on a reorder.
    "--resident-experts", "--vram-reserve", "--pool-workers", "--hybrid-rows", "--batch-groups", "--gpus",
    // not "--vision": that one is a bare flag the front end passes (serve/server.py appends it for a config with a
    // "vision" section), and this build honours it - see the parser below.  A value-taking entry here would have eaten
    // the flag after it.
    "--defrag-thold", "--tune", "--vision-tokens", "--rope-scaling", "--rope-scale",
    "--seed", "--temp", "--top-k", "--top-p", "--min-p", "--repeat-penalty", "--presence-penalty",
    "--frequency-penalty", "--penalty-last-n", "--expert-vram", "--prefill-head", "--serve-log", "--session",
    "--session-dir", "--api-key", "--host", "--port", "--draft-window", "--dflash", "--lookup-max-t",
    "--ubatch-size", "--logits-file", "--threads-batch", "--numa", "--device",
    "--image-max-tokens", "--chat-template", "--reasoning", "--effort", "--sym", "--chat-template-file",
    "--slot", "--main-gpu", "--session-file", "--dump-logits",
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
        "  --pack DIR               a latent pack directory (Edge0 145B / 120B); --native still names its skeleton\n"
        "  --expert-cache auto|MiB  expert residency ceiling; auto derives it from the memory budget\n"
        "  --vram-reserve-mib N     headroom the residency plan leaves unused (default 700)\n"
        "  --kv-resident N          tokens of KV kept resident (default: all of them)\n"
        "  --expert-profile FILE    the pack ranking used to choose which experts stay resident\n"
"  --latent-device          keep the latent pack's expert blobs in device memory so the block reads them in place\n"
                "  --npu off|auto|head|dense|all|split  Hexagon NPU placement: none, find only, output head, dense, whole\n"
                "                           graph, or split (the last --npu-layers layers + head on the HTP)\n"
                "                           (head needs a per-buffer budget above ggml-hexagon's 1 GiB default on a\n" 
                "                            large-vocab model; see docs/NPU-FEASIBILITY.md before raising it)\n"
                "  --gpu vulkan|opencl|cpu  which compute backend runs the graph (default: vulkan when present)\n"
                "  --npu-layers N           --npu split: last N blocks plus the head on the HTP\n"
                "  --npu-mtp                run the MTP draft head on the HTP, not the GPU\n"
                "  --kv f16|q8_0|q4_0       KV cache type; a quantized one needs --flash-attn on\n"
                "  --shader-family auto|f16|fp32  Vulkan shader family; fp32 is the prefill-first one (decode pays)\n"
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
        else if (a == "--ppl") { if (take()) o.ppl = v; }
        else if (a == "--ppl-chunk") { if (take()) parse_int(v, o.ppl_chunk); }
        else if (a == "--ppl-chunks") { if (take()) parse_int(v, o.ppl_chunks); }
        else if (a == "--native" || a == "--model" || a == "--model-path" || a == "-m") { if (take()) o.model = v; }
        else if (a == "--max-context" || a == "--ctx-size" || a == "-c") { if (take()) parse_int(v, o.n_ctx); }
        // `--prefill` is Strata's prompt-chunk flag and maps to llama.cpp's batch.  `--batch` is NOT that: in Strata's
        // protocol it is the number of concurrent request slots, which this engine answers by running one at a time -
        // reading it as the chunk size would turn a 4-slot run into a 4-token prefill chunk.  It is consumed below with
        // the other flags this engine does not use.
        else if (a == "--prefill") { int n = 0; if (take() && parse_int(v, n) && n > 0) o.n_batch = n; }
        else if (a == "--batch-size" || a == "-b") { if (take()) parse_int(v, o.n_batch); }
        // The one place a slot count is honoured: it sizes the context's sequence budget and, in the benchmark, makes the
        // N sequences decode in one batch.  `--parallel`/`--batch` are the server's spelling of the same thing (Strata's
        // protocol passes `--batch N`, see serve/replicas.py) and now mean what they say: N requests are admitted at once
        // and step in one batch.
        else if (a == "--batch-slots") { int n = 0; if (take() && parse_int(v, n) && n > 0) o.slots = n; }
        else if (a == "--parallel" || a == "--batch") { int n = 0; if (take() && parse_int(v, n) && n > 1) o.parallel = n; }
        else if (a == "--bench-dual") { int n = 0; if (take() && parse_int(v, n) && n > 0) o.bench_dual = n; }
        else if (a == "--dual-contexts") { int n = 0; if (take() && parse_int(v, n) && n > 0) o.dual_contexts = n; }
        else if (a == "--perf-cap") { if (take()) parse_int(v, o.perf_cap); if (o.perf_cap < 1) o.perf_cap = 1; if (o.perf_cap > 100) o.perf_cap = 100; }
        else if (a == "--threads" || a == "-t") { if (take()) parse_int(v, o.threads); }
        else if (a == "--n" || a == "--max-new" || a == "-n") { if (take()) parse_int(v, o.max_new); }
        else if (a == "--tokens") { if (take()) o.tokens = parse_ids(v); }
        else if (a == "--prompt" || a == "-p") { if (take()) o.prompt = v; }
        else if (a == "--kv" || a == "--kv-type" || a == "--cache-type-k" || a == "--cache-type-v") { if (take()) o.kv = v; }
        else if (a == "--dry-run") { o.dry_run = true; }
        else if (a == "--mmproj") { if (take()) o.mmproj = v; }
        // `--vision` is the front end's switch for pictures, and it is passed to the ENGINE as well as to the reader:
        // the engine only needs to know that a GENI line may carry them, because the projector runs in its own process.
        else if (a == "--vision") { o.vision = true; }
        else if (a == "--mtp") { if (take()) o.mtp = v; }
        else if (a == "--lazy-mode") { if (take()) o.lazy = parse_tri(v); }
        else if (a == "--load-mode") { if (take()) { o.load_mode = v; o.load_mode_set = true; } }
        else if (a == "--dump-logits") { if (take()) o.dump_logits = v; }
        else if (a == "--split-cpu") { if (take()) o.split_cpu = (float) std::atof(v); }
    else if (a == "--split-mode") {
        if (take()) {
            if (std::strcmp(v, "layer") == 0 || std::strcmp(v, "row") == 0 || std::strcmp(v, "none") == 0) o.split_mode = v;
            else std::fprintf(stderr, "[strata-vk] --split-mode %s is not layer, row or none\n", v);
        }
    }
    else if (a == "--tensor-split") {
        // Shares for the devices, in the order they are named: the graphics device first, then the CPU.  A trailing
        // extra share is how the released engine spells "the CPU gets the rest", and llama.cpp normalises anyway.
        if (take()) {
            o.tensor_split.clear();
            std::string cur;
            for (const char * c = v; ; ++c) {
                if (*c == ',' || *c == '\0') {
                    if (!cur.empty()) o.tensor_split.push_back((float) std::atof(cur.c_str()));
                    cur.clear();
                    if (*c == '\0') break;
                } else {
                    cur.push_back(*c);
                }
            }
        }
    }
        else if (a == "--compat") { if (take()) o.compat = v; }
        else if (a == "--pack") { if (take()) o.pack = v; }
        else if (a == "--expert-profile") { if (take()) o.expert_profile = v; }
        else if (a == "--latent-device") { o.latent_device = true; }
        else if (a == "--expert-cache") {
            if (take()) {
                if (std::strcmp(v, "auto") == 0) { o.expert_cache_mib = 0; }
                else { o.expert_cache_mib = (size_t) std::strtoull(v, nullptr, 10); }
            }
        }
        else if (a == "--vram-reserve-mib") { if (take()) o.vram_reserve_mib = (size_t) std::strtoull(v, nullptr, 10); }
        else if (a == "--kv-resident") { if (take()) o.kv_resident = (size_t) std::strtoull(v, nullptr, 10); }
        else if (a == "--ple-io") { if (take()) o.ple_io = v; }
        else if (a == "--pcie-frac") { if (take()) o.pcie_frac = std::atof(v); }
        else if (a == "--pcie-mode") { if (take()) o.pcie_mode = v; }
        else if (a == "--pool-workers") { if (take()) o.pool_workers = std::atoi(v); }
    else if (a == "--cpu-moe") { o.cpu_moe = true; }
        else if (a == "--n-cpu-moe") { if (take()) parse_int(v, o.n_cpu_moe); }
        else if (a == "--moe-cache-size") { int mib = 0; if (take() && parse_int(v, mib) && mib >= 0) o.moe_cache_mib = (size_t) mib; }
        else if (a == "--moe-cache-cpu-share") { if (take()) o.moe_cache_cpu_share = (float) std::atof(v); }
        else if (a == "--flash-attn" || a == "-fa") { if (take()) { o.flash_attn = parse_tri(v); o.flash_attn_set = true; } }
        else if (a == "--shader-family") {
            if (take()) {
                const std::string fam = v;
                if      (fam == "f16"  || fam == "fp16") o.shader_family = 1;
                else if (fam == "fp32" || fam == "f32")  o.shader_family = 2;
                else if (fam == "auto" || fam == "default") o.shader_family = 0;
                else { std::fprintf(stderr, "[strata-vk] --shader-family takes auto, f16 or fp32, not '%s'\n", fam.c_str()); std::exit(2); }
            }
        }
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
        else if (a == "--bench-ctx") {
        if (i + 1 < argc && parse_int(argv[i + 1], o.bench_ctx) && o.bench_ctx > 0) { ++i; }
    }
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







