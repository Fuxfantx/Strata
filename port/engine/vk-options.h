// vk-options.h - the command line, a superset-compatible subset of the released engine's.
#pragma once

#include "llama.h"

#include "vk-common.h"

#include <string>
#include <vector>

namespace strata {

// Strata's tri-state flags; "Auto" is a policy the engine resolves, not a third behaviour.
enum class Tri { Off, On, Auto };

// Which compute backend carries the graph.  Vanilla Vulkan and OpenCL are two drivers for the same Adreno, and naming
// both to llama.cpp only makes it split layers between them, so the engine names exactly one (or the CPU).
enum class GpuBackend { Auto, Vulkan, OpenCL, CPU };

GpuBackend  gpu_backend_from(const std::string & name);
const char * gpu_backend_name(GpuBackend b);

// Which device runs the dense half of the graph: Off, Auto (when a Hexagon NPU is present) or Dense (insist).
enum class NpuMode {
    Off,    // never look for an NPU
    Auto,   // find it, report it, place nothing
    Head,   // only the output head: one big batched matmul, one dispatch, no per-layer boundary
    Dense,  // the dense tensor patterns (per-layer: costs a cross-device sync per layer)
    All,    // every tensor - the whole graph, which only pays for a model that fits the HTP budget
    Split,  // layer-parallel: the last --npu-layers layers (head included) on the HTP, the rest on the GPU
};

struct Options {
    std::string model;
    int  n_ctx      = kDefaultCtx;
    int  n_batch    = kDefaultBatch;
    // --batch-slots N: how many independent sequences the context may hold.  0 keeps the single-sequence engine exactly
    // as it was; N >= 1 is the concurrent-decode path, which the benchmark drives so that N = 1 and N > 1 run through the
    // same code and can be compared directly.
    int  slots      = 0;
    // --parallel N (the server's spelling is `--batch N`): N request slots in `--serve`.  Each slot owns one KV sequence
    // and they step together, one token each, in a single batch, so N requests share one copy of the weights.
    int  parallel   = 0;
    // --bench-dual N: N decode steps per context, on --dual-contexts separate llama_contexts that share one model.
    int  bench_dual     = 0;
    int  dual_contexts  = 2;
    int  threads    = 0;
    // 1-100: share of wall time the engine may spend working, so the rest of the machine stays usable.
    // Host-resident experts (llama.cpp's --cpu-moe): the expert tensors stay in host memory and the device keeps a bounded
    // cache of the ones a token actually uses. `moe_cache_mib` 0 leaves the upstream default (no cache, experts on the device).
    bool   cpu_moe    = false;
    int    n_cpu_moe  = 0;
    size_t moe_cache_mib = 0;
    size_t expert_bytes  = 0;   // filled from ModelFacts; drives the automatic expert placement
    size_t disk_bytes    = 0;   // filled from ModelFacts: the on-disk N-Gram/PLE table, not resident
    float  moe_cache_cpu_share = 0.0f;
    // Share of the model's blocks placed on the CPU backend (0 = none). Item (9b)'s layer split: a second device in
    // llama_model_params::devices, so the scheduler owns the cross-device copies.
    float split_cpu = 0.0f;
    int  perf_cap   = 100;  // opt-in only: --perf-cap N reserves N% of the machine; 100 = full speed (the default)
    // --shader-family: 0 auto (the per-model policy below), 1 f16 shaders, 2 fp32 shaders.  On this device the two are a
    // real trade: f16 is the better decode, fp32 the better prefill (PERFORMANCE.md section 1).
    int  shader_family = 0;
    int  max_new    = 0;              // --n, one-shot generation
    std::string prompt;               // --prompt: tokenised with the model's own vocab, for a run without the app
    bool serve      = false;
    bool version    = false;
    bool geometry   = false;          // --geometry: list the devices and exit
    bool detok      = false;          // --detok: read --tokens back as text and exit (no context is created)
    bool npu_probe  = false;          // --npu-probe: report what the NPU path found and exit
    bool bench_pp   = false;
    bool bench_tg   = false;
    int  bench_n    = 128;
    std::vector<llama_token> tokens;
    std::string kv  = "f16";
    bool dry_run    = false;   // print the effective policy and exit, without loading the weights
    std::string compat;               // --compat FILE: a companion pack whose tensors replace the base's
    std::string mmproj;               // --mmproj FILE: the projector (the picture reader runs in its own process)
    bool vision = false;              // --vision: the front end says a GENI line may carry pictures
    std::string mtp;                  // --mtp FILE: the nextn head that drafts (see vk-mtp.h)
    Tri  lazy       = Tri::Auto;      // the engine forwards Strata's policy, it does not choose
    std::string load_mode = "direct";  // --load-mode mmap|direct|none: measured, see vk-engine-load.cpp
    bool load_mode_set = false;        // the user named --load-mode, so a pack that does not fit must not override it
    std::string dump_logits;           // --dump-logits FILE: write the prefill's last logits (n_vocab f32)
    // Flash attention: see the measurement note in vk-engine-load.cpp.  Off by default; `--flash-attn on` is what makes a
    // quantized KV cache possible.
    Tri  flash_attn = Tri::Off;
    bool flash_attn_set = false;   // the user named --flash-attn, so the engine must not choose for them
    // Prompt-lookup speculation.  Off by default, and that is a measurement, not a policy: a draft is verified by one
    // target pass over k+1 tokens, and on this backend such a pass costs about 2.4 single-token decode steps for k=2
    // (tiny-qwen: 60.5 tok/s with 9/26 drafts kept against 94.8 without), so even a mostly-accepted draft loses.  The
    // backward history scan is no longer part of the argument - the n-gram index in vk-engine.cpp answers it in O(1) -
    // so `--lookup on` is cheap to try on a pack whose prefill is fast relative to its decode.
    Tri  lookup     = Tri::Off;
    NpuMode    npu  = NpuMode::Auto;
    int  npu_layers = 0;              // --npu split: how many of the last layers (plus the head) go to the HTP
    bool npu_mtp    = false;          // --npu-mtp: run the MTP draft head on the HTP instead of the GPU
    GpuBackend gpu  = GpuBackend::Auto;
};

Options parse_args(int argc, char ** argv);

/** COMMA- or space-separated token ids, as the protocol's GEN line and `--tokens` carry them. */
std::vector<llama_token> parse_ids(const std::string & text);

} // namespace strata


