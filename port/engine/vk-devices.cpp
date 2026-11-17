// vk-devices.cpp - device enumeration, the measured Adreno tuning, and the Hexagon NPU plan.
//
// Two Adreno drivers are reachable on this machine (Vulkan and OpenCL) plus the Hexagon HTP.  The engine names exactly
// one graphics backend in `llama_model_params::devices`; naming more than one makes llama.cpp split layers between two
// drivers for the same GPU, which is slower and, on this part, wrong.
//
// The NPU half follows port/docs/NPU-FEASIBILITY.md: the experts stay on the Adreno (TQ2_T is not an HTP type, and
// MUL_MAT_ID over 512 experts is the decode budget), and the dense tensors move to the HTP through
// `llama_model_params::tensor_buft_overrides` - the same mechanism as llama.cpp's `-ot pattern=HTP0`.
#include "vk-devices.h"

#include "gguf.h"

#include <cstdlib>
#include <cstring>
#include <string>

#if defined(_WIN32)
#define NOMINMAX            // windows.h defines min/max as macros, which breaks std::min/std::max
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace strata {

bool f16acc_from_policy = false;
namespace {

struct TuningKnob {
    const char * name;
    const char * why;
};

// Measured on the X2-90 (port/docs/ADRENO.md, interleaved rounds, medians):
//  - async submission returns different answers for the same prompt on this driver, so it stays off;
//  - the bf16 shaders are slower than the f16 fallback here;
//  - the fp16 shader variants cost prefill a factor of four on any model whose prefill runs on the int-dot mat-mat path
//    (q5_0-heavy tiny-qwen: 729 -> 2875 tok/s at 512 tokens, decode unchanged at 99.9%, and the fp32 accumulate is the
//    more accurate of the two).  On a trellis expert pack it measures neutral-to-slightly-negative, so it is applied
//    only when the model actually has that shape - see model_prefill_is_dense_matmul().
//  - GGML_VK_DISABLE_MMVQ: the int8-dot matvec re-reads the quantised weights once per batch column; the fp32-dequant
//    one decodes them once per row and reuses them, so a short multi-token pass (a 2-8 token prompt suffix) is 21-38%
//    faster with it off.  One token is unaffected (this driver already takes the fp32-dequant path there), and prefill
//    uses the mat-mat tiles either way.
// GGML_VK_FORCE_MMVQ and GGML_VK_DISABLE_FUSION are deliberately absent: both measure neutral inside the noise, so the
// backend chooses.
const TuningKnob kVulkanTuning[] = {
    { "GGML_VK_DISABLE_ASYNC",    "async submission returns different answers for the same prompt" },
    { "GGML_VK_DISABLE_BFLOAT16", "the bf16 shaders are slower than the f16 fallback here" },
    { "GGML_VK_DISABLE_MMVQ",     "the int8 matvec re-reads the weights per batch column; the fp32 one does not" },
};

const TuningKnob kIntDotTuning[] = {
    { "GGML_VK_DISABLE_F16", "the fp16 shader variants cost the int-dot matmul a factor of four on this driver" },
};

void set_env_once(const char * name, const char * value) {
    if (std::getenv(name) != nullptr) return;   // a user's own value always wins
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 0);
#endif
}

const char * backend_of(ggml_backend_dev_t dev) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    return reg != nullptr ? ggml_backend_reg_name(reg) : "";
}

bool is_npu_name(const char * name) {
    // ggml-hexagon names its devices HTP0, HTP1, ...; nothing else does.
    return name != nullptr && std::strncmp(name, "HTP", 3) == 0;
}

bool looks_like_npu(ggml_backend_dev_t dev) {
    if (is_npu_name(ggml_backend_dev_name(dev))) return true;
    const char * desc = ggml_backend_dev_description(dev);
    return desc != nullptr && (std::strstr(desc, "Hexagon") != nullptr || std::strstr(desc, "HTP") != nullptr);
}

// The tensor-name regexes a placement mode moves, null-terminated.  The experts are deliberately absent from the dense
// set: `ffn_*_exps` and `ffn_*_chexps` keep their place on the GPU, because TQ2_T is not an HTP type.
const char * const kAllPatterns[] = {
    ".*", nullptr,
};

// The output head alone.  One tensor, one dispatch per token, no per-layer boundary, and for a 248k-vocab model the
// largest single batched matmul in the graph - the shape the HTP is 8-12x faster at.
const char * const kHeadPatterns[] = {
    "^output\\.", nullptr,
};

const char * const kDensePatterns[] = {
    "(^|\\.)output\\.",              // the output head
    "\\.ffn_(up|down|gate)\\.",      // the dense FFN projections
    "\\.ffn_(up|down|gate)_shexp",   // the shared expert of a mixture-of-experts block
    "\\.ffn_gate_inp\\.",            // the router
    "\\.attn_(q|k|v|qkv|output|out|gate)\\.", // the attention projections, including qwen4exp's attn_gate
    nullptr,
};

bool has_backend(GpuBackend backend) {
    const char * want = gpu_backend_name(backend);
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type t = ggml_backend_dev_type(d);
        if (t != GGML_BACKEND_DEVICE_TYPE_GPU && t != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        if (looks_like_npu(d)) continue;
        if (std::strcmp(backend_of(d), want) == 0) return true;
    }
    return false;
}

} // namespace

const char * gpu_backend_name(GpuBackend b) {
    switch (b) {
        case GpuBackend::Vulkan: return "Vulkan";
        case GpuBackend::OpenCL: return "OpenCL";
        default:                 return "CPU";
    }
}

GpuBackend gpu_backend_from(const std::string & name) {
    if (name == "vulkan" || name == "vk") return GpuBackend::Vulkan;
    if (name == "opencl" || name == "cl") return GpuBackend::OpenCL;
    if (name == "cpu")                   return GpuBackend::CPU;
    return GpuBackend::Auto;
}

std::string exe_dir() {
    char buf[4096] = {};
#if defined(_WIN32)
    const unsigned long n = GetModuleFileNameA(nullptr, buf, sizeof(buf));
    const std::string path(buf, n > 0 && n < sizeof(buf) ? n : 0);
#else
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf));
    const std::string path(buf, n > 0 ? (size_t) n : 0);
#endif
    const size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? std::string(".") : path.substr(0, slash);
}

SkelDirGuard::SkelDirGuard(bool enable) {
    if (!enable) return;
    const std::string dir = exe_dir();
    char buf[4096] = {};
#if defined(_WIN32)
    const unsigned long n = GetCurrentDirectoryA(sizeof(buf), buf);
#else
    if (getcwd(buf, sizeof(buf)) == nullptr) return;
    const size_t n = std::strlen(buf);
#endif
    if (n == 0 || n >= sizeof(buf)) return;
    saved_.assign(buf, n);
    if (saved_ == dir) return;                       // already there
#if defined(_WIN32)
    if (!SetCurrentDirectoryA(dir.c_str())) return;
#else
    if (chdir(dir.c_str()) != 0) return;
#endif
    active_ = true;
    logf("[strata-vk] NPU: running the load from %s, where FastRPC finds the HTP skel\n", dir.c_str());
}

SkelDirGuard::~SkelDirGuard() {
    if (!active_) return;
#if defined(_WIN32)
    SetCurrentDirectoryA(saved_.c_str());
#else
    if (chdir(saved_.c_str()) != 0) { /* nothing left to do about it */ }
#endif
}

GpuBackend resolve_backend(GpuBackend want) {
    if (want != GpuBackend::Auto) return want;
    // Vulkan first: on this driver it is the backend with the measured kernel work behind it (the trellis TQ2_T path
    // only exists there), so it is the one that can run every model in the catalogue.
    if (has_backend(GpuBackend::Vulkan)) return GpuBackend::Vulkan;
    if (has_backend(GpuBackend::OpenCL)) return GpuBackend::OpenCL;
    return GpuBackend::CPU;
}

const char * dev_kind(enum ggml_backend_dev_type type) {    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:   return "CPU";
        case GGML_BACKEND_DEVICE_TYPE_GPU:   return "GPU";
        case GGML_BACKEND_DEVICE_TYPE_IGPU:  return "iGPU";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "accelerator";
        default:                             return "other";
    }
}

// The per-model shader route, in one place. Every rule here is measured in ADRENO.md; keeping them together is what stops
// one from being applied twice or forgotten at a call site, which happened while they lived in the tuning call itself.
struct ModelPolicy {
    bool fp32_shaders = false;   // dense packs: the fp32 shaders are up to 2.2x faster on the int-dot mat-mat
    bool int_dot_off  = false;   // q2_0-only packs: their dp4a fp16 shader crashes this driver on a multi-token prefill
    // The f16-accumulate cooperative matrix was on everywhere on the strength of "identical token streams".  That is not
    // true on this driver: against the CPU backend the logits correlate 0.02 and the top token changes (PERFORMANCE.md
    // 20.50).  Off by default; STRATA_VK_F16ACC=1 still selects it for a shape-only benchmark.
    bool coop_f16acc  = false;
};

static ModelPolicy decide_policy(bool dense_model, bool q20_only) {
    ModelPolicy p;
    p.fp32_shaders = dense_model;
    p.int_dot_off  = q20_only && !p.fp32_shaders;
    return p;
}

void apply_adreno_tuning(GpuBackend backend, bool dense_model, bool q20_only, bool lowbit_pack, int shaders) {
    // STRATA_VK_ADRENO_TUNING=0 runs every knob at the backend's own default, which is what a sweep needs to measure the
    // defaults against the tuning.
    if (const char * off = std::getenv("STRATA_VK_ADRENO_TUNING"); off != nullptr && std::strcmp(off, "0") == 0) {
        logf("[strata-vk] STRATA_VK_ADRENO_TUNING=0: running without the measured Adreno defaults\n");
        return;
    }
    if (backend != GpuBackend::Vulkan && backend != GpuBackend::Auto) return;
    for (const TuningKnob & k : kVulkanTuning) set_env_once(k.name, "1");

    ModelPolicy policy = decide_policy(dense_model, q20_only);
    // The shader family is a real trade, so it is a first-class choice: f16 shaders are the better decode, fp32 shaders
    // the better prefill on this device (Gyro-S 10.7 against 5.5 decode, 19.3 against 46.6 prefill, PERFORMANCE.md §1).
    // `--shader-family` wins; STRATA_VK_F16=on|off stays for an A/B without touching the command line.
    if (shaders != 0) {
        policy.fp32_shaders = shaders == 2;
        policy.int_dot_off  = q20_only && !policy.fp32_shaders;
        logf("[strata-vk] --shader-family %s\n", shaders == 2 ? "fp32" : "f16");
    } else if (const char * f = std::getenv("STRATA_VK_F16")) {
        policy.fp32_shaders = std::strcmp(f, "on") != 0;
        policy.int_dot_off  = q20_only && !policy.fp32_shaders;
    }
    (void) lowbit_pack;   // kept for the call site's sake: the cooperative path is the default for every pack

    if (policy.fp32_shaders) {
        for (const TuningKnob & k : kIntDotTuning) set_env_once(k.name, "1");
    }
    if (policy.int_dot_off) {
        // This disables dp4a for the whole device, decode included, for a crash that only happens on a multi-token prefill: a
        // q2_0-only pack pays it on every matvec of every token.  Kept because the alternative is a crash, but said out loud.
        logf("[strata-vk] int-dot off for the whole device: q2_0 prefill crashes this driver, decode pays for it\n");
        set_env_once("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT", "1");
    }
    // Its kernel accumulates in f16, which on this driver does not just fail the strict element-wise oracle: it returns a
    // different distribution (PERFORMANCE.md 20.50).  Off by default, and an explicit STRATA_VK_F16ACC=1 still reaches it
    // for a shape-only benchmark.
    if (policy.coop_f16acc) {
        f16acc_from_policy = getenv("STRATA_VK_F16ACC") == nullptr;
        set_env_once("STRATA_VK_F16ACC", "1");
    }

    // Report the mode in force, not the constant: host-resident experts and an explicit STRATA_VK_F16ACC both move it, and
    // a log that says otherwise mislabels every A/B recorded afterwards.  The backend reads the variable the same way
    // (unset or "0" is off), so this is the backend's answer rather than a restatement of the policy.
    const char * f16acc_env = getenv("STRATA_VK_F16ACC");
    const bool   f16acc_on  = f16acc_env != nullptr && std::strcmp(f16acc_env, "0") != 0;
    logf("[strata-vk] policy: fp32 shaders %s, int-dot %s, cooperative matrix %s\n",
         policy.fp32_shaders ? "on" : "off", policy.int_dot_off ? "off" : "on",
         f16acc_on ? "f16-accumulate" : "off");
}

// Everything the engine decides from the model header, in one metadata pass: whether the prefill is a dense mat-mul
// (which drives the fp16 shader choice) and how many blocks the NPU layer split divides.  Reading it twice would just be
// the same cost twice.
// Per-core nominal clock from the registry: the class with the higher figure is the big-core class on this SoC.  Measured
// with a pinned single-thread loop: the 3437 MHz logical CPUs run 1.7-2.2x the 3072 MHz ones, so high clock == big core here.
static std::vector<int> core_clocks(int total) {
    std::vector<int> mhz((size_t) total, 0);
#ifdef _WIN32
    for (int i = 0; i < total; ++i) {
        char path[128];
        std::snprintf(path, sizeof(path), "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\%d", i);
        DWORD value = 0, size = sizeof(value);
        if (RegGetValueA(HKEY_LOCAL_MACHINE, path, "~MHz", RRF_RT_REG_DWORD, nullptr, &value, &size) == ERROR_SUCCESS) {
            mhz[(size_t) i] = (int) value;
        }
    }
#else
    (void) total;
#endif
    return mhz;
}

AffinityPlan plan_affinity(int perf_cap) {
    AffinityPlan plan;
    const int total = (int) std::thread::hardware_concurrency();
    if (total <= 0) return plan;
    perf_cap = std::max(1, std::min(100, perf_cap));

    const std::vector<int> mhz = core_clocks(total);
    const int top = *std::max_element(mhz.begin(), mhz.end());
    std::vector<int> bigs, smalls;
    for (int i = 0; i < total; ++i) {
        if (top > 0 && mhz[(size_t) i] >= top) { bigs.push_back(i); } else { smalls.push_back(i); }
    }
    if (bigs.empty() || smalls.empty()) {            // topology unknown: a plain share of the cores, no pinning
        plan.n_threads = std::max(1, total * perf_cap / 100);
        return plan;
    }

    if (perf_cap >= 100) {                            // no reservation: every core is fair game
        for (int i = 0; i < total; ++i) plan.cpus.push_back(i);
        plan.n_threads = total;
        return plan;
    }
    const int share   = (total * (100 - perf_cap) + 99) / 100;   // ceil, without <cmath>
    const int reserve = std::max(share, 3);                     // never fewer than one big + two small
    std::vector<char> taken((size_t) total, 0);
    int need = reserve;
    int nb = 0, ns = 0;
    for (int i = 0; i < (int) bigs.size() && need > 0 && nb < 1; ++i)   { taken[(size_t) bigs[(size_t) i]] = 1; --need; ++nb; }
    for (int i = 0; i < (int) smalls.size() && need > 0 && ns < 2; ++i) { taken[(size_t) smalls[(size_t) i]] = 1; --need; ++ns; }
    for (int i = 0; i < (int) bigs.size() && need > 0; ++i) {
        if (!taken[(size_t) bigs[(size_t) i]]) { taken[(size_t) bigs[(size_t) i]] = 1; --need; }
    }
    for (int i = 0; i < (int) smalls.size() && need > 0; ++i) {
        if (!taken[(size_t) smalls[(size_t) i]]) { taken[(size_t) smalls[(size_t) i]] = 1; --need; }
    }

    for (int i = 0; i < total; ++i) {
        if (!taken[(size_t) i]) plan.cpus.push_back(i);
    }
    plan.n_threads = (int) plan.cpus.size();
    if (plan.n_threads < 1) { plan.n_threads = 1; plan.cpus.assign(1, 0); }
    return plan;
}

// The pool the CPU backend runs the graph with. It is the big-core class alone, minus one core, and both halves are measured:
// every node of every decode step opens and closes a barrier, so a heterogeneous pool pays its slowest core on each of them, and
// the six 3072 MHz cores cost more at the barrier than they return in work; and with a worker on every core of the class the
// engine's own main thread and the OS contend with the pool and the barrier skews again (12 workers 10.73 tok/s, spread 10%;
// 11 workers 11.65 tok/s, spread 2%).  `want_threads` is --threads, 0 meaning "the class minus one"; the result is clamped to
// the mask either way. STRATA_CPU_MASK (a hex affinity mask) and STRATA_CPU_THREADS override both, for measurement.
AffinityPlan plan_cpu_affinity(int want_threads) {
    AffinityPlan plan;
    const int total = (int) std::thread::hardware_concurrency();
    if (total <= 0) return plan;

    if (const char * env = std::getenv("STRATA_CPU_MASK")) {
        const unsigned long long mask = std::strtoull(env, nullptr, 16);
        for (int i = 0; i < total && i < 64; ++i) {
            if (mask & (1ull << i)) plan.cpus.push_back(i);
        }
    }
    if (plan.cpus.empty()) {
        const std::vector<int> mhz = core_clocks(total);
        const int top = *std::max_element(mhz.begin(), mhz.end());
        for (int i = 0; i < total; ++i) {
            if (top > 0 && mhz[(size_t) i] >= top) plan.cpus.push_back(i);
        }
        if (plan.cpus.empty() || (int) plan.cpus.size() == total) {   // topology unknown: no pinning, every core
            plan.cpus.clear();
            for (int i = 0; i < total; ++i) plan.cpus.push_back(i);
        }
    }

    // Asking for more threads than the class has means asking for the little cores as well, which is the old behaviour and still
    // the right one for a CPU prefill; the default (want_threads == 0) stays inside the class.
    if (want_threads > (int) plan.cpus.size()) {
        plan.cpus.clear();
        for (int i = 0; i < total; ++i) plan.cpus.push_back(i);
    }

    int n = want_threads > 0 ? want_threads : (int) plan.cpus.size() - 1;
    if (const char * env = std::getenv("STRATA_CPU_THREADS")) n = std::atoi(env);
    plan.n_threads = std::max(1, std::min(n, (int) plan.cpus.size()));
    return plan;
}

ModelFacts read_model_facts(const std::string & path) {
    ModelFacts f;
    gguf_init_params gp = { /*no_alloc*/ true, /*ctx*/ nullptr };
    ggml_context * meta = nullptr;
    gp.ctx = &meta;
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (g == nullptr) {
        logf("[strata-vk] cannot read the model header (%s): keeping the backend's own shader choice\n", path.c_str());
        return f;
    }
    f.readable = true;
    char arch[128] = {};
    if (const int64_t id = gguf_find_key(g, "general.architecture"); id >= 0 && gguf_get_val_str(g, id) != nullptr) {
        std::strncpy(arch, gguf_get_val_str(g, id), sizeof(arch) - 1);
    }
    const std::string prefix = arch[0] != '\0' ? arch : "llama";
    if (const int64_t id = gguf_find_key(g, (prefix + ".block_count").c_str()); id >= 0) {
        f.block_count = (int) gguf_get_val_u32(g, id);
    }
    const int64_t n = gguf_get_n_tensors(g);
    for (int64_t i = 0; i < n; ++i) {
        const char * name = gguf_get_tensor_name(g, i);
        const size_t bytes = gguf_get_tensor_size(g, i);
        if (name != nullptr && std::strstr(name, "_exps") != nullptr) {
            ++f.experts;
            f.expert_bytes += bytes;
        } else if (name != nullptr && std::strstr(name, "per_layer_token_embd") != nullptr) {
            // read from the file per token, not resident
            f.disk_bytes += bytes;
        } else {
            (void) bytes;   // dense tensors need no accounting: nothing reads their total
        }
        switch (gguf_get_tensor_type(g, i)) {
            case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1:
            case GGML_TYPE_Q8_0: case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K:
                ++f.int_dot;
                break;
            case GGML_TYPE_Q2_0:
                ++f.int_dot;
                ++f.low2;
                break;
            case GGML_TYPE_TQ2_T: case GGML_TYPE_PTQ1_0: case GGML_TYPE_TQ1_0: case GGML_TYPE_TQ2_0:
                ++f.trellis;
                break;
            default:
                break;
        }
    }
    gguf_free(g);
    if (meta != nullptr) ggml_free(meta);
    // A dense model's prefill is the int-dot mat-mat, which the fp32 shaders make four times faster.  A mixture-of-experts
    // pack runs its prefill through the grouped mat-mul instead, and there the same setting measures 1.9x slower, so it keeps
    // the fp16 shaders - trellis packs included: with the matrix cores on they measure pp430 60.2 / tg24 3.68 in fp16 against
    // 276.6 / 3.03 in fp32, and the recorded baseline is 9.3 / 3.6, so fp16 is the mode that beats both halves for them.
    f.dense_matmul = f.experts == 0;
    f.q20_only    = f.low2 > 0 && f.trellis == 0;
    f.lowbit_pack = f.low2 > 0 || f.trellis > 0;
    logf("[strata-vk] model: %d blocks, %d tensors on the int-dot mat-mul path (%d q2_0), %d trellis, %d expert stacks -> "
         "fp32 shaders %s\n", f.block_count, f.int_dot, f.low2, f.trellis, f.experts,
         f.dense_matmul ? "on (measured up to 3.9x prefill here)"
                        : (f.q20_only ? "off, int-dot off instead (q2_0)" : "left to the backend"));
    return f;
}

void configure_npu_budget(NpuMode mode) {
    // Runtime repacking hands the DSP a tiled weight layout. The host layout round-trips exactly
    // (GGML_HEXAGON_REPACK_CHECK reports 0 of 144643072 bytes differing on the tiny head), but the DSP never answers the
    // first tiled batch: the process sits in the queue wait until it is killed. The untiled path answers, so the NPU
    // turns the tiled path off; docs/NPU-FEASIBILITY.md records the finding.
    set_env_once("GGML_HEXAGON_RT_REPACK", "0");
    if (mode == NpuMode::Off || mode == NpuMode::Auto) return;    // The HTP backend reads these when it is loaded, and setting either one explicitly overrides the backend's own
    // measurement.  On this machine any explicit value tried made session creation fail with 0x80000406 while the
    // unset defaults work, so the engine does not touch them unless asked: STRATA_VK_NPU_VMEM / _MBUF (MiB).
    if (const char * v = std::getenv("STRATA_VK_NPU_VMEM")) set_env_once("GGML_HEXAGON_VMEM", v);
    if (const char * m = std::getenv("STRATA_VK_NPU_MBUF")) set_env_once("GGML_HEXAGON_MBUF", m);
}

NpuPlan plan_npu(NpuMode mode, const ModelFacts & facts, int layers) {
    NpuPlan plan;
    plan.requested = mode != NpuMode::Off;
    if (mode == NpuMode::Off) return plan;

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU) continue;
        if (!looks_like_npu(d)) continue;
        plan.dev  = d;
        plan.buft = ggml_backend_dev_buffer_type(d);
        break;
    }

    if (plan.dev == nullptr) {
        if (mode == NpuMode::Dense) {
            logf("[strata-vk] --npu dense was asked for, but this build found no Hexagon NPU: the dense tensors stay "
                 "on the shared GPU.  `--npu-probe` reports what was searched for.\n");
        }
        return plan;
    }
    if (plan.buft == nullptr) {
        logf("[strata-vk] the NPU %s has no buffer type: dense tensors stay on the shared GPU\n",
             ggml_backend_dev_name(plan.dev));
        plan.dev = nullptr;
        return plan;
    }
    if (mode == NpuMode::Auto) {
        logf("[strata-vk] NPU: %s found and reported; the dense tensors stay on the GPU (measured slower on this "
             "machine - docs/ADRENO.md).  --npu dense places them there explicitly.\n",
             ggml_backend_dev_name(plan.dev));
        return plan;
    }

    // Placement is refused by default.  On this build the HTP session opens and the first decode never returns with
    // either pack here (a q6_K head: 417 MB on the 35B-A3B, 1.0 GB on the 27B), so an explicit placement hangs the app
    // instead of being slow - docs/NPU-FEASIBILITY.md records it.  STRATA_VK_NPU_FORCE=1 tries anyway, on a machine
    // where the placement is known to complete.
    if (std::getenv("STRATA_VK_NPU_FORCE") == nullptr) {
        logf("[strata-vk] NPU: placement is off: the last measurement of it on this machine opened the DSP session and "
             "then stopped answering on the first decode.  The tensor placement would be %s.  Set "
             "STRATA_VK_NPU_FORCE=1 to try it (docs/NPU-FEASIBILITY.md).\n",
             mode == NpuMode::Head ? "the output head"
             : mode == NpuMode::Split ? "the last layers plus the head"
             : mode == NpuMode::Dense ? "the dense tensors"
             : "the whole graph");
        return plan;
    }

    if (mode == NpuMode::Split) {
        // Layer-parallel, not tensor-parallel: the last N blocks and the output head live on the HTP, everything before
        // them on the GPU.  That is one device crossing per token instead of one per layer, which is what made the
        // per-tensor placements (-33% dense, -40% whole graph) lose.
        plan.layers  = std::max(0, layers);
        plan.n_layer = facts.block_count;
        if (plan.layers == 0) {
            logf("[strata-vk] --npu split needs --npu-layers N (0 = only the head, which is --npu head)\n");
            plan.dev = nullptr;
            return plan;
        }
        logf("[strata-vk] NPU: %s takes the last %d of %d layers plus the head; the rest stay on the GPU\n",
             ggml_backend_dev_name(plan.dev), plan.layers, plan.n_layer);
        return plan;
    }

    const bool all = mode == NpuMode::All;
    const char * const * table = all ? kAllPatterns : mode == NpuMode::Head ? kHeadPatterns : kDensePatterns;
    for (; *table != nullptr; ++table) plan.patterns.push_back(*table);
    for (const char * p : plan.patterns) plan.overrides.push_back({ p, plan.buft });
    plan.overrides.push_back({ nullptr, nullptr });   // the loader stops at a null pattern

    const char * what = all ? "the whole graph (a batched phase only - decode pays a sync per op)"
                       : mode == NpuMode::Head ? "the output head only"
                       : "the dense tensors, the experts and the KV cache stay on the GPU";
    logf("[strata-vk] NPU: %s takes %s\n", ggml_backend_dev_name(plan.dev), what);
    return plan;
}

std::vector<ggml_backend_dev_t> gpu_devices(const NpuPlan & npu, GpuBackend backend) {
    std::vector<ggml_backend_dev_t> out;
    GGML_UNUSED(npu);
    if (backend == GpuBackend::CPU) {
        // The CPU is named explicitly too: leaving `devices` empty hands llama.cpp its default list, which is every GPU
        // device - and the HTP reports itself as one, so a "CPU only" run would open a DSP session.
        if (ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) out.push_back(cpu);
        return out;
    }
    const char * want = gpu_backend_name(backend);
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(d);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        // ggml-hexagon reports itself as a GPU, so llama.cpp would happily split layers onto the NPU.  It is a tensor
        // placement target, never a layer-split target.
        if (looks_like_npu(d)) continue;
        if (std::strcmp(backend_of(d), want) != 0) continue;
        out.push_back(d);
    }
    return out;
}

void print_devices(const NpuPlan & npu, bool to_stdout) {
    FILE * out = to_stdout ? stdout : stderr;
    const char * prefix = to_stdout ? "" : "[strata-vk] ";
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        std::fprintf(out, "%sdevice %zu: %s | %s | %s | %s\n", prefix, i,
                     ggml_backend_dev_name(d) ? ggml_backend_dev_name(d) : "?",
                     dev_kind(ggml_backend_dev_type(d)), backend_of(d),
                     ggml_backend_dev_description(d) ? ggml_backend_dev_description(d) : "");
    }
    if (!npu.requested) return;
    std::fprintf(out, "%sNPU: %s\n", prefix,
                 npu.found() ? (npu.active() ? ggml_backend_dev_name(npu.dev) : "found, not used (--npu dense to place)")
                             : "no Hexagon NPU on this machine");
}

} // namespace strata



