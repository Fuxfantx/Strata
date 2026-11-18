// vk-engine-load.cpp - the load path: devices, the model, the context.
//
// Split from vk-engine.cpp so that the one decision that can strand a user - placing tensors on the NPU - is in one
// place with its fallback: a model that the NPU cannot take is loaded again without it.
#include "vk-engine.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX     // windows.h's min/max macros would break every std::min/std::max below
#endif
#include <windows.h>
#endif

#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <thread>

namespace strata {

// set when --dry-run printed the policy, so a null model is a success and not a load failure
bool dry_run_finished = false;

// The engine publishes the choices that depend on free memory at load time to the backend libraries through the
// environment.  A value the user set always wins, so this only fills in a default.
static void set_env_default(const char * name, const char * value) {
    if (std::getenv(name) != nullptr) return;
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 0);
#endif
}

// STRATA_VK_EVAL=<substring> dumps a checksum per graph node whose name matches, which is how the CUDA/Vulkan and the CPU
// paths are compared node by node (see PORT-STATE.md, the qwen4exp defect).
struct EvalDumpState {
    std::string filter;
};

static bool eval_dump_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (EvalDumpState *) user_data;
    if (ask) {
        // Only a matching node needs its own submit and synchronisation; answering true for every node turned one decode graph
        // into 2824 submits with a device drain each, which cost more than the dump was worth.
        return st->filter.empty() || std::strstr(t->name, st->filter.c_str()) != nullptr;
    }
    if (!st->filter.empty() && std::strstr(t->name, st->filter.c_str()) == nullptr) {
        return true;   // false would make the scheduler abort the rest of the split
    }
    const size_t nbytes = ggml_nbytes(t);
    std::vector<uint8_t> raw(nbytes);
    ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
    double sum = 0.0, abs_sum = 0.0, lo = 0.0, hi = 0.0;
    if (t->type == GGML_TYPE_F32) {
        const float * f = (const float *) raw.data();
        const size_t n = nbytes / sizeof(float);
        lo = hi = n ? (double) f[0] : 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double v = f[i];
            sum += v;
            abs_sum += v < 0 ? -v : v;
            lo = v < lo ? v : lo;
            hi = v > hi ? v : hi;
        }
    } else {
        for (size_t i = 0; i < nbytes; ++i) {
            sum += raw[i];
        }
        abs_sum = sum;
    }
    fprintf(stderr, "[eval] %-10s %-12s n=%lld abs=%.9g\n", t->name, ggml_op_name(t->op),
            (long long) ggml_nelements(t), abs_sum);
    if (t->op == GGML_OP_MUL_MAT && t->src[0] != nullptr) {
        const ggml_tensor * s0 = t->src[0];
        fprintf(stderr, "[evsrc] %-10s wtype=%s k=%lld buf=%s\n", t->name, ggml_type_name(s0->type),
                (long long) s0->ne[0], s0->buffer != nullptr ? ggml_backend_buffer_name(s0->buffer) : "-");
    }
    fflush(stderr);
    return true;
}


namespace {

enum llama_lazy_mode lazy_mode(Tri t) {
    switch (t) {
        case Tri::Off: return LLAMA_LAZY_MODE_OFF;
        case Tri::On:  return LLAMA_LAZY_MODE_ON;
        default:       return LLAMA_LAZY_MODE_AUTO;
    }
}

enum llama_flash_attn_type flash_attn(Tri t) {
    switch (t) {
        case Tri::Off: return LLAMA_FLASH_ATTN_TYPE_DISABLED;
        case Tri::On:  return LLAMA_FLASH_ATTN_TYPE_ENABLED;
        default:       return LLAMA_FLASH_ATTN_TYPE_AUTO;
    }
}

void read_arch(const llama_model * model, std::string & out) {
    char buf[128] = {};
    if (llama_model_meta_val_str(model, "general.architecture", buf, sizeof(buf)) > 0 && buf[0] != '\0') {
        out = buf;
        return;
    }
    char desc[256] = {};
    llama_model_desc(model, desc, sizeof(desc));
    out = desc[0] != '\0' ? desc : "unknown";
}

} // namespace

// Exactly one graphics backend is named to llama.cpp.  `tensor_buft_overrides` can only place a tensor in a buffer type
// a scheduler backend owns, and the NPU must not be given *layers*: llama.cpp treats a device that reports 0/0 memory as
// "use host memory instead" and hands it half the model.  Naming only the chosen graphics device, with an explicit
// tensor_split of 0 for the NPU, is the documented way to say "no layers here".
//
// `--npu split` is the exception, and it is deliberate: there the HTP *is* a layer-split target, listed last so it takes
// the highest-numbered blocks plus the output head.  That keeps the whole crossing count at one per token (the HTP hands
// finished logits to the host) instead of the one-per-layer that the tensor placements pay and lose on.
std::vector<ggml_backend_dev_t> Engine::plan_devices(const Options & opt, const NpuPlan & npu) {
    backend_ = resolve_backend(opt.gpu);
    std::vector<ggml_backend_dev_t> devices = gpu_devices(npu, backend_);
    if (backend_ != GpuBackend::CPU && devices.empty()) {
        logf("[strata-vk] no %s device is registered in this build; falling back to the CPU\n",
             gpu_backend_name(backend_));
        backend_ = GpuBackend::CPU;
        devices = gpu_devices(npu, backend_);
    }
    split_.clear();
    if (npu.split() && backend_ != GpuBackend::CPU) {
        // llama.cpp divides n_layer+1 units (the blocks plus the output layer) by the cumulative split, and rounds each
        // unit to the device whose slot it lands in.  Weighting the HTP by the unit count it should own puts exactly
        // `layers` blocks and the head there.
        const int units = npu.n_layer > 0 ? npu.n_layer + 1 : npu.layers + 1;
        const int want  = std::min(npu.layers + 1, units);
        devices.push_back(npu.dev);
        split_.assign(devices.size(), 0.0f);
        for (size_t i = 0; i + 1 < split_.size(); ++i) split_[i] = (float) (units - want) / (float) units;
        split_.back() = (float) want / (float) units;
        logf("[strata-vk] NPU split: %d of %d units (last %d blocks + head) on %s, the rest on %s\n",
             want, units, npu.layers, ggml_backend_dev_name(npu.dev), ggml_backend_dev_name(devices.front()));
    } else if (opt.split_cpu > 0.0f && backend_ != GpuBackend::CPU) {
        // A layer split with the CPU backend: the scheduler then owns the activations crossing the boundary. Moving tensors with
        // buft_overrides instead leaves the CPU reading Vulkan buffers (Assertion !isnan(x) in ggml-cpu) - HYBRID.md.
        const ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu != nullptr) {
            const float share = std::max(0.0f, std::min(1.0f, opt.split_cpu));
            devices.push_back(cpu);
            split_.assign(devices.size(), 0.0f);
            for (size_t i = 0; i + 1 < split_.size(); ++i) split_[i] = 1.0f - share;
            split_.back() = share;
            logf("[strata-vk] layer split: %.0f%% of the blocks on the CPU backend\n", share * 100.0f);
        } else {
            logf("[strata-vk] --split-cpu: no CPU backend registered\n");
        }
    } else if (npu.active()) {
        devices.insert(devices.begin(), npu.dev);
        split_.assign(devices.size(), 0.0f);
        for (size_t i = 1; i < split_.size(); ++i) split_[i] = 1.0f / (float) (split_.size() - 1);
    }
    devices.push_back(nullptr);   // null-terminated, as llama_model_params::devices wants
    return devices;
}

llama_model * Engine::load_model(const Options & opt, const NpuPlan & npu) {
    devices_ = plan_devices(opt, npu);

    constexpr size_t kGiB = 1024ull * 1024ull * 1024ull;

    // Two numbers decide the placements below and the load mode: the bytes in the pack, and the RAM free right now.
    size_t ram_avail = 0;
#ifdef _WIN32
    {
        MEMORYSTATUSEX ms{};
        ms.dwLength = sizeof(ms);
        if (GlobalMemoryStatusEx(&ms)) ram_avail = (size_t) ms.ullAvailPhys;
    }
#endif
    // A sharded pack's tensor table only covers the first shard, so the decision uses the bytes of every shard.
    size_t pack_bytes = 0;
    {
        const std::string & p = opt.model;
        const size_t dash = p.rfind("-of-");
        const size_t dot  = p.rfind(".gguf");
        if (dash != std::string::npos && dash > 6 && dot != std::string::npos && dot > dash) {
            const std::string head = p.substr(0, dash - 6);              // strip "-00001"
            const std::string tail = p.substr(dot);                      // ".gguf"
            const std::string of   = p.substr(dash + 4, dot - dash - 4); // the shard count
            const int n_shards = std::atoi(of.c_str());
            for (int i = 1; i <= std::max(n_shards, 1); ++i) {
                char buf[32];
                std::snprintf(buf, sizeof(buf), "-%05d-of-", i);
                const std::string shard = head + buf + of + tail;
                std::error_code ec;
                const auto sz = std::filesystem::file_size(shard, ec);
                if (!ec) pack_bytes += (size_t) sz;
            }
        }
        if (pack_bytes == 0) {
            std::error_code ec;
            pack_bytes = (size_t) std::filesystem::file_size(p, ec);
        }
    }
    if (pack_bytes == 0) pack_bytes = opt.expert_bytes;
    // What has to be resident is the pack minus the tables that stay on disk (the N-Gram/PLE table), not the file size.
    const size_t ple_bytes = opt.disk_bytes;
    const size_t resident  = pack_bytes > ple_bytes ? pack_bytes - ple_bytes : pack_bytes;

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers   = backend_ == GpuBackend::CPU ? 0 : 999;   // everything the backend will take
    // Direct I/O reads the file straight into the destination buffers; mmap maps it and then copies.  Measured on this
    // machine, direct is faster for every model here because the copy dominates: Gyro-M 87.3 -> 51.5 s, the Q2 pack
    // 88.5 -> 51.2 s, 35B-A3B 8.2 -> 6.2 s, 27B 8.9 -> 5.5 s.  mmap stays available with --load-mode mmap.
    mp.load_mode      = opt.load_mode == "direct" ? LLAMA_LOAD_MODE_DIRECT_IO
                      : opt.load_mode == "none"   ? LLAMA_LOAD_MODE_NONE
                                                  : LLAMA_LOAD_MODE_MMAP;
    // Except for the packs that do not fit: direct I/O reads every byte of the file into a buffer, including the
    // N-Gram/PLE table that mmap can leave in the file, so on a pack larger than RAM it is what fills the RAM.  Measured
    // on Gyro-S (54.47 GiB pack, 15.27 GiB of it that table, 40.5 GiB free): direct 4.1 tok/s and 54.5 GiB resident,
    // mmap 6.1-6.4 tok/s and 0.6 GiB free at the end of the load, with the disk idle during decode.
    if (!opt.load_mode_set && ram_avail > 0 && pack_bytes > ram_avail) {
        mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        logf("[strata-vk] --load-mode: the pack (%.1f GiB) does not fit in RAM available now (%.1f GiB), so it is mapped "
             "rather than read: only the tensors that run become resident, and the ones marked read-on-demand stay in the "
             "file\n", (double) pack_bytes / (double) kGiB, (double) ram_avail / (double) kGiB);
    }
    // The CPU backend's repack path turns a K-quant weight into an interleaved copy that an 8-row gemv reads, and it is worth
    // a lot: on the 35B-A3B the dense matvec's op time goes 820 -> 80 ms, and the same run measures 33.8 -> 38.9 tok/s. It is
    // gated here rather than left on, because a repacked tensor is anonymous memory where a mapped one was a file page, and that
    // is the difference between a pack running and not. What the gate has to size is therefore **the part that gets copied**, not
    // the pack: only the MUL_MAT-typed dense tensors are admitted (the expert stacks are trellis, the per-layer embedding table is
    // a GET_ROWS table, and neither has a repack kernel), so the copy is pack minus experts minus the read-on-demand table.
    // Gyro-S: 54.5 GiB pack, 24.0 of experts and 26.8 of table → 3.7 GiB copied, and measured 11.80 -> 12.10 tok/s decode
    // (medians of four) and 18.01 -> 20.34 prefill. 27B and 35B-A3B already fitted whole. Q2_0 (67 GiB, 31.6 of experts) would
    // copy ~8.6 GiB and is NOT measured; treat it as the first thing to check if that pack misbehaves.
    const size_t repack_bytes = pack_bytes > (opt.expert_bytes + opt.disk_bytes)
                              ? pack_bytes - (opt.expert_bytes + opt.disk_bytes) : 0;
    set_env_default("STRATA_CPU_REPACK", (ram_avail == 0 || repack_bytes + kGiB <= ram_avail) ? "1" : "0");
    // The MTP head is a separate nextn-only GGUF, grafted into this model at load time.
    // The graft is driven by `mtp_path` being non-empty, not by `load_mtp` (llama-model-loader.cpp:809), so both have to
    // be cleared when this build cannot run the head - otherwise 1.2-2.2 GB of nextn tensors are loaded for nothing.
    const bool mtp_wanted = !opt.mtp.empty() && mtp_supported_by_build();
    mp.load_mtp       = mtp_wanted;
    mp.mtp_path       = mtp_wanted ? opt.mtp.c_str() : nullptr;
    mp.check_tensors  = false;
    mp.lazy_mode      = lazy_mode(opt.lazy);
    mp.devices               = devices_.data();
    // Host-resident experts, straight from llama.cpp's own mechanism: the same regex `common.h` uses for --cpu-moe, placed on
    // the CPU buffer type, merged with any NPU placement (the loader stops at a null pattern, so the merged list is
    // re-terminated here).
    std::vector<llama_model_tensor_buft_override> buft_overrides;
    if (npu.active()) {
        for (const auto & o : npu.overrides) {
            if (o.pattern != nullptr) buft_overrides.push_back(o);
        }
    }
    std::vector<std::string> expert_patterns;   // kept alive for the duration of the load
    // Upstream Strata's two policies (EXPERT-CACHE.md): if the experts do not fit the RAM available now with 1 GiB to spare,
    // keep them in host memory and give the device a bounded cache instead of failing or thrashing; otherwise leave the pack
    // as it is. Any explicit expert flag turns both off.
    bool   host_experts  = opt.cpu_moe || opt.n_cpu_moe > 0;
    moe_cache_mib_ = opt.moe_cache_mib;
    const bool expert_flags_set = host_experts || opt.moe_cache_mib > 0;
    if (!expert_flags_set && opt.expert_bytes > 0) {
        logf("[strata-vk] expert policy: pack %.2f GiB, experts %.2f GiB, RAM available %.2f GiB\n",
             (double) pack_bytes / (double) kGiB, (double) opt.expert_bytes / (double) kGiB,
             (double) ram_avail / (double) kGiB);
        if (ram_avail > 0 && resident + kGiB > ram_avail) {
            if (opt.expert_bytes + kGiB < ram_avail) {
                // The pack does not fit but the experts do, so they stay on the device and mmap pages the rest on demand.
                // Q2_0 (62.2 GiB pack, 31.6 GiB of experts, 40 GiB free) measured 3.3 tok/s with the host staging below
                // and 14.3 this way, because one token touches only ~0.6 GiB of experts and the page cache keeps that
                // set: a fixed 4 GiB device bank over 31.6 GiB of host experts misses often enough to spend half the wall
                // time in the loader, and a bigger bank only thrashes harder (8 GiB: 2.9 tok/s, 767% CPU).
                logf("[strata-vk] the resident set (%.1f GiB) does not fit in RAM available now (%.1f GiB), but the experts "
                     "(%.1f GiB) do: leaving them on the device and letting mmap page in what runs.\n",
                     (double) resident / (double) kGiB, (double) ram_avail / (double) kGiB,
                     (double) opt.expert_bytes / (double) kGiB);
            } else {
                host_experts  = true;
                moe_cache_mib_ = 4096;   // the measured-good size (ADRENO.md); llama.cpp splits it per device
                logf("[strata-vk] the resident set (%.1f GiB: pack %.1f GiB less %.1f GiB of on-disk N-Gram/PLE, experts %.1f GiB) does "
                     "not fit in RAM available now (%.1f GiB, 1.0 GiB kept free) and neither do the experts: keeping them in host "
                     "memory with a %zu MiB device cache.\n",
                     (double) resident / (double) kGiB, (double) pack_bytes / (double) kGiB, (double) ple_bytes / (double) kGiB,
                     (double) opt.expert_bytes / (double) kGiB, (double) ram_avail / (double) kGiB, moe_cache_mib_);
            }
        }
    }
    if (moe_cache_mib_ > 0 && opt.gpu == GpuBackend::CPU) {
        // The cache is device memory for host-resident experts. The CPU backend has no device to put it on, and llama.cpp's
        // cache throws when its device list is empty (llama-moe-cache.cpp), so `--gpu cpu` alone used to fail in the context
        // instead of running. Drop the automatic size here; the experts stay in the model, which is the normal CPU path.
        logf("[strata-vk] moe cache off: the CPU backend has no device to cache experts into\n");
        moe_cache_mib_ = 0;
        host_experts   = false;
    }
    if (host_experts) {
        // The f16-accumulate cooperative pipelines miscompute on this arch as soon as a batch carries more than a
        // handful of tokens and the expert matmuls run on the CPU: Gyro-S and Gyro-M abort in the CPU swiglu's NaN
        // check at 12 tokens (8 is fine), while the same model on device experts is clean at 6, 12 and 32, and the
        // vanilla driver path - which never enables f16acc - runs the same host-expert prompt to the end.
        // STRATA_VK_F16ACC=0 makes that 12-token run pass, so the host-expert path turns it off; the backend re-reads
        // the variable on every call, so setting it here takes effect for the graph. An explicit value still wins.
        if (f16acc_from_policy) {
#if defined(_WIN32)
            _putenv_s("STRATA_VK_F16ACC", "0");
#else
            setenv("STRATA_VK_F16ACC", "0", 1);
#endif
            f16acc_from_policy = false;
            logf("[strata-vk] host experts: STRATA_VK_F16ACC=0 (f16-accumulate miscomputes multi-token with CPU experts)\n");
        }
        const ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        ggml_backend_buffer_type_t cpu_buft = cpu_dev != nullptr ? ggml_backend_dev_buffer_type(cpu_dev) : nullptr;
        if (cpu_buft == nullptr) {
            logf("[strata-vk] --cpu-moe: no CPU backend to place the experts on\n");
        } else if (opt.cpu_moe || opt.n_cpu_moe <= 0) {
            expert_patterns.emplace_back("\\.ffn_(up|down|gate|gate_up)_(ch|)exps");
        } else {
            for (int i = 0; i < opt.n_cpu_moe; ++i) {
                expert_patterns.emplace_back("blk\\." + std::to_string(i) + "\\.ffn_(up|down|gate|gate_up)_(ch|)exps");
            }
        }
        for (const std::string & p : expert_patterns) buft_overrides.push_back({ p.c_str(), cpu_buft });
        const std::string where = (opt.cpu_moe || opt.n_cpu_moe <= 0) ? std::string("all layers")
                                                                    : ("first " + std::to_string(opt.n_cpu_moe) + " layers");
        logf("[strata-vk] experts in host memory: %s (device cache %zu MiB, cpu share %.2f)\n",
             where.c_str(), moe_cache_mib_, (double) opt.moe_cache_cpu_share);
    }
    if (!buft_overrides.empty()) buft_overrides.push_back({ nullptr, nullptr });
    mp.tensor_buft_overrides = buft_overrides.empty() ? nullptr : buft_overrides.data();
    mp.tensor_split          = split_.empty() ? nullptr : split_.data();

    // An HTP session opens while the tensors are being allocated, and FastRPC resolves the skel by bare name against
    // the current directory, so the load has to run from the engine's own directory.  Paths are made absolute first for
    // the same reason: they would otherwise be resolved against a directory that is not the caller's.
    const bool npu_load = npu.active() || npu.split();
    const std::string model_path = npu_load ? std::filesystem::absolute(opt.model).string() : opt.model;

    logf("[strata-vk] loading %s ...\n", model_path.c_str());
    load_t0_ = Clock::now();
    SkelDirGuard skel_dir(npu_load);
    if (opt.dry_run) {
        // What the flags resolved to, in one block: placement, load mode, the tuning switches and the warmups.  The
        // policy combinations here have cost this project real rounds (host experts x f16acc, --cpu-moe leaving the MoE
        // cache at 0), and none of them is visible without reading the code or grepping the log.
        const char * f16acc = std::getenv("STRATA_VK_F16ACC");
        const char * rm_mult = std::getenv("STRATA_VK_TQ_RM_MULT");
        const char * rm_mult_id = std::getenv("STRATA_VK_TQ_RM_MULT_ID");
        logf("[strata-vk] dry run\n");
        logf("  backend        : %s\n", backend_ == GpuBackend::CPU ? "cpu" : "vulkan");
        logf("  pack           : %.2f GiB (experts %.2f, on-disk N-Gram/PLE %.2f), RAM available %.2f GiB\n",
             (double) pack_bytes / (double) kGiB, (double) opt.expert_bytes / (double) kGiB,
             (double) ple_bytes / (double) kGiB, (double) ram_avail / (double) kGiB);
        logf("  load mode      : %s%s\n",
             mp.load_mode == LLAMA_LOAD_MODE_DIRECT_IO ? "direct" : mp.load_mode == LLAMA_LOAD_MODE_NONE ? "none" : "mmap",
             opt.load_mode_set ? " (explicit)" : " (auto)");
        logf("  expert place   : %s\n", host_experts ? "host memory + device cache" : "device (mmap on demand if needed)");
        if (host_experts) {
            logf("  moe cache      : %zu MiB, cpu share %.2f\n", moe_cache_mib_, (double) opt.moe_cache_cpu_share);
        }
        logf("  f16 accumulate : %s\n", f16acc == nullptr ? "default (on where supported)" : f16acc);
        logf("  tq rows/wg     : dense %sx, id %sx of the standard row count\n",
             rm_mult == nullptr ? "20" : rm_mult, rm_mult_id == nullptr ? "20" : rm_mult_id);
        logf("  ctx / batch    : %d / %d, kv %s, threads %d, flash-attn %s\n", opt.n_ctx, opt.n_batch, opt.kv.c_str(),
             opt.threads, opt.flash_attn == Tri::On ? "on" : opt.flash_attn == Tri::Off ? "off" : "auto");
        logf("  warmup         : %s (decode %s, prefill %s)\n",
             std::getenv("STRATA_VK_NO_WARMUP") != nullptr ? "off" : "on",
             std::getenv("STRATA_VK_NO_WARMUP") != nullptr ? "no" : "yes",
             std::getenv("STRATA_VK_NO_WARMUP") != nullptr ? "no" : "yes");
        logf("  lookup / mtp   : %s / %s\n", opt.lookup == Tri::Off ? "off" : "on", mtp_wanted ? "on" : "off");
        dry_run_finished = true;
        return nullptr;
    }
    if (!opt.compat.empty()) {
        // A companion pack: a small GGUF, marked strata.companion=true in its own header, whose tensors replace the
        // base pack's same-named ones.  It is handed over as an extra file, which is the loader's existing multi-file
        // path, so no new API is needed (PERFORMANCE.md section 5.0.1).
        const std::string companion_path = std::filesystem::absolute(opt.compat).string();
        logf("[strata-vk] companion pack %s: its tensors replace the base pack's same-named ones\n", companion_path.c_str());
        const char * paths[2] = { model_path.c_str(), companion_path.c_str() };
        return llama_model_load_from_splits(paths, 2, mp);
    }
    return llama_model_load_from_file(model_path.c_str(), mp);
}

bool Engine::load(const Options & opt, const NpuPlan & npu) {
    opt_ = opt;
    npu_ = npu;

    llama_backend_init();
    print_devices(npu, false);

    model_ = load_model(opt, npu);
    if (model_ == nullptr && npu.active()) {
        // A placement the NPU cannot serve must not cost the user the model.  On this machine its session open fails
        // with 0x80000406 before any tensor moves, which is the case this covers.
        logf("[strata-vk] the NPU could not take the tensors it was given; loading on the %s alone.  `--npu-probe` "
             "reports the device and docs/NPU-FEASIBILITY.md records what stops it.\n", gpu_backend_name(backend_));
        npu_ = NpuPlan{};
        npu_.requested = npu.requested;
        model_ = load_model(opt, npu_);
    }
    if (model_ == nullptr) {
        logf("[strata-vk] cannot load the model: %s\n", opt.model.c_str());
        return false;
    }
    const double load_ms = ms_since(load_t0_);
    vocab_ = llama_model_get_vocab(model_);
    read_arch(model_, arch_);
    logf("[strata-vk] model loaded in %.1f s: %s, %d layers, n_embd %d, n_ctx_train %d, template %s\n",
        load_ms / 1000.0, arch_.c_str(), llama_model_n_layer(model_), llama_model_n_embd(model_),
        llama_model_n_ctx_train(model_), llama_model_chat_template(model_, nullptr) ? "yes" : "none");

    int n_ctx = opt.n_ctx;
    const int train_ctx = llama_model_n_ctx_train(model_);
    if (train_ctx > 0 && n_ctx > train_ctx) {
        logf("[strata-vk] context %d is past the model's training context %d; using %d\n", n_ctx, train_ctx, train_ctx);
        n_ctx = train_ctx;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = (uint32_t) n_ctx;
    cp.n_batch         = (uint32_t) std::min(opt.n_batch, n_ctx);
    cp.n_ubatch        = cp.n_batch;
    cp.n_seq_max       = (uint32_t) std::max(std::max(1, opt.slots), opt.parallel);
    // The four values that decide whether llama.cpp divides the context among sequences: log them, so a later
    // "n_ctx is not divisible by n_seq_max" can be read against what was actually requested instead of guessed at.
    logf("[strata-vk] context params: n_ctx %u, batch %u, slots %d, parallel %d, n_seq_max %u\n",
         cp.n_ctx, cp.n_batch, opt.slots, opt.parallel, cp.n_seq_max);
    // Keep 20% of the cores (with a one-big-two-small floor) genuinely free: the engine's threadpools are pinned to the rest
    // (plan_affinity in vk-devices.cpp), so a browser and Office always find a core.  The CPU backend is the exception: it runs
    // every node of the graph itself, so it takes the whole big-core class and leaves the little cores to the rest of the
    // machine (plan_cpu_affinity, and PERFORMANCE.md for why the little cores are not worth a barrier wait).
    const bool cpu_backend = opt.gpu == GpuBackend::CPU;
    const AffinityPlan plan = cpu_backend ? plan_cpu_affinity(opt.threads) : plan_affinity(opt.perf_cap);
    cp.n_threads       = opt.threads > 0 ? std::min(opt.threads, plan.n_threads) : plan.n_threads;
    // Two pools with two jobs.  Decode opens and closes a barrier per node, so it wants one core left for the engine's own main
    // thread and the OS (that is what plan_cpu_affinity's "class minus one" is for, PERFORMANCE.md section 8).  A batch pass does
    // not: it is throughput-bound, its matmuls are long, and the extra core is a straight 5% there (PERFORMANCE.md section 8.3),
    // so the batch pool takes every core in the mask - but never more, or the pool oversubscribes the mask it is pinned to.
    cp.n_threads_batch = cpu_backend && opt.threads <= 0
                       ? (int) plan.cpus.size()
                       : cp.n_threads;
    // The device-side cache for host-resident experts (llama_context_params, upstream's --moe-cache-size).
    if (moe_cache_mib_ > 0) {
        cp.moe_cache_size      = moe_cache_mib_ * 1024ull * 1024ull;
        cp.moe_cache_cpu_share = opt.moe_cache_cpu_share;
    }
    // Flash attention is a switch, not a default. Measured on the X2-90 (ADRENO.md): -32% decode at a short position,
    // +9% at 2 048, 4.5x prefill on an attention-bound model at 2 048 and -16% on a compute-bound one at the same length.
    // The sign depends on the position actually reached, which the engine cannot know when it creates the context, and a
    // n_ctx-based guess was measured 3.3x *worse* on decode at a 32 768-token window with a fresh conversation. So the
    // caller decides.
    // Flash attention pays for itself from ~8k context upward (ADRENO.md has the 27B and tiny measurements): prefill
    // 90.6 -> 164.1 tok/s at 4k on the 27B, 240 -> 1748 on tiny, at the cost of ~14% decode at a short position, which a
    // long prompt amortises many times over. A `--flash-attn` on the command line still decides by itself.
    Tri fa_choice = opt.flash_attn;
    if (!opt.flash_attn_set && n_ctx >= 4096 && llama_model_n_layer(model_) >= 32) fa_choice = Tri::On;
    cp.flash_attn_type = flash_attn(fa_choice);
    // The MTP draft head consumes this context's hidden state row by row, which needs the output buffer reserved at
    // creation (see vk-mtp.cpp).  Unmasked: every row of the verify pass is wanted, indexed by position.
    if (!opt.mtp.empty() && mtp_supported_by_build()) {
        cp.embeddings_nextn        = true;
        cp.embeddings_nextn_masked = false;
    }
    cp.no_perf         = false;
    cp.offload_kqv     = true;
    cp.op_offload      = true;
    // STRATA_VK_KV_UNIFIED=1 (experiment): llama_context_default_params leaves this false, while every llama.cpp tool that
    // runs more than one sequence goes through common_params, which sets it true.  16384 cells for 2 sequences is what the
    // non-unified cache reports; the question is whether it can serve a second sequence's prompt correctly.
    if (std::getenv("STRATA_VK_KV_UNIFIED") != nullptr) {
        cp.kv_unified = true;
        logf("[strata-vk] kv_unified on: the cache holds one cell pool for every sequence\n");
    }
    // llama.cpp refuses a quantised V cache without flash attention; a request for one must not cost the user the model,
    // so the engine says why and stays on f16.
    if (opt.kv == "q8_0" || opt.kv == "int8" || opt.kv == "q4_0") {
        if (cp.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
            logf("[strata-vk] a %s KV cache needs flash attention, which is off (it costs 18%% decode on this driver); "
                 "staying on f16\n", opt.kv.c_str());
        } else {
            const ggml_type t = opt.kv == "q4_0" ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0;
            cp.type_k = t;
            cp.type_v = t;
        }
    }

    if (const char * e = std::getenv("STRATA_VK_LOOKUP_NM")) {
        int n = 0, m = 0;
        if (std::sscanf(e, "%d,%d", &n, &m) == 2 && n > 0 && m > 0) { lookup_n_ = n; lookup_m_ = m; }
    }
    // Speculation has to be able to drop the suffix a rejected draft wrote.  A full-attention cache always can; a
    // recurrent or hybrid cache needs per-token snapshots, requested here, and llama.cpp clamps the request to 0 for an
    // architecture that cannot roll back - which is why the gate below reads the resolved value instead of guessing.
    // Only a recurrent or hybrid cache needs per-token snapshots to roll a rejected draft back; a full-attention cache
    // can drop a suffix on its own.  Requesting the ring unconditionally made --lookup cost ~8% on models that never use
    // it (35B-A3B: 40.5 -> 37.2 tok/s with zero drafts), so the request follows the model's kind instead of the flag.
    const bool model_needs_rollback = llama_model_is_recurrent(model_) || llama_model_is_hybrid(model_);
    // Only a drafter that can actually run earns the rollback ring: a bare --mtp that cannot load a head (vk-mtp.h:31-34), or
    // an --lookup left on Auto, otherwise pays for it and drafts nothing.  Measured at about 2 ms per token when it was paid.
    const bool mtp_runnable = !opt.mtp.empty() && mtp_supported_by_build();
    if ((opt.lookup == Tri::On || mtp_runnable) && model_needs_rollback) {
        cp.n_rs_seq = (uint32_t) std::max(lookup_m_, 4);
    }

    logf("[strata-vk] creating a %d-token context (batch %u, %d threads, kv %s) ...\n",
         n_ctx, cp.n_batch, cp.n_threads, opt.kv.c_str());
    Clock::time_point t0 = Clock::now();
    static EvalDumpState eval_state;
    if (const char * e = std::getenv("STRATA_VK_EVAL")) {
        eval_state.filter = e;
        cp.cb_eval = eval_dump_cb;
        cp.cb_eval_user_data = &eval_state;
    }
    ctx_ = llama_init_from_model(model_, cp);
    if (ctx_ == nullptr) {
        logf("[strata-vk] cannot create the context\n");
        return false;
    }
    n_ctx_ = (int) llama_n_ctx(ctx_);
    batch_ = llama_batch_init((int32_t) cp.n_batch, 0, 1);
    batch_cap_ = (int) cp.n_batch;
    // --parallel N: one KV sequence per slot, and each slot's cache is the context divided by the sequence count that
    // llama.cpp actually settled on (it rounds n_ctx down to a multiple of n_seq_max * 256).
    if (opt.parallel > 1) {
        const int seq = (int) std::max(1u, llama_n_seq_max(ctx_));
        slot_st_.resize((size_t) opt.parallel);
        slot_ctx_ = n_ctx_ / seq;
        logf("[strata-vk] %d request slots, %d tokens of cache each (batch %d)\n", opt.parallel, slot_ctx_,
             batch_cap_);
        // Opt-in on purpose: the path is correct (slot 0 matches a solo run token for token, PERFORMANCE.md 20.50) but
        // a 32-row step costs 32 single-row steps, so it buys no aggregate throughput on this backend yet.
        logf("[strata-vk] %d slots: correct but no aggregate gain yet (a 32-row step costs 32 solo steps; "
             "PERFORMANCE.md 20.50), so serve/server.py does not use it\n", opt.parallel);
    }
    // `llama_n_rs_seq` reports what was actually configured: 0 means the architecture cannot roll back, or the request
    // was clamped away, and only then does speculation become impossible.
    spec_ok_ = !model_needs_rollback || llama_n_rs_seq(ctx_) > 0;
    if (const char * e = std::getenv("STRATA_VK_LOOKUP")) {
        if (std::strcmp(e, "0") == 0) spec_ok_ = false;
    }
    if (spec_ok_ && opt.lookup != Tri::Off) {
        logf("[strata-vk] prompt lookup on: %d-gram, up to %d drafted tokens, greedy requests only (rollback ring %u)\n",
             lookup_n_, lookup_m_, llama_n_rs_seq(ctx_));
    }
    logf("[strata-vk] context ready in %.1f s (%d tokens)\n", ms_since(t0) / 1000.0, n_ctx_);

    // A mapped pack is page-cold, and the first graph pays for it: one 11 MB tensor reads at 23 MB/s the first time and
    // at cache speed afterwards, which made the first decode step 3.35 s against a 99.8 ms steady state (the same op
    // measured 96 x 4985 us, then 96 x 2.5 us). One single-token forward here moves that into the load, where a slow
    // start is expected, instead of leaving the user's first token to wait for the disk.
    if (getenv("STRATA_VK_NO_WARMUP") == nullptr) {
        const Clock::time_point tw = Clock::now();
        llama_token warm = llama_vocab_bos(vocab_);
        if (warm == LLAMA_TOKEN_NULL || warm < 0) {
            warm = 0;
        }
        // One forward, single token: it walks every layer's weights, so it faults in the whole mapped pack. A
        // multi-token forward would also warm the cooperative mat-mat pipelines, but it leaves a hybrid model's memory
        // in a state the next decode rejects (M-RoPE asserts on the position order even after llama_memory_clear), so
        // the first real prefill keeps paying for its pipelines once per process.
        batch_.n_tokens = 1;
        batch_.token[0]   = warm;
        batch_.pos[0]     = 0;
        batch_.n_seq_id[0] = 1;
        batch_.seq_id[0][0] = 0;
        batch_.logits[0]  = 1;
        if (llama_decode(ctx_, batch_) == 0) {
            llama_memory_clear(llama_get_memory(ctx_), true);
            logf("[strata-vk] warmed the decode graph in %.2f s (page faults on the mapped pack, now part of the load)\n",
                 ms_since(tw) / 1000.0);
        } else {
            logf("[strata-vk] warmup forward failed, the first token will pay for the page faults\n");
        }
        batch_.n_tokens = 0;

    }

    // The prefill graph takes different pipelines from a decode step, and the first real prompt compiles them: a six
    // token prompt spent 2.4 s on its first run after the decode warmup above. A multi-token forward in this context
    // leaves a hybrid model's memory in a state the next decode rejects (M-RoPE asserts on the position order even
    // after llama_memory_clear), so the pipelines are warmed in a throwaway context instead: the pipeline cache belongs
    // to the device, the context only supplies a graph, and this one is created small enough to cost nothing.
    if (getenv("STRATA_VK_NO_WARMUP") == nullptr) {
        const Clock::time_point tw = Clock::now();
        llama_context_params wcp = cp;
        // One sequence, and a context that is already a multiple of 256: `n_ctx_seq` is `n_ctx / n_seq_max` padded up to
        // 256, so 16 tokens over 32 sequences became an 8192-token context. `n_outputs_max` also defaults to `n_batch`,
        // while every decode reserves `max(n_outputs, n_seq_max)` rows, so the 32 sequences of a batched run asserted
        // here against this 16-token batch.
        wcp.n_ctx     = 256;
        wcp.n_seq_max = 1;
        wcp.n_batch   = 16;
        wcp.n_ubatch  = 16;
        wcp.moe_cache_size = 0;   // the cache belongs to the real context's budget
        llama_context * wctx = llama_init_from_model(model_, wcp);
        if (wctx != nullptr) {
            llama_batch wb = llama_batch_init(16, 0, 1);
            llama_token tok = llama_vocab_bos(vocab_);
            if (tok == LLAMA_TOKEN_NULL || tok < 0) {
                tok = 0;
            }
            for (int i = 0; i < 16; ++i) {
                wb.token[i]      = tok;
                wb.pos[i]        = (llama_pos) i;
                wb.n_seq_id[i]   = 1;
                wb.seq_id[i][0]  = 0;
                wb.logits[i]     = i == 15 ? 1 : 0;
            }
            wb.n_tokens = 16;
            if (llama_decode(wctx, wb) == 0) {
                logf("[strata-vk] warmed the prefill pipelines in %.2f s (a throwaway 16-token context)\n",
                     ms_since(tw) / 1000.0);
            } else {
                logf("[strata-vk] prefill warmup forward failed, the first prompt will compile its pipelines\n");
            }
            llama_batch_free(wb);
            llama_free(wctx);
        } else {
            logf("[strata-vk] no prefill warmup: the throwaway context did not fit\n");
        }
    }

    // The MTP head drafts the next tokens from this model's hidden state.  Without one the engine is complete and simply
    // does not speculate; with one that cannot load it says why and runs on.
    if (!opt.mtp.empty() && !mtp_supported_by_build()) {
        logf("[strata-vk] MTP: drafting measures slower than plain decoding on this GPU (5.8 against 7.0 tok/s on the "
             "27B, 11.9 against 36.2 on the 35B-A3B); running without drafts (set STRATA_VK_MTP=1 to use it anyway)\n");
    }
    if (!opt.mtp.empty() && mtp_supported_by_build()) {
        // Where the head runs is a real choice on this machine: on the GPU it costs 1.2-2.2 GB of the same memory the
        // target model is reading through, and on the HTP it costs a FastRPC round trip per draft but nothing of the
        // GPU's.  `--npu-mtp` picks the HTP; the fallback is the graphics device the target already uses, because the
        // backend's own default list is every graphics device in the build and would split the head across three.
        std::vector<ggml_backend_dev_t> mtp_devices = { nullptr };
        std::string mtp_where = "the GPU";
        if (opt.npu_mtp && npu_.dev != nullptr) {
            mtp_devices = { npu_.dev, nullptr };
            mtp_where = ggml_backend_dev_name(npu_.dev);
        } else if (!devices_.empty() && devices_.front() != nullptr) {
            mtp_devices = { devices_.front(), nullptr };
        } else if (ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) {
            mtp_devices = { cpu, nullptr };
        }
        bool ok = mtp_.load(opt.mtp, ctx_, n_ctx, cp.n_threads, mtp_devices.data());
        if (!ok && opt.npu_mtp && npu_.dev != nullptr) {
            // The DSP's mapped-vmem budget is ~3 GiB, so a large head can simply not fit; say so and use the GPU.
            logf("[strata-vk] MTP: %s on %s (%s); placing the head on the GPU instead\n",
                 "cannot load the head", mtp_where.c_str(), mtp_.status());
            mtp_.unload();
            std::vector<ggml_backend_dev_t> gpu_only = { nullptr };
            if (!devices_.empty() && devices_.front() != nullptr) gpu_only = { devices_.front(), nullptr };
            ok = mtp_.load(opt.mtp, ctx_, n_ctx, cp.n_threads, gpu_only.data());
            mtp_where = "the graphics device";
        }
        if (!ok) logf("[strata-vk] MTP: %s (running without drafts)\n", mtp_.status());
        else logf("[strata-vk] MTP head on %s\n", mtp_where.c_str());
    }
    return true;
}

std::vector<llama_token> Engine::tokenize(const std::string & text) const {
    std::vector<llama_token> out;
    if (vocab_ == nullptr) return out;
    const bool add_special = llama_vocab_get_add_bos(vocab_);
    out.resize(text.size() + 8);
    int n = llama_tokenize(vocab_, text.c_str(), (int32_t) text.size(), out.data(), (int32_t) out.size(), add_special, true);
    if (n < 0) {
        out.resize((size_t) -n);
        n = llama_tokenize(vocab_, text.c_str(), (int32_t) text.size(), out.data(), (int32_t) out.size(), add_special, true);
    }
    out.resize(n > 0 ? (size_t) n : 0);
    return out;
}

std::string detokenize_ids(const llama_vocab * vocab, const std::vector<llama_token> & ids) {
    std::string out;
    if (vocab == nullptr) return out;
    for (const llama_token id : ids) {
        char buf[256];
        const int n = llama_token_to_piece(vocab, id, buf, (int) sizeof(buf), 0, true);
        if (n < 0) {
            std::vector<char> big((size_t) -n);
            const int m = llama_token_to_piece(vocab, id, big.data(), (int) big.size(), 0, true);
            if (m > 0) {
                out.append(big.data(), (size_t) m);
            }
            continue;
        }
        out.append(buf, (size_t) n);
    }
    return out;
}

std::string Engine::detokenize(const std::vector<llama_token> & ids) const {
    return detokenize_ids(vocab_, ids);
}

} // namespace strata


