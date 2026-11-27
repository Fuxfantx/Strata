// vk-engine-load.cpp - the load path: devices, the model, the context.
//
// Split from vk-engine.cpp so that the one decision that can strand a user - placing tensors on the NPU - is in one
// place with its fallback: a model that the NPU cannot take is loaded again without it.
#include "vk-engine.h"

#include "vk-companion.h"

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

// ---------------------------------------------------------------------------------------------------------------
// The residency plan for a latent pack.
//
// A latent pack's weights live beside its skeleton, so the engine can say what it will keep resident before it loads
// anything.  The numbers are the same arithmetic `port/tools/edge0-pack.py --plan` prints, so the two can be compared
// line for line; they are formulas over the documented geometry, not measurements on this machine.

namespace {

std::string read_text_file(const std::string & path) {
    std::string out;
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) { return out; }
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) { out.append(buf, n); }
    std::fclose(f);
    return out;
}

// the handful of numbers model.json carries, without pulling in a JSON library
long long json_int(const std::string & text, const char * key, long long fallback) {
    const std::string pat = std::string("\"") + key + "\"";
    size_t at = text.find(pat);
    if (at == std::string::npos) { return fallback; }
    at = text.find(':', at);
    if (at == std::string::npos) { return fallback; }
    return std::strtoll(text.c_str() + at + 1, nullptr, 10);
}

size_t file_size_or_zero(const std::string & path) {
    std::error_code ec;
    const auto sz = std::filesystem::file_size(path, ec);
    return ec ? 0 : (size_t) sz;
}

}  // namespace

void log_pack_plan(const Options & opt) {
    const std::string dir = opt.pack;
    const std::string meta = read_text_file(dir + "/model.json");
    if (meta.empty()) {
        logf("[strata-vk] pack %s has no model.json; the residency plan is not available\n", dir.c_str());
        return;
    }
    const long long layers  = json_int(meta, "layers", 48);
    const long long experts = json_int(meta, "experts", 512);
    const long long latent  = json_int(meta, "latent_dim", 0);

    // D/U: the format is the pack's own choice, and the q8 file is the larger one, so report the one that will be used
    const bool q4 = read_text_file(dir + "/latent_dense_q8.bin").empty();
    const std::string du_name = q4 ? "latent_dense_q4.bin" : "latent_dense_q8.bin";
    const size_t du_bytes = file_size_or_zero(dir + "/" + du_name);

    // the expert blob size comes from the pack's own manifest header, or from the file size itself
    size_t blob = 1630720;
    const size_t experts_bytes = file_size_or_zero(dir + "/experts.bin");
    if (experts_bytes && layers && experts) { blob = experts_bytes / (size_t) (layers * experts); }

    const size_t skeleton = file_size_or_zero(dir + "/model-skeleton.gguf");
    const size_t ple      = file_size_or_zero(dir + "/ple-fp8.gguf");
    const size_t profile  = file_size_or_zero(dir + "/" + (opt.expert_profile.empty() ? "profile.bin"
                                                                                    : opt.expert_profile));

    // KV: only the full-attention layers keep a cache; a fourth of them for this family
    const long long n_full = (layers + 3) / 4;
    const long long n_lin  = layers - n_full;
    const double bits_per_value = (opt.kv == "q4_0" || opt.kv == "q4_1") ? 18.0 / 32.0
                                 : (opt.kv == "q8_0" ? 34.0 / 32.0 : 2.0);
    const double kv = (double) n_full * 2 * 256 * (double) opt.n_ctx * 2.0 * bits_per_value;
    const double rec = (double) n_lin * (128.0 * 48.0 * 4.0 + 4.0 * 6144.0 * 4.0);
    const double compute = 3.0 * 1024 * 1024 * 1024;   // documented reserve, the same constant the tool uses
    const double fixed = (double) skeleton + (double) du_bytes + kv + rec + compute;
    const double budget = 36.0 * 1024 * 1024 * 1024;   // the knob's default ceiling for the whole engine
    const double left = budget - fixed;
    const long long total_blobs = layers * experts;
    const long long fit = blob ? (long long) (left > 0 ? left / (double) blob : 0) : 0;
    const long long fit_blobs = fit < total_blobs ? fit : total_blobs;
    const long long per_layer = layers ? fit_blobs / layers : 0;

    logf("[strata-vk] pack plan for %s (formulas, not measurements):\n", dir.c_str());
    logf("  skeleton %.2f GiB, D/U %s %.2f GiB, KV %lld layers q=%s %.2f GiB, recurrent %.3f GiB, compute %.2f GiB\n",
         (double) skeleton / 1073741824.0, q4 ? "q4" : "q8", (double) du_bytes / 1073741824.0, n_full, opt.kv.c_str(),
         kv / 1073741824.0, rec / 1073741824.0, compute / 1073741824.0);
    logf("  PLE %.2f GiB (%s, mapped and read by row), profile %zu bytes, latent width %lld\n",
         (double) ple / 1073741824.0, opt.ple_io.c_str(), (size_t) profile, latent);
    logf("  budget %.2f GiB, fixed %.2f GiB -> experts %.2f GiB: %lld of %lld blobs resident (%.1f%%),"
         " about %lld of %lld per layer\n",
         36.0, fixed / 1073741824.0, (left > 0 ? left : 0.0) / 1073741824.0, fit_blobs, total_blobs,
         total_blobs ? 100.0 * (double) fit_blobs / (double) total_blobs : 0.0, per_layer, experts);
    if (opt.expert_cache_mib) {
        logf("  --expert-cache %zu MiB caps the expert set\n", opt.expert_cache_mib);
    }
    if (!opt.pcie_mode.empty() || opt.pool_workers) {
        logf("  accepted from the pack template, not used yet: pcie-mode %s, pcie-frac %.2f, pool-workers %d\n",
             opt.pcie_mode.empty() ? "(none)" : opt.pcie_mode.c_str(), opt.pcie_frac, opt.pool_workers);
    }
}

// set when --dry-run printed the policy, so a null model is a success and not a load failure
bool dry_run_finished = false;

// The companion's name: the pack's own file name with `.f16companion.gguf` instead of `.gguf`.  It sits beside the
// pack by default, where its name makes the pairing obvious and it can never be mistaken for a model (the shard scan
// looks for `-00001-of-NNNNN`); STRATA_VK_COMPANION_DIR puts it somewhere else, which a read-only model directory
// needs.  The name is derived from the pack's, so two different packs with the same file name would share one cache
// entry - deleting the file is always safe, it is only an optimisation.
static std::string f16_companion_path(const std::string & model) {
    const size_t slash = model.find_last_of("/\\");
    const std::string dir  = slash == std::string::npos ? std::string(".") : model.substr(0, slash);
    const std::string file = slash == std::string::npos ? model : model.substr(slash + 1);
    const size_t dot = file.rfind(".gguf");
    const std::string stem = dot == std::string::npos ? file : file.substr(0, dot);
    std::filesystem::path where(dir);
    if (const char * env = std::getenv("STRATA_VK_COMPANION_DIR"); env != nullptr && env[0] != '\0') {
        where = env;
        std::error_code ec;
        std::filesystem::create_directories(where, ec);
    }
    where /= stem + ".f16companion.gguf";
    return where.string();
}

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

// STRATA_VK_ROW_SPLIT=<cpu share> cuts a slice of rows out of the dense weights that dominate a decode step, so that
// the CPU can compute that slice while the GPU computes the rest of the same matmul (PERFORMANCE.md 5.5.17): the
// companion writer emits `<name>_low` with the tail rows, the loader creates those tensors, and
// llm_graph_context::build_lora_mm multiplies with both halves and concatenates.  The result is exact - the rows are
// only distributed, not re-quantised - and the share is what decides where the split runs.
// STRATA_VK_ROW_SPLIT_TENSORS overrides which tensors are sliced (comma separated name substrings).
static const std::vector<CompanionSlice> & row_split_slices(const std::string & model_path) {
    static const std::vector<CompanionSlice> slices = [] (const std::string & path) {
        std::vector<CompanionSlice> out;
        const char * rs = std::getenv("STRATA_VK_ROW_SPLIT");
        const char * es_env = std::getenv("STRATA_VK_EXPERT_SPLIT");
        if (rs == nullptr && es_env == nullptr) return out;
        const double cpu_share = rs != nullptr ? std::strtod(rs, nullptr) : 0.0;
        if (!(cpu_share > 0.0) && es_env == nullptr) return out;

        std::vector<std::string> pats;
        if (const char * env = std::getenv("STRATA_VK_ROW_SPLIT_TENSORS")) {
            const std::string all(env);
            size_t p = 0;
            while (p <= all.size()) {
                size_t q = all.find(',', p);
                if (q == std::string::npos) q = all.size();
                if (q > p) pats.push_back(all.substr(p, q - p));
                p = q + 1;
            }
        } else {
            // the dense weights of one decode step, largest first
            for (const char * d : { "attn_qkv.weight", "attn_q.weight", "attn_gate.weight", "ssm_out.weight",
                                    "attn_output.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                                    "ffn_down_shexp.weight", "output.weight" }) {
                pats.push_back(d);
            }
        }

        size_t hit = 0;
        for (const CompanionTensorInfo & t : companion_tensor_list(path)) {
            if (!(cpu_share > 0.0) || t.ne1 < 2 || t.ne2 != 1 || t.ne3 != 1) continue;
            bool want = false;
            for (const std::string & p : pats) {
                if (t.name.find(p) != std::string::npos) { want = true; break; }
            }
            if (!want) continue;
            int64_t keep = (int64_t) std::llround((double) t.ne1 * (1.0 - cpu_share));
            keep = std::max<int64_t>(1, std::min<int64_t>(t.ne1 - 1, keep));
            out.push_back({ t.name, t.name + "_low", 1, keep });
            ++hit;
        }

        // STRATA_VK_EXPERT_SPLIT=<cpu share> slices the expert stacks along the expert dimension instead of by row.
        // The graph turns an id outside its half into -1 (which mul_mat_id zeroes), so the two products add up.
        if (es_env != nullptr) {
            const double expert_share = std::strtod(es_env, nullptr);
            if (expert_share > 0.0) {
                size_t ehit = 0;
                for (const CompanionTensorInfo & t : companion_tensor_list(path)) {
                    if (t.ne2 < 2 || t.ne3 != 1) continue;
                    if (t.name.find("_exps.weight") == std::string::npos &&
                        t.name.find("_exps.weight") == std::string::npos) {
                        if (t.name.find("exps") == std::string::npos) continue;
                    }
                    int64_t keep = (int64_t) std::llround((double) t.ne2 * (1.0 - expert_share));
                    keep = std::max<int64_t>(1, std::min<int64_t>(t.ne2 - 1, keep));
                    out.push_back({ t.name, t.name + "_low", 2, keep });
                    ++ehit;
                }
                logf("[strata-vk] expert split: %zu stack(s) sliced, %.0f%% of each stack's experts on the CPU\n",
                     ehit, expert_share * 100.0);
            }
        }

        if (cpu_share > 0.0) {
            logf("[strata-vk] row split: %zu tensor(s) sliced, %.0f%% of each tensor's rows on the CPU\n",
                 hit, cpu_share * 100.0);
        }
        return out;
    }(model_path);
    return slices;
}



// STRATA_VK_EVAL=<substring> dumps a checksum per graph node whose name matches, which is how the CUDA/Vulkan and the CPU
// paths are compared node by node (see PORT-STATE.md, the qwen4exp defect).
struct EvalDumpState {
    std::string filter;
};


// A quantised KV cache is only usable when the backend can run flash attention over it.  The check builds the same
// operation the graph will build and asks the device, rather than assuming that every backend has that kernel.
static bool vk_quantised_attn_supported(ggml_type kv_type) {
    if (ggml_backend_dev_count() == 0) {
        // the probe can run before anything has enumerated the backends
        ggml_backend_load_all();
    }
    // The probe has to ask the device the graph will actually run on: the registry lists more than one GPU-class
    // device here and HTP0 comes first, so "the first GPU" is the wrong question to ask.
    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        const char * name = d != nullptr ? ggml_backend_dev_name(d) : nullptr;
        if (name != nullptr && std::strncmp(name, "Vulkan", 6) == 0) {
            dev = d;
            break;
        }
    }
    if (dev == nullptr) {
        return false;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (backend == nullptr) {
        return false;
    }
    ggml_init_params ip = { 1u << 20, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    bool ok = false;
    if (ctx != nullptr) {
        ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 8, 1, 1);
        ggml_tensor * k = ggml_new_tensor_4d(ctx, kv_type, 128, 256, 1, 1);
        ggml_tensor * v = ggml_new_tensor_4d(ctx, kv_type, 128, 256, 1, 1);
        ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 256, 1);
        ggml_tensor * o = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f / 128.0f, 0.0f, 0.0f);
        ok = o != nullptr && ggml_backend_supports_op(backend, o);
        if (std::getenv("STRATA_VK_KV_TRACE") != nullptr) {
            logf("[strata-vk] kv probe: device %s, type %s, op %p, supported %d\n",
                 ggml_backend_dev_name(dev), ggml_type_name(kv_type), (void *) o, (int) ok);
        }
        ggml_free(ctx);
    }
    ggml_backend_free(backend);
    return ok;
}

static bool eval_dump_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (EvalDumpState *) user_data;
    // "*" is for STRATA_VK_EVAL_GRAPH alone: walk every node to print the graph, read nothing back.  A readback per
    // node is what made the full dump expensive (one submit and drain each), and the structure does not need it.
    const bool graph_only = st->filter == "*";
    if (ask) {
        // Only a matching node needs its own submit and synchronisation; answering true for every node turned one decode graph
        // into 2824 submits with a device drain each, which cost more than the dump was worth.
        return st->filter.empty() || graph_only || std::strstr(t->name, st->filter.c_str()) != nullptr;
    }
    if (!graph_only && !st->filter.empty() && std::strstr(t->name, st->filter.c_str()) == nullptr) {
        return true;   // false would make the scheduler abort the rest of the split
    }
    if (!graph_only) {
        const size_t nbytes = ggml_nbytes(t);
        std::vector<uint8_t> raw(nbytes);
        ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
        double sum = 0.0, abs_sum = 0.0;
        long long n_finite = 0, n_pinf = 0, n_ninf = 0, n_nan = 0;
        if (t->type == GGML_TYPE_F32) {
            const float * f = (const float *) raw.data();
            const size_t n = nbytes / sizeof(float);
            for (size_t i = 0; i < n; ++i) {
                const float v = f[i];
                // Classify before accumulating: a masked tensor is legitimately full of -inf, so summing absolute values
                // reports infinity whether or not anything is wrong.  The counts say which case it is.
                if (v != v) {
                    n_nan++;
                    continue;
                }
                if (v == INFINITY) {
                    n_pinf++;
                    continue;
                }
                if (v == -INFINITY) {
                    n_ninf++;
                    continue;
                }
                n_finite++;
                sum += v;
                abs_sum += v < 0 ? -v : v;
            }
        } else {
            for (size_t i = 0; i < nbytes; ++i) sum += raw[i];
            abs_sum = sum;
            n_finite = (long long) nbytes;
        }
        fprintf(stderr, "[eval] %-10s %-12s n=%lld abs=%.9g finite=%lld +inf=%lld -inf=%lld nan=%lld\n",
                t->name, ggml_op_name(t->op), (long long) ggml_nelements(t), abs_sum,
                n_finite, n_pinf, n_ninf, n_nan);
        if (t->op == GGML_OP_MUL_MAT && t->src[0] != nullptr) {
            const ggml_tensor * s0 = t->src[0];
            fprintf(stderr, "[evsrc] %-10s wtype=%s k=%lld buf=%s\n", t->name, ggml_type_name(s0->type),
                    (long long) s0->ne[0], s0->buffer != nullptr ? ggml_backend_buffer_name(s0->buffer) : "-");
        }
    }
    // STRATA_VK_EVAL_GRAPH=1: the node and its sources, as ops and names.  The fusion rules in the backend are all
    // about adjacency (a MUL_MAT followed by an ADD, a unary followed by a MUL), and this is what says whether the
    // graph a model actually builds has those adjacencies - asking the fusion code instead only reports the ones that
    // were rejected, not the ones that never met.
    static const bool dump_graph = std::getenv("STRATA_VK_EVAL_GRAPH") != nullptr;
    if (dump_graph) {
        fprintf(stderr, "[node] %-10s %-14s [%lld,%lld,%lld,%lld] src:", t->name, ggml_op_name(t->op),
                (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
        for (int s = 0; s < GGML_MAX_SRC && t->src[s] != nullptr; ++s) {
            fprintf(stderr, " %s(%s", t->src[s]->name, ggml_op_name(t->src[s]->op));
            if (t->src[s]->op == GGML_OP_MUL_MAT || t->src[s]->op == GGML_OP_MUL_MAT_ID) {
                const ggml_tensor * w = t->src[s]->src[0];
                fprintf(stderr, ",%s", w != nullptr ? ggml_type_name(w->type) : "-");
            }
            fprintf(stderr, ")");
        }
        fprintf(stderr, "\n");
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
    } else if (!opt.tensor_split.empty() && backend_ != GpuBackend::CPU) {
        // A tensor split names the devices it shares the model between.  With one graphics device in this build the
        // second share is the CPU, which is the same device list `--split-cpu` builds; the shares themselves then
        // decide the geometry together with `--split-mode`.
        const ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu != nullptr) {
            devices.push_back(cpu);
            split_.assign(opt.tensor_split.begin(), opt.tensor_split.end());
            if (split_.size() < devices.size()) split_.resize(devices.size(), 0.0f);
            logf("[strata-vk] tensor split: %zu device(s), shares", split_.size());
            for (float f : split_) logf(" %.2f", (double) f);
            logf("\n");
        } else {
            logf("[strata-vk] --tensor-split: no CPU backend to share with\n");
        }
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
    std::vector<std::string> latent_patterns;
    std::vector<std::string> hot_patterns;   // likewise: the patterns below are handed over as pointers
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
        // This path used to force STRATA_VK_F16ACC=0 here: with CPU-resident experts Gyro-S and Gyro-M hit the CPU
        // swiglu's NaN check at 12 tokens.  That was the trellis coop pipeline reading a staged A block its shader
        // never wrote (`ggml-vulkan.cpp`, the per-type SPIR-V fix), not the placement, so nothing is forced off now.
        const ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        // STRATA_VK_ROW_SPLIT=<cpu share> cuts the output head's rows in two so that one half can run on the CPU while the
    // other runs on the GPU (PERFORMANCE.md 5.5.17): the companion writer emits `output.weight_low` with the tail rows,
    // the graph multiplies with both halves and concatenates, and the override below parks the tail in host memory.
    // The keep count is decided here and handed to the model through the environment, so both sides agree without a
    // new model parameter.
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
    const auto & slices = row_split_slices(opt.model);
    if (!slices.empty() && cpu_buft != nullptr) {
        // every tail lives in host memory, so the scheduler puts its matmul on the CPU
        buft_overrides.push_back({ "_low$", cpu_buft });
    }
        const std::string where = (opt.cpu_moe || opt.n_cpu_moe <= 0) ? std::string("all layers")
                                                                    : ("first " + std::to_string(opt.n_cpu_moe) + " layers");
        logf("[strata-vk] experts in host memory: %s (device cache %zu MiB, cpu share %.2f)\n",
             where.c_str(), moe_cache_mib_, (double) opt.moe_cache_cpu_share);
    }
    if (!opt.pack.empty()) {
        // A latent pack's expert blobs are 40.8 GiB in total, so they must not be copied onto the device: they stay in
        // host memory as mapped views and the latent op reads the rows a token needs.  The D/U pair and the labels are
        // placed with them, because the same op consumes all four.
        const ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        ggml_backend_buffer_type_t cpu_buft = cpu_dev != nullptr ? ggml_backend_dev_buffer_type(cpu_dev) : nullptr;
        if (cpu_buft == nullptr) {
            logf("[strata-vk] latent pack: no CPU buffer type; the expert blobs would have to fit the device\n");
        } else {
            latent_patterns.emplace_back("\\.latent_(d|u|labels)(\\.weight)?$");
            latent_patterns.emplace_back("\\.experts_blob$");
            // The blobs are 40.8 GiB, so the default keeps them in host memory where the CPU reference reads them.  On a
            // unified-memory device the graphics heap is the same RAM, and placing them there lets the latent block read
            // them in place; this is opt-in because it moves 40 GiB through the device allocator.
            ggml_backend_buffer_type_t latent_buft = cpu_buft;
            if (opt.latent_device) {
                // by name: `ggml_backend_dev_by_type(GPU)` answers with whatever accelerator registered first, and on this
                // machine that is the Hexagon backend, whose buffers cannot run the latent op at all
                ggml_backend_dev_t dev = nullptr;
                for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                    ggml_backend_dev_t d = ggml_backend_dev_get(i);
                    ggml_backend_reg_t r = ggml_backend_dev_backend_reg(d);
                    const char * rn = r != nullptr ? ggml_backend_reg_name(r) : nullptr;
                    if (rn != nullptr && std::string(rn).find("Vulkan") != std::string::npos) { dev = d; break; }
                }
                // host memory the device can read, not device memory: a 40 GiB device allocation fails outright
                if (dev != nullptr) {
                    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
                    auto get_host = (ggml_backend_buffer_type_t (*) () )
                        ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_host_buffer_type");
                    if (get_host != nullptr) {
                        latent_buft = get_host();
                    }
                }
                if (dev != nullptr && ggml_backend_dev_buffer_type(dev) != nullptr && latent_buft == cpu_buft) {
                    latent_buft = ggml_backend_dev_buffer_type(dev);
                }
                if (latent_buft == cpu_buft) {
                    logf("[strata-vk] --latent-device: no graphics device buffer type, keeping the blobs in host memory\n");
                }
            }
            for (const std::string & p : latent_patterns) {
                buft_overrides.push_back({ p.c_str(), latent_buft });
            }
            // A layer whose blob is device-visible only runs on the device if the rest of the op's inputs are there too:
            // D/U, labels and the expert blob are all consumed by the same op, so a scheduler that finds one of them in
            // host memory keeps the whole op on the CPU.
            static const int hot_layers = [] {
                const char * v = std::getenv("STRATA_LATENT_HOT");
                return v != nullptr ? std::atoi(v) : 0;
            }();
            if (hot_layers > 0) {
                ggml_backend_buffer_type_t host_buft = nullptr;
                for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                    ggml_backend_dev_t d = ggml_backend_dev_get(i);
                    ggml_backend_reg_t r = ggml_backend_dev_backend_reg(d);
                    const char * rn = r != nullptr ? ggml_backend_reg_name(r) : nullptr;
                    if (rn != nullptr && std::string(rn).find("Vulkan") != std::string::npos) {
                        auto get_host = (ggml_backend_buffer_type_t (*) () )
                            ggml_backend_reg_get_proc_address(r, "ggml_backend_vk_host_buffer_type");
                        if (get_host != nullptr) {
                            host_buft = get_host();
                        }
                        break;
                    }
                }
                if (host_buft != nullptr) {
                    hot_patterns.emplace_back("\\.latent_(d|u|labels)(\\.weight)?$");
                    hot_patterns.emplace_back("\\.experts_blob_hot$");
                    // at the FRONT: llama.cpp takes the first override that matches a tensor name, and the general
                    // latent patterns added just above would otherwise claim every D/U, labels and blob tensor
                    size_t at = 0;
                    for (const std::string & p : hot_patterns) {
                        buft_overrides.insert(buft_overrides.begin() + (long) at++, { p.c_str(), host_buft });
                    }
                    logf("[strata-vk] latent pack: %d layer(s) keep their experts in device-visible memory\n", hot_layers);
                } else {
                    logf("[strata-vk] STRATA_LATENT_HOT set but the host buffer type is unavailable; nothing moved\n");
                }
            }
            logf("[strata-vk] latent pack: D/U, labels and the expert blobs in %s\n",
                 latent_buft != cpu_buft ? "device memory (read in place)" : "host memory (read by row)");
        }
    }
    if (!buft_overrides.empty()) buft_overrides.push_back({ nullptr, nullptr });
    mp.tensor_buft_overrides = buft_overrides.empty() ? nullptr : buft_overrides.data();
    mp.tensor_split          = split_.empty() ? nullptr : split_.data();
    // llama.cpp's own split mode.  ROW is tensor parallelism: every tensor is cut by rows across the same device list
    // `--split-cpu` builds, so both devices take part in each matmul instead of each owning whole blocks.
    const char * eff_split_mode = "layer";
    if (opt.split_mode == "row") {
        // llama.cpp takes the split buffer type from the backend registry and throws when it is missing, which
        // aborts the load.  Ask first and say so: this backend does not offer one, so a caller that asked for tensor
        // parallelism gets the layer split it would otherwise have got silently, plus this line.
        bool can_row = split_.size() > 1;
        for (size_t i = 0; can_row && i < devices_.size(); ++i) {
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(devices_[i]);
            can_row = ggml_backend_reg_get_proc_address(reg, "ggml_backend_split_buffer_type") != nullptr;
            if (!can_row) {
                logf("[strata-vk] --split-mode row: %s has no split buffer type (llama.cpp refuses a row split "
                     "there), falling back to a layer split\n", ggml_backend_dev_name(devices_[i]));
            }
        }
        mp.split_mode = can_row ? LLAMA_SPLIT_MODE_ROW : LLAMA_SPLIT_MODE_LAYER;
        eff_split_mode = can_row ? "row" : "layer";
    } else if (opt.split_mode == "none") {
        mp.split_mode = LLAMA_SPLIT_MODE_NONE;
        eff_split_mode = "none";
    } else {
        mp.split_mode = LLAMA_SPLIT_MODE_LAYER;
        eff_split_mode = "layer";
    }
    if (split_.size() > 1) {
        logf("[strata-vk] split mode: %s over %zu device(s)\n", eff_split_mode, split_.size());
    }

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
             opt.threads,
                      !opt.flash_attn_set ? "auto" : opt.flash_attn == Tri::On ? "on"
                                        : opt.flash_attn == Tri::Off ? "off" : "auto");
        logf("  warmup         : %s (decode %s, prefill %s)\n",
             std::getenv("STRATA_VK_NO_WARMUP") != nullptr ? "off" : "on",
             std::getenv("STRATA_VK_NO_WARMUP") != nullptr ? "no" : "yes",
             std::getenv("STRATA_VK_NO_WARMUP") != nullptr ? "no" : "yes");
        logf("  lookup / mtp   : %s / %s\n", opt.lookup == Tri::Off ? "off" : "on", mtp_wanted ? "on" : "off");
        if (opt.npu_mtp && !mtp_wanted) {
            // The dry run is where an operator checks a flag combination, and --npu-mtp chooses where a head runs without
            // loading one: without --mtp, or with MTP still gated off, it would otherwise do nothing and say nothing.
            logf("  --npu-mtp      : no effect (%s)\n",
                 opt.mtp.empty() ? "no --mtp file was given" : "MTP drafting is off in this build (STRATA_VK_MTP=1 turns it on)");
        }
        std::string comp_note = "none (no BF16 tensors)";
        if (!opt.compat.empty()) {
            comp_note = opt.compat;
        } else if (opt.bf16_bytes > 0) {
            const std::string comp = f16_companion_path(model_path);
            std::error_code comp_ec;
            comp_note = comp + (std::filesystem::exists(comp, comp_ec) ? " (cached)" : " (will be built on this load)");
        }
        logf("  companion      : %s\n", comp_note.c_str());
        dry_run_finished = true;
        return nullptr;
    }
    // A BF16 tensor is run at two thirds of the F16 rate on this driver (PERFORMANCE.md 5.5.5): 387 of them cost the Q2
    // pack 19% of its prefill.  Converting them to F16 is not a lossy step for weights - a BF16 value with a 7-bit
    // mantissa is exactly representable in F16 whenever its magnitude is in F16's normal range - and the conversion is
    // written once, beside the pack, as the same kind of companion `--compat` takes.  `STRATA_VK_NO_COMPANION=1` and an
    // explicit `--compat` both leave it alone; a read-only directory just means the pass reports and is skipped.
    std::string auto_companion;
    if (opt.compat.empty() && opt.bf16_bytes > 0 && std::getenv("STRATA_VK_NO_COMPANION") == nullptr) {
        const std::string comp = f16_companion_path(model_path);
        std::error_code ec;
        // A cached companion is only reusable when it was sliced the same way: same names, same row counts.  Both
        // directions matter, a file with slices must not be reused without them and the other way round.
        const std::vector<CompanionTensorInfo> comp_tensors = companion_tensor_list(comp);
        size_t comp_low = 0;
        for (const CompanionTensorInfo & t : comp_tensors) {
            if (t.name.size() > 4 && t.name.compare(t.name.size() - 4, 4, "_low") == 0) ++comp_low;
        }
        bool split_matches = comp_low == row_split_slices(model_path).size();
        for (size_t i = 0; split_matches && i < row_split_slices(model_path).size(); ++i) {
            const CompanionSlice & sl = row_split_slices(model_path)[i];
            split_matches = false;
            for (const CompanionTensorInfo & t : comp_tensors) {
                if (t.name != sl.name) continue;
                const int64_t got = sl.dim == 2 ? t.ne2 : t.ne1;
                split_matches = got == sl.keep_rows;
                break;
            }
        }
        if (split_matches && row_split_slices(model_path).empty()) {
            int64_t h0 = 0, h1 = 0, p0 = 0, p1 = 0;
            if (companion_tensor_rows(comp, "output.weight", h0, h1) &&
                companion_tensor_rows(model_path, "output.weight", p0, p1) && h1 != p1) {
                split_matches = false;
            }
        }
        if (std::filesystem::exists(comp, ec) && companion_matches(comp, model_path) && split_matches) {
            auto_companion = comp;
        } else {
            if (std::filesystem::exists(comp, ec)) {
                // A companion left over from a pack that has since been replaced: converting the new pack over it is
                // the only safe thing (the file is derived, nothing else reads it).
                logf("[strata-vk] companion: %s was built from a different pack; rebuilding it from this one\n",
                     comp.c_str());
            }
            const Clock::time_point tc = Clock::now();
            const std::vector<CompanionPick> picks =
                companion_picks_of_type(model_path, GGML_TYPE_BF16, GGML_TYPE_F16);
            const std::vector<CompanionSlice> & slices = row_split_slices(model_path);
            std::string err;
            if ((!picks.empty() || !slices.empty()) && write_companion(model_path, comp, picks, err, slices)) {
                std::error_code sz_ec;
                const auto sz = std::filesystem::file_size(comp, sz_ec);
                logf("[strata-vk] companion: %zu BF16 tensor(s) converted to F16 in %.1f s (%.0f MiB), cached as %s\n",
                     picks.size(), ms_since(tc) / 1000.0, sz_ec ? 0.0 : (double) sz / (1024.0 * 1024.0), comp.c_str());
                auto_companion = comp;
            } else {
                logf("[strata-vk] companion: the BF16 -> F16 pass did not run (%s); the pack runs as it is, %.0f%% "
                     "slower on prefill\n", err.c_str(), 19.0);
            }
        }
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
    if (!auto_companion.empty()) {
        logf("[strata-vk] companion %s: its F16 tensors replace the pack's same-named BF16 ones\n", auto_companion.c_str());
        const char * paths[2] = { model_path.c_str(), auto_companion.c_str() };
        return llama_model_load_from_splits(paths, 2, mp);
    }
    return llama_model_load_from_file(model_path.c_str(), mp);
}

bool Engine::load(const Options & opt, const NpuPlan & npu) {
    if (!opt.pack.empty()) {
        // the loader reads this itself, because the tensor table has to know about the pack's files before create_tensor()
        _putenv_s("STRATA_LATENT_PACK", std::filesystem::absolute(opt.pack).string().c_str());
    }
    if (!opt.pack.empty()) {
        log_pack_plan(opt);
    }
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
    // The output tensor is [n_vocab, n_outputs_max] and it is f32: at 248 320 tokens of vocabulary a 2 048-row batch
    // already makes it 2.0 GB, and a 3 072-row ubatch asked for 3.05 GB, which the Vulkan backend refused - the
    // scheduler then found `result_output` (MUL_MAT [248320,3072]) unassigned and aborted on
    // `GGML_ASSERT(*cur_backend_id != -1)` while the context was being created.  This engine never asks for more than
    // one row of logits in a prefill (the last row) and one per sequence in the slot path, so the reserve can be told
    // what the caller actually asks for instead of reserving the whole batch.  16 also covers a speculative verify pass
    // (the sampled token plus a full draft, and the draft length is a measured knob).  llama.cpp grows the output
    // buffer on demand if a batch ever asks for more, so this is a reservation, not a limit on what can be requested.
    cp.n_outputs_max   = std::max(cp.n_seq_max, 16u);
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
    if (opt.npu_mtp && (opt.mtp.empty() || !mtp_supported_by_build())) {
        // --npu-mtp only chooses where the head runs; it does not load one.  Without --mtp, or with MTP still gated off,
        // the flag would otherwise do nothing and say nothing.
        logf("[strata-vk] --npu-mtp: %s, so this flag has no effect\n",
             opt.mtp.empty() ? "no --mtp file was given" : "MTP drafting is off in this build (STRATA_VK_MTP=1 turns it on)");
    }
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
        } else if (opt.gpu != GpuBackend::CPU &&
                   !vk_quantised_attn_supported(opt.kv == "q4_0" ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0)) {
            // The Vulkan flash-attention kernel has no quantised K/V path: every FLASH_ATTN_EXT case with type_K of
            // q4_0 or q8_0 reports "not supported" for this device, so such a cache would run an unsupported op and
            // produce wrong tokens quietly.  Say so and stay on f16, which is the configuration the tokens agree on.
            logf("[strata-vk] this backend has no flash attention for a quantised K/V cache (test-backend-ops reports "
                 "every q4_0/q8_0 case as unsupported); staying on f16\n");
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
        // "all" is the portable way to ask for every node: an empty environment value disappears in most shells, so
        // the word stands in for the empty filter the callback treats as "match everything".
        eval_state.filter = std::strcmp(e, "all") == 0 ? std::string() : std::string(e);
        cp.cb_eval = eval_dump_cb;
        cp.cb_eval_user_data = &eval_state;
    } else if (std::getenv("STRATA_VK_EVAL_GRAPH") != nullptr) {
        // Structure only: walk every node, read nothing back (see eval_dump_cb).
        eval_state.filter = "*";
        cp.cb_eval = eval_dump_cb;
        cp.cb_eval_user_data = &eval_state;
    }
    ctx_ = llama_init_from_model(model_, cp);
    if (ctx_ == nullptr) {
        logf("[strata-vk] cannot create the context\n");
        return false;
    }
    n_ctx_ = (int) llama_n_ctx(ctx_);
    n_embd_ = (int) llama_model_n_embd(model_);
    batch_ = llama_batch_init((int32_t) cp.n_batch, 0, 1);
    batch_cap_ = (int) cp.n_batch;
    // --parallel N: one KV sequence per slot, and each slot's cache is the context divided by the sequence count that
    // llama.cpp actually settled on (it rounds n_ctx down to a multiple of n_seq_max * 256).
    // --parallel N (serve) and --batch-slots N (the bench) are the same machinery: both size the request slots and both
    // feed n_seq_max, so either flag has to create the slots.  Only --parallel used to, which left `--batch-slots 64
    // --serve` advertising `batch_slots=0` and made multi-slot serving unreachable from serve/server.py.
    const int serve_slots = std::max(opt.parallel, opt.slots);
    if (serve_slots > 1) {
        const int seq = (int) std::max(1u, llama_n_seq_max(ctx_));
        slot_st_.resize((size_t) serve_slots);
        slot_ctx_ = n_ctx_ / seq;
        logf("[strata-vk] %d request slots, %d tokens of cache each (batch %d)\n", serve_slots, slot_ctx_,
             batch_cap_);
        // Correct and worth it: one step per slot over one copy of the weights measures 30.5 tok/s aggregate at 32
        // slots and 33.6 at 64, against 10.25 for a single stream (PERFORMANCE.md 5.3).  The old note here said the
        // opposite; the cooperative expert matmul is what changed it.
        logf("[strata-vk] %d slots: serve/server.py uses them when it reads batch_slots>=2\n", serve_slots);
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
        std::vector<ggml_backend_dev_t> gpu_only = { nullptr };
        if (!devices_.empty() && devices_.front() != nullptr) gpu_only = { devices_.front(), nullptr };
        std::vector<ggml_backend_dev_t> mtp_devices = gpu_only;
        std::string mtp_where = "the GPU";
        const bool want_npu = opt.npu_mtp && npu_.dev != nullptr;
        if (want_npu) {
            mtp_devices = { npu_.dev, nullptr };
            mtp_where = ggml_backend_dev_name(npu_.dev);
        } else if (mtp_devices.front() == nullptr) {
            if (ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) {
                mtp_devices = { cpu, nullptr };
            }
        }
        // A session opening is not the same as the graph running.  On this pack the nextn layer's *fused hyper-connection*
        // node has no HTP kernel at that shape, so llama.cpp's scheduler assigns it to the CPU and leaves the rest of the
        // layer on the HTP - the device split that docs/features/NPU-ASSETS.md recorded as "opens the session, then stops
        // answering on the first decode".  The scheduler says so on its warning channel, which this engine captures, so
        // the split is caught here and the head goes to the GPU instead of hanging on the first draft.
        bool ok;
        std::string npu_fail;
        if (want_npu) {
            watch_backend_logs("is assigned to device");
            ok = mtp_.load(opt.mtp, ctx_, n_ctx, cp.n_threads, mtp_devices.data());
            if (ok && backend_log_watch_hit()) {
                npu_fail = "the HTP cannot run the whole nextn graph (one node has no kernel there, so the layer would be "
                           "split across HTP and CPU)";
                ok = false;
            }
            clear_backend_log_watch();
        } else {
            ok = mtp_.load(opt.mtp, ctx_, n_ctx, cp.n_threads, mtp_devices.data());
        }
        if (!ok && want_npu) {
            if (npu_fail.empty()) {
                // The DSP's mapped-vmem budget is ~3 GiB, so a large head can simply not fit.
                logf("[strata-vk] MTP: cannot load the head on %s (%s); placing the head on the GPU instead\n",
                     mtp_where.c_str(), mtp_.status());
            } else {
                logf("[strata-vk] MTP: %s on %s; placing the head on the GPU instead\n",
                     npu_fail.c_str(), mtp_where.c_str());
            }
            mtp_.unload();
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

    if (std::getenv("STRATA_HY4_TRACE") != nullptr) {
        const int32_t n_vocab = llama_vocab_n_tokens(vocab_);
        std::string ids;
        for (size_t i = 0; i < out.size(); ++i) {
            ids += std::to_string(out[i]) + (i + 1 < out.size() ? "," : "");
            if (out[i] < 0 || out[i] >= n_vocab) {
                ids += "(<-- outside the " + std::to_string(n_vocab) + " entry vocabulary)";
            }
        }
        logf("[strata-vk] tokenized %zu bytes into %zu ids: %s\n", text.size(), out.size(), ids.c_str());
    }
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


