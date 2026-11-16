// vk-devices.h - the devices the engine can run on, the Adreno knobs, and the NPU (Hexagon) plan.
#pragma once

#include "ggml-backend.h"
#include "llama.h"

#include "vk-options.h"

#include <string>
#include <vector>

namespace strata {

const char * dev_kind(enum ggml_backend_dev_type type);

// What the engine decides from the model header before any backend exists.
struct ModelFacts {
    bool readable     = false;
    bool dense_matmul = true;   // prefill runs on the int-dot mat-mat, where the measured fp32-shader tuning pays
    int  block_count  = 0;      // layers the NPU split divides
    int  int_dot      = 0;      // tensors on the int-dot path
    int  trellis      = 0;
    int  low2         = 0;      // q2_0 tensors: their int-dot fp16 shader crashes this driver on a multi-token prefill
    bool q20_only     = false;  // 2-bit int-dot tensors and no trellis: fp16 without the int-dot shader is the fast side
    int  experts      = 0;      // `*_exps` stacks: a mixture-of-experts pack
    bool lowbit_pack  = false;  // any 2-bit or trellis tensors: the cooperative-matrix path pays for itself there
    // Bytes of the expert stacks (`*_exps`) against everything else: the two figures upstream's -            --
    // resident-experts decision is made from, and the ones that decide whether this pack can be held in RAM.
    size_t expert_bytes = 0;
    size_t dense_bytes  = 0;
    // Tensors that live on disk and are read per token rather than held in RAM: the N-Gram / PLE table of the 3.8 Flash Next
    // packs (`per_layer_token_embd`).  Subtracting them is what makes "does this pack fit" the right question.
    size_t disk_bytes   = 0;
};

ModelFacts read_model_facts(const std::string & path);

std::string exe_dir();

// FastRPC on Windows resolves the HTP skel (`libggml-htp-v<N>.so`) as a bare name against the process's current
// directory: the identical URI opens a session with the skel's directory as cwd and fails with 0x80000406 without it,
// and an absolute path in the URI is not accepted either.  So the loader has to run with that directory as cwd, and
// every path it is given has to be absolute first.  This restores the previous directory when it goes out of scope.
class SkelDirGuard {
public:
    explicit SkelDirGuard(bool enable);
    ~SkelDirGuard();
    SkelDirGuard(const SkelDirGuard &) = delete;
    SkelDirGuard & operator=(const SkelDirGuard &) = delete;

private:
    std::string saved_;
    bool active_ = false;
};

// The Adreno tuning this build runs with.  Every value is measured on the X2-90 (port/docs/ADRENO.md); the engine sets a
// variable only when the environment does not already name it, so a user can override any of them.
void apply_adreno_tuning(GpuBackend backend, bool dense_model, bool q20_only = false, bool lowbit_pack = false);

// true when the policy above was the one that enabled STRATA_VK_F16ACC (and the user did not), so a placement that
// cannot live with it can turn it off without overriding an explicit request
extern bool f16acc_from_policy;

// Which CPUs the engine may use, and how many worker threads that allows.
// Reserves max(ceil(cpus * (100 - perf_cap) / 100), 1 big + 2 small) cores: the requirement is 20% free with a floor of one
// big and two small cores, so the browser and Office always find a fast core.  `cpus` is the list the engine may use.
struct AffinityPlan {
    int              n_threads = 0;
    std::vector<int> cpus;
};
AffinityPlan plan_affinity(int perf_cap);

// What the engine decided about the dense tensors: the Hexagon NPU takes them when one is present.
struct NpuPlan {
    bool                       requested = false;   // --npu was not "off"
    ggml_backend_dev_t         dev       = nullptr; // the HTP device, null when there is none
    ggml_backend_buffer_type_t buft      = nullptr;
    int                        layers    = 0;       // --npu split: last N layers (plus the head) run on the HTP
    int                        n_layer   = 0;       // the model's block count, for the split fraction
    std::vector<const char *>  patterns;            // the tensor-name regexes that move, borrowed from a static table
    std::vector<llama_model_tensor_buft_override> overrides; // `patterns` as the loader wants them, null-terminated

    bool found()  const { return dev != nullptr; }
    bool active() const { return dev != nullptr && !overrides.empty(); }
    bool split()  const { return dev != nullptr && layers > 0; }
};

// Sets the HTP mapping budget from the environment before the backend loads; a no-op unless tensors will be placed.
void configure_npu_budget(NpuMode mode);

NpuPlan plan_npu(NpuMode mode, const ModelFacts & facts, int layers = 0);

// The graphics devices the graph runs on: every device of the chosen backend, in llama.cpp's order.  Empty for CPU.
// This is `llama_model_params::devices`, which is also what keeps llama.cpp from splitting layers onto another backend.
std::vector<ggml_backend_dev_t> gpu_devices(const NpuPlan & npu, GpuBackend backend);

// Resolves Auto to the backend that is actually present, preferring Vulkan (measured faster on this driver, see
// port/docs/ADRENO.md).
GpuBackend resolve_backend(GpuBackend want);

void print_devices(const NpuPlan & npu, bool to_stdout);

} // namespace strata


