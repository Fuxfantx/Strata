# Patches worth offering upstream

Everything here already lives in this tree - `port/llama-overlay-vanilla/` for the llama.cpp side, `port/engine/` for the
engine. This note is the shortlist that is **not** fork-specific: each entry says what it changes, why the measurement
justifies it, and where it is. Numbers are from `ADRENO.md` on this machine.

## Backend (llama.cpp, `ggml/src/ggml-vulkan`)

1. **Cooperative-matrix capability detection without an f32-accumulate assumption.**
   `ggml_vk_instance_init` disables every matrix-core path when the device has no f32-accumulate f16 matrix
   (`f32acc 0`), even when it does report an f16-accumulate one. Qualcomm's Adreno X2-90 is exactly that device. The change
   lets a device with `coopmat_acc_f16_support && !coopmat_acc_f32_support` build the `*_f16acc` pipelines, accounts the
   staging at 2 bytes per element in the shared-memory check (26624 of 32768 bytes for the 64x64 tile), and lets
   `ggml_vk_get_mul_mat_mat_f16acc` return true there. Measured: 27B pp512 77.5 -> 163.9 tok/s, 35B-A3B 148.6 -> 198.9 (and
   598 with flash attention), with identical CPU-vs-Vulkan token streams.

   > **Withdrawn (2026-11-18).** "Identical token streams" was a 32-token comparison and it is false: the kernel's logits
   > correlate 0.02 with the CPU backend's and the top token changes (`PERFORMANCE.md` §4.1). The engine no longer enables
   > it by default. The capability-detection patch itself is still right - it is what makes the f32-accumulate cooperative
   > matrix reachable, which is the default path today.

2. **Pipeline-map lookup falls back to the `f16acc` key.** With the capability above, a plain request finds nothing because
   only `*_f16acc` variants exist; the lookup retries with `f16acc = true` and warns nothing. (`GGML_ASSERT(mmp_map !=
   nullptr)` is what it replaces.)

3. **`prefer_f16_b` should be a shape rule, not a vendor list.** Upstream prefers an f16 B for Intel only. Two additions are
   needed for correctness and speed elsewhere:
   * prefer it only when a real token batch is present (`ne[1] || ne[2] || ne[3] > 1`), because an f16 B disqualifies the
     matvec kernel and decode regresses (tiny 130.7 -> 112.3 tok/s when that was missing);
   * require tile-legal dimensions (`k % 128 == 0`, `m % 64 == 0`, `n >= 64` for the dense path; plus `n_mats >= 16` and
     `k >= 256` for the grouped path), because the cooperative shader has no tail handling. This is the whole accuracy
     remedy: the oracle goes from 20/38 (garbage on partial tiles) to 38/38 in both precision modes with zero cost on the
     model shapes.

4. **Create the cooperative pipelines in both shader arms.** The block that creates them sat inside the fp32 arm; the packs
   that run in fp16 mode (every mixture-of-experts pack here) never reached it, so their grouped matmuls logged
   `prefer_f16_b=1 f16_avail=0` and stayed on the dp4a path. A self-contained creation block in the fp16 arm (dense and
   grouped) is worth 35B-A3B prefill 195 -> 502 tok/s and Gyro-M 34 -> 197.

5. **Open defect worth reporting: direct backend graphs return zeros for the cooperative kernel.** `mm-probe` and
   `test-backend-ops` submit a single-node graph straight to the backend and get an all-zero result for every shape, while
   llama.cpp's scheduler path computes the same shapes correctly. It is why the oracle could not certify this kernel before
   the shape gate; a fix in the prealloc/reformat planning of the direct path is the missing piece.

## Engine (this port, `port/engine`)

6. **Per-model shader route from the GGUF header** (`vk-devices.cpp`, `ModelFacts`): dense models want the fp32 shaders,
   packs whose int-dot tensors are all 2-bit want the fp16 shaders with `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1` (their dp4a
   fp16 shader crashes this driver on a multi-token prefill), trellis packs want the fp16 shaders for decode. This is worth
   2-3x on the low-bit packs and costs nothing.

7. **Flash attention chosen from the context length and the model size** (`vk-engine-load.cpp`): off below 8k context and for
   small models (a 24-layer model loses 38% decode with it), on above 8k for models with 32+ layers, where it is worth 81% on
   a 4k prefill (f16 KV) and 31% with q4_0 KV. `--flash-attn` always wins.

8. **`--perf-cap` is opt-in only** (`vk-common.h`, `vk-options.*`): it sleeps the owed idle share per step and pins the
   process away from the reserved cores. It exists because a benchmark otherwise consumes the whole machine, but note for
   anyone reusing it: it cannot cure stutter caused by paging (see `ADRENO.md`), and millisecond-accurate sleeps need the
   spin-loop that is there because Windows' timer granularity is ~15.6 ms.

9. **A pack larger than RAM is mapped, not read** (`vk-engine-load.cpp`): `--load-mode direct` reads every byte of the file   into a buffer, including the N-Gram/PLE table that mmap can leave in the file, so on an oversized pack it is exactly
   what fills the RAM. When the user did not name `--load-mode` and the pack is bigger than `ullAvailPhys`, the engine now
   maps it. Measured on Gyro-S (54.47 GiB pack, 15.27 GiB of it that table, 40.3 GiB free): 4.1 -> 6.1-6.4 tok/s decode,
   with 0.6 GiB still free at the end of the load and the disk idle during decode. Small packs keep the direct path, which
   loads them 1.5-1.7x faster (`ADRENO.md`).

10. **Host experts turn the f16-accumulate cooperative matrix off** (`vk-devices.cpp`, `vk-engine-load.cpp`): the policy   above opts into `STRATA_VK_F16ACC` for prefill speed, and with experts on the CPU that combination miscomputes as soon
   as a batch carries more than a handful of tokens - Gyro-S and Gyro-M abort in `ggml-cpu`'s swiglu NaN check at 12
   tokens (8 is fine), while the same model on device experts is clean at 6, 12 and 32 and vanilla `llama-completion
   --cpu-moe` finishes the same prompt because it never enables f16acc. `f16acc_from_policy` records that the engine (not
   the user) turned it on, and the host-expert branch sets it back to 0 with a log line. Measured: `--bench-pp 64
   --cpu-moe` went from an abort to 6.82 tok/s on Gyro-S and 7.94 on Gyro-M, with the device-expert path unchanged
   (90.9 tok/s at pp64).

11. **One single-token forward at the end of the load** (`vk-engine-load.cpp`, `STRATA_VK_NO_WARMUP` disables it): a
   mapped pack is page-cold and the first graph pays for it - the same `MUL_MAT_VEC q8_0 m=10240 n=1 k=320` measured
   96 x 4985 us on the first graph and 96 x 2.5 us on the next one, which is 11 MB at 23 MB/s against cache speed, and
   made the first decode step 3.35 s against a 99.8 ms steady state. The forward costs 5.25 s once, inside the load, and
   moves the user-visible `prompt` from ~5100 ms to **111 ms** (decode n=8 went from 6.5 to 11.0 tok/s purely from not
   averaging that stall in). A multi-token warmup would also warm the cooperative pipelines, but leaves a hybrid model's
   memory in a state the next decode rejects (M-RoPE asserts on position order even after `llama_memory_clear`), so only
   the single-token one is here; item 12 warms the prefill pipelines the safe way.

12. **A throwaway 16-token context warms the prefill pipelines** (`vk-engine-load.cpp`, same `STRATA_VK_NO_WARMUP`): the
   prefill graph takes different pipelines from a decode step and the first six-token prompt spent 2.4 s compiling them.
   Doing the multi-token forward in the real context corrupts a hybrid model's memory, so the load creates a second
   context with `n_ctx`/`n_batch` of 16 and no MoE cache, decodes 16 tokens, and frees it - the pipeline cache belongs to
   the device, the context only supplies a graph. Measured: the first real prompt went from 2423 ms to **780 ms** for
   2.87 s of extra load time, with the reference tokens, Paris and pp64 (92.95) unchanged.

13. **An oversized pack keeps its experts on the device when they fit** (`vk-engine-load.cpp`): the host-expert staging
   is for packs whose *experts* are bigger than RAM, but it used to trigger whenever the whole resident set was. For Q2_0
   (62.2 GiB pack, 31.6 GiB of experts, 40 GiB free) that was the wrong side of the line: one token touches only ~0.6 GiB
   of experts and the page cache keeps that set, while the staging pays a 4 GiB device bank against 31.6 GiB of host
   experts and spends 56% of the wall in the loader. Leaving them on the device measured **3.3 -> 14.2 tok/s decode and
   17.1 -> 40.4 tok/s at pp64**; a larger bank is worse, not better (8 GiB: 2.9 tok/s and 767% CPU, because the bank and
   the page cache evict each other). Gyro-M gains the same way (pp64 65.3 -> **95.5**), Gyro-S is untouched because its
   resident set fits, and `--cpu-moe` still forces the old placement.

14. **Four shaders that the backend depends on are now tracked in the overlay** (`ggml/src/ggml-vulkan/vulkan-shaders/`):
   `dsv4_hc_pre.comp`, `dsv4_hc_post.comp`, `dsv4_hc_comb.comp` and `lightning_indexer.comp` existed only under
   `third_party/llama.cpp`, so the next re-fetch of upstream would have dropped them and silently changed every qwen4exp
   forward pass - the C++ side (`ggml-vulkan.cpp`) was in the overlay but its shaders were not. Copying them brings the
   overlay to 58 files with zero mismatches against `third_party`; the build is byte-identical in behaviour (the
   reference 32-token stream is unchanged).

15. **`GGML_OP_GATED_MUL`: one op for `x * sigmoid(gate)`** (`ggml.h`, `ggml.c`, `ggml-cpu/ops.{cpp,h}`,
   `ggml-cpu/ggml-cpu.c`, `ggml-vulkan/ggml-vulkan.cpp`, `ggml-vulkan-push-constants.h`,
   `ggml-vulkan-types.h`, `vulkan-shaders/gated_mul.comp`, `vulkan-shaders-gen.cpp`, `src/models/qwen4exp.cpp`).
   The three gate sites in qwen4exp emitted `sigmoid` followed by `mul`; the backend's `SIGMOID_MUL` fusion requires the
   two nodes to be adjacent, and this fork's graph builder interleaves sibling consumers between them, so the pair cost
   two dispatches each. The op takes `x` and the *raw* gate and applies the sigmoid internally, in the same broadcast
   form as `ggml_mul`'s second operand (one value per row when `gate->ne[0] == 1`). Op count 99 -> 100, hence the two
   `static_assert`s in `ggml.c`. Measured on Gyro-S decode: 2728 -> 2668 dispatches per token and 9.60 -> 9.70 tok/s
   (median of three runs), reference 32-token stream identical.

16. **`STRATA_VK_GRAPH_EDGES=<from>,<to>`** (`ggml-vulkan.cpp`): prints each node's real sources inside a node range,
   which is how the non-adjacency above was diagnosed. Gated by the environment variable, no behaviour change.

## Trellis types: TQ2_T / TQK6 / TQK7

**Provenance.** The Vulkan shaders, the reference codecs and the CPU dot kernel are transplanted from
`https://github.com/agentionai/llama.cpp` (MIT licence), branch `main` at `dc255f2a`, with the Vulkan matvec tuning of
`perf/strix-vulkan` at `3c75be02` (the word-view state extraction in `tqk.glsl`). One tuning value diverges from the fork:
`rm_tq` (rows per workgroup in the matvec) is `20 * rm_stdq`, not 1, reachable through `STRATA_VK_TQ_RM_MULT`. The range
was swept on Gyro-S decode at n=16: 2x gives 6.1 tok/s, 4x 7.8, 8x 9.1, 12x 9.8, 16x 9.9, **20x 10.4**, 24x 10.1, 32x 9.5,
so the workgroups were far too fine-grained (a 4.5x change in rows per workgroup bought 70%). A workgroup unpacks a whole
6-bit row from the stream, so fewer, fatter ones amortise the unpack setup. Prefill is unchanged (pp64 88.8, pp512 217.4)
and the TQK oracles stay green (MUL_MAT 37/37, MUL_MAT_ID 75/75). The earlier 2x figure came from sweeping only {1,2},
which was the mistake this corrects. The tree
already carried **TQ2_T (id 144)** from ggml-org upstream (`32bdc3a8`): block layout, the 4x2048 fp16 codebook,
`quantize_row_tq2_t_ref`/`dequantize_row_tq2_t`, `ggml_vec_dot_tq2_t_q8_0`, `dequant_tq2_t.comp`, the
`mul_mat_vec`/`mul_mat_vec_id` TQ2_T branch and the `types.glsl`/`dequant_funcs.glsl` hooks. Only the two bit-packed siblings were
missing, so the transplant is the **TQK6/TQK7 delta** of the same format:

* `ggml-common.h`: `QK_TQK`, `block_tqk6`/`block_tqk7`, `tqk_state()` and its bit helpers, static asserts.
* `ggml-quants.c`: `tqk_set_chunk`, `quantize_row_tqk*_ref`, `dequantize_row_tqk*`, `quantize_tqk6/7` (identical value formula to
  TQ2_T, the state window is the only difference).
* `ggml-cpu/quants.c`: the `ggml_tq_state_i16` table, its `GGML_TQ_I16_SCALE = 2^13` fixed point and
  `ggml_vec_dot_tqk{6,7}_q8_0`, plus the one-time `ggml_cpu_tq_init()` call. Scalar only: the x86 SIMD variants are of no use on
  ARM64.
* Vulkan: `tqk.glsl` (byte and word views of the circular 16-bit window), `dequant_tqk6/7.comp`, the `accumulate_tqk` branch of
  `mul_mat_vec.comp` (four 16-bit loads per lane per row), the `mul_mm_funcs.glsl` staging branch for the mat-mat path, the
  `dequant_funcs.glsl` `tqk_state`/`dequantize` hooks, and the generator + pipeline wiring in `ggml-vulkan.cpp`.

**Verification.** `test-backend-ops -o MUL_MAT -p type_a=tqk{6,7}` 37/37 and `-o MUL_MAT_ID` 75/75 for both types, against the
CPU reference (which the oracle reaches through the same `vec_dot`, so the independent check is the token comparison below). The
pack `Gyro-S-TQ1_0.gguf` loads and generates, and the same eight tokens come out of three placements of the same weights: host
experts (CPU `ggml_vec_dot_tqk*`), device experts with a 24 GiB cache, and device experts with mmap paging - the default
placement now maps the pack (see engine item 9) and decodes at 6.1-6.5 tok/s, against 4.1-4.2 with the whole pack in
buffers. The Hadamard hint needs no work: `ggml_vk_can_use_fwht` already honours `GGML_HINT_SRC0_IS_HADAMARD` and
the tree carries the fused FWHT pipeline for the 136 rotated expert tensors.

**Defects found on the way, both reproducible without the trellis types (Gyro-M):** the host-expert MoE cache reads its device bank
past the rows it allocated for a multi-token prefill (`ffn_moe_gate-1 bad=1542/409600`), and `GGML_OP_LIGHTNING_INDEXER` returns
all-NaN for some placements. `GGML_NAN_SCAN=1` (CPU) and `GGML_VK_NAN_SCAN=<substr|*>` (Vulkan) are the two scans that localise
them; both are env-gated and cost nothing when unset. `GGML_VK_GRAPH_PRINT=1` is the third knob: it dumps the shape and name of
every elementwise node in each graph, which is what `GGML_VK_PERF_LOGGER` cannot show, because a fused dispatch (the eight
`[2560,1]` adds of `ffn_moe_out` become one `MULTI_ADD`) is logged under the fused name only.

**Correction on the indexer (round 16):** the `LIGHTNING_INDEXER` "NaN" was `-inf`. Those scores are masked-out pools, which is
the intended value, and the scan was counting every non-finite value as a NaN. The scan now reports `nan= / -inf= / +inf=` and
the index of the first bad element, and `GGML_VK_NAN_SCAN_PIN=<substr>` pins the matching nodes as graph outputs first, so a
post-graph scan reads their own bytes instead of a buffer the allocator already recycled.

Two real defects the same round did find, both outside the kernels: the unfused indexer-mask fallback in `qwen4exp.cpp`
(`set_rows` + `cumsum` + dump rows) miscomputes when a ubatch carries more than one token, and it is the path every backend
takes when `LLM_FUSED_OP_TOP_K_MASK` does not resolve (the CPU one among them); and every fused-op probe in `llama-context.cpp`
is built with `n_tokens_per_seq = 1`, so the fused paths were never validated on a multi-token prefill. `STRATA_NO_FUSED_LID=1`
and `STRATA_NO_FUSED_TOPK_MASK=1` isolate the two indexer fusions the way `STRATA_NO_FUSED_HC=1` isolates the hc one.

## Backend fix worth offering upstream: the shader generator deadlocks on Windows

`execute_command()` in `vulkan-shaders-gen.cpp` drains `stdout` to EOF and only then reads `stderr`. A `glslc` child that fills the
64 KiB stderr pipe before closing stdout blocks forever, and so does the parent: a build wedges with every compiler process idle at
~0.1 s of CPU and no progress (seen here with 12 `glslc` processes, no disk activity, 2 minutes of wall time). Routing both streams
into one pipe and reading that single stream removes the deadlock; keep the diagnostics by copying the captured text into
`stderr_str`.

17. **`STRATA_VK_TIME=1`: where a decode step's wall time goes** (`ggml/src/ggml-vulkan/ggml-vulkan.cpp`,
   `ggml/src/ggml-vulkan/ggml-vulkan-common.h`).  A stopwatch over a graph's recording, six buckets inside it (the
   dispatch helper, `ggml_vk_sync_buffers`, the fusion chain, `ggml_vk_ctx_end`, `ggml_vk_submit` and the command pool
   cleanup), the fence wait in `ggml_backend_vk_synchronize`, and one line per token.  Off by default with no overhead.

   It first looked as if a decode step were CPU recording-bound (Gyro-S recording 100.7 ms and waiting 0.001 ms for a
   91 ms token), and the buckets show that reading was wrong: the dispatch helper costs 0.49 us per call, a barrier
   0.04 us, the whole fusion chain 0.16 us per node, and cleanup 0.12 ms, which is about 3 ms of host work in a 72 ms
   window.  `STRATA_VK_SUBMITS=N`, which caps how many command buffers a graph is split into, confirms it: 1, 4 and 26
   submissions all give the same throughput.  With `GGML_VK_SERIALIZE_SUBMISSIONS=1` a Q2_0 token falls from 15.5 to
   9.5 tok/s, which exposes about 40 ms of GPU work against roughly 25 ms of process CPU, so a token is 40 ms of GPU
   plus 25 ms of CPU, partially overlapped.  `port/docs/PERFORMANCE.md` §6 records the correction; the knobs
   stay because proving what is not the bottleneck is worth an environment variable.

18. **The dp4a mat-mat shader is taken from the fp32 module** (`ggml-vulkan.cpp`, `ggml_vk_load_shaders`).
    Both modules are compiled into one binary and the creation order decides which one a pipeline slot gets
    (`create_mm_pipelines` returns early when the slot is already `initialized && compiled`), so the fp16-module
    `matmul_*_q8_1` lines silently beat the fp32-module ones that follow them. Pointing the three fp16-arm
    `sg_create_mmq` runs at the `*_fp32_len`/`*_fp32_data` symbols - types, pipeline keys and tile configs
    unchanged - moves only the int-dot **mat-mat** to fp32 while the mat-vec and every elementwise shader stay on
    the fp16 module. Measured on Gyro-S (interleaved 3 rounds, `ggml-vulkan.dll` swap, medians): **pp512
    16.01 -> 49.80 tok/s (+211%)**, tg16 10.17 -> 10.12 (noise), bench prompt 5 840 -> 3 050 ms;
    `test-backend-ops -o MUL_MAT -b Vulkan0` 3/3 and `OK`; the two modules agree on logits at correlation
    0.999424 with the same argmax. The reason the fp32 module is faster is the one `--shader-family fp32` already
    showed (ADRENO.md: "the fp16 shader variants cost the int-dot matmul a factor of four"), now obtained without
    paying that switch's decode cost.
