# The Adreno X2-90, as measured

Everything here is measured on this machine, with the command or environment variable that produces it named next to the
number. `PERFORMANCE.md` has the resulting state, the build and the gate list.

## Hardware facts that bound every design

```
STRATA_VK_COOPMAT_LOG=1
[coopmat] vendor 0x5143 arch 10 coopmat 0 (m/n/k 64/64/16, f32acc 0 f16acc 1) int 1 (m/n/k 64/64/32) shmem 32768 intdot 1
```

* **The matrix cores exist in two flavours**: int8 (64x64x32, int32 accumulate) and **f16 accumulation**
  (64x64x16). There is **no f32-accumulate cooperative matrix** (`f32acc 0`), and **no `dot2`** (the f16-product /
  f32-accumulate dot), so every f16 formulation on this device accumulates in f16. That single fact decides the accuracy
  discussion below.
* `subgroup 64`, **32 KiB** of shared memory, integer dot product available, no bf16.
* Decode is memory-bound and prefill is compute-bound: the numbers that bound both are in the next two sections.

## The two ceilings that decide what is worth trying

| what | measured | how |
| --- | --- | --- |
| CPU read bandwidth | **154.6 GB/s** (was recorded as 137-145) | `port/tools/membw.cpp`, three-channel LPDDR, 14 threads, median of 3 |
| GPU streaming read | ~108 GB/s | same tool through a Vulkan reduce |
| CPU + GPU at the same time | 96.9 + 80.4 = **177 GB/s** | both running interleaved |
| Specification | 228 GB/s | - |

**Why 228 is not reachable, measured rather than assumed (2026-11-18).** `membw.cpp` now sweeps thread count and
per-thread unroll depth, so the curve says whether the shortfall is request concurrency or the memory system itself:

| threads | 2 | 3 | 4 | 6 | 8 | 10 | 12 | 14 | 18 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GB/s (8 GiB/run, depth 4, median of 3) | 101 | 126 | 147 | 149 | 140 | 147 | 147 | **155** | 149 |

and at a fixed 6/12/18 threads, depth 4 / 8 / 16 gives 141/134/137, 129/111/111, 133/119/119 GB/s - deeper unrolling
does **not** help and can hurt. So the shortfall is not per-core miss concurrency: **four threads already saturate
(~147 GB/s) and nothing past 14 threads adds anything** (155 GB/s, 66% of the specification). The practical wall for
one engine is therefore ~150 GB/s, the GPU adds enough to reach 177 GB/s together (78%), and the remaining 22-34% is
the memory system's own read efficiency on this UMA part, not something a kernel can ask for.

Two corrections follow. First, the old figure was measured at 18 threads, which is not the best point: the curve peaks
around 14 and 18 costs ~4%. Second, "the practical wall is ~60%" was too pessimistic for the CPU alone (~68% at the
best thread count) while staying right for one engine at a time.

So the practical wall is ~60-68% of the specified number per engine and 78% for both together, and no single engine
reaches the specification alone. The 27B decode already moves
**~112 GB/s** of weight traffic (14.46 GB per token at 7.3 tok/s), i.e. it is at the wall; the 35B-A3B moves
**~83 GB/s** (38.96 tok/s x ~2.1 GB/token) because its small-op tail keeps the memory system from being saturated.


Prefill on the int8-dot (dp4a) tiles alone measured **~4 TFLOP/s**, about 31% of the int8 peak, and that is what the
cooperative matrix was brought in for.

## Decode: where the bandwidth goes

Per-op figures come from the backend's perf logger (`GGML_VK_PERF_LOGGER=1`); the class split is `PERFORMANCE.md`'s
ablation. What matters for design:

* The dense packs are at the memory wall, so **decode work is graph and dispatch work, not bandwidth work**. The host gap was
  once blamed on command recording; instrumenting the backend's encode loop (`GGML_VK_ENCTIME`) showed **1.3-2.6 ms for a
  3687-node MoE graph**, so recording is not the cost. The gap is **not** the GPU going idle either: `STRATA_VK_TIME=1` on Gyro-S
  puts **95.4 of a token's 100.5 ms inside the blocked submit with no fence wait and only 29 ms of host CPU**, i.e. the GPU is
  ~95% **busy** and only moving 43 GB/s over the whole token while its matmuls alone run at 118 GB/s. The small ops are busy and
  latency-bound, so merging nodes does not help (`PERFORMANCE.md` §2 has the measurements and what shape would).
* Cutting nodes was measured and was **neutral**: the per-layer-embedding convolution used to emit a `cont` and a `cast` per
  tap, and folding them to one per layer changed nothing measurable because the graph optimizer already folded them.
* The multi-row matvec (upstream's MMVQ) is already the default where it helps: forcing it off costs the MoE **7%**
  (36.4 against 38.9 tok/s) and leaves the 27B unchanged.

## Prefill on the int8-dot tiles, and why the 32 KiB is not the limit

`mm-probe` gives per-shape throughput on the real layer shapes (q4_0, B = q8_1, GFLOP/s):

| m | int8-dot 64x64 | f16-accumulate cooperative 64x64x16 |
| --- | --- | --- |
| 6144 | 1182 | **2699** |
| 10240 | 1154 | **2729** |
| 17408 | 1199 | **2750** |
| 248320 | 1040 | **2664** |

The 128x128 int8-dot tile needs **28672 of 32768 bytes** and therefore fits, and it is **disabled** - by upstream's Qualcomm
vendor tuning, not by the shared-memory check. Making the dp4a flags own their decisions (assign instead of clear, and stop
copying the generic vendor-tuned flags over them) selected it, and it measured **17.9 tok/s against 74.5** for the 64x64 tile
on the 27B: **0.25x**. The vendor choice is right, the shmem budget is not the prefill limit, and the wider tile is not a
lever. (The trace that pinned this: `[tiles] q4_0 int s/m/l=110` while `[shm] ... total=28672 limit=32768 ok`.)

## The cooperative matrix, as landed

The f16-accumulate pipelines (`matmul_quant_f16_f16acc`) are created **next to** the dp4a ones, in **both** the fp32 and the
fp16 shader arm, and are keyed `{type, F16, id, f16acc=true}`. Three selection rules make them reachable and keep decode
fast:

1. `ggml_vk_get_mul_mat_mat_f16acc` returns true for a device with f16-accumulate matrices even though `coopmat_support` is
   false, and the pipeline-map lookup falls back to the `f16acc` key;
2. the b-type choice prefers an f16 B **only for real token batches** (`ne[1] > 1 || ne[2] > 1 || ne[3] > 1`) - an f16 B
   disqualifies the matvec kernel, and checking only `ne[1]` reads a `MUL_MAT_ID` batch as one column (that mistake cost
   Gyro-M half its prefill);
3. the engine turns the path on per model from the GGUF header, with `STRATA_VK_F16ACC=0` as the override.

Default results (same commands as `PERFORMANCE.md`):

| model | prefill | decode | before this path |
| --- | --- | --- | --- |
| 27B Q4_0 | **163.9** | 7.30 | 77.9 / 7.0 |
| 35B-A3B | **198.9** | 38.96 | 151.7 / 38.1 |
| tiny | **4301** | 126.5 | 2826 / 118.9 |
| Q2_0 pack | **31.9** | 13.20 | ~9 / 12.4 |
| Gyro-M | **65.3** | 3.90 | 9.3 / 3.6 |

The 2-bit and trellis packs needed one more piece each: `q2_0`/`tq2_t`/`ptq1_0` had to be added to the cooperative type list
(the generic quant shader dispatches on a type spec constant and has a runtime `q2_0` branch, so no new shader was needed),
and the fp16 arm needed its own creation block because its dependencies sit in three different scopes.

### Where the five packs stand now (2026-11-16, `--tokens 9707 --n 16` and `--bench-pp`)

| model | decode | pp64 | pp512 | against the table above |
| --- | --- | --- | --- | --- |
| Gyro-S (trellis) | **9.6** | 85.1 | 211.9 | 4.2 -> 9.6 decode |
| Gyro-M (trellis) | **8.7** | 100.9 | — | 3.90 -> 8.7 decode, 65.3 -> 100.9 pp64 |
| Q2_0 | **14.0** | 41.5 | — | 13.20 -> 14.0 decode, 31.9 -> 41.5 pp64 |
| 27B Q4_0 | 7.8 | 133.6 | 162.3 | unchanged (7.30 / 163.9) |
| 35B-A3B | **43.1** | 187.6 | **612.7** | 38.96 -> 43.1 decode, 198.9 -> 612.7 pp512 |

The two big movers are not new kernels: Gyro-M/Q2_0 come from an oversized pack keeping its experts on the device
instead of staging them through a 4 GiB bank (see `PERFORMANCE.md`, phase 2 round 2), and 35B-A3B's prefill from
the cooperative path no longer being gated on a 64-token batch (round 21). Gyro-S's decode is the trellis matvec's rows
per workgroup (round 29). The bench now also prints the first decode step separately, because averaging a cold first
token into the rate hid a 3.35 s stall for about ten rounds.

## The model-wide `-inf` was a shader-selection bug, not a tile bug (2026-11-19)

The two corrections below this section end with "the cooperative kernel correlates 0.02 and stays off". The kernel was
right and the *selection* was wrong. Quantized matmuls have two kinds of SPIR-V in this backend:

* one shared shader for the legacy and K-quant types that dispatches on the `MmTypeA` specialization constant
  (`mul_mm_funcs.glsl`, the `MmTypeA ==` chain: Q1_0, Q2_0, Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K,
  Q8_0, TQ1_0, TQ2_0 - fourteen cases, and no trellis case);
* per-type SPIR-V for the LUT and trellis types, compiled with `DATA_A_<type>` (`matmul_tqk6_f16_f16acc_cm1`,
  `matmul_id_subgroup_tqk6_f16_f16acc_cm1`, ... - the generator emits these from the `is_lut_quant(tname)` branch).

The creation block for the f16-accumulate cooperative pipelines handed the **shared** shader to every type in its list,
and that list contained TQK6, TQK7, TQ2_T and PTQ1_0. For those four the `MmTypeA` chain matches nothing, so the
`load_a_to_shmem` body never writes `buf_a`, the MMA reads uninitialised shared memory, and the output is `-inf` -
which is exactly the "first 64 elements of every tensor" pattern: one 64-row tile of garbage that the layer then
spreads. Nothing about the tile, the store, the row map or the scratch was ever wrong.

Why every gate missed it, which is the part worth keeping:

* `test-backend-ops -o MUL_MAT_ID` does have tqk6/tqk7 cases, but at `k=128`; the cooperative path is gated on
  `k >= 256`, so they always took the dp4a shader. Its `k=2880` case is q4_0, which the shared shader *does* handle.
  After the fix the same oracle passes 1168/1168 with `STRATA_VK_F16ACC=1`.
* `mm-probe --verify` uses q4_0.
* the model was the only harness whose matmuls are trellis-typed, and it reported `-inf` without saying where.

Measured after the fix (same binary, paired on this machine):

| | cooperative on | off | |
| --- | --- | --- | --- |
| pp512 | **85.55 tok/s** | 42.51 | +101% |
| tg16, one stream | 10.25 | 10.19 | neutral |
| 32 slots, aggregate | **31.09** | 25.47 | +22% |
| logits vs CPU (`--dump-logits`) | corr 0.998239, argmax 77 | 0.998352, argmax 77 | |
| 32-token continuation | identical to the reference stream | identical | |

Two consequences for the rest of this document. The 32-slot number is a *kernel* result and not a memory-system one,
so "batched decode is at 95% of the bandwidth ceiling" describes the dp4a kernel, not the hardware. And dense
cooperative is still a 5% loss on this pack (80.98 against 85.55), so only the grouped half is on by default.

## The accuracy question, answered by measurement

> **Correction (2026-11-18).** The paragraph below claimed that the f16-accumulate kernel "reproduces the token stream:
> CPU-vs-Vulkan token streams are identical on all five packs". That was measured with 32-token greedy continuations and
> it is false. With `--dump-logits` on the full vocabulary, the kernel's logits correlate **0.02** with the CPU backend's
> and pick a different token, on all three packs tried (Gyro-S, 35B-A3B, 27B; `PERFORMANCE.md` §4.1 has the table). The
> kernel is off by default (`ModelPolicy::coop_f16acc = false`) and reachable only with `STRATA_VK_F16ACC=1`.
> The error table below is real and was the visible half of the same problem.

The f16-accumulate kernel computes in f16, so it cannot pass an element-wise comparison against an f32 reference. What it
was believed to do is reproduce the token stream: **CPU-vs-Vulkan token streams are identical on all five packs**. The error profile,
measured with `mm-probe --verify` (q4_0, m=512, n=64, max relative error against the f32 reference):

| k | f16-accumulate cooperative | dp4a for comparison |
| --- | --- | --- |
| 32 | 11.3x | - |
| 128 | 14.0x | - |
| 512 | 40.5x | - |
| 2048 | 49.8x | - |
| 5120 | 138.6x | 9.1x at k=4096 |

The error is there at **k=32**, where a correct f16 accumulation of 32 products should be near 1e-3: the weights are ~0.01 and
the activations ~1, so the partial sums wander far above the final value and f16's 11-bit mantissa loses about 1% of the
*partial* magnitude per step. It is accumulation, not staging, and chunking the accumulation into f32 promotions would need
chunks of 16 or less - which erases the gain.

`mm-probe --verify` is a diagnostic, not a gate: its CPU reference dequantises to f32 while the GPU path runs q8_1
activations, so even the correct dp4a path shows 1-13% relative spread on near-zero outputs. llama.cpp's oracle compares
like-with-like and is the gate - in the precision mode being claimed, because `test-backend-ops` leaves fp16 on by default and
therefore never exercises the fp32 shaders the engine uses for dense models.

## Tried and ruled out

| idea | why it is out |
| --- | --- |
| int8 cooperative matrix (64x64x32) | correct (oracle green) but **2x slower** than the dp4a tiles: 512/581/575 against 902/1144/1142 GFLOP/s |
| f16-accumulate matrix cores with device-wide fp16 | the fp16 shaders cost the elementwise ops ~3x, which ate the kernel gain (51.5 against 76.9 tok/s prefill before the fp32-elementwise design) |
| 128x128 int8-dot tile | fits, selected, measures 0.25x |
| `dot2` mixed precision | not present on this device |
| split-K partials in shared memory | prefill has n = 512; nothing to split |
| persistent workgroup reusing the activation tile | not implemented; the only remaining structural prefill idea |
| NPU (Hexagon HTP) placement | slower than the GPU in every placement measured; see `NPU-FEASIBILITY.md` |
| MTP drafting | a draft costs a 1.7-2.2 GB head forward; only one draft per round pays, and it loses at every length on the MoE |
| hybrid CPU+GPU placement | no placement wins here; `HYBRID.md` has the matrix |

## How to measure next time

* `STRATA_VK_COOPMAT_LOG=1` - what the backend decided about the matrix cores at device creation.
* `GGML_VK_PIPELINE_STATS=matmul` - which pipeline actually ran (this is how "the matrix cores are unused" was proved, and
  later how the cooperative path was confirmed).
* `GGML_VK_PERF_LOGGER=1` - per-op time and bytes.
* `mm-probe` - one shape, per-shape throughput and a rough element-wise check (`--verify` is a diagnostic, see above).
* `test-backend-ops -o MUL_MAT|MUL_MAT_ID -p type_a=<type>` - the like-with-like oracle; run it in the precision mode being
  claimed.
* `--gpu cpu` on the same prompt - the token-equality gate that covers the f16-accumulating kernel.

## The f16-accumulate kernel cannot be certified by a standalone backend graph (and what that means)

Following the "formal verification before benchmark" rule, the matrix-core path was put in front of the two harnesses that
never exercise it during normal model runs. The result is not a precision problem - it is a harness-visible correctness gap:

* `STRATA_VK_F16ACC=1 test-backend-ops -o MUL_MAT -p type_a=q4_0` gives **20/38** (the same in fp32 and fp16 shader modes),
  and the failures are the cases where the cooperative pipeline is selected: `m=16,n=9,k=256` reports `ERR = 76.0` against a
  5e-4 bound and `m=2880,n=32,k=2880` reports `ERR = 964.7`. The cases with `n <= 8` pass because they take a different
  kernel.
* `STRATA_VK_F16ACC=1 mm-probe -t q4_0 ... --verify` returns **all zeros** for every shape tried, including the 27B's own
  matmul shape (`m=10240, n=512, k=5120`).
* The same build under the engine produces **identical CPU-vs-Vulkan token streams** on all five packs at 165-167 tok/s, and
  `GGML_VK_PIPELINE_STATS` shows `matmul_coop_f16_f16acc_aligned_1` - the family this block creates (renamed this round from
  `matmul_quant_f16_f16acc` so the stats cannot be confused with the non-cooperative f16 shader of the same base name).

So the kernel is correct inside llama.cpp's graph and produces zeros/garbage when a graph is submitted to the backend
directly. That is a backend bug in the direct-graph path (the likely difference is the prealloc/reformat planning that
`ggml_backend_sched` triggers), and it is why the oracle cannot certify this path at any tolerance: it never gets a
meaningful result to compare.

> **Correction (2026-11-18).** The paragraph above is wrong, and the harnesses were right. Run inside the engine's own
> graph, the kernel's logits correlate 0.02 with the CPU backend's and the top token changes (`PERFORMANCE.md` §4.1).
> The 20/38 with `ERR = 76.0` / `964.7` below was the real signal, not a direct-graph artifact. Keep this section as the
> record of how long a weak gate (32 greedy tokens) can hide a kernel that is not doing the whole job.

The measured accuracy of the two families, where it can be measured at all (`mm-probe --verify`, m=512, n=64):

| path | NMSE at k=512 | NMSE at k=5120 |
| --- | --- | --- |
| dp4a | 2.0e-7 | 2.3e-7 |
| f16-accumulate cooperative | not measurable (all zeros) | not measurable (all zeros) |

Dropping the "aligned" pipeline variant was tried as a lazy shape-alignment fix: the oracle still gives 20/38 and the engine
drops from 165 to 117 tok/s, so it was reverted. The aligned variant is the fast one and is not the cause.

**What the lazy path is, then**: fix the backend's direct-graph path so the cooperative pipeline produces real output there.
That single fix (a) lets the oracle measure the kernel, (b) protects any caller that submits a graph without
`ggml_backend_sched`, and (c) is a prerequisite for the accuracy question - only after it does comparing the kernel's NMSE
against the oracle's per-case bound mean anything. If it then misses by a factor of ~1e3 (f16 accumulation vs 1e-7), the two
remaining options are the chunked f32-promotion shader (real cost) and a per-backend tolerance hook in the oracle, which is
the designed extension point (`test_mul_mat::max_nmse_err(backend)` already carries WebGPU and HTP overrides).

## CPU topology and the reservation the engine keeps

Per-logical nominal clocks from the registry, then a pinned single-thread loop to confirm which class is the fast one:

| logical CPUs | nominal | pinned loop (same work) |
| --- | --- | --- |
| 0-5 (6) | 3072 MHz | 56-88 ms |
| 6-17 (12) | 3437 MHz | 29-46 ms |

So **the high-clock group (12 cores) is the big class** - it runs 1.7-2.2x the low-clock group - and CPU0-5 are the small
cores on this part. The engine therefore reserves `max(ceil(cpus x (100 - perf_cap)/100), 1 big + 2 small)` cores, two big
and two small at the 80% default, and pins its whole process to the rest (`SetProcessAffinityMask` in `vk-main.cpp`, because
`ggml_threadpool_new` lives in the dynamically loaded CPU backend and cannot be linked here). That leaves the reserved cores
genuinely free for a browser and Office instead of free only on average.

## Why the machine stutters during a benchmark: paging, not scheduling

The 20% reservation did not fix the stutter, so it was measured instead. Free physical memory, pages/sec and disk read rate were
sampled once per second while a pack ran (`Get-CimInstance Win32_OperatingSystem` plus `Get-Counter`):

| run | free RAM, min | pages/sec, max | disk read, max | prefill |
| --- | --- | --- | --- | --- |
| Q2_0 pack, 32k ctx / batch 512 / f16 KV | **15 MB** (from 41.6 GB) | 258 153 | 3 755 MB/s | 46.9 |
| Q2_0 pack, 8k ctx / batch 128 / **q4_0 KV** | **62 MB** | 315 991 | 3 550 MB/s | 46.2 |
| 27B Q4_0, 8k ctx / batch 128 / q4_0 KV | 24.6 GB | 187 165 (load only) | 3 440 MB/s | 127.4 |

So the stutter is the operating system running out of physical memory: the Q2 pack is 67 GB and Gyro-M is 62 GB on a 47.6 GB
machine, so during those runs free RAM sits at 15-62 MB, the page-fault rate is 2.6-3.2 x 10^5 per second and the SSD is
streaming 3.5 GB/s. No share of the CPU can fix that - a browser needs pages, not cores - which is why the reservation did
nothing.

The lean profile (8k context, q4_0 KV, 128-token prefill chunk) saves roughly 50-100 MB, not the tens of GB that would be
needed: the KV and the compute buffer are not the problem, the weights are. The packs that fit (27B 14.5 GB, 35B-A3B, tiny)
leave 24 GB free and do not thrash at all.

Practical consequences:

* benchmark the oversized packs (Q2, Gyro-M) when the machine is otherwise idle; their numbers are paging-bound anyway;
* the packs that fit can be benchmarked while browsing - the lean profile is not needed for them, and batch 512 measures
  ~2% faster than 128 on the 27B;
* the flag name `--resident-budget-gib` exists in the help text but is not implemented; a resident-set cap for the expert
  weights is the lever that would make the oversized packs coexist with other work, and it is engine work, not a knob.

## The cooperative tile ceiling is 64x64

The tile list is created with 32x32, 64x64 and 128x128 candidates, and on this device only the first two can exist. Forcing
the list past the dp4a vendor flags (they were excluding the large one for reasons that belong to a different shader family)
made the backend actually try to build `matmul_coop_f16_f16acc_aligned_2` and the driver answered:

```
ggml_vulkan: compute pipeline creation failed for matmul_coop_f16_f16acc_aligned_2:
  vk::Device::createComputePipeline: ErrorUnknown
FATAL ERROR: Internal assertion error
```

so the 128x128 cooperative tile is not available here - 64x64 is the ceiling, and the exclusion is now explicit in the code
with that reason instead of being a side effect of the dp4a tile flags. (The 128x128 **dp4a** tile is a different story: it
fits shared memory and measures 0.25x of the 64x64 one, which is why the vendor tuning disables it.)

What is left of the tile work is the 64x64 double-buffered variant: the cooperative loop loads a k-tile, barriers, runs the
mma, barriers again, so the next tile's loads are not overlapped with compute. Two staging buffers fit in 32 KiB at this tile
size, but it is a shader change (prefetch into the second buffer, barrier order, loop rotation) rather than a configuration
one, and it is measured next.

## The int8 cooperative matrix is exact and slower - it stays gated to AMD

The backend can build int8 cooperative pipelines (`matmul_q4_0_q8_1_cm1` and friends, int32 accumulate), but both their
creation and the tile choice were gated to AMD RDNA3/4. Relaxing that gate on this device and measuring:

| path (27B pp512 / MoE pp512) | prefill |
| --- | --- |
| f16-accumulate cooperative (`STRATA_VK_F16ACC=1`) | **175.4** / **478.8** |
| int8 cooperative (`STRATA_VK_F16ACC=0`, B = q8_1) | 83.3 / 152.6 |

Correctness first, as the rule says: with the int8 pipelines selectable, `test-backend-ops` stayed **38/38** on MUL_MAT and
**82/82** on MUL_MAT_ID in the default mode, so the int8 path really is exact. It is simply about half the f16-accumulate rate
here, which matches the earlier kernel-level figure (int8 cooperative ~2x slower than dp4a, and dp4a is itself ~2.3x behind
the f16-accumulate cooperative tiles). The gate is back in place, since enabling it only adds startup compiles.

## Flash attention is now chosen from the context and the model size

Measured A/B, all on this machine (27B unless stated, `--bench-pp 4096 --max-context 16384`):

| config | prefill 4096 |
| --- | --- |
| f16 KV, FA off | 90.6 |
| f16 KV, FA on | **164.1** |
| q4_0 KV, FA off | 91.5 |
| q4_0 KV, FA on | **120.4** |
| tiny (same shape, ctx 16384) | 240 off / **1748** on |

So FA is worth 0.8-7x on prefill once the prompt is long - the user's case, with a 256K window and q4_0 KV, gains 31% on prefill
even with the quantised cache. Costs, measured the same way:

| shape | FA off | FA on |
| --- | --- | --- |
| 27B pp512 (default ctx 32768) | 164-175 | 155.8 |
| 27B decode 16, ctx 16384, q4_0 KV | **7.60** | 6.56 |
| 27B decode 16, ctx 32768, f16 KV | 6.50 | **7.39** |
| 35B-A3B pp512 | 478-502 | **598.4** |
| tiny decode 64 (ctx 16K or 32K) | **126.9** | 78.2 |

The short-prompt numbers are mixed but small, the long-prompt numbers are large, and the one consistent regression is a small
model whose decode is dominated by attention overhead. The engine therefore defaults FA **on** when `n_ctx >= 8192` and the
model has at least 32 layers, which is exactly the 27B / 35B-A3B / Q2 / Gyro-M set and not tiny (24 layers); `--flash-attn
on|off` still decides by itself, and `--flash-attn auto` leaves it to llama.cpp.

## ubatch, descriptor reuse, memory priority and the repack path

Item (4)'s remaining knobs, measured on the 27B (pp512 + tg16 unless noted):

| knob | prefill | decode |
| --- | --- | --- |
| baseline | 170.9 | 7.37 |
| `GGML_VK_DISABLE_DESCRIPTOR_REUSE=1` | 173.6 | 7.35 |
| `GGML_VK_ENABLE_MEMORY_PRIORITY=1` | 173.7 | 7.40 |

Both are inside run-to-run noise (the 27B moves 164-175 between runs), so neither is adopted.

The prefill chunk (`--prefill`, the engine's `n_batch`) is **best at its 512 default** and degrades as it grows:

| `--prefill` | 27B prefill 2048 tokens |
| --- | --- |
| 512 (default) | **164.8** |
| 1024 | 153.8 |
| 2048 | 134.8 |

That matches the note in `vk-common.h`: a 2048-token batch makes the Vulkan backend allocate a 3.2 GiB compute buffer on this
part, and the larger chunk costs more than the saved launches. No change.

There is **no runtime-repack path in this tree**: the repacked `Q4_0_4_4/4_8/8_8` types are gone and the Vulkan backend has no
repack knob (the full `GGML_VK_*` list has no repack entry), so that sub-item is not available to measure.

## Context size is neutral; the flash-attention threshold was too high

Holding everything else fixed, the context size itself does not move TPS on the 27B (pp512 / tg16): 4k 165.2 / 7.22, 8k
173.7 / 7.27, 32k 172.8 / 7.29, 64k 171.9 / 7.27. The 4k row was slower only because the flash-attention policy required 8k to
turn it on - and FA pays at 4k too:

| shape at ctx 4096 | FA off | FA on |
| --- | --- | --- |
| 27B pp4096 | 89.1 | **156.9** |
| 27B tg16 | 7.29 | 7.33 |
| 35B-A3B pp4096 | 295.7 | **758.7** |

So the threshold is now **4096 tokens** (still with the 32-layer floor that keeps tiny on the non-FA path). 27B pp4096 at
ctx 4096 measures 157.4 with the new default, and the default-context pp512 is unchanged at 172.2.

## The direct-graph zero output is specific to the cooperative kernel's dispatch

The direct-graph bug (a single-node `MUL_MAT` graph returns all zeros through `mm-probe` and `test-backend-ops` while
llama.cpp's scheduler path computes the same shapes correctly) was narrowed with a temporary log at the single dispatch choke
point (`ggml_vk_dispatch_pipeline`), which divides the element counts by each pipeline's `wg_denoms`:

| run (q4_0, m=512, n=64, k=512, `STRATA_VK_F16ACC=1`) | dispatched pipeline | elements | workgroups |
| --- | --- | --- | --- |
| as is | `matmul_coop_f16_f16acc_aligned_1` | (512, 64, 1) | (8, 1, 1) |
| with `GGML_VK_DISABLE_COOPMAT=1` | not the cooperative one (dp4a path) | - | - |

So the grid, the push-constant size (68 bytes, the 17-u32 struct) and the pipeline choice are all consistent; the zeros are
inside the cooperative kernel's execution on the direct path only, and the scheduler path - which runs the same pipeline on
the same shapes and produces identical tokens - differs in how the buffers are bound (offsets/ranges of the descriptor set,
and whether A went through the dequant-to-f16 reformat). The next probe is the descriptor buffer offsets of the two paths, not
the grid, and not the shader.

## The 64x64 cooperative tile is the right one (32x32 measured against it)

`STRATA_VK_MAT_TILE=<index>` forces a matmul tile index, which is how the two usable cooperative tiles were compared directly
(they are the only ones that exist - the 128x128 creation fails on this driver):

| tile | 27B pp512 | 35B-A3B pp512 | 27B tg16 |
| --- | --- | --- | --- |
| 0 (32x32) | 104.0 | 372.4 | 7.28 |
| **1 (64x64)** | **163.6** | **659.5** | **7.32** |

So the selector's existing choice (index 1 for anything above 64x64) is also the fastest, by 57% on the 27B and 77% on the
MoE. The knob stays as a measurement flag, off unless the variable is set.

> **Correction (2026-11-19).** Those "32x32" entries are tile *classes* of the non-f16acc comparison - they differ in the
> BM/BN warp tile, not in the matrix shape, which stays 64x64. A genuine 32x32 matrix core is not available to this
> driver: creating an f16-accumulate pipeline with `TM=TN=32` (the tile the MoE padding would want, since an expert owns
> ~10 rows) fails with
> `Invalid mask for Layout[] ... @llvm.qgpu.coopmat.layout.annotate.v64f16(...) layout value: i32 17` and
> `createComputePipeline: ErrorUnknown`. The only usable cooperative shape here is **64x64** with k=32, so the 84% empty
> columns of a sparse-expert prefill cannot be narrowed away with this matrix core.

## The remaining `GGML_VK_*` knobs are neutral or worse on this machine

Swept on 35B-A3B (baseline: decode 39.77 tok/s, prefill 608.6):

| knob | decode | prefill |
| --- | --- | --- |
| `GGML_VK_FORCE_MMVQ=1` | 39.67 | 605.2 |
| `GGML_VK_DISABLE_MULTI_ADD=1` | 39.02 | 596.6 |
| `GGML_VK_MAX_NODES_PER_SUBMIT=1` | **26.84** | 599.8 |
| `GGML_VK_PREFER_HOST_MEMORY=1` | 39.93 | 600.5 |
| `GGML_VK_SUBALLOCATION_BLOCK_SIZE=256 MiB` | 39.05 | 615.1 |

Everything is inside the run-to-run spread except `MAX_NODES_PER_SUBMIT=1`, which is dramatically worse and confirms that
batching submissions per graph is what makes decode work here at all. The engine's current defaults are therefore left alone;
there is no free knob left on this list, and the remaining TPS work is in shaders and scheduling, not configuration.

## Why 64x64 double buffering cannot be built here (and what would fit)

The k-tile loop stages A and B in shared memory, then `barrier()`s before the compute: the next tile's loads do not overlap
with the mma. Double buffering would remove that stall, but it does not fit. With the cooperative tile at BM=BN=64, BK=32,
`SHMEM_STRIDE = BK/2 + 4 = 20` f16x2 vectors, one row costs 20 x 4 bytes x 2 halves (the vectors are `FLOAT_TYPEV2` = 8
bytes here, and the staging is `FLOAT_TYPEV2` arrays):

| item | bytes |
| --- | --- |
| A staging (64 x 20 x 8) | 10 240 |
| B staging (64 x 20 x 8) | 10 240 |
| current total | 20 480 |
| device shared memory | **32 768** |
| free | 12 288 |
| full double buffering (2 x A + 2 x B) | **40 960** - short by 8 192 |

So the 64x64 tile cannot hold two staging sets; the only forms that fit are **one-operand** double buffering (A x 2 + B = 30
720, which hides about half the staging traffic) and full double buffering at the 32x32 tile (2 x 5 120 + 2 x 5 120 = 20 480),
but that tile measures 57% slower on the 27B and 77% slower on the MoE before any overlap gain (`STRATA_VK_MAT_TILE=0` against
the default 1). Item (3) therefore closes with two quantified limits: the 128x128 cooperative pipeline cannot be created by this
driver, and the 64x64 tile cannot be double buffered in 32 KiB.

## Single-operand double buffering fits but costs 25% - measured, reverted

Two full staging sets do not fit in 32 KiB, so the remaining option was to double only A (2 x 10 240 + 10 240 = 30 720 plus the
2 048-byte epilogue stage = exactly 32 768). That variant was built: the next k-tile's A loads are issued into the idle half
before this tile's `coopMatLoad`/`coopMatMulAdd`, B stays single-buffered behind the existing barrier, and the pipeline is
created without complaint.

| 27B pp512 | tok/s |
| --- | --- |
| single-buffered (current) | 164-172 |
| A double-buffered | **129.8** |

It is 25% slower, and the reason is occupancy: raising the workgroup's shared-memory use from 20 480 to 32 768 bytes cuts the
number of resident workgroups per compute unit, and with it the latency hiding that the single-buffered version was already
getting for free. The change was reverted and the overlay is back to 44 files with 0 mismatches.

Item (3) now has three measured limits and no unexplored option: the 128x128 cooperative pipeline cannot be created, two full
staging sets do not fit, and the one that does fit is slower.

## Async submission is off for a reason that matters to every measurement

The engine sets `GGML_VK_DISABLE_ASYNC=1` (and notices that async gave different answers historically). Re-running the A/B without
that knob showed something more practical: the call returns before the GPU has finished, so the engine's own timings become
meaningless.

| with async submission | reported |
| --- | --- |
| tiny pp512 | **117 506 tok/s** ("prompt 4 ms") |
| 35B-A3B pp512 | **1 482 tok/s** ("prompt 9 ms") |
| 27B pp512 | 185.5 (against 164-172) |
| decode (all three) | unchanged, 7.3 / 39.5 / 125.8 |

The prefill numbers are impossible for this machine, which is the point: the benchmark measured the submit, not the work. Token
streams stayed identical on the short prompts, but any comparison built on those timings is void. So the knob stays, and any
future work that enables async submission must put an explicit fence wait after the graph before it is measured.

## The matmul epilogue is already fused upstream - measured, not assumed

Item (5) asked for elementwise ops to be fused into the matmul epilogue for the MoE tail. The backend already does it through
`MAT_VEC_FUSION_FLAGS_{BIAS0,BIAS1,SCALE0,SCALE1}`: the grouped matvec looks ahead at the next one or two nodes and turns
`GGML_OP_MUL` into SCALE0, `GGML_OP_ADD_ID` into BIAS0, and a second fused node into SCALE1. Comparing with
`GGML_VK_DISABLE_FUSION=1` on 35B-A3B decode shows how much that is worth:

| 35B-A3B decode | dispatches per token | tok/s |
| --- | --- | --- |
| fusion disabled | **1 424** | 35.72 |
| fusion as shipped | **331** | ~39.7 |

So fusion removes 77% of the dispatches and about 11% of the decode time already. What remains in the 331 is a different
shape of problem: the gated-delta-net chain per layer (`CONCAT`, `CPY`, `GATED_DELTA_NET`, `L2_NORM`, `SIGMOID`, `SILU`,
`CLAMP`) plus `GLU` - no epilogue pattern covers those, they would need one fused kernel for the whole GDN step, which is a
much larger change than an epilogue and is not attempted here.

## Submissions are already pipelined: item (8) has no headroom

Command-buffer reuse was the last configuration-level decode idea, so it was measured before being built. A temporary counter in
the queue submit path (`STRATA_VK_SUBMIT_STATS=1`, since removed) showed 35B-A3B decode issuing **~46 submissions per token at
~50 us of CPU time each** - 2.3 ms of the 26 ms token, 8.8% if it were serial. It is not serial: cutting the submissions by 17%
(1536 -> 1280 over the run, by raising `GGML_VK_MAX_NODES_PER_SUBMIT` to 512 or 4096) changed nothing:

| max nodes per submit | submissions | decode |
| --- | --- | --- |
| 100 (default) | 1 536 | 39.41 |
| 512 | 1 280 | 39.33 |
| 4096 | 1 280 | 39.12 |

The remaining submissions are forced by buffer syncs, the CPU already runs ahead of the GPU, and the 27B behaves the same
(7.18 at 4096 against 7.24-7.31 default). So there is nothing to win by recording command buffers per graph shape, and the item
closes as measured.

## Host-resident experts: upstream already has the mechanism, the engine now exposes it

The paging problem (the 62-67 GB packs against 47.6 GB of RAM) does not need a new loader: llama.cpp already carries an expert
cache. `tensor_buft_overrides` can keep the expert tensors in host memory (the same regex `common.h` uses for `--cpu-moe`,
`\\.ffn_(up|down|gate|gate_up)_(ch|)exps`), and `llama_context_params::moe_cache_size` / `moe_cache_cpu_share` keep a bounded
device cache of the experts a token actually uses. The engine now passes all of it through: `--cpu-moe`, `--n-cpu-moe N`,
`--moe-cache-size MiB`, `--moe-cache-cpu-share F`.

Measured on 35B-A3B and the Q2_0 pack:

| run | result |
| --- | --- |
| 35B-A3B decode, all experts on the device | 39.7 tok/s |
| 35B-A3B decode, `--cpu-moe --moe-cache-size 2048` | **22.2 tok/s** |
| Q2_0 pack pp64, all experts on the device | 46.3 tok/s, load 64.7 s |
| Q2_0 pack pp64, `--cpu-moe --moe-cache-size 4096` | 41.4 tok/s, load 85.1 s |

So streaming experts costs 44% of MoE decode here and 10% of the low-bit pack's prefill, and it does **not** by itself free the
Q2 pack: free physical memory still bottoms out (1 MB), because the dense part of a 67 GB pack is already larger than RAM. The
free-memory metric is also the wrong probe for this mode - with experts in host memory most of that memory is evictable page
cache, which free-memory counters do not distinguish - so the next measurement for this mode has to sample the disk-read rate
and the page-fault rate, not free RAM.

Token equality for `--cpu-moe` is verified on 35B-A3B (`--cpu-moe --moe-cache-size 1024`, four tokens): identical tokens
against both the CPU reference and the all-device Vulkan path, so the upload-on-miss path computes the same thing.

### Host-resident experts on the 67 GB pack, measured with the right probe

Free physical memory was the wrong probe (host-resident experts are evictable page cache), so the same run was repeated with
`\Memory\Pages/sec` and `\PhysicalDisk(_Total)\Disk Read Bytes/sec` sampled once a second:

| Q2_0 pack, pp64 | all experts on the device | `--cpu-moe --moe-cache-size 4096` |
| --- | --- | --- |
| prefill | 46.9 tok/s | 48.2 tok/s |
| disk read, peak | 3 755 MB/s | **2 237 MB/s** |
| disk read, median | 675 MB/s | 1 049 MB/s |
| pages/sec, peak | 258 153 | 332 800 |
| load time | 64.7 s | 85.1 s |

So the mode lowers the worst disk burst (by ~40%) but not the sustained rate, leaves prefill inside the noise, and costs 30% more
load time. It is a mitigation for peak stutter, not a cure: the pack's dense half alone exceeds RAM, and no expert placement
changes that.

## Decode already demands more than the streaming ceiling, so the shader micro-optimisations left have no room

The last TPS items on the list (matvec shared-memory activation residency, further coop tuning) presume the decode kernels are
leaving bandwidth on the table. The arithmetic says they are not:

| pack | weights | decode | implied read rate |
| --- | --- | --- | --- |
| 27B Q4_0 | 14.46 GiB | 7.47 tok/s | **116 GB/s** |
| 35B-A3B | 18.36 GiB | 39.6 tok/s | 781 GB/s if every weight were read - it is not, 8 of 256 experts are |

The 27B reads its whole weight set once per token, so its implied rate is 116 GB/s of pure read. The port's streaming figure
(120 GB/s measured, 108 GB/s for a read+write chain) is the same number: decode is **at the ceiling**, which means the activation
re-reads a residency patch would remove are not a measurable fraction of the traffic. That item is closed as not worth doing,
joining the ones already measured out (persistent kernels ~2%, command-buffer reuse 0%, int8 cooperative -50%, double buffering
-25% or impossible).

What is left on the roadmap is either under one percent (the GDN fusions, the CONCAT+CPY pair) or not a TPS item at all (the
chunked-f32 accuracy option, the direct-graph zero-output bug, the 0.25 split crash the owner deferred). The only measured way to
go beyond one device's ceiling is concurrency: two engines reach 53.8 tok/s aggregate on 35B-A3B against 39.3 alone (HYBRID.md).

## The cooperative kernel cannot be made oracle-certifiable here (chunked f32 included)

Item (11) still lists "chunked-f32 accumulation" as the way to make the cooperative matrix pass the strict element-wise oracle. It
cannot work on this device, for two independent reasons:

* the hardware has **no f32-accumulate cooperative matrix** (`f32acc 0`): `coopMatMulAdd` can only accumulate in f16, which is why
  the tile list is the `*_f16acc` family in the first place;
* the f32 partial buffer that chunking would need does not fit: a 64x64 f32 accumulator is 16 KiB, and the staging already takes
  20 480 of the 32 768 bytes (`ADRENO.md`, double buffering). A smaller partial buffer would change the tile, and the 32x32 tile
  measures 57-77% slower.

So the coop path stays "token-identical on the five packs, and the oracle passes only because illegal shapes fall back to dp4a".
That is the complete statement of what is certified and what is not, and no shader change records it more strongly.

The inert `ssm_conv_window` pipeline was removed with this: it could never match (the graph interleaves the nodes) and only cost a
shader compile at startup. The design stays in GDN-FUSION.md for a future CONCAT+CPY two-node version, which would need a simpler
kernel.

## The direct-graph cooperative zero output: what is ruled out, and the workaround

Repro (no model needed):

    STRATA_VK_F16ACC=1 build/bin/mm-probe -b Vulkan0 -t q4_0 -m 512 -n 64 -k 512 --verify
    -> nmse 1.000e+00, 32768 of 32768 elements outside 1e-3, i.e. the result buffer is all zeros

Everything that reaches the GPU was compared against the scheduler path, which computes the same kernel correctly:

| | direct (zeros) | scheduler (correct, 27B pp64) |
| --- | --- | --- |
| pipeline | matmul_coop_f16_f16acc_aligned_1 | same |
| grid | (8, 1, 1) = 512/64 | (160, 1, 1) = 10240/64 |
| pc m, n, k | 512, 64, 512 | 10240, 64, 5120 |
| pc strides a/b/d | 512, 512, 512 | 5120, 5120, 10240 |
| pc batch strides | 262144, 32768, 32768 | 52428800, 327680, 655360 |
| pc k_split, padded_n | 512, 64 | 5120, 64 |
| descriptor A | buffer set, offset 0, range 147456 | buffer set, offset 1043005440, range 29491200 |
| descriptor B | buffer set | buffer set |

Every field is internally consistent with the shapes, the tile divides the grid exactly, and the bindings are real buffers with
sane ranges - yet the direct path writes nothing. So the defect is below the API layer this port can inspect: either the
descriptor set the backend builds for a standalone graph differs from the one it builds under the scheduler, or the shader exits
before its stores for a reason tied to that dispatch. Settling it needs a GPU capture (RenderDoc) or a shader-side counter, neither
of which this port has.

**Workaround**, and the reason the engine is unaffected: the scheduler path (every model run) computes correctly, so only
direct-graph users - `mm-probe`, `test-backend-ops` - meet the zeros, and `GGML_VK_DISABLE_COOPMAT=1` makes them take the dp4a
path and pass. The oracle therefore certifies the cooperative tile only through the shape gate that guarantees the model shapes
reach it, exactly as `ADRENO.md` already states.

## Chunked f32 cannot certify the cooperative kernel: the floor is the f16 accumulator itself

Chunking was the remaining idea for item (4): accumulate a few hundred k elements in the cooperative accumulator, store the partial
as f32, zero it, repeat, and add the partials at the end. It cannot work, for a reason that has nothing to do with the chunk size:

* the cooperative accumulator is f16 (the device has no f32-accumulate cooperative matrix), so **every partial is rounded to an
  11-bit significand before it is ever stored**. The per-element relative error of that rounding is 2^-11 ~ 4.9e-4, and the
  mean-square relative error of a sum of such rounded values is ~(4.9e-4)^2 / 3 ~ 8e-8 - right at the oracle's 1e-7 bound
  *before any accumulation error is added*, and the accumulation over k adds more (the measured figures were 11.3x the f16
  reference error at k=32 and 138.6x at k=5120);
* a shared-memory f32 partial buffer would fit (the existing `coopmat_stage` is exactly that, 2 KiB for TM x TN x warps), so the
  plumbing was never the obstacle - the precision of the partial is.

So the cooperative path is certified by **token equality on the five packs**, not by the element-wise oracle, and no amount of
chunking changes that. The only exact path on this device is the integer one (dp4a / int8 cooperative, whose int32 accumulation is
exact) - and the int8 cooperative kernel measured at half the f16-accumulate rate (item 7), so the fast path stays f16.

Item (4) is closed with this argument; the two-extra-dispatch design it was going to need is not built.

## Matvec activation residency (item 7): the kernel is already at 97% of the read ceiling

The idea was to keep the activation vector resident in shared memory so the matvec does not re-read it per row block. The
efficiency says there is nothing to recover:

| model | weight bytes | decode | implied read rate | ceiling |
| --- | --- | --- | --- | --- |
| 27B Q4_0 | 14.46 GiB | 7.51 tok/s | 116 GB/s | ~120 GB/s (streaming, ADRENO) |
| 35B-A3B | 18.36 GiB total, ~2.5 GB active | 39.3 tok/s | ~99 GB/s of active weights | same |

116/120 is 97% of what a pure read stream achieves on this part, and the activation traffic a residency patch would remove is a few
MiB per token against 14.5 GiB of weights - under 0.1% of the bytes moved. A residency change would therefore not move the number
by even a tenth of a percent, while adding a workgroup-wide staging step to every matvec.

Item (7) is closed as measured-not-worth-doing. What is left in the GDN area is item (5) - the CONCAT+CPY pair, which is 30
dispatches per token (about 0.3%, since the copy itself is only 96 KiB per layer) and needs a new four-binding concat pipeline plus
its wiring - and item (6), GDN stages 2 and 3, roughly 1.2% together but each needing a new graph op with a CPU reference.

### The coop zero output is real, and it is specific to a single-op graph

The probe itself was cleared first. `mm-probe --verify` on the default (int-dot) path reports
`nmse 1.981e-07 (ratio 2.0)`, so its CPU reference and the device output come from the same deterministic inputs and the comparison
works. With `STRATA_VK_F16ACC=1` the same probe reports `nmse 1.000e+00, 32768 of 32768 outside 1e-3` - the device output is zero
while the reference is not, and the same happens with async submission disabled.

Three further facts narrow it:

* `test-backend-ops` MUL_MAT passes 38/38 in both precision modes, but its shapes fail the cooperative gate (`k >= 256`,
  `n_mats >= 16`), so those 38 cases never dispatch the cooperative pipeline - they certify the int-dot path, not this one;
* the engine (27B, 35B-A3B) runs the same cooperative pipeline inside a model graph and its tokens are identical to the CPU path;
* therefore the defect is specific to a **single-op graph**: the cooperative dispatch writes nothing when the graph holds one node,
  and computes correctly when the same pipeline runs among many nodes.

That points at the per-graph state the cooperative path depends on (the f16 reformat of the right-hand operand and its "last
pipeline used" bookkeeping), which a single-node graph never primes. The workaround stands (`GGML_VK_DISABLE_COOPMAT=1` for
single-op tools, or run through `test-backend-ops` with shapes that reach the gate).

## Decode emits about three full pipeline barriers per dispatch

Instrumenting `ggml_vk_sync_buffers` (a full pipeline barrier: same-stage to same-stage, shader read/write plus transfer on both
sides, no buffer ranges - i.e. a cache flush) gives **5 897 calls for a 4-token decode run after the prompt**, of which 978 belong to
the prompt itself: **4 919 for 4 tokens = ~1 230 barriers per token** against the 398 dispatches per token measured with the perf
logger. Three barriers per dispatch is the strongest candidate for the "waiting" that every micro-fusion in this session failed to
reclaim: the per-op times the perf logger reports include those barrier waits, which is why removing dispatches did not move the
decode rate.

The redundant-barrier counter (skip a barrier when nothing was recorded since the previous one) could not be made reliable in the
overlay (the counter lives in the .cpp while the discrete dispatch site is a template in ggml-vulkan-common.h, which the overlay does
not carry), so the split between "needed" and "duplicate" barriers is still unmeasured. The next step is a census by call site
(the ~30 `ggml_vk_sync_buffers` callers plus the buffer-upload helpers) to see whether decode is re-uploading a tensor per layer -
a per-layer upload would explain both the three barriers per dispatch and the wasted time.

### The graph barriers are dependency-driven, not redundant: a per-call-site census

A census keyed by source line (`__LINE__` folded into the call, so no call site had to be edited) settles where the barriers come
from. Over a prompt plus two decode tokens the census totals 3 367, and after the prompt the same ~1 230 per token holds. The
breakdown:

| call site | calls |
| --- | --- |
| the graph loop's `need_sync` branch (ggml-vulkan.cpp:13130) | 2 450 per two tokens |
| the f16 reformat helper (ggml-vulkan.cpp:6797) | 532 |
| two buffer-copy helpers (8252, 9325) | 78 + 10 |
| others | 60 + 3 |

So the barriers come from the graph loop, and that loop **already** tracks dependencies precisely: `overlaps_unsynced()` compares
byte ranges of the just-written and about-to-be-read buffers, and the barrier is only emitted when they intersect. A deferred-barrier
experiment (skip the flush when the current node is a view, reshape, permute or transpose, which record no commands) was built and
measured: the barrier count did not move by a single one, because `need_sync` never fires for those nodes. The barriers are real
dependencies over shared state buffers, not waste.

Two consequences, both honest negatives:

* the "1 230 barriers per token" figure is the backend doing its job, not a bug to remove;
* scheduling independent ops onto another queue has nothing to overlap here: the graph is a dependency chain over shared state, and
  `ggml_backend_vk_graph_compute` already submits it as one command buffer with per-dependency barriers.

The remaining lever would be **buffer-scoped barriers** - `pipelineBarrier` with `vk::BufferMemoryBarrier` entries for just the
overlapping ranges instead of global shader-read/write access flags, which flush every cache. That is a change to
`ggml_vk_sync_buffers` and to every caller (the overlapping buffers have to be passed in), and its upside is unmeasured.

### What one barrier costs, and why buffer-scoped barriers are not worth the surgery

The marginal cost was measured directly, by emitting `N` extra empty barriers per graph-node iteration (always safe: an extra barrier
only over-synchronises) and timing decode. The run iterates **21 441 nodes** for a prompt plus 16 tokens, and every iteration already
pays a barrier, so the slope is a clean per-barrier number:

| extra barriers per node | decode |
| --- | --- |
| 0 | 404.0 ms (39.60 tok/s) |
| 1 | 415.8 ms (38.48) |
| 2 | 422.3 ms (37.89) |
| 4 | 417.4 ms (38.33) - noise dominates above one extra |

One extra barrier per iteration costs 11.8 ms over 21 441 iterations = **about 0.55 us per barrier** (the two-barrier point gives
0.43 us, so the range is 0.4-0.55 us). Applying that to the ~1 230 barriers a decode token already emits gives **~0.5-0.7 ms of a
25 ms token, i.e. 2.0-2.7%**.

That is the entire ceiling for making barriers cheaper: buffer-scoped barriers (`vk::BufferMemoryBarrier` over the overlapping ranges
instead of global shader read/write access flags) would have to drive the cost to zero to recover those 2%, and they cannot - the
barriers still have to order the same dependencies. A realistic capture is half of it, around 1%, in exchange for changing
`ggml_vk_sync_buffers` and its roughly 30 call sites. **Not implemented**: the measured ceiling sits below the 2% bar this fork uses
for a backend change.

Instrumentation removed; the default path was never touched (the knob defaulted to zero extra barriers).

