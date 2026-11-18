<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h3 align="center">Strata on Windows on ARM.</h3>
<p align="center">A Vulkan engine for Snapdragon X2 and Adreno X2, built correctness first.</p>

<p align="center">
  <a href="#what-this-branch-is">What it is</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#measured-on-this-machine">Measured</a> ·
  <a href="#status">Status</a> ·
  <a href="#docs">Docs</a> ·
  <a href="#gates">Gates</a>
</p>

<p align="center"><sub><b>English</b> · <a href="README.zh-CN.md">简体中文</a></sub></p>

**This is the `windows-arm64-vulkan` branch of [Strata](https://github.com/Niko1221/Strata).** Upstream's engine is a
hand-written CUDA / HIP / SYCL kernel set and none of it runs on an Adreno GPU, so this branch carries one addition:
**`strata-vk`**, a Vulkan engine component that speaks Strata's `--serve` protocol, plus the overlay that teaches the
vendored `llama.cpp` the model families and the low-bit types the packs use. The front end around it (the desktop app,
the Terminal, `setup.py`, the Python server) is upstream's and drives this engine without changes.

Everything specific to this branch lives in `port/`. The port was developed and measured on one machine: a Snapdragon X2
Elite Extreme laptop. Treat its numbers as that machine's, not as a hardware requirement.

## What this branch is

| Part | Where | What it does |
| --- | --- | --- |
| The engine | `port/engine/` | `strata-vk`: loading and placement, prefix reuse, sampling, the token loop, the `--serve` line protocol, `--parallel` request slots |
| The overlay | `port/llama-overlay-vanilla/` | the tracked diff against vanilla `llama.cpp` at the pinned commit: model families (`qwen35`, `qwen35moe`, `qwen4exp`, trellis packs), low-bit types and their Vulkan kernels, the MTP/NextN graft, the loader fallbacks |
| CPU and GPU in one build | same build | the Vulkan backend and the CPU backend (with the repack kernels) are both in `strata-vk`, so either can be selected with `--gpu`, and `test-backend-ops` can compare them |
| The picture encoder | `port/engine/strata-vision.cpp` | `strata-vision`: llama.cpp's mtmd on an mmproj file, speaking the front end's `ENC <image> <out>` line and writing the `.sve` record the engine reads; CPU or Vulkan, chosen with `--gpu` |
| Tools | `port/tools/` | fetch and build scripts, the measurement harnesses (`batch-serve.py`, `energy-ab.py`, `two-serve.py`, `gguf-tokenizer.py`), the gates (`check-encoding.py`, `audit-docs.py`, `check-pin.py`) |
| Docs | `port/docs/` | [index](port/docs/README.md): usage, performance, backend options, maintainer notes, upstream patches, device facts |

## Quick start

```powershell
# 1) fetch the pinned llama.cpp and compose the overlay into third_party/llama.cpp
powershell -NoProfile -ExecutionPolicy Bypass -File port/tools/fetch-llama.ps1
# 2) build the engine (and, optionally, the backend correctness oracle)
powershell -NoProfile -ExecutionPolicy Bypass -File port/tools/build-engine.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File port/tools/build-engine.ps1 -Target test-backend-ops
# 3) run
build\bin\strata-vk.exe --native C:\models\pack.gguf --prompt "Hello" --n 32
```

The engine needs the **ARM64** Vulkan loader beside the exe; `build-engine.ps1` stages it. Without it `ggml-vulkan.dll`
fails to load, the engine silently registers the CPU backend alone, and the output is slow and wrong. `--geometry` must
list `Vulkan0`.

Flags, measurement switches and the full gate list: [port/docs/USAGE.md](port/docs/USAGE.md).

## Measured on this machine

Snapdragon X2 Elite Extreme (18 Oryon cores, 47.6 GB unified memory, 3-channel LPDDR5X at 228 GB/s specified, Adreno
X2-90 on Vulkan 1.4, driver `30.0.228.10000`), Windows 11 ARM64. Model: `Gyro-S-TQ1_0.gguf` (54.47 GiB, `qwen4exp`,
trellis experts, 0.469 GiB of experts per token out of 3.86 GB read per token). Decode is `--bench-tg 16`, prefill
`--bench-pp 512`. Medians of repeated runs, same binary, one machine.

| what | measured | notes |
| --- | --- | --- |
| **Vulkan**, prefill pp512 | **85.55 tok/s** | default settings; **+101%** over the same build with `STRATA_VK_F16ACC=0` (42.51), measured in pairs on 2026-11-19 |
| **Vulkan**, decode, one stream | 10.25 tok/s | ~37.9 GB/s of weight traffic; unchanged by the cooperative path |
| **Vulkan**, decode, 32 request slots | **31.09 tok/s aggregate** | 2.4x; the same build without the cooperative path is 25.47 (**+22%**) |
| **CPU** (`--gpu cpu`, big-core class) | 10.5-11.3 tok/s decode, 21 tok/s prefill | still the fastest single stream for a short reply |
| **Vulkan + CPU as two processes** | 29.1 tok/s aggregate (25.3 + 3.8) | measured on the dp4a kernel, not re-measured since |
| read bandwidth, CPU | **154.6 GB/s** (14 threads) | 68% of the 228 GB/s specification; 4 threads already reach 147 |
| read bandwidth, Vulkan | ~108 GB/s | |
| read bandwidth, both at once | 177 GB/s | 78% of specification |

Prefill came from the cooperative matrix, and the reason it is only now the default is a shader-selection bug, not a
tile problem: the trellis types (TQK6, TQK7, TQ2_T, PTQ1_0) have **no `MmTypeA` branch** in the shared quant shader, so
the pipelines built for them from that shader never wrote the staged `A` block and the MMA read uninitialised shared
memory - that was the model-wide `-inf`. Each trellis type now gets its own SPIR-V, which is what took pp512 from
42.51 to 85.55 tok/s and 32 slots from 25.47 to 31.09. The oracle never caught it because its trellis shapes are
below the `k >= 256` cooperative gate.

Two things this table is still saying. **Decode is near the memory wall when batched** - 32 slots move about as much
weight traffic per step as the GPU can stream - so batching has limited room left, and the remaining gap on a single
stream has to come from the trellis expert matvec, which is 52% of a token and which the cooperative path cannot
reach because decode is one column. **The 32-slot figure is a kernel result, not a hardware one**: it rose 22% with a
kernel change, so the old "95% of the bandwidth ceiling" reading was measuring the dp4a kernel, not the memory system.

The cooperative matrix is now the correct fast path. What is left of it is measured:

- The tile is 64x64 because that is what this device reports; `128x128` fails pipeline creation and `32x32` measures
  57-77% slower.
- Dense cooperative stays off: 80.98 against 85.55 tok/s on the same pack, a 5% loss.
- About 84% of the cooperative MMA columns are empty: 512 tokens select all 512 experts, so an expert has ~10
  rows in a 64-column tile. It is a step function - 20 rows is still one tile - which is why pp1024 and pp2048
  measure neutral against pp512. Both ways out are measured shut: the trellis types have no int8 matmul shader to
  hang a narrow dp4a tile on, and the driver rejects a 32x32 matrix core outright (`Invalid mask for Layout[]`).
  85-90 tok/s is this kernel set's ceiling; going past it needs a different weight layout.
  [PERFORMANCE.md](port/docs/PERFORMANCE.md) §5.2 has the evidence.

[PERFORMANCE.md](port/docs/PERFORMANCE.md) has every number with the run that produced it, including what has been
ruled out.

## Status

- **Correct first.** The default paths are verified against the CPU backend: same argmax, logits correlating 0.9982 on
  this pack, and a 32-token continuation that is token-for-token identical to the recorded reference stream. The 0.999
  in the gate below is the level the 35B-A3B pack reaches; on Gyro-S the two backends sit at 0.9982 and the argmax
  still agrees.
- **Prefill**: the default path is now the cooperative matrix, 85.55 tok/s against 42.51 for the same build with it
  off. The trellis types had been built from a shader that cannot decode them, which is what produced the `-inf`;
  each gets its own SPIR-V now. What is left of prefill is the 84% of empty cooperative rows at pp512 (see above).
- **Decode**: one stream is memory- and instruction-latency-bound (10.25 tok/s, 37.9 GB/s of weight traffic), and the
  cooperative path cannot help it because a decode step carries one column. Every byte-reduction route has been
  measured and closed: Q4_K on attention correlates 0.978 and changes the token at position 3; even Q5_0/Q5_K on the
  over-precision tensors correlates 0.991 and changes it at position 7. On the CPU side the bottleneck is the trellis
  dot product, which a measured 1.33x kernel rewrite would only partly reach (its real rate is 4.6x below the
  microbenchmark, so the step is not purely instruction-bound).
- **Batching**: `--parallel N` / `--batch-slots N` runs N requests in one process on one copy of the weights, and it
  does pay: 30.51 tok/s aggregate at 32 slots and **33.62-36.65 at 64** (3.3-3.6x a single stream), measured on the
  cooperative path. The curve is monotone to 64 slots now - the 48-slot dip belonged to the dp4a kernel - and 96 slots fails to
  allocate on 47.6 GB. 8 slots is still a pit (`mul_mat_vec_max_cols = 8`). Either flag now creates the slots, so
  `serve/server.py` can actually reach them (`batch_slots=64`), and 64 slots produce a slot-0 stream identical to a
  solo run token for token.
- **NPU**: the Hexagon path builds, but no DSP session opens on this driver (`remote_handle64_open` returns
  `AEE_ERPC 0x200` for every module name tried), and every placement that could be measured before that was slower
  than the GPU. Reviving it needs a QNN backend, not a fixed URI.
  [NPU-FEASIBILITY.md](port/docs/features/NPU-FEASIBILITY.md) has the measurements.
- **Pictures**: the path is built and has been driven end to end - `port/engine/strata-vision.cpp` is the encoder
  (llama.cpp's mtmd on the mmproj file, CPU or Vulkan), the engine takes `--vision` and a `GENI` line that names the
  embeddings file, and `serve/server.py` spawns it by itself. What is verified is the plumbing, not the answers: a
  picture request returns 200 with its image tokens in the prompt, and the encoder measures 5.2 s per picture on the
  CPU and 1.6 s on the GPU with cosine 0.99997 between them. What is **not** shown is image understanding: this
  pack's chat answers are degraded on both backends alike (Vulkan and CPU emit the same ids for the same prompt), so
  a picture question cannot be judged good by the answer it gets. [VISION.md](port/docs/features/VISION.md) says
  exactly what was measured and what was not.
- **Batching through pictures**: `BGENI` (the multi-slot spelling) answers `ERR` for pictures; only the one-at-a-time
  path carries them.
- **DeepSeek-V4.1**: not supported by this component, and the engine says so on the serve line.

## Docs

[`port/docs/README.md`](port/docs/README.md) is the index. Start from what you are:

| You are | Read |
| --- | --- |
| **End user**: just run it | the quick start above, then [USAGE.md](port/docs/USAGE.md) |
| **Contributor**: make it faster | [PERFORMANCE.md](port/docs/PERFORMANCE.md) (current numbers, what was measured and ruled out) and [BACKEND-OPTIONS.md](port/docs/BACKEND-OPTIONS.md) (what is left) |
| **Contributor**: pictures | [features/VISION.md](port/docs/features/VISION.md) (the `.sve` contract, the three backends, what was measured and what was not) |
| **Anyone**: what is known to be broken | [KNOWN-ISSUES.md](port/docs/KNOWN-ISSUES.md) |
| **Maintainer**: change the vendored tree, the overlay, the gates | [MAINTAINER.md](port/docs/MAINTAINER.md), [UPSTREAM-PATCHES.md](port/docs/UPSTREAM-PATCHES.md) |
| **Curious about this machine** | [ADRENO.md](port/docs/ADRENO.md) (measured Adreno X2-90 facts) |

The one rule in `port/docs`: **no number without a measurement on this machine**. Where something is a hypothesis it
says so.

## Gates

Every change passes all of these before it is committed; the commands are in [USAGE.md §5](port/docs/USAGE.md):

1. CPU and Vulkan agree: correlation >= 0.999 on the logits and the same argmax (`--dump-logits` on both backends),
   **and the same generated ids for a real (multi-token) prompt** run once with `--gpu vulkan` and once with
   `--gpu cpu`. The second half is the cheap way to tell "the model did that" from "the port did that"; it is how the
   poor chat answers on this pack were traced to the pack and not here.
2. `test-backend-ops -o MUL_MAT -b Vulkan0`: zero failures.
3. `check-encoding.py`: zero files with encoding damage. `audit-docs.py`: zero document problems.
4. `check-pin.py`: the overlay still matches `third_party/llama.cpp`.

[GitHub Sponsors](https://github.com/sponsors/Niko1221) · [Buy me a coffee](https://buymeacoffee.com/strataengine) ·
[MIT license](LICENSE) · [Upstream Strata](https://github.com/Niko1221/Strata)
