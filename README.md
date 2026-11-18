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
| Tools | `port/tools/` | fetch and build scripts, the measurement harnesses (`batch-serve.py`, `energy-ab.py`, `two-serve.py`), the gates (`check-encoding.py`, `audit-docs.py`, `check-pin.py`) |
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
| **Vulkan**, prefill pp512 | **43-50 tok/s** | default settings; **+211%** over the same build before 2026-11-18 |
| **Vulkan**, decode, one stream | 10.2 tok/s | ~37.7 GB/s of weight traffic |
| **Vulkan**, decode, 32 request slots | **26.7 tok/s aggregate** | 2.73x; **103 GB/s = 95%** of this GPU's 108 GB/s ceiling |
| **CPU** (`--gpu cpu`, big-core class) | 10.5-11.3 tok/s decode, 21 tok/s prefill | still the fastest single stream for a short reply |
| **Vulkan + CPU as two processes** | **29.1 tok/s aggregate** (25.3 + 3.8) | the CPU half collapses to 37% of solo: it is instruction-bound, not bandwidth-bound |
| read bandwidth, CPU | **154.6 GB/s** (14 threads) | 68% of the 228 GB/s specification; 4 threads already reach 147 |
| read bandwidth, Vulkan | ~108 GB/s | |
| read bandwidth, both at once | 177 GB/s | 78% of specification |

Two things this table is saying. **Prefill is now 2.4x the CPU** on default settings, and that came from one change:
the int-dot mat-mat takes its shader from the **fp32 module** while the mat-vec and elementwise shaders stay on the
fp16 one, so the prefill half of `--shader-family fp32` is free and the decode cost of that switch is not paid.
**Decode is at the memory wall when batched** - 103 of 108 GB/s - so batching cannot be pushed further, and the
remaining gap to a higher number has to come from the CPU side (see Status).

The cooperative matrix path is **not** a faster correct option yet, and the reasons are measured:

- The f16-accumulate cooperative tile used to report 221 tok/s at pp512 with logits correlating 0.02 with the CPU.
  That kernel's `WN` was 32 against a `TN` of 64, so `cms_per_col = WN/TN` truncated to zero and the MMA loop, the
  epilogue and the stores compiled to nothing: it was a staging-only kernel, not an approximation. The tile is now
  built from the device's own matrix shape and it computes, but the **grouped** (`MUL_MAT_ID`) variant then writes
  `-inf` into exactly the first 64 elements of a neighbouring buffer at model shapes. The **dense** variant is
  correct (identical argmax, bit-finite) and measures **neutral**, so it buys nothing on its own.
- `STRATA_VK_INT8_COOPMAT=1`, which the docs used to describe as "exact but not faster", never created a pipeline at
  all: the creation block sits under `coopmat_support`, which is false on this device because it has no
  f32-accumulate matrix. The exact int8 matrix core (64x64x32, int32 accumulate) is therefore still unused.

[PERFORMANCE.md](port/docs/PERFORMANCE.md) has every number with the run that produced it, including what has been
ruled out.

## Status

- **Correct first.** The default paths are verified against the CPU backend: same argmax, logits correlating 0.9984 on
  this pack, and a 32-slot batched stream that is 64/64 tokens identical to a solo run. The 0.999 in the gate below is
  the level the 35B-A3B pack reaches; on Gyro-S the two backends sit at 0.9984 and the argmax still agrees.
- **Prefill**: the default path is the fastest correct one on this machine, 43-50 tok/s, and it is limited by the
  int8-dot tiles. The cooperative matrix is the way past it and is described above: correct on the dense half,
  broken on the grouped half, worth 145 tok/s when both are on.
- **Decode**: one stream is memory- and instruction-latency-bound (10.2 tok/s, 37.7 GB/s); batched it reaches
  103 GB/s, which is the GPU's ceiling. Every route beyond that has been measured and closed: reducing the per-token
  bytes breaks the output (Q4_K on attention correlates 0.978 and changes the token at position 3; even Q5_0/Q5_K on
  the over-precision tensors correlates 0.991 and changes it at position 7), the CPU+GPU split adds 9% and not more
  because the CPU side cannot turn its share of bandwidth into tokens, and the CPU side's own bottleneck is the
  trellis dot product at 1.78 instructions per weight.
- **Batching**: `--parallel N` / `--batch-slots N` runs N requests in one process on one copy of the weights, and it
  does pay: 2.73x at 32 slots. Note 8 slots is a pit (`mul_mat_vec_max_cols = 8`), 48 slots measured a dip, and the
  curve is not monotone. `serve/server.py` is not wired to it yet.
- **NPU**: the Hexagon path builds, but no DSP session opens on this driver (`remote_handle64_open` returns
  `AEE_ERPC 0x200` for every module name tried), and every placement that could be measured before that was slower
  than the GPU. Reviving it needs a QNN backend, not a fixed URI.
  [NPU-FEASIBILITY.md](port/docs/features/NPU-FEASIBILITY.md) has the measurements.
- **Pictures**: the engine reads text only; the app's image path needs upstream's `strata-vision` encoder.
- **DeepSeek-V4.1**: not supported by this component, and the engine says so on the serve line.

## Docs

[`port/docs/README.md`](port/docs/README.md) is the index. Start from what you are:

| You are | Read |
| --- | --- |
| **End user**: just run it | the quick start above, then [USAGE.md](port/docs/USAGE.md) |
| **Contributor**: make it faster | [PERFORMANCE.md](port/docs/PERFORMANCE.md) (current numbers, what was measured and ruled out) and [BACKEND-OPTIONS.md](port/docs/BACKEND-OPTIONS.md) (what is left) |
| **Maintainer**: change the vendored tree, the overlay, the gates | [MAINTAINER.md](port/docs/MAINTAINER.md), [UPSTREAM-PATCHES.md](port/docs/UPSTREAM-PATCHES.md) |
| **Curious about this machine** | [ADRENO.md](port/docs/ADRENO.md) (measured Adreno X2-90 facts) |

The one rule in `port/docs`: **no number without a measurement on this machine**. Where something is a hypothesis it
says so.

## Gates

Every change passes all of these before it is committed; the commands are in [USAGE.md §5](port/docs/USAGE.md):

1. CPU and Vulkan logits agree: correlation >= 0.999 and the same argmax (`--dump-logits` on both backends).
2. `test-backend-ops -o MUL_MAT -b Vulkan0`: zero failures.
3. `check-encoding.py`: zero files with encoding damage. `audit-docs.py`: zero document problems.
4. `check-pin.py`: the overlay still matches `third_party/llama.cpp`.

[GitHub Sponsors](https://github.com/sponsors/Niko1221) · [Buy me a coffee](https://buymeacoffee.com/strataengine) ·
[MIT license](LICENSE) · [Upstream Strata](https://github.com/Niko1221/Strata)
