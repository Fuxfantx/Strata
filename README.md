<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="app/host/public/brand/strata-lockup-dark.svg">
    <img src="app/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h3 align="center">Strata on Windows on ARM.</h3>
<p align="center">A Vulkan engine for Snapdragon X2 and Adreno X2, correctness first.</p>

<p align="center">
  <a href="#what-this-branch-is">What it is</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#measured-on-this-machine">Measured</a> ·
  <a href="#status">Status</a> ·
  <a href="#docs">Docs</a> ·
  <a href="#gates">Gates</a>
</p>

<p align="center"><sub><b>English</b> · <a href="README.zh-CN.md">简体中文</a></sub></p>

**This is the `windows-arm64-vulkan` branch of [Strata](https://github.com/Niko1221/Strata).**
Upstream's engine is a hand-written CUDA / HIP / SYCL kernel set, and none of it runs on an Adreno GPU. This branch
therefore carries one addition: **`strata-vk`**, a Vulkan engine component that speaks Strata's `--serve` protocol,
plus the overlay that teaches the vendored `llama.cpp` the model families and the low-bit types these packs use.
The front end around it (the desktop app, the terminal, `setup.py`, the Python server) is upstream's and drives this
engine without changes.

Everything specific to this branch lives in `port/`. The port was developed and measured on one machine, a Snapdragon
X2 Elite Extreme laptop; treat its numbers as that machine's, not as a hardware requirement.

## What this branch is

| Part | Where | What it does |
| --- | --- | --- |
| The engine | `port/engine/` | `strata-vk`: loading and placement, prefix reuse, sampling, the token loop, the `--serve` line protocol, `--parallel` request slots |
| The overlay | `port/llama-overlay-vanilla/` | the tracked diff against vanilla `llama.cpp` at the pinned commit: model families (`qwen35`, `qwen35moe`, `qwen4exp`, trellis packs), low-bit types and their Vulkan kernels, the MTP/NextN graft, the loader fallbacks |
| CPU and GPU in one build | same build | the Vulkan backend and the CPU backend (with the repack kernels) are both in `strata-vk`, so either can be selected with `--gpu`, and `test-backend-ops` can compare them |
| The picture encoder | `port/engine/strata-vision.cpp` | `strata-vision`: llama.cpp's mtmd on an mmproj file, speaking the front end's `ENC <image> <out>` line and writing the `.sve` record the engine reads; CPU or Vulkan, chosen with `--gpu` |
| Tools | `port/tools/` | fetch and build scripts, the measurement harnesses, the gates (`check-encoding.py`, `audit-docs.py`, `check-pin.py`) |
| Docs | `port/docs/` | [index](port/docs/README.md): usage, performance, maintainer notes, shader conventions, upstream patches, device facts |

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

The engine needs the **ARM64** Vulkan loader beside the exe; `build-engine.ps1` stages it. Without it
`ggml-vulkan.dll` fails to load, the engine silently registers the CPU backend alone, and the output is slow and
wrong. `--geometry` must list `Vulkan0`.

Flags, measurement switches and the full gate list: [port/docs/USAGE.md](port/docs/USAGE.md).

## Packs measured here

Three Qwen3.8-Flash-Next packs have been driven end to end on this branch. Others should load, but only these three
have measured numbers behind them.

<details>
<summary><b>Qwen3.8-Flash-Next Q2</b> — the 2-bit pack, the fastest of the three</summary>

| What | Value |
| --- | --- |
| Quantisation | `q2_0` for the expert and projection tensors, `f16` for attention projections |
| Weights on disk | about 41 GB, two shards |
| Measured decode | about 17-22 tok/s on one stream, depending on the window |
| Measured prefill | about 173 tok/s at 512 tokens, 155-183 at 2048 |

Per-family GPU time on the decode graph is dominated by `mul_mat_vec` and `mul_mat_id` for the experts; the
projections that stay in `f16` are what the prefill tile change below mostly helps.

</details>

<details>
<summary><b>Gyro-S</b> — trellis experts, lower memory, slower</summary>

Trellis packs (`TQK6`, `TQK7`, `TQ2_T`, `PTQ1_0`) each compile to their own SPIR-V. The per-type firmware was a
correctness fix before it was a speed one: a shared quant shader cannot decode the trellis types, and using one
produced a model-wide `-inf` that the correctness oracle did not catch, because its trellis shapes sit below the
`k >= 256` cooperative gate.

</details>

<details>
<summary><b>Gyro-M</b> — the same trellis family, larger</summary>

Measured for loading and for correctness; the numbers above are from the Q2 pack unless stated otherwise.

</details>

## Measured on this machine

Snapdragon X2 Elite Extreme (18 Oryon cores, 47.6 GB unified memory, Adreno X2-90 on Vulkan 1.4, driver
`30.0.228.10000`), Windows 11 ARM64.

| What | Measured | Notes |
| --- | --- | --- |
| **Vulkan**, prefill `--bench-pp 512` | **about 173 tok/s** | median of repeated runs; the 2048-token case measures 155-183 |
| **Vulkan**, decode, one stream, 8192 context | **17-22 tok/s** | the spread is the measurement window, not a setting |
| **Vulkan**, decode, 262144 context, same prompt | 21-22 tok/s | same stream as the 8192 case, token for token |
| read bandwidth, CPU | about 155 GB/s | 14 threads, 68% of the 228 GB/s specification |

Two cautions that come from measuring this machine for a long time:

- **Prefill readings drift.** The same binary on the same prompt can read 149 or 173 tok/s in adjacent rounds, so any
  prefill change has to be judged from interleaved rounds and a median, never from one run.
- **Decode readings are stable** (about 0.6% spread), so a decode change can be judged from two or three runs.

## Status

- **Correct first.** The default paths are verified against the CPU backend: the same argmax, logits correlating
  0.9982 on this pack, and a multi-token continuation that is token for token identical to the recorded reference
  stream.
- **Prefill** was the largest measured win of the last pass: the per-thread output tile of the float and the quantised
  matrix multiply is a startup setting, and moving it from `4x4` to `8x4` (float) and `8x8` (quantised) lifted
  prefill from about 123 to about 173 tok/s with byte-identical output. Reverting is a single environment
  variable ([USAGE.md](port/docs/USAGE.md)).
- **Decode** is bound by the weight traffic of a single column: about 1.7 GB per token on the Q2 pack. Every
  byte-reduction route measured so far changes the output text, and the port keeps text identical as a hard rule, so
  those routes are closed rather than pending.
- **Batching**: `--parallel N` runs N requests in one process on one copy of the weights. It helps; the numbers for
  this pack are not re-measured since the tile change, and the old 32/64-slot figures in `PERFORMANCE.md` are
  historical.
- **CPU + GPU together**: the engine has a `--split-cpu` layer split, and it is measured **slower** on this machine
  (about -41% at 15% of the layers on the CPU) as well as text-changing, so it is not the default and not recommended.
- **Pictures**: the plumbing is verified end to end; image understanding is not claimed. See
  [VISION.md](port/docs/features/VISION.md).
- **NPU**: the Hexagon path builds, but no DSP session opens on this driver. See
  [NPU.md](port/docs/features/NPU.md).
- **DeepSeek-V4.1**: not supported by this component, and the engine says so on the serve line.

## Docs

[`port/docs/README.md`](port/docs/README.md) is the index. Start from what you are:

| You are | Read |
| --- | --- |
| **End user**: just run it | the quick start above, then [USAGE.md](port/docs/USAGE.md) |
| **Contributor**: make it faster | [PERFORMANCE.md](port/docs/PERFORMANCE.md) (current numbers, methodology, what was measured and ruled out) |
| **Contributor**: pictures | [features/VISION.md](port/docs/features/VISION.md) |
| **Anyone**: what is known to be broken | [KNOWN-ISSUES.md](port/docs/KNOWN-ISSUES.md) |
| **Maintainer**: change the vendored tree, the overlay, the gates | [MAINTAINER.md](port/docs/MAINTAINER.md), [UPSTREAM-PATCHES.md](port/docs/UPSTREAM-PATCHES.md) |
| **Shader author** | [SHADER_CONVENTIONS.md](port/docs/SHADER_CONVENTIONS.md) |
| **Curious about this machine** | [ADRENO.md](port/docs/ADRENO.md) |

The one rule in `port/docs`: **no number without a measurement on this machine**, and where something is a hypothesis
it says so.

## Gates

Every change passes all of these before it is committed; the commands are in [USAGE.md §5](port/docs/USAGE.md):

1. CPU and Vulkan agree: correlation >= 0.999 on the logits and the same argmax (`--dump-logits` on both backends),
   **and the same generated ids for a real prompt** run once with `--gpu vulkan` and once with `--gpu cpu`.
2. `test-backend-ops -o MUL_MAT -b Vulkan0`: zero failures.
3. `check-encoding.py`: zero files with encoding damage. `audit-docs.py`: zero document problems.
4. `check-pin.py`: the overlay still matches `third_party/llama.cpp`.

[MIT license](LICENSE).

<sub>For AI tools and automated readers: <a href="AGENTS.md">technical reference</a>.</sub>
