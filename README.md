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

Snapdragon X2 Elite Extreme (18 Oryon cores, 47.6 GB unified memory, Adreno X2-90 on Vulkan 1.4, driver
`30.0.228.10000`), Windows 11 ARM64. Model: `Gyro-S-TQ1_0.gguf` (54.47 GiB, `qwen4exp`, trellis experts). Decode is
`--bench-tg 16`, prefill `--bench-pp 512`. Medians of repeated runs, same binary.

| backend | decode | prefill pp512 |
| --- | --- | --- |
| **CPU** (`--gpu cpu`, big-core class) | **11.34 tok/s** | 21.06 tok/s |
| **Vulkan**, default shaders | 10.68 tok/s | 19.33 tok/s |
| **Vulkan**, `--shader-family fp32` (fp32 shaders) | 5.48 tok/s | **46.63 tok/s** |

So on this pack the CPU is the better single-stream choice, and the GPU wins only the prefill-heavy case, where the
fp32 shader family is 2.2x the CPU. Every number in this table is produced with the **correct** kernel set: the Vulkan
logits correlate **0.999** with the CPU backend and pick the same token, which is the gate that matters (§5 of USAGE).
`--shader-family fp32` is also the family whose 16-token stream matched the CPU's token for token on the 35B pack; the
f16 family diverges on a near-tie, so "identical to the reference" and "fastest prefill" point at the same setting here.

A faster Vulkan path exists and is **not** used by default: the f16-accumulate cooperative matrix measures 221 tok/s at
pp512 on the same pack, but its logits correlate **0.02** with the CPU's and it picks a different token on every pack
tried. It is reachable with `STRATA_VK_F16ACC=1` for shape-only benchmarks. [PERFORMANCE.md §4.1](port/docs/PERFORMANCE.md)
has the numbers; repairing that kernel is the largest open GPU item.

## Status

- **Correct first.** The default paths are verified against the CPU backend: same logits to 0.999 correlation, same
  argmax, and the 32-slot batched stream is 64/64 tokens identical to a solo run.
- **Vulkan**: loads and answers for `qwen35`, `qwen35moe`, `qwen4exp` (including the trellis `Gyro` packs and the
  `Q2_0` `GSQ-RCO` pack), with the CPU as the reference backend. Decode is not faster than the CPU on this machine yet.
- **CPU**: the fastest single-stream backend here, on the big-core class only (`-t 18` costs decode and gains nothing).
- **Batching**: `--parallel N` runs N requests in one process on one copy of the weights. It is correct and it does not
  pay on this machine: 32 slots step in 3178 ms, which is 32 x a solo step, i.e. 1.00x aggregate. `serve/server.py` is
  not wired to it.
- **NPU**: the Hexagon path builds, but no DSP session opens on this driver (`remote_handle64_open` returns
  `AEE_ERPC 0x200` for every module name tried), and every placement that could be measured before that was slower than
  the GPU. Reviving it needs a QNN backend, not a fixed URI. [NPU-FEASIBILITY.md](port/docs/features/NPU-FEASIBILITY.md)
  has the measurements.
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
