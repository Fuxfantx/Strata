# The Windows-ARM64 Vulkan port

`strata-vk` is a Vulkan-first engine for GGUF models on Snapdragon X2 / Adreno X2, built as a small front end
(`port/engine`) over a pinned vanilla `llama.cpp` (`third_party/llama.cpp`) that carries a set of patches
(`port/llama-overlay-vanilla`). Everything specific to this port lives in those two directories; the vendored tree stays
reproducible from its pin.

## Build and run

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File port/tools/fetch-llama.ps1   # pin + overlay -> third_party/llama.cpp
powershell -NoProfile -ExecutionPolicy Bypass -File port/tools/build-engine.ps1 # -> build/bin/strata-vk.exe
build\bin\strata-vk.exe --geometry
build\bin\strata-vk.exe --native D:\models\pack.gguf --bench-pp 512
build\bin\strata-vk.exe --native D:\models\pack.gguf --prompt "The capital of France is" --n 16
```

`--geometry` must list `Vulkan0`. A run that lists only `CPU` means the Vulkan backend did not load; the engine also
prints the expert placement, the f16-accumulate decision and the warmup plan, and `--dry-run` prints that resolved
policy without reading the weights.

## What it supports

| pack family | notes |
| --- | --- |
| trellis (Gyro-S / Gyro-M): `TQK6`, `TQK7`, `TQ2_T` | 2.125 and 1.5 bpw types, matvec + cooperative-matmul paths |
| ternary `PTQ1_0` | 1.5 bpw, same paths |
| `Q2_0`, `Q4_0`, `Q8_0`, `q5_K`, `q6_K` | int8-dot and cooperative-matrix paths |
| qwen4exp (3.8-Flash-Next) | hyper-connections, gated delta net, sparse indexer, per-layer embeddings |
| qwen35moe / qwen3moe (35B-A3B) | MoE with device or host experts |
| MTP / NextN heads | grafted through the loader, or loaded as a separate context |

## Measured on this machine

Snapdragon X2 Elite Extreme (18 cores, 47.6 GB unified memory, Adreno X2-90), Windows 11 ARM64. `--tokens 9707
--n 32` for decode, `--bench-pp` for prefill; medians of repeated runs.

| model | decode tok/s | prefill tok/s |
| --- | --- | --- |
| Gyro-S (trellis MoE, 54.5 GB) | 9.7 | 212 (pp512) |
| Gyro-M (trellis MoE, 61.8 GB) | 8.7 | 101 (pp64) |
| Q2_0 (62.2 GB, device experts) | 14.0 | 41.5 (pp64) |
| 27B Q4_0 (dense) | 7.8 | 162 (pp512) |
| 35B-A3B (MoE) | 43.1 | 613 (pp512) |

Token streams are identical to the CPU backend on every pack. The device facts behind these numbers, the tuning knobs and
what each one does are in `ADRENO.md`; the campaign that produced them, including everything measured and ruled out, is in
`OPTIMIZATION-BACKLOG.md`.

## Where to look next

| you are | read |
| --- | --- |
| running the engine | this file, then `ADRENO.md` for the flags that matter on this device |
| changing the engine or the overlay | `features/` for hyper-connections, the gated delta net, expert placement and the NPU, then `MAINTAINER.md` for the rules the overlay has to follow |
| maintaining the pin or the fork | `MAINTAINER.md`, then `UPSTREAM-PATCHES.md` for the provenance of every patch |
| the product around the engine | the upstream documents in `docs/`: `AI_REFERENCE.md` for models, hardware and APIs, `DETAILS.md` for the engine and every measured number, `CONTRIBUTING.md` for the rules a change has to follow |

## Layout

| path | what it is |
| --- | --- |
| `port/engine/` | the `strata-vk` front end: CLI, session/serve, per-model tuning, placement |
| `port/llama-overlay-vanilla/` | every file this port changes or adds, at its real path; copied verbatim onto the pin |
| `port/tools/` | `fetch-llama.ps1`, `build-engine.ps1`, the gates (`check-encoding.py`, `audit-docs.py`, `check-fork-features.py`), the measurement harnesses (`bench_runs.py`, `mm-probe`) |
| `port/tests/` | the engine's stdout contract |
| `third_party/llama.cpp` | vanilla `ggml-org/llama.cpp` at the pinned commit with the overlay applied |
