# Strata on Windows ARM64: the Vulkan engine

This branch carries one thing on top of upstream: **`strata-vk`**, a Vulkan-first engine for Snapdragon X2 / Adreno X2,
and the overlay that teaches the vendored `llama.cpp` the model families and the low-bit types it needs. The front end
around it — the desktop app, the Terminal, `setup.py` and the Python server — is upstream's and drives this engine
without changes (the engine component's stdout contract passes 10/10).

Everything specific to this branch lives in `port/`. There is no other divergence: the app, the Terminal, `serve/` and
`setup.py` are the files upstream ships.

## Measured on this machine

Snapdragon X2 Elite Extreme (18 cores, 47.6 GB unified memory, Adreno X2-90), Windows 11 ARM64. Decode is
`strata-vk --tokens 9707 --n 32`, prefill is `strata-vk --bench-pp`; medians of repeated runs.

| model | decode tok/s | prefill tok/s |
| --- | --- | --- |
| Gyro-S (trellis MoE, 54.5 GB) | 9.7 | 212 (pp512) |
| Gyro-M (trellis MoE, 61.8 GB) | 8.7 | 101 (pp64) |
| Q2_0 (62.2 GB, device experts) | 14.0 | 41.5 (pp64) |
| 27B Q4_0 (dense) | 7.8 | 162 (pp512) |
| 35B-A3B (MoE) | 43.1 | 613 (pp512) |

The 32-token stream is identical to the CPU backend on every pack.

## Build and run

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File port/tools/fetch-llama.ps1   # pin + overlay -> third_party/llama.cpp
powershell -NoProfile -ExecutionPolicy Bypass -File port/tools/build-engine.ps1  # -> build/bin/strata-vk.exe
build\bin\strata-vk.exe --geometry
build\bin\strata-vk.exe --native D:\models\pack.gguf --prompt "The capital of France is" --n 16
build\bin\strata-vk.exe --native D:\models\pack.gguf --bench-pp 512
```

`--geometry` must list `Vulkan0`. The engine prints the expert placement, the f16-accumulate decision and the warmup
plan at start, and `--dry-run` prints that resolved policy without reading the weights.

The app and the Terminal use the engine through `setup.py`, which downloads a release engine or builds one; building
this branch's engine is what `build-engine.ps1` does.

## What the engine supports

| pack family | notes |
| --- | --- |
| trellis `TQK6`, `TQK7`, `TQ2_T` (Gyro) | 2.125 / 1.5 bit weights, matvec and cooperative-matrix paths |
| ternary `PTQ1_0` | 1.5 bit, same paths |
| `Q2_0`, `Q4_0`, `Q8_0`, `q5_K`, `q6_K` | int8-dot and cooperative-matrix paths |
| qwen4exp (3.8-Flash-Next) | hyper-connections, gated delta net, sparse indexer, per-layer embeddings |
| qwen35moe (35B-A3B) | MoE with device or host experts |
| MTP / NextN heads | grafted through the loader, or loaded as a separate context |

## Layout

| path | what it is |
| --- | --- |
| `port/engine/` | the `strata-vk` front end: CLI, session/serve, per-model tuning, expert placement, NPU discovery |
| `port/llama-overlay-vanilla/` | the 60 files applied to the pinned `llama.cpp`, at their real paths |
| `port/tools/` | `fetch-llama.ps1`, `build-engine.ps1`, the gates, the measurement harnesses, `check-pin.py` |
| `port/docs/` | the port's own documents: quickstart, maintainer rules, the Adreno measurements, per-patch provenance |
| `apps/`, `serve/`, `strata_cli/`, `strata_control/`, `setup.py` | the front end, as upstream ships it |

## Where to look next

| you are | read |
| --- | --- |
| running the engine | `port/docs/README.md`, then `port/docs/ADRENO.md` for the flags that matter on this device |
| changing the engine or the overlay | `port/docs/features/` and `port/docs/MAINTAINER.md` |
| maintaining the pin | `port/docs/MAINTAINER.md` and `port/tools/check-pin.py` |
| using Strata itself | `docs/` (English and Simplified Chinese) |

MIT licensed, on upstream's terms; the port's own notes are in `port/docs/`.
