# Strata on Windows on ARM (Snapdragon X2 Elite Extreme / Adreno X2-90)

This folder is the Windows-on-ARM port of Strata 0.2.13: a native ARM64 engine component that speaks Strata's engine
protocol, the Control changes that let the app find it, and the scripts that package both into the desktop app.

It is **not** a fork of Strata's engine. Strata's engine is a hand-written CUDA/HIP/SYCL kernel set, and none of it runs on
an Adreno GPU. The port keeps Strata's product and its contracts (the `strata --serve` line protocol, Control's component
manifest, the app's bundled-engine install) and puts the model work on **ggml's Vulkan backend**, built from vanilla
llama.cpp at pin `32bdc3a89e6528d3e419024c833c5dabf32c7541` rather than from Strata's `third_party/strata-ggml`, with the
trellis types and the NextN/MTP graft ported onto that pin as the 38-file overlay in `llama-overlay-vanilla/`.

The measured state, the gates and the pitfalls are in [`docs/README.md`](docs/README.md). The build procedure is
[`../BUILD-WOA.md`](../BUILD-WOA.md). This file is the map.

## What is here

| Path | What it is |
| --- | --- |
| `engine/vk-*.cpp`, `engine/vk-*.h` | The engine, in seven parts: `vk-options` (the command line), `vk-devices` (the Adreno tuning and the NPU plan), `vk-engine` + `vk-engine-load` (load, prefix reuse, generate), `vk-serve` (the line protocol), `vk-mtp` (the NextN head), `vk-common`, `vk-main` |
| `engine/CMakeLists.txt` | Builds `strata-vk` inside the llama.cpp checkout (`-DSTRATA_VK_DIR=...`) |
| `llama-overlay-vanilla/` | The 38 files this port changes or adds to vanilla llama.cpp, at their real paths: the `TQ2_T`/`PTQ1_0` trellis types and their Vulkan kernels, the MTP graft and loader fallbacks, the recurrent-state fast path, the engine target |
| `tools/fetch-llama.ps1` | Downloads the pinned upstream commit and copies the overlay over it, producing `third_party/llama.cpp` exactly (4 188 files) |
| `tools/build-engine.ps1` | Configures `build\`, builds `strata-vk` and stages the ARM64 `vulkan-1.dll` beside the exe |
| `tools/package_component.py` | Writes the schema-2 engine component and verifies it with the app's own checker |
| `packaging/stage-app.ps1` | Stages `resources/node`, `resources/control`, `resources/engine-bundle` for the Tauri app |
| `tests/serve_contract.mjs` | Drives `--serve` over stdin/stdout exactly as the app does (10 checks) |
| `probe/mm-probe.cpp` | The kernel instrument: per-shape matvec/mat-mat timing, device-to-host handoff |
| `docs/README.md` | The port as it stands: layout, build, measured numbers, where to look next |
| `docs/ADRENO.md` | The Adreno hardware bounds, the tuning defaults and why, the trellis tile result |
| `docs/HYBRID.md` | Where the CPU, GPU and NPU end up, what a decode token is made of, and what was measured against it |
| `docs/NPU-FEASIBILITY.md` | The Hexagon path: how the host half is built, what it is worth, and its hazards |
| `docs/VANILLA-REBASE.md` | How the vanilla base is composed and the procedure to swap it again |
| `docs/OVERLAY-VANILLA.md` | The overlay, file by file |

## Build

```powershell
powershell -ExecutionPolicy Bypass -File port/tools/build-engine.ps1
build\bin\strata-vk.exe --geometry      # must list Vulkan0, not only CPU
```

The script fetches the pin first if `third_party\llama.cpp` is absent. `-Build` takes another build directory and
`-Target` another target (`mm-probe` is the kernel instrument, `test-backend-ops` the correctness oracle).

The one trap worth repeating here: the engine needs the **ARM64** Vulkan loader beside `strata.exe`, and
`C:\Windows\System32\vulkan-1.dll` is the x64 loader on this machine. Without a good copy `ggml-vulkan.dll` fails to load,
the engine silently registers the CPU backend alone, and the output is slow and wrong. `build-engine.ps1` stages it;
`BUILD-WOA.md` has the check.

## Run

```powershell
$exe   = "C:\Users\Fuxfantx\A-Works\DSH Works\Strata\build\bin\strata-vk.exe"
$model = "C:\Users\Fuxfantx\C-Res\Qwen\3.6-35A3\Qwen3.6-35B-A3B-Q4_0.gguf"

& $exe --native $model --prompt "The quick brown fox jumps over the lazy dog." --n 32
& $exe --native $model --bench-tg 32                      # decode only, warm
& $exe --native $model --bench-pp 512                     # prefill only
& $exe --native $model --serve --host 127.0.0.1 --port 8099   # the app's line protocol
```

`--serve` also takes `--session-dir`, `--api-key`, `--chat-template`, `--reasoning` and the generation flags the app
sends; `--help` lists everything, including the NPU (`--npu`, `--npu-layers`, `--npu-mtp`, `--npu-probe`), drafting
(`--lookup`, `--mtp`) and backend (`--gpu`) flags. The protocol is the one `tests/serve_contract.mjs` checks.

## Measured (this machine, engine defaults)

| workload | prefill | decode |
| --- | --- | --- |
| tiny-qwen (dense 0.6B) | 4 301 tok/s (512) | 126.5 tok/s (64) |
| 27B Q4_0 dense | 163.9 tok/s (512) | 7.30 tok/s (16) |
| 35B-A3B (`qwen35moe`) | 198.9 tok/s (512) | 38.96 tok/s (32) |
| Gyro-M (62 GB trellis MoE) | 65.3 tok/s (430) | 3.90 tok/s (24) |
| GSQ-RCO `Q2_0` pack (67 GB) | 31.9 tok/s (64) | 13.20 tok/s (32) |

These are the defaults, which use the f16-accumulate cooperative matrix (see `ADRENO.md`); the same commands without that
path measured 2 826 / 118.9, 77.9 / 7.0, 151.7 / 37.8, 17.4 / 3.3 and ~9 / 12.4. `STRATA_VK_F16ACC=0` restores the older
path for any model. `PORT-STATE.md` carries the method, the gate list and the interleaved A/B results behind each engine
change. One note about Gyro-M: it is 62 GB on a 47.6 GB machine, so its decode pages part of the active expert rows in from
disk - earlier notes quote 3.3 tok/s for that reason, and 3.90 is what this tree measures now.

## What works, what does not

- **27B Q4_0 dense and 35B-A3B MoE** (`qwen35`, `qwen35moe`): load, answer, and are the two models above. The MoE's
  decode is 38.96 tok/s on ~2.1 GB of active weights per token, so it is dispatch-bound rather than bandwidth-bound
  (`HYBRID.md` sizes the ops).
- **Gyro-M / Gyro-S** (TQ2_T trellis, `prism.hadamard` rotation): load and answer coherently, 65.3 tok/s prefill at 430
  tokens, decode as in the table above.
- **Qwen3.8-Flash-Next** (qwen4exp): the Gyro-M trellis pack and the ISTA **GSQ-RCO** `Q2_0` pack both run on Vulkan
  with output identical to the CPU backend. GSQ-RCO is a quantisation method over standard GGUF types, not a rotation
  (the two packs carry the same 1223 tensor names); what broke them was a shader bug in the port's `q8_0` float matvec,
  which decoded the quant bytes as unsigned and is fixed in `dequant_funcs.glsl` (`PORT-STATE.md` has the entry). The Q2
  pack now prefills at 31.9 tok/s and decodes at 13.20; its `q2_0` tensors need the fp16 shaders with the integer dot
  product off, which the engine selects from the pack's header.
- **DeepSeek-V4.1**: not supported by this component, and the engine says so (`ERR ...` on the serve line). Strata's
  DeepSeek engine is a separate CUDA/HIP program.
- **Pictures**: the engine reads text only; the app's image path needs Strata's `strata-vision` encoder.
- **NPU**: the Hexagon backend builds from this tree and, with the vendor-signed ops library staged by
  `port/tools/fetch-signed-htp.ps1`, `--npu dense` completes - the old DSP hang is gone and `GGML_HEXAGON_RT_REPACK=1` now
  returns a clean `NO-SUPPORT`. Every placement measured is still slower than the GPU on this machine (27B decode 4.1-4.7
  against 7.3 tok/s, 35B-A3B 10-14 against 39), so the engine reports the device and leaves it unused unless
  `STRATA_VK_NPU_FORCE=1`. `NPU-FEASIBILITY.md` has the measurements and the signing wall.
- **Desktop app**: built for ARM64; the engine travels inside the installer and is installed on first start.

## Integration with the desktop app, verified rather than assumed

The app does not start an arbitrary program: it installs an engine **component** and runs the program the component's
manifest names. Two tools close that loop.

`port/tools/package_component.py` builds the package the app expects - `0.2.13-windows-arm64-vulkan/` with an
`engine-component.json` (schema 2: a SHA-256 and byte count per file) - then verifies it with the app's own reader and
checker (`strata_control.engine_component.read_manifest` + `verify(full=True)`). It refuses to build at all when the app's
matrix would not accept the combination. Run it with an **ARM64** CPython from the repository root, so `strata_control`
is importable; an x64 Python reports `windows-x86_64` and the app then refuses Vulkan entirely:

```powershell
python -c "import platform; print(platform.machine())"    # must print ARM64, not AMD64
python port\tools\package_component.py --verify
```

`port/tests/serve_contract.mjs` drives `strata-vk.exe --serve` over stdin/stdout exactly as the app does and checks the
things the front end parses: READY arriving, INFO being non-empty, one `GEN <id>` per token, the DONE line carrying
fifteen fields after the word DONE with matching counts, a STOP mid-generation coming back with a finish reason, and
QUIT ending the process. Ten checks, no dependencies beyond node.

What that loop found and fixed, in the port's own words: the serve path used to drop every generated token (the callback
ignored its argument), the package must name its program **`strata.exe`** (that is what saved run configurations refer
to), and packaging from an x64 Python reports `windows-x86_64`, which the app refuses for Vulkan - use the app's ARM64
interpreter.

**Installed on this machine**: `%LOCALAPPDATA%\Strata\components\engine\0.2.13-windows-arm64-vulkan` holds the current
build and passes `verify(full=True)`; the previous package is backed up in
`%LOCALAPPDATA%\Strata\backup-engine-20261113-201811`. The component is unsigned (`signature: null`): the app installs it,
but a machine with `STRATA_REQUIRE_SIGNED_COMPONENTS` set would refuse it - the same signing wall as the NPU skels.

## Where this lives

- Fork: <https://github.com/Fuxfantx/Strata>, branch **`windows-arm64-vulkan`** (upstream `main` merged; one conflict in
  `tools/strata_mcp.py` resolved by keeping upstream's card assembly and appending the Adreno fallback).
- Installer: produced by `npm --prefix apps/desktop run tauri build` (Tauri's NSIS bundler); this tree has no
  `target/release/bundle` directory, so the installer is built on demand, not shipped here.
- Engine component: `%LOCALAPPDATA%\Strata\components\engine\0.2.13-windows-arm64-vulkan`, reported by
  `strata components engine` as `usable: true`, platform `windows-arm64`, backend `vulkan`.
