# The Hexagon NPU: current state

Short page, current state only. The measurement records behind it are
[NPU-FEASIBILITY.md](NPU-FEASIBILITY.md) (the driver investigation and what the path was worth when sessions opened)
and [NPU-ASSETS.md](NPU-ASSETS.md) (what is installed on this machine and how the skels are signed).

## Where it stands (2026-11-19)

- **No DSP session opens from a locally built skel.** `remote_handle64_open` returns `AEE_ERPC (0x200)` for every
  module name tried, including a name that is in no allow-list, and including Qualcomm's own host library from the
  GenieX package. An allow-list explanation was measured and refuted; a signed catalog was built and did not change
  it.
- **The hardware and the driver are not the problem.** The machine carries the QNN stack Microsoft ships for it - a
  signed `libQnnHtpV81Skel.so` in the driver store, `QnnHtp.dll`, `QnnSystem.dll`, and `onnxruntime_providers_qnn.dll`
  - and GenieX (installed here) runs GGUF models through a `ggml-hexagon.dll` plus its own signed
  `libggml-htp-v81.so`. So the DSP is reachable through QNN and not through the raw FastRPC door this port uses. A
  future attempt is a QNN backend, not a fixed URI.
- **`dspqueue_create` is the next wall when a session does open.** With this tree's own skel, or with Qualcomm's
  signed one, an earlier measured run reached `HTP0 new session ... handle <non-null>` and then failed at
  `dspqueue_create: 0x14`, with the hardware summary reading `threads 0, hvx 0, hmx 0, vtcm 0 MB`. That shape is a
  host/skel build mismatch at least as much as a driver refusal, and it has not been separated yet.
- **When sessions did open, the path was 0.34x the GPU for decode** (35B-A3B: 12.5 against 36.4 tok/s; prefill 139.8
  against 148.6), and the tiled "fast" path - runtime repacking - hung on both skel builds (`GGML_HEXAGON_ARCH=79`
  answers one token then hangs on a 512-token prefill; v81 never answers). That is why the engine keeps the NPU off
  by default and why keeping the NPU off is not a loss.

## What is in the tree

- `third_party/llama.cpp/ggml/src/ggml-hexagon/` - the DSP backend: HVX and HMX tiled matmuls, flash attention, GDN,
  SSM scan, the lightning indexer, `dsv4-hc`, allreduce, repack, and the surrounding elementwise ops. Its type table
  covers Q1_0…Q8_0, IQ2_XXS, IQ4_NL, MXFP4 and NVFP4 - **not** the trellis types, so this model's experts cannot run
  there without a repack.
- The host side already honours the Hadamard hint on the DSP (`HTP_OP_HADAMARD`), which the Vulkan backend implements
  as a separate FWHT pass.
- `port/engine/vk-devices.cpp` builds the NPU plan and places dense tensors when `--npu dense` is asked for;
  `--npu-mtp` can place a draft head there (see [KNOWN-ISSUES](../KNOWN-ISSUES.md) for why that flag is a no-op
  unless MTP is forced).

## What not to repeat

- Do not expect the signed-catalog route to unblock the raw FastRPC door. The catalog build works
  (`HEXAGON_HTP_CERT` + `inf2cat` + `signtool`, and `bcdedit /set TESTSIGNING ON` with the certificate in both trust
  stores), and the measured result with it was the same `0x200`.
- Do not measure NPU throughput with GPU utilisation counters: on this machine the GPU's counter reads a pipeline
  this port does not use, and the NPU has no Windows utilisation counter at all.
- Do not put the dense tensors on the NPU to "fill an idle device": every placement measured before the session
  broke was slower than the Adreno, and the boundaries between devices cost more than the offload saves.
