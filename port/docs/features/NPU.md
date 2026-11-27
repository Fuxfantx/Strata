# The Hexagon NPU, and the capture tools on this machine

*状态：**记录**。结论已并入 [KNOWN-ISSUES.md](../KNOWN-ISSUES.md)；本节把三份 NPU/抓帧记录（原 `NPU.md`、`NPU.md`、`NPU.md`）合并为一份，细节与逐次运行在 [PERFORMANCE-LOG.md](../PERFORMANCE-LOG.md)。 / Status: a record. The conclusion is in [KNOWN-ISSUES.md](../KNOWN-ISSUES.md); this page merges the three former NPU and capture notes.*

## Where the NPU stands

- **No DSP session opens from a locally built skel.** `remote_handle64_open` returns `AEE_ERPC (0x200)` for every
  module name tried, including one that is in no allow-list and including Qualcomm's own host library from the GenieX
  package. A signed catalog was built with `HEXAGON_HTP_CERT` + `inf2cat` + `signtool` (and `bcdedit /set TESTSIGNING ON`,
  certificate in both trust stores); the result was the same `0x200`, so the allow-list explanation is refuted.
- **The hardware and the driver are fine.** The machine ships the QNN stack Microsoft provides for it - a signed
  `libQnnHtpV81Skel.so` in the driver store, `QnnHtp.dll`, `QnnSystem.dll`, `onnxruntime_providers_qnn.dll` - and
  GenieX (installed at `C:\Users\Fuxfantx\C-Res\geniex-sdk`, v0.11.0) runs GGUF models through `ggml-hexagon.dll`
  plus its own signed `libggml-htp-v81.so`. The DSP is reachable through QNN, not through the raw FastRPC door this
  port uses; a future attempt is a **QNN backend**, not a fixed URI.
- **`dspqueue_create` is the next wall.** When an earlier run did open a session (`HTP0 new session ... handle`), it
  then failed at `dspqueue_create: 0x14` with `threads 0, hvx 0, hmx 0, vtcm 0 MB` - a host/skel build mismatch at
  least as much as a driver refusal, not separated yet.
- **When sessions did open, the path was slower than the GPU.** Qwen3.6-35B-A3B: 12.5 against 36.4 tok/s decode
  (0.34x), prefill 139.8 against 148.6; the tiled "fast" path hung on both skel builds (`GGML_HEXAGON_ARCH=79` answers
  one token then hangs on a 512-token prefill, v81 never answers). Keeping the NPU off is therefore not a loss.

## What is in the tree

- `third_party/llama.cpp/ggml/src/ggml-hexagon/` - the DSP backend: HVX/HMX tiled matmuls, flash attention, GDN, SSM
  scan, lightning indexer, `dsv4-hc`, allreduce, repack, elementwise ops. Its type table covers Q1_0…Q8_0, IQ2_XXS,
  IQ4_NL, MXFP4, NVFP4 - **not** the trellis types, so trellis experts cannot run there without a repack.
- `port/engine/vk-devices.cpp` builds the NPU plan and places dense tensors for `--npu dense`; `--npu-mtp` can place a
  draft head there (a no-op unless MTP is forced, see [KNOWN-ISSUES](../KNOWN-ISSUES.md)).

## What not to repeat

1. Do not expect a signed catalog to unblock the raw FastRPC door - it was built and measured, same `0x200`.
2. Do not measure NPU throughput with GPU utilisation counters: this port's GPU counter reads a pipeline it does not
   use, and the NPU has no Windows utilisation counter.
3. Do not place dense tensors on the NPU to "fill an idle device": every placement measured before the session broke
   was slower than the Adreno, and the device boundary costs more than the offload saves.

## Capture tools on this machine

| What | Where / state |
| --- | --- |
| RenderDoc 1.47 (x86-64 only) | `tools\rdoc\RenderDoc_1.47_64` (a second copy at `C:\strata-nospace\tools\rdoc`), byte-identical between the two; not in Add/Remove Programs |
| ARM64 RenderDoc | one binary exists on the machine; a capture of this process has **not** been validated on this driver |
| `STRATA_VK_RDC*` | loads the layer and reports what it finds; treat any capture as unverified |
| Qualcomm assets | `C:\Users\Fuxfantx\C-Res\geniex` (source, v0.11.0), `geniex-sdk` (prebuilt, windows-arm64), `hexagon-sdk` 6.7.0.1 |

The practical consequence for profiling: on this machine there is no working frame capture of the Vulkan process, so
attribution comes from the timing buckets and the per-family counters described in [USAGE.md §4](../USAGE.md), not from
a capture.
