# Known issues

What is known to be broken, slow or unverified on this branch, with the measurement that says so. Nothing here is a
plan; each line is a fact of the current tree. Where an item was measured on one machine, it says so.

Every number below comes from a run recorded in the document next to it.

## Correctness and coverage

- **Chat answers on the trellis packs are poor, and it is the pack, not the engine.** `Paris is the capital of
  France.` continues normally (`\n\nParis (French pronunciation: [`), while `The capital of France is` stops after a
  space and a 15-token chat prompt answers `n\n`. The test that separates the two: the same prompt on `--gpu vulkan`
  and on `--gpu cpu` returns the **same ids** (Vulkan `77,198`, CPU `77,198`), and the CPU path is llama.cpp's own
  kernels. So the port reproduces the reference; this 1.63-bit-per-weight pack is what behaves that way. No
  higher-precision pack of the same model has been tested here.
- **Pictures are plumbed, not proven.** `strata-vision` encodes, the engine takes `--vision` plus the `.sve` on a
  `GENI` line, and `serve/server.py` drives it: a picture request returns 200 with its image tokens in the prompt.
  What cannot be shown is image *understanding* - the pack's chat answers are degraded on both backends, so an
  answer about a picture cannot be judged. `port/docs/features/VISION.md` §3.4 has the evidence.
- **`BGENI` does not carry pictures.** The multi-slot spelling answers `ERR`; pictures go through the one-at-a-time
  path only.
- **DeepSeek models are refused on purpose** - by the front end (`strata_control/retired.py`) and, since this
  branch's change, by the engine itself, before any tensor is read. The engine is not able to run them.

## Performance

- **Single-stream decode is at the format's floor.** 10.0-10.25 tok/s, and every route measured to move it came back
  flat or worse: the codebook path is 6% of a token, the activation hoist was a no-op, residency and rows per
  workgroup are flat, an explicit 4-row unroll was 14% *worse* at one of the three shapes, and replacing the three
  shared windows with direct word reads is neutral once the addresses are right. The wall is the trellis decode rate
  (~47-50 G weights/s, the same number on the matvec, the cooperative mat-mat and prefill's expert phase).
  [PERFORMANCE.md](PERFORMANCE.md) §5.4.
- **Prefill's remaining 15% needs a different weight layout.** At pp512, 84% of the cooperative MMA columns are
  empty (512 tokens touch all 512 experts, so an expert has ~10 rows in a 64-column tile). Both ways out are measured
  shut: no int8 matmul shader exists for the trellis types, and the driver rejects a 32x32 matrix core. The ceiling
  of this kernel set is 85-95 tok/s.
- **More than 64 request slots does not fit.** 96 slots fails in `vk::Device::allocateMemory` on this 47.6 GB
  machine. Each slot's context is `n_ctx / N`, so 64 slots at 32k leaves ~512 tokens per request.
- **The two-process CPU+GPU figure is stale.** 29.1 tok/s aggregate (25.3 + 3.8) was measured on the dp4a kernel and
  has not been re-measured since the cooperative path landed.

## Devices

- **The NPU does not open a session from a locally built skel.** `remote_handle64_open` returns `AEE_ERPC (0x200)`
  for every module name, including names in no allow-list, while the QNN stack Microsoft ships for this machine
  (`QnnHtp.dll` + a signed V81 skel, used by GenieX) does work. So the gate is the door, not the hardware: a QNN
  backend would be needed, not a fixed URI. When sessions did open on an earlier driver the path measured 0.34x the
  GPU's decode, so it is not a throughput win either. [features/NPU.md](features/NPU.md).
- **`--npu-mtp` is a no-op unless MTP is forced.** MTP drafting measures 31% slower on this pack, so the engine
  turns it off and says so; `--npu-mtp` chooses *where* a head would run. If it is forced and the head's graph needs
  an op the HTP cannot run, the engine now watches llama.cpp's placement line and drops the head onto the GPU rather
  than hanging on the first decode.

## Environment

- **`serve/server.py` needs a configuration this checkout does not carry.** It wants `exe`, `args`, `cwd`,
  `tokenizer`, `model_name` (and a `vision` section for pictures). `port/tools/gguf-tokenizer.py` writes the
  tokenizer directory from a GGUF; a run config must be written by hand or by `setup.py`.
- **The picture encoder's directory is under `%TEMP%`.** The server prefers a space-free parent, this repository's
  path contains a space, so it falls back to the system temp directory. On a machine where a binary built from this
  checkout may not write outside the workspace (this one), the encoder cannot write its `.sve` there and the request
  fails with `EACCES`; a space-free junction into the workspace plus `TEMP=<that junction>` works around it.
  [features/VISION.md](features/VISION.md) §3.3.
- **The RenderDoc capture path is untested on this driver.** `STRATA_VK_RDC` loads the DLL and reports what it finds;
  no capture has been validated here.
