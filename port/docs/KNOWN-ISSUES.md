# Known issues

What is known to be broken, slow or unverified on this branch, with the measurement that says so. Nothing here is a
plan; each line is a fact of the current tree. Where an item was measured on one machine, it says so.

Every number below comes from a run recorded in the document next to it.

## Correctness and coverage

- **Turning the integer dot product off breaks q8_0's mat-mat fallback, not q2_0.** With
  `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1`, `test-backend-ops -o MUL_MAT` fails four q8_0 cases that take the mat-mat
  path (`m=16 n=9 k=256` ERR 1.69, `m=2880 n=32 k=2880` 1.42, `m=32 n=509 k=2112` 1.53, `m=64 n=2 k=256 bs=[3,2]`
  1.54) while every q2_0 case passes. The switch is device-wide, so a q2_0 pack - whose attention projections are
  q8_0 - came out as noise (logits correlation 0.003 against the CPU backend). The engine no longer sets it;
  `STRATA_VK_INTDOT_OFF=1` still reaches it for a bisect. [PERFORMANCE-LOG.md](PERFORMANCE-LOG.md) §5.5.
- **The q2_0 prefill crash that rule was written for is not reproduced.** `mm-probe` at the pack's expert shapes
  (`MUL_MAT_ID`, all 512 experts, n = 512 and 2048) computes and verifies; so does a 1024-token prefill of the whole
  model. What does crash is `--prefill 3072`, and it is two separate walls. The first was the reserved output tensor:
  `[n_vocab, n_outputs_max]` with `n_outputs_max` defaulting to `n_batch` is 2.0 GB at 2 048 rows and 3.05 GB at
  3 072, which the backend refuses; the engine now reserves 16 rows, which is all it ever asks for. The second is the
  driver's 65 535-workgroup limit: at 3 072 tokens the GDN materialises a 3072x3072 tensor and its elementwise
  dispatch asks for 73 728 workgroups. The batch ceiling for this model on this device is therefore 2 048.
  [PERFORMANCE-LOG.md](PERFORMANCE-LOG.md) §5.5.3.
- **The grouped matmul is billed per token in the model, not per tile.** The isolated shape charges per expert tile,
  which is why a larger ubatch looked like the way to fill it; in the model the expert phase is 1.96 ms/token at
  ubatch 1 024 and 1.83 at 2 048 (measured by skipping `matmul_id`), so doubling the batch buys 2.5%, not the 10% the
  tile arithmetic predicts. [PERFORMANCE-LOG.md](PERFORMANCE-LOG.md) §5.5.3.

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

- **`GGML_VK_PERF_LOGGER` misattributes by one node, so its per-name table cannot be used to rank work.** On the Q2
  pack's pp512 it puts `SCALE` first (186 × 2.8 ms = 486-522 ms, 10-13% of the run) and `MUL_MAT_ID_VEC` last
  (96 × 36 us); skipping `scale_f32` outright moves pp1024 by 0.1% (8 071.8 against 8 067.2 ms), so that time belongs
  to the matmul recorded just before it.  The concurrent mode (`GGML_VK_PERF_LOGGER_CONCURRENT=1`) has the same
  mispairing, it only makes the table's sum line up better (104% of total against 118%).  What was usable, while the ablation switch existed, was its wall-clock difference and only for a family whose
  removal does not change the data - skipping `topk_moe` "saved" 25% only because the destroyed routing collapsed the
  expert phase to one tile, and skipping `gated_delta_net` or `swiglu` was *slower* than not skipping. That switch has
  since been removed from the tree; the conclusions it produced stand as recorded in the log.  [PERFORMANCE-LOG.md](PERFORMANCE-LOG.md) §5.5.6.

- **A BF16 tensor costs this driver a fifth of the prefill, and the engine now works around it by converting.** The
  Q2 pack carries 387 BF16 tensors (1.18 GiB); converting them to F16 - exact for weights, since a BF16 mantissa fits
  in F16 - takes pp1024 from 106.2 to 126.4 tok/s with an identical token stream. The engine writes
  `<pack>.f16companion.gguf` (1.41 GiB, about 2 s, beside the pack) on the first load and reuses it.
  `STRATA_VK_NO_COMPANION=1` opts out, `STRATA_VK_COMPANION_DIR=<dir>` moves it, and `--compat` takes precedence.
  Two consequences worth knowing: **a new file appears next to the model**, and the companion is keyed on the pack's
  file name, so two different packs with the same file name would share one cache entry (deleting the file is always
  safe). [PERFORMANCE-LOG.md](PERFORMANCE-LOG.md) §5.5.5.

- **A Qwen3.8-Flash-Next decode token is 2 512 dispatches.** Counted by differencing two `STRATA_VK_PIPE_LOG=1`
  runs (48 layers, 52.3 per layer). When this was measured (2026-11-21) the Q2 pack read 15.25 tok/s, which is 4.37 GiB of weights at 66.6 GB/s
  (62% of the ~108 GB/s the GPU streams) plus about 18 ms of fixed per-dispatch cost. The dispatch count and the
  per-family shares below are properties of the graph and have not changed; the throughput now reads higher, see
  [PERFORMANCE.md](PERFORMANCE.md). The largest single family is
  **483 `mul_mat_vec_bf16_f32_f32`** per token - ten small hyper-connection matvecs per layer, about a fifth of the
  dispatches. Fusing or re-laying those out is the next decoder item; skipping the whole `matmul` family changes the
  wall clock by 1%, because the bytes are in the `mul_mat_vec` family that filter does not match.
  [PERFORMANCE-LOG.md](PERFORMANCE-LOG.md) §5.5.
- **Self-speculation cannot pay on these packs.** A 2-row and a 4-row step cost 1.93x and 3.08x a 1-row step
  (132.1 / 210.9 ms against 68.5), i.e. every extra row re-reads the whole weight set. So a verify pass over k drafts
  is not cheaper per accepted token than plain decoding, whatever the drafter's acceptance rate. Splitting the 4-row
  step by family says the same thing from the other side: the dense weights amortise 2.4x rather than 4x, and the
  ~2 000 small dispatches get 1.75x *worse* per row (11.1 ms to 77.9 ms).
- **Energy per token cannot be measured on this machine.** `\Power Meter(*)\Power` exists in `Get-Counter` and reads
  0, there is no `Energy Meter` counter set, and `root\wmi BatteryStatus` reports `PowerOnline=True` with
  `DischargeRate=0` under AC. Bytes moved per token stands in as the algorithmic-efficiency figure:
  4.37 GiB/token for the Q2 pack, 3.60 GiB for Gyro-S.

- **Single-stream decode on the trellis pack (Gyro-S) sits at the format's floor.** 10.0-10.25 tok/s on 2026-11-19, and every route measured to move it came back
  flat or worse: the codebook path is 6% of a token, the activation hoist was a no-op, residency and rows per
  workgroup are flat, an explicit 4-row unroll was 14% *worse* at one of the three shapes, and replacing the three
  shared windows with direct word reads is neutral once the addresses are right. The wall is the trellis decode rate
  (~47-50 G weights/s, the same number on the matvec, the cooperative mat-mat and prefill's expert phase).
  [PERFORMANCE-LOG.md](PERFORMANCE-LOG.md) §5.4.
- **Prefill still has headroom, and it is kernel efficiency, not the weight layout.** The older claim that this
  kernel set peaks at 85-95 tok/s was superseded on 2026-11-24: the per-thread output tile of the float and
  quantised matrix multiply is a startup setting, and moving it to `8x4` / `8x8` lifted prefill from about 123
  to about 173 tok/s with byte-identical output. What remains short is the multiply itself, which measures
  0.46-0.71 TFLOPS against a roughly 2.7 TFLOPS float peak. [PERFORMANCE.md](PERFORMANCE.md).
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

### 不在本机范围内：Gyro-S 与 Gyro-M

这两款模型已从本机删除，**后续任何涉及 GS / GM 的验证与调优都跳过**，其他部分照常进行。 之前围绕它们测到的数字仍
留在 `PERFORMANCE-LOG.md` 里作为历史记录，但不再复跑、也不再作为默认路径的依据。 默认建议与文档示例因此只应引用当
前仍在机器上的两款公开模型（Qwen3.8-Flash-Next 与 Qwen3.6-35B-A3B），以及本轮的 Edge0 145B / 120B 工作。

### 改动 `ggml-vulkan-types.h` 后必须整块重编后端（未确认）

给 `vk_device` 增删字段（例如收敛 latent 的管线字段）时，只做增量 `cmake --build` 出现过三种不同的失败形态：F32 夹具
在派发后静默故障、探针在打印任何结果前退出、以及同一改动"有时通过有时不通过"。每次都只有把文件恢复到与头文件一致的
状态才恢复。**推测**（未证实）是头文件改动没有触发全部包含它的翻译单元重编，导致同一结构体在不同 TU 里布局不一致。

因此：改 `ggml-vulkan-types.h` 后请**重编整个后端目标**（例如 `--clean-first` 或先删除对应的对象目录），不要依赖增量
构建的依赖判断。这条在证实之前只作为操作纪律，不作为结论。

### 一个残留的 latent 进程会让后续运行在提交队列时失败

试过把 40 GiB 的专家映射放进设备可见内存之后，紧接着的商品路径验证报
`vk::Queue::submit: ErrorInitializationFailed`，而**结束残留进程后同一命令立即恢复**（`25,15451,369,279,`）。也就是说前一次运行占住的
设备内存没有释放，后一次运行在提交队列时才失败，错误信息并不指向真正的原因。

因此：跑过 latent / 大内存实验之后，先确认没有残留的引擎进程，再做商品路径验证；看到 `ErrorInitializationFailed` 时先查进程，而不是先怀疑代码。
