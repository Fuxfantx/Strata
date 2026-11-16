# The Hexagon NPU

State, this build: the HTP backend **builds and runs**, opens DSP sessions and reports the device. With the vendor-signed ops
library staged by `port/tools/fetch-signed-htp.ps1`, `--npu dense` completes, and `GGML_HEXAGON_RT_REPACK=1` now fails cleanly
with a `dsp-rsp NO-SUPPORT` instead of hanging the DSP. Every placement measured is still slower than the GPU on this machine
(27B decode 4.11 head / 4.69 split4 / 0.57 dense against 7.34 tok/s on the GPU; 35B-A3B 13.9 / 10.3 / 5.22 against 37.6), so
the NPU stays off unless `STRATA_VK_NPU_FORCE=1`.

The rest of this note records the investigation that got there, including the hang it started from: with the older skels the
first packet of a **runtime-repacked (tiled) weight** never came back - the host sat in the queue wait until killed - while
`GGML_HEXAGON_RT_REPACK=0` completed, which placed the defect in the DSP-side tiled-matmul path rather than in the transport.
The host side is provably fine: `GGML_HEXAGON_REPACK_CHECK=1` repacks and unpacks in place and reports
`output.weight: 0 of 144643072 bytes differ` on the tiny head, i.e. the tiled layout round-trips exactly.

Two more facts place the defect. The shipped skels differ: with `GGML_HEXAGON_ARCH=79` the tiled path answers a
one-token batch (the v81 skel never does), yet the same v79 configuration hangs on a 512-token prefill, so the tiled path
is shape-dependent on both. And the untiled path, which works, is much slower than the GPU: 35B-A3B decode 12.5 against
36.4 tok/s and prefill 139.8 against 148.6, 27B decode 3.7 against 6.7. That is why the NPU stays **off by default**;
the engine disables the tiled path so that a forced placement at least terminates.

The sections below are the earlier measurement of the same path, kept as the record of what it was worth when it did
complete; their numbers are not current.

## The signed ops library, and the vision-encoder ops

The DSP only loads a **signed** ops library, and the signing unit is the whole library (`libggml-htp-v{73,75,79,81,85}.so`
plus the `libggml-htp.cat` catalog), not individual operators. There is no separate "vision encoder" package anywhere:
on the ggml path a vision encoder is an mmproj that runs as ordinary ggml ops (conv2d, conv2d_1d, im2col_3d, matmul) on
the same HTP skel; on Qualcomm's QNN path it is a signed *context binary* shipped per model. So "signed vision-encoder
ops" means "a signed skel whose `hte` op table includes conv2d", and that is exactly what was missing here.

Qualcomm publishes such a catalog in the **GenieX** Windows SDK (`qualcomm/GenieX`, release v0.11.0): the package
`geniex-sdk-windows-arm64-v0.11.0.zip` contains `lib/llama_cpp/libggml-htp.cat` plus the five skels, Microsoft-signed for
llama.cpp at `93cf736a` (no test-signing, no cert import). `port/tools/fetch-signed-htp.ps1` downloads it, verifies the
release's own sha256 and stages the files beside the exe (and into the engine component with `-Component`):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File port/tools/fetch-signed-htp.ps1 -Component
```

What the op tables actually contain (counted in the binaries):

| skel | `conv2d` / `conv2d_1d` | `im2col_3d` | `repack` (fork's tiled path) |
| --- | --- | --- | --- |
| `build\bin` copy from 11/5 (stale) | 0 | 0 | 74 |
| current DSP-side source, freshly built | 57 / 4 | 16 | 0 |
| GenieX v0.11.0, Microsoft-signed | 57 / 4 | 16 | 0 |

So the current source already builds the vision-encoder ops; the installed copy was eleven days old and carried the
fork's tiled repack ops instead. That stale pair is also why the tiled path hung: the op existed in the old skel and
answered nothing. With the signed skels staged, the host library and the skel agree:

- `--npu dense` on tiny completes (dense tensors on `HTP0`, 16-17 tok/s decode, prompt ~1.0 s): the signed skels are
  ABI-compatible with this `ggml-hexagon.dll`.
- `GGML_HEXAGON_RT_REPACK=1` now fails **cleanly** — `dsp-rsp NO-SUPPORT` then `synchronize failed` — because the signed
  skel does not implement the fork's repack op, instead of hanging in the queue wait.

NPU placement is still slower than the GPU for token generation (16-17 against 126 tok/s on tiny), so it stays opt-in.
`port/tools/build-engine.ps1` now stages the skels by **content hash** rather than "only if absent", so a stale catalog
cannot silently return.

## What works

| capability | state |
| --- | --- |
| host backend `ggml-hexagon.dll` | builds from the Hexagon SDK, loads, registers `HTP0` |
| DSP session | opens (`HTP0 allocating new session : domain 3 phys-idx 0 virt-idx 0`) |
| arch detected | `Hexagon Arch version v81`, `DMA64 enabled` |
| kernels | the trellis types are not implemented for HTP; the head matmul runs on standard types |
| MTP head on the DSP | does not fit (1.2 GB), and is not attempted |
| placement of the head (q6_K, 417 MB / 1.0 GB) | opens the session, then stops answering on the first decode |

`--geometry` on a working build lists four devices: `Vulkan0`, `GPUOpenCL`, `HTP0`, `CPU`.

## Build recipe

The DSP-side artefacts are **prebuilt and signed** (`libggml-htp-v73/v75/v79/v81/v85.so` plus `libggml-htp.cat`, shipped
beside the exe). Only the host library is built, so no certificate and no `signtool` are involved:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File port/tools/build-engine.ps1
```

That script finds the SDK, passes the definitions, builds `ggml-hexagon` first and stages the signed skels. Two details it
encodes:

* **`PREBUILT_LIB_DIR` must be defined at top level.** The SDK's `hexagon_fun.cmake` runs
  `string(FIND ${PREBUILT_LIB_DIR} ...)` as soon as it is included, and that variable is normally supplied by the skel
  sub-build. Without it the configure fails inside the SDK's own cmake.
* **Build only the `ggml-hexagon` target.** The `htp-v7x`/`htp-v8x` skels are `ExternalProject`s that need the DSP
  toolchain and a signing certificate; they are not dependencies of the host library, so they are never built.

## What it is worth, measured

| placement | tiny-qwen | 27B |
| --- | --- | --- |
| output head only (`--npu head`, default) | 100% of a 32-token prefill on the head, no loss | +3.4% |
| layer split CPU/GPU/NPU | -2 to -3% prefill per layer moved | same |
| dense placement (`--npu dense`) | -33% | -81% |

So the head is the only placement that paid when placement completed, and it is **not** the default: it is refused by
default since placement began hanging in this build (see the state at the top). A cooperative (concurrent)
design was studied and rejected: the paths share one command stream, there is no independent queue to overlap, and the
measured split costs more than the parallel section gains.

## Pitfalls

**FastRPC resolves the skel by bare name against the process's current directory.** `htp_iface_open` returns
`0x80000406` unless the engine's directory is the skel's directory during the load; this port changes directory for the
duration (`SkelDirGuard`). The DSP search-path API returns `0x14` (unsupported) on this driver, so it is not an option.

**An over-large mapping request wedges the DSP until reboot.** Anything that asks for more than the rpcmem pool kills the
session and the next call hangs. The rpcmem size is fixed by the `cdsprpc` driver; the constant in the backend must not be
raised past what the pool holds.

**A failed session is not an error path you can retry.** Once a session dies, the process must be restarted; the backend
will not recover it. That is why the engine only ever places the head by default and requires an explicit flag for
anything larger.

**The signed skels are part of the deployment.** They are not produced by this build; if they are missing from beside the
exe, `HTP0` will not register even though the host backend loads. `build-engine.ps1` stages them from the installed engine
component.
