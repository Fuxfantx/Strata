# The Adreno X2-90, as measured

Durable facts about the device in this laptop, and the limits they put on this port. Everything here was measured on this
machine; the run behind each number, and the experiments that were ruled out on the way, are in
[`PERFORMANCE-LOG.md`](PERFORMANCE-LOG.md) Appendix A. The current performance of the port is in
[`PERFORMANCE.md`](PERFORMANCE.md).

## The device

| Fact | Value | How it was established |
| --- | --- | --- |
| GPU | Adreno X2-90, Vulkan 1.4, driver `30.0.228.10000` | `vulkaninfo` and the engine's banner |
| Compute units | 8 | device properties |
| Subgroup | 64, with arithmetic | `--geometry` probe and the cooperative-matrix probe |
| Shared memory | 32 KiB per workgroup | device properties |
| Memory | 47.6 GB unified, no separate VRAM | operating system and the engine's allocation probes |
| Integer dot product | available | `intdot 1` in the probe below |
| Cooperative matrix | only `64x64x16` f16-accumulate and `64x64x32` int8 | the probe below; requesting any other shape fails pipeline creation |
| `f32`-accumulate cooperative matrix | **absent** | same probe (`f32acc 0`) |
| `bf16` | **absent** | device properties |
| Stream read ceiling, GPU | about 100-108 GB/s | `torch`-free read kernel measured through the engine; the platform specifies 228 GB/s, so the measured ceiling is roughly half of it |
| Stream read ceiling, CPU | about 155 GB/s at 14 threads | same measurement on the CPU backend |

The cooperative-matrix probe, which is the single most load-bearing measurement here:

```text
STRATA_VK_COOPMAT_LOG=1
[coopmat] vendor 0x5143 arch 10 coopmat 0 (m/n/k 64/64/16, f32acc 0 f16acc 1) int 1 (m/n/k 64/64/32) shmem 32768 intdot 1
```

`coopmat 0` on this line means the f32-accumulate form is not available and the cooperative path has to accumulate in
f16. That single fact is why the cooperative path cannot be certified against the CPU oracle, and it is why that path is
gated to shapes where its error stays inside the acceptance in `PERFORMANCE.md`.

## What the device cannot do, and what that closes

| Limit | Consequence |
| --- | --- |
| No f32-accumulate cooperative matrix | the cooperative f16 path cannot be made oracle-certifiable here, so it stays behind an accuracy gate rather than becoming the only path |
| Only `64x64` cooperative tiles | `128x128` fails pipeline creation; smaller tiles measured 57-77% slower, so the tile is not a free parameter |
| Half of the specified bandwidth in practice | a memory-bound plan has to be judged against about 100 GB/s, not 228 GB/s |
| 32 KiB of shared memory | the tiling of the matrix kernels is bounded by it; the larger configurations measured worse |
| No `bf16` | the f16 formulations are the only 16-bit ones available |

## How to measure here

The machine is a laptop with a shared memory bus, so the measurement discipline matters as much as the device facts:

1. **Prefill readings drift** - the same binary on the same prompt read 149 and 173 tok/s in adjacent rounds (about 12-16%).
   Judge a prefill change from interleaved rounds and a median, never from one run, and report the spread with it.
2. **Decode readings are stable** - about 0.6% across rounds, so two or three runs are enough to judge a decode change.
3. **One engine at a time.** Two `strata-vk` processes share the GPU and both slow down; one of them can also look hung
   (a low thread count and no new stderr while the disk is idle). Check for a leftover process before measuring.
4. **A 262 144-token context takes minutes to load and warm**; that is not a hang.
5. **Do not page.** A benchmark that pushes the resident set past physical memory measures the page file, not the GPU;
   the configs used in the docs fit in 47.6 GB with room to spare.
6. **Compare inside one window.** A number from an hour ago is not a baseline on this machine; a same-day, same-window run
   of the build being compared against is.

The engine prints its own device report at startup, so any claim above can be re-derived on this machine:

```powershell
build\bin\strata-vk.exe --native C:\models\pack.gguf --tokens 9707 --n 4 --max-context 8192 --kv q4_0 --flash-attn on
```
