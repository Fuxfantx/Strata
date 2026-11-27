# CPU + GPU together: measured, and not recommended

*状态：**记录**。结论已并入 [KNOWN-ISSUES.md](../KNOWN-ISSUES.md) 与 [PERFORMANCE.md](../PERFORMANCE.md)；逐次实验细节在 [PERFORMANCE-LOG.md](../PERFORMANCE-LOG.md)。 / Status: a record. The conclusion is in [KNOWN-ISSUES.md](../KNOWN-ISSUES.md); details are in the log.*

The engine can run more than one worker: a GPU copy plus a second process on the CPU backend, and a `--split-cpu`
layer split inside one process. Both were built and measured. Both are **slower than a single GPU stream** on this
machine, and the layer split also changes the generated text. Neither is part of the recommended path.

## Crossing a device boundary costs what it costs

| What | Measured |
| --- | --- |
| One round trip, 40 KB down + up, submitted and synchronised | **0.639 ms** per crossing (2026-11-24; down from 0.800 before the barrier work) |
| A per-layer split, 48 layers | 48 x 0.639 ms = **30.7 ms per token** of pure synchronisation |
| What the target (30 tok/s) allows per token | 33 ms |

So even if the transferred work were free, a per-layer split would spend about 93% of the budget on synchronisation.

## The layer split, measured

`--split-cpu <share>` puts that share of the blocks on the CPU backend. On the Q2 pack, one stream:

| Share | Decode | Generated text |
| --- | --- | --- |
| 0 (GPU only) | **22.2 tok/s** | `25,15451,369,279,` |
| 0.15 | 13.0 tok/s | `25,198,262,687,` |
| 0.30 | 9.5 tok/s | `25,198,262,220,` |

Two conclusions, both measured rather than assumed: the CPU side of a decode chain is about an order of magnitude
slower per matvec, and a chain is as slow as its slowest link; and the split **changes the text**, because the two
backends accumulate differently. The port keeps user-visible text identical as a hard rule, so this route fails that
rule before throughput is even considered.

## The two-process route

Two processes (one GPU, one CPU) do add throughput - about **+37% aggregate** in the measurement recorded in the log,
and the CPU side on this part reaches parity with a single GPU stream for some packs. That is a *throughput* answer,
not a latency one: it helps a server with several requests, and does nothing for one request. It also needs the
requests to be independent, and the oversized packs only pay for it when the mapped pack is already in page cache.

## What a future attempt would need

1. **A coarser boundary.** Synchronise per group of layers, or per expert block, so the fixed 0.639 ms is amortised over
   much more work than one layer.
2. **Text identity first.** Any split must reproduce the single-device token stream, or it belongs behind a switch that
   the docs do not recommend.
3. **An idle device.** The GPU is not bandwidth-saturated on a single stream (about 41 GB/s of a measured ~100), so the
   premise "the CPU can work while the GPU waits" is plausible; the measurements above say the *implementation*, not
   the premise, is what fails.

## Recorded, not recommended

The `--split-cpu` flag stays because it is how the negative result is reproduced; it is not a performance option.
