# Performance — current numbers, what moved them, and what is closed

**本文件只写当前结论：现在的数字、把它们推动的改动、以及已被测量关掉的路线。**
Every number below was measured on this machine (Snapdragon X2 Elite Extreme, Adreno X2-90, Windows 11 ARM64) unless
a line says otherwise. The raw chronology, including the runs that were later withdrawn, is in
[`PERFORMANCE-LOG.md`](PERFORMANCE-LOG.md); treat that file as evidence, not as a guide.

## 1. Where the numbers stand / 当前数字

Model: Qwen3.8-Flash-Next Q2 (`q2_0` projection and expert tensors, `f16` attention projections).

| what | measured | how |
| --- | --- | --- |
| prefill, `--bench-pp 512` | **about 173 tok/s** | median of interleaved rounds, current defaults |
| prefill, `--bench-pp 2048` | **155-183 tok/s** | same, wider spread |
| decode, one stream, 8192 context | **17-22 tok/s** | 8 serial tokens, steady state |
| decode, one stream, 262144 context | 21-22 tok/s | same prompt and stream as above |

Two cautions, both measured:

- **Prefill drifts.** The same binary on the same prompt read 149 and 173 tok/s in adjacent rounds. Any prefill change
  needs interleaved rounds (A B A B) and a median; single runs are not evidence.
- **Decode is stable**, about 0.6% spread across rounds, so two or three runs are enough to judge a decode change.

## 2. Where a token's time goes / 一个 token 的时间去向

Decode is GPU bound: the host side of the graph costs about 6-8 ms per token against a graph that takes 46-59 ms. The
per-family shares below come from the profiler over three graphs:

| family | share of GPU time | notes |
| --- | --- | --- |
| `mul_mat_vec` (the f16 projections) | about 22% | one column per token, so a matvec, not a matmul |
| `mul_mat_id` (the experts) | about 5% measured, larger on other packs | gathers the selected experts per layer |
| elementwise and normalisation families | the rest | fused where possible; 2182 fusions per token across 9 patterns |

About 90% of decode dispatches have grids of 320 workgroups or fewer, and the median elementwise grid is a single
workgroup. Those are latency bound, not bandwidth bound, which is why merging them was worth trying and, in the end,
not available (section 4).

## 3. What moved the numbers / 已落地且测到收益的改动

| change | effect | evidence |
| --- | --- | --- |
| narrower per-buffer memory barriers instead of one global barrier | decode +6%, prefill +18% | paired runs, same binary |
| compute-only stage when a command needs no transfer | part of the above | same |
| split-K reduction through an event pair instead of a full barrier | prefill +2.9% | paired runs; `STRATA_VK_SPLIT_EVENT_BARRIER` reverts it |
| per-command-buffer barrier de-duplication | 383 skips per 8 tokens | counter, same run |
| empty dirty-set fast path | 275 skips per 8 tokens | counter, same run |
| per-thread output tile `4x4` to `8x4` (float) and `8x8` (quantised) | **prefill about 123 to about 173 tok/s** | interleaved rounds, byte-identical output; `STRATA_VK_MM_TILE` / `STRATA_VK_MMQ_TILE` revert it |
| per-buffer prealloc mask flags | neutral throughput, fewer flags cleared | paired runs |

The tile change is the largest single win of the last pass and the cheapest to revisit: the geometry is a startup
value, so a future device or pack can be re-tuned by environment variable instead of a rebuild.

## 4. Closed routes, and the number that closed each / 已关闭的路线

Each line is a route that looked promising and was measured shut. The numbers are from the runs recorded in
[`PERFORMANCE-LOG.md`](PERFORMANCE-LOG.md).

| route | why it is closed |
| --- | --- |
| cooperative matrix (`coopmat2`) | the device reports a 64x64 tile; 128x128 fails pipeline creation and 32x32 measures slower; dense cooperative is about 5% slower than the default path |
| larger matmul blocks (128x128 and friends) | 152-162 tok/s against 183 for the current block, so 64x64 is right on this chip |
| `f16` to integer projection | changes the output text, and text identity is a hard rule here |
| lossy 8-bit or 4-bit substitutions on the over-precision tensors | correlated 0.978-0.991 and changed a token at position 3-7 |
| two-stage projection, long-k single plane | both change the text and one measures slower |
| pre-scale folding into the consumer | a probe found 3 adjacent pairs and **0** safe fusions |
| SCALE batching through a task table | every SCALE in this model is in place, so deferring it is unsafe |
| ADD batching through a task table | instrumented: every flush carried **one** task, so the batch never merged anything; the earlier apparent gain was noise |
| persistent kernels | same prerequisite as the two lines above: no adjacent dispatches to merge |
| command replay | blocked by command-buffer lifetime; the descriptor bank is frozen but inert |
| lazy or sequence-based dependency tracking | three variants produced wrong output |
| CPU + GPU split (`--split-cpu`) | 15% of the layers on the CPU costs about 41% of decode throughput **and** changes the text |
| cache warming, multi-queue submission | no effect, or an artefact of the measurement harness |
| per-layer CPU/GPU split | the host round trip measures 0.639 ms, so 48 layers would spend about 30.7 ms per token on synchronisation alone |

## 5. How to measure here / 这台机器上的测量方法

1. **One engine instance at a time.** Two instances contend for the same GPU; one appears idle while the other drops
   from 17 to 6 tok/s. Check for stale `strata-vk` processes before every round.
2. **Interleave.** Run A, B, A, B and compare medians. Report the spread next to the median.
3. **Treat the window as part of the result.** If the same configuration reads 12% apart, the effect has to beat that
   before it counts.
4. **Gate on text, not only on speed.** Every change above is judged together with the recorded output stream, so a
   faster-but-different result is a rejected change rather than a win.

## 6. What is left / 还剩什么

- **Prefill**: the float projection still runs at roughly a quarter of the device's theoretical float rate, so kernel
  work on the matmul (tiles, per-thread outputs, shared-memory staging) is the only route with an order of magnitude
  left in it. The tile sweep took the cheap half of that.
- **Decode**: bound by the weight traffic of a single column, about 1.7 GB per token on this pack. Every route that
  reduces those bytes changes the text, so the remaining work is in the expert matvec kernel itself, not in the
  amount of data.
- **Batching**: `--parallel N` shares one copy of the weights across requests. The numbers for this pack have not been
  re-measured since the tile change; the 32 and 64 slot figures in the log predate it.
