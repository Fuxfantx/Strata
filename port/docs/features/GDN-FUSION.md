# Fusing the gated-delta-net step: a closed route

*状态：**记录**。结论已并入 [PERFORMANCE.md](../PERFORMANCE.md) 的"已关闭路线"；本文件保留关键数字与结论，逐次实验细节在 [PERFORMANCE-LOG.md](../PERFORMANCE-LOG.md)。 / Status: a record. The conclusion is in the closed-routes list of [PERFORMANCE.md](../PERFORMANCE.md); details are in the log.*

The gated-delta-net (GDN) chain is the only place where a decode step still runs a long sequence of small ops: on
Qwen3.6-35B-A3B the chain is about 30 layers of `CONCAT` + `CPY` + `L2_NORM` + elementwise work, roughly 331 dispatches
per token. Fusing them looked like the largest remaining structural lever. It is not: every stage below was built and
measured, and the route is closed.

## What was measured, and what closed it

| Attempt | Result |
| --- | --- |
| Stage 1: fold the conv window into one kernel | The pair is correct and token-identical, but the measured gain is parity or slower. The per-op time attributed to those small ops is mostly **wait**, so removing the op does not remove the time |
| Dispatch-count experiment: shrink 331 dispatches to ~60 by removing non-matmul nodes | 35.72 against **39.7 tok/s** (1 424 against 331 dispatches) - fewer dispatches, slower. This is the measurement that closed the "small ops are the cost" hypothesis |
| `CONCAT` + `CPY` fusion | Blocked by one predicate: the adjacent `CPY` is a *different* copy, not the concat result (the dump showed it; the decode dump is in the log) |
| Stage 2: the pair as its own op, on the CPU first | Correct and token-identical, 9.8 tok/s (30 host round-trips), then 10.4 tok/s with `STRATA_GDN_PAIR=1`; the Vulkan kernel exists but its first dispatch faulted, and the fault was in the dispatch layer (a missing descriptor-set reservation), not the shader |
| Stage 3: fold the delta-rule input | Closed by arithmetic: the operands are produced after the point the fold would need them |
| In-place writing at the graph level | Slower |
| Fusing the normalisation into the delta kernel | No gain, and it changes the tokens |

## The state of the tree

The kernel and its wiring were **removed**; the default path never ran them, so nothing in the shipped path depends on
this work. The only trace is in the log.

## What this closes, in one sentence

On this device, a decode step's cost is the weight traffic and the latency of its dependencies, not the number of
dispatches around the GDN chain - so fusing that chain cannot pay, however clean the fusion is.
