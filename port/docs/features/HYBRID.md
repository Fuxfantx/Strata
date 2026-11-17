# The CPU + GPU + NPU path, as measured

> **Status (2026-11-18).** A measurement record from the hybrid investigation. Its verdicts are in `PERFORMANCE.md`
> §4: layer splitting, `--cpu-moe` and cross-engine token-level parallelism all lost on this machine (the two-process
> shape gains 33% throughput at 2.44x the energy per token), and the NPU contributes nothing (no DSP session opens).
> The token rates below are the ablations taken **before** two later changes: the cooperative-matrix default, which was
> withdrawn for being numerically wrong, and the epilogue fixes in the CPU port. The same commands now measure
> 151.7 / 77.9 / 17.4 / 2 826 prefill and 37.8 / 7.0 / 3.3 / 118.9 decode on the shipped default (correct kernels,
> `PERFORMANCE.md` §1). What the note establishes - what a token is made of and how the three engines divide the work -
> is unchanged.

A token runs on the Adreno. The other two engines have one job each that pays, and one role each that does not. This note
is the measurement behind that sentence, so the next attempt starts from numbers instead of from the idea.

## Reading the utilisation numbers

Windows' `GPU Engine` counters do not describe a Vulkan-compute workload on this driver: during a 512-token prefill at
148.9 tok/s the only non-zero instance is `engtype_3d` at 3.2%, and during decode at 39.6 tok/s it sums to about 12%.
The same runs are doing roughly 4 TFLOP/s of int8 tile work (prefill) and about 83 GB/s of weight traffic (decode:
39.6 tok/s x ~2.1 GB per token, against the 110-130 GB/s this part sustains), so the device is not idle - the counter is
measuring a pipeline this port does not use. Read tokens/s, or the port's own per-op figures, instead. The NPU has no
Windows utilisation counter at all, so 0% there says nothing either.

## Where the hybrid stands after the trellis fixes

Both Qwen3.8-Flash-Next packs now execute as **one scheduler split on Vulkan**: attention, the dense blocks and the
trellis experts all live in device buffers, and the CPU side is a single gather into the per-layer embedding table
(27 GB, kept in an mmap'd host buffer because the pack is larger than the 47.6 GB of RAM). Per decode step the GPU split
costs 287 ms on the trellis pack and 74 ms on the Q2 pack; the CPU split costs 26-460 microseconds. There is nothing
left to move for these packs: the only remaining host-resident tensor is the table, and putting it on the device too
would need 62 GB of device buffers on a 47.6 GB machine.

That changes the hybrid question. The NPU is still not a usable third unit in this build: the HTP backend opens a DSP
session and then stops answering on the first decode (reproducible, so placement is refused by default), the trellis
types have no HTP kernels, and the one placement that ever paid when it completed was the output head (+3.4% on the
27B). A CPU+GPU+NPU split is therefore not on the table until the DSP-side hang is explained; the useful unit on this
machine is the GPU, with the CPU left as the paging path for weights that do not fit.

## What each engine is worth on this part

| engine | measured | instrument |
| --- | --- | --- |
| Adreno X2-90, streaming | 120 GB/s (f32 add, 768 MB touched) | `mm-probe --chain 1 -k 67108864` |
| Adreno X2-90, one 7 MB q4_0 matvec | 44 us of work after the 100 us submit/fence, i.e. ~160 GB/s | `mm-probe -t q4_0 -m 4864 -k 2560` |
| Adreno X2-90, small matvec in a graph | 7 us marginal per dispatch (1 490 of them per MoE token) | ablation + fit, see below |
| CPU, 18 Oryon cores | 74.4 GB/s matvec, 594.9 GFLOP/s GEMM | `ADRENO.md` |
| Hexagon HTP v81 | session opens (8 HVX, 8 MB VTCM), placement hangs | `NPU-FEASIBILITY.md` |

The SoC has one LPDDR5X pool. A CPU read and a GPU read contend for the same bandwidth, which is why a *sequential*
CPU+GPU split of a bandwidth-bound op cannot win, and why a *concurrent* one only wins while the GPU is leaving
bandwidth unused - and it does leave it unused on small ops (the MoE token moves 1.86 GB in 27.8 ms, ~67 GB/s of a
~130 GB/s part).

## The cost of crossing a device boundary

`mm-probe --handoff N` (added by this work) times one split boundary: the device computes an N-float tensor, the host
reads it back, the host writes it back, in a loop.

| N floats | compute | + readback | + upload |
| --- | --- | --- | --- |
| 2 048 (8 KiB) | 69.9 us | 166.9 us (+97.0) | 143.0 us |
| 5 120 (20 KiB) | 81.7 us | 159.4 us (+77.6) | 139.6 us |
| 17 408 (68 KiB) | 92.5 us | 146.8 us (+54.3) | 154.8 us |
| 65 536 (256 KiB) | 103.8 us | 155.8 us (+51.9) | 174.2 us |

The readback is 52-97 us and barely moves with the size: it is the fence, not the bytes. That is the number that decides
the question, because the glue nodes a CPU side would take over cost 18 us (a 5 120-wide RMS norm) to ~100 us (a router
chain) on the GPU. A split boundary is therefore more expensive than the node it moves, before the CPU does any work.
Per-op CPU offload loses on this device by construction; it is not a matter of a better CPU kernel.

## Where a decode token's time goes

Qwen3.6-35B-A3B, 36 tok/s = 27.8 ms/token, ~1.86 GB of active weights per token:

- ~1 490 Vulkan dispatches x ~7 us = ~10 ms of dispatch and the barriers between them;
- ~1.86 GB at the part's practical ~110 GB/s = ~17 ms of DRAM.

`GGML_VK_ABLATE=<mask>` (a measurement-only patch, not in the tree) drops a whole op class from the graph with wrong
results by design. Dropping every non-matmul node moved decode from 37.6 to 52.1 tok/s (+39%). The per-class numbers are
approximate because the masks are op-typed, not tree-typed: SOFT_MAX/SCALE/UNARY/ADD/MUL together are +10.7% but cover
the router, the GDN l2 norms and the gating chains at once; RMS_NORM +5.5%, SSM ops +4.9%, GET_ROWS +3.4%, CPY +2.4%,
CONCAT +1.1%, GLU 0%. The classes overlap, so they sum to less than the whole. The conclusion is not "move the glue to the
CPU": it is "give the GPU fewer, fatter nodes".

### The recurrent-state gather, and the guards it needed

The largest single item was the recurrent-state gather: with one sequence and one state row, `build_rs` gathers the only
row the index input can name - a 2-3 MiB copy per GDN layer per token, 60 of them. The graph now reads the cache tensor
itself in that case (`src/llama-graph.cpp`), the tokens stay bit-identical, and the crash that used to hide behind it was
a chain of buffer assumptions worth recording because it is what made the item look blocked:

* the state cache is on the device (`cache_s_l*` is a `Vulkan0` buffer), so a view of it is an ordinary device view - the
  "it is host memory" explanation was wrong;
* the meta split state asked `ggml_backend_buffer_get_usage(tensor->buffer)` for a tensor with no buffer yet; a
  buffer-less tensor now takes the unsplit state instead of querying a device (`ggml-backend-meta.cpp`);
* the last crash was not in the allocator at all. `llama_context` calls `set_inputs` after allocating the graph, and with
  the gather gone nothing consumes the state-copy index tensor, so the graph never allocates it - and `set_input` wrote it
  anyway. `llm_graph_input_rs::set_input` and `llm_graph_input_mem_hybrid::set_input` now skip a copy tensor with no
  buffer, which is the case `llama-context.cpp`'s own FIXME describes;
* measured on the same binary, interleaved: the 35B-A3B decodes **37.7 tok/s against 36.1** with the gather (+4.6%), the
  27B 7.04 against 7.00 (+0.6%) and prefills 77.9 against 75.8 tok/s (+2.8%), tokens bit-identical on both models.

The next graph items (in-place GDN state write, conv ring buffer, l2-norm fold, `kqv` cont removal) see the same
buffer-less-input pattern, so they start from a tree that already handles it.

What did not help: `lldb.exe` ships with the VS 18 LLVM tools but exits with `unable to find 'python311.dll'` (and even
with an ARM64 3.11 DLL beside it, it stops on a delay-load failure), and a `-gline-tables-only` clang-cl build without
`/DEBUG` has nothing for `llvm-symbolizer`. What did work was naming the caller: a module-plus-offset print mapped
against the DLL's export table, then `typeid(*input).name()` on the graph inputs.

## What the hybrid is, then

- **GPU**: the whole graph. Matmuls and glue.
- **CPU**: the sampler, the prompt-lookup drafter, the tokeniser, the protocol. All of it already there, all of it
  overlapping the GPU's work by nature (it runs while the GPU is busy).
- **NPU**: reported, refused for placement unless `STRATA_VK_NPU_FORCE=1`, because placement hangs (see
  `NPU-FEASIBILITY.md`).

The two designs that could still add another engine's throughput, with their measured or estimated ceilings:

| design | ceiling | why it is not here |
| --- | --- | --- |
| per-op CPU glue | negative | a 52-97 us handoff against 18-100 us of node cost |
| concurrent CPU+GPU split of one matvec | negative on the 27B (already at 100 GB/s of ~130), ~1.2-1.4x on the MoE's small ops | needs an op the scheduler can split by row across two backends, and a combine; row splits are exact but every layer boundary costs a handoff |
| chunk-pipelined layer split for prefill | ~+15% (a 5:1 layer split balanced for the CPU's 5x slower GEMM, pipelined across chunks) | llama.cpp's scheduler runs one split at a time; this needs a chunk-level pipeline inside the engine, and 3.4 s of MoE prefill is the only place the ~100 us boundaries amortise |
| two sequences, one per engine | throughput only, not latency | the engine serves one request at a time by contract, and the app starts one engine per chat |

The path that remains for decode is more graph surgery on the GPU, not another device. Two items are in: the identity
state read (4.6% on the MoE) and the in-place state write, where the gated-delta-net kernel writes the new state straight
into the recurrent cache view instead of a tail that a copy then moved (+2.4%, interleaved 38.12 against 37.23 tok/s,
tokens bit-identical). What is left - folding the l2-norm scale into the norm and dropping the `kqv` cont at one token -
is worth a few percent; the conv-state copy measures flat when removed, so its ring buffer is not worth an operator change.

## What a multi-token pass costs, and why speculation still loses

A verification pass is `n = 1 + drafts` tokens through the same graph. Measured (`--bench-pp n`, medians):

| n | 27B | 35B-A3B | tiny |
| --- | --- | --- | --- |
| 1 | 146.6 ms | 29.8 ms | 7.9 ms |
| 2 | 183.3 ms (1.25x) | 40.2 ms (1.35x) | 10.0 ms (1.27x) |
| 3 | 253.5 ms (1.73x) | 51.2 ms (1.72x) | 11.0 ms (1.39x) |
| 4 | 328.3 ms (2.24x) | 62.1 ms (2.08x) | 12.9 ms (1.63x) |

Those are with `GGML_VK_DISABLE_MMVQ` (the fp32-dequant matvec) and the multi-column weight reuse; before both, the 3-token
pass on the 27B was 322.7 ms and on the 35B-A3B 82.4 ms. So a 3-token pass costs ~1.7 single steps: profitable only if the
drafter lands better than ~1 of 2 drafts. Measured prompt-lookup on a repetitive code prompt: **2 of 11 drafts kept**
(18%), 5.6 tok/s against 7.0 without, so it stays opt-in. The verify pass is no longer the obstacle; acceptance is.

The MTP head now loads (it did not before: the graft merges the whole head block, but the appended block asks only for the
six `nextn.*` tensors, so `done_getting_tensors` threw 866 against 851; the check now tolerates the unused remainder when
a head was grafted - `src/llama-model.cpp`), and it drafts correctly: **8 of 12 kept (67%) on the 27B**, 8 of 29 on the
35B-A3B with a real prompt. It is still slower than plain decoding - 5.8 against 7.0 tok/s on the 27B, 11.9 against 36.2
on the MoE - because every drafted token pays a full forward of the 1.7-2.2 GB head, and the verification round costs
roughly a 5-token pass. So it stays opt-in. One caveat found while measuring: with MTP enabled the greedy continuation
differs from plain decoding, so the MTP path is not output-equivalent to the target alone.

## Numbers a future attempt has to beat

| attempt | measured |
| --- | --- |
| 128x128 MMQ tile, forced on Qualcomm | 0.40x prefill (tiny 1 110 against 2 786 tok/s; MoE 37 against 146) |
| int8 MMVQ forced for decode matvecs (`GGML_VK_FORCE_MMVQ=1`) | 0.65x MoE decode (35.1 -> 22.8 tok/s), 0.74x tiny (123.6 -> 91.7): the fp32 dequant matvec is the better decode kernel on this driver |
| async submission / two queues | wrong answers for the same prompt (`ADRENO.md`) |
| `KHR_coopmat` | wrong results, not just slow |
| fp16 shader variants on a dense model | 0.26x prefill |
| NPU head placement | opens the DSP session and stops answering on the first decode (417 MB and 1.0 GB q6_K heads) |
| NPU dense / layer-split placement | -33% / -2 to -3% per moved layer (`NPU-FEASIBILITY.md`) |
| the recurrent state write as a `vkCmdCopyBuffer` instead of a dispatch | neutral (36.53 against 36.71 tok/s); dropping the write entirely is +3.5%, so the traffic is the cost, not the launch |
| the conv state write removed entirely | neutral (36.24 against 36.41), so it is not worth an operator change |

## Can the CPU add throughput while the GPU runs? Measured: yes, about 63%

Item (9) assumes the CPU has spare bandwidth while the GPU is at its wall. That was measured with the same kernel on both
sides (`mm-probe --chain 1 -k 67108864`, a 256 MiB add chain, 200 repetitions each):

| run | CPU | GPU | aggregate |
| --- | --- | --- | --- |
| CPU alone | 62.3 GB/s | - | 62.3 |
| GPU alone | - | 68.5 GB/s | 68.5 |
| both at once | 48.8 GB/s | 63.0 GB/s | **111.8 GB/s** |

The CPU loses 22% and the GPU 8%, but the sum is **1.63x** the best single-device figure. So the memory system does have room for
both engines, and the premise behind item (9) holds.

What that does *not* mean: llama.cpp runs one graph on one backend at a time, so splitting a single token across the two
devices would serialise them (GPU layers, then CPU layers, with a sync) and gain nothing - the CPU's layers are also the slow
ones. The 63% is only reachable by **overlapping independent work**:

1. **two slots, one per backend** (server mode: a GPU slot and a CPU slot, each taking a request). The CPU slot is much slower
   per token, but the aggregate throughput when two requests are in flight is what the measurement above bounds;
2. **an executor that overlaps layers of one token** (CPU computes layer N+1 while the GPU computes layer N). Nothing in this
   tree does that today; it is a new executor with per-layer fences, not a placement change.

Item (9) is therefore re-scoped: option 1 is a server-mode feature worth up to +63% aggregate and is implementable in the
engine; option 2 is a new executor and is out of scope for a placement/config change.

## Decode on the CPU is at parity with the GPU here, and two engines add 37%

The bandwidth result above says the CPU has room; what matters for item (9a) is what decode actually does on each side:

| model, decode | CPU | GPU |
| --- | --- | --- |
| 27B Q4_0 | **7.40 tok/s** | 7.24 |
| 35B-A3B | 35.77 | 39.32 |

The CPU is at parity on the 27B and 91% on the MoE. That is the bandwidth ceiling showing up directly: the GPU's advantage in
this port is **prefill** (cooperative matrix: 172 against the CPU's much lower pp512), not decode.

So two concurrent engines, one per backend, on the same pack:

| 35B-A3B, two requests | tok/s |
| --- | --- |
| GPU alone | 39.3 |
| CPU alone | 35.8 |
| **both at once** | **33.8 + 20.0 = 53.8 (+37% against the GPU alone)** |

Both sides lose throughput to the shared memory system (GPU -14%, CPU -44%), but the aggregate is 37% above the best single
device. That is the number item (9a) is built on: one GPU slot plus one CPU slot, each with its own model instance.

Cost to keep in view: a second model instance is a second copy of the weights. The packs that fit do (27B 14.5 GiB, MoE 18.4 GiB,
tiny), the 54-62 GiB packs do not, and on this UMA part "device memory" is the same RAM, so the copy is real RAM either way.

### The oversized packs *can* pay it, because a mapped pack is shared page cache

The paragraph above is wrong for the packs that are loaded with `--load-mode mmap` (every pack larger than RAM), and it is wrong
in the one direction that matters: two processes that map the *same file* share its pages. There is no second copy — there is one
page cache and two readers of it. Measured on Gyro-S (54.5 GiB pack, 47.6 GiB machine, 38 GiB free), `--bench-tg 32` each:

| Gyro-S, one token stream | decode |
| --- | --- |
| GPU alone | 9.14 tok/s |
| CPU alone (12 big cores) | 8.10 tok/s |
| **both at once, one shared map** | **9.34 + 3.40 = 12.74 tok/s (+39% against the GPU alone)** |

So the aggregate shape is the same +39% the 35B-A3B showed, on the pack that was said to be unable to pay for it. What it does cost
is memory: free physical memory bottomed out at **0.7 GiB** during the pair, against 30 GiB for the CPU engine alone, and that is
why the CPU side falls to 3.4 tok/s — it is not losing to the GPU, it is losing to the page fault rate. The GPU is unaffected
(9.34 against 9.14: it is latency-bound, not bandwidth-bound, so the other process's faults do not touch it).

⇒ Item (9a) is available on the oversized packs after all. The gate is `apps/host/server/native-pool.mjs` (which is not in this
tree — `HYBRID.md`'s earlier claim that it was "built, tested, 6 cases" does not hold at this commit, and building it is the work
that remains). The rule it needs is not "does the second copy fit" but **"is the pack mapped?"** — if it is, a second worker costs
page cache pressure, not RAM. The honest starting numbers are the row above: aggregate +39%, second request ~3.4 tok/s.

## What item (9a) actually needs: a concurrent protocol, not a second backend

The measurement says a GPU slot plus a CPU slot is worth +37% aggregate, but the engine cannot expose it today, and the reason
is not the backends:

* `strata --serve` is one `Engine` (one model, one context, one sequence) and one line-in/line-out loop
  (`vk-serve.h`: `int serve(Engine & engine, CommandQueue & q)`), with the stdin reader on its own thread only so STOP/QUIT are
  seen during a long decode;
* the protocol answers one command with one answer line, and `serve/server.py` indexes into the DONE fields positionally, so two
  requests in flight would interleave their `T`/`PP`/`DONE` lines and the client would mis-parse them.

So the work is: (1) a pool of engines, one per backend, in the engine; (2) a protocol that can carry more than one request at a
time. The minimal version that stays backwards compatible is a request id:

```
GEN id ...            ... T id .. / PP id .. / DONE id ...
```

with a request without an id behaving exactly as today (the contract tests keep passing), and `POOL` reporting
`POOL 2 vulkan cpu`. The engine then runs one thread per engine and multiplexes by id; the client must accept ids before it sees
any gain - without that, the second backend only makes requests *faster on average*, not concurrent.

Two things worth recording before the work starts:

* the second engine costs a second copy of the weights (27B 14.5 GiB, MoE 18.4 GiB, tiny; the 54-62 GiB packs cannot pay it),
  and on this UMA part that is real RAM rather than a separate pool;
* (9b) - overlapping layers inside one request - needs no protocol change but a new executor with per-layer fences, because
  llama.cpp runs a graph on one backend at a time. It is the only form of the split that helps a *single* request.

So (9a) is engine + protocol work with a client-side dependency, and (9b) is a new executor. Neither is a placement flag, which
is why this note exists before either is started.

### Two routes to (9a), and which one is cheaper

| route | what changes | gain | cost |
| --- | --- | --- | --- |
| **A. app-side pool** | the host launches two engines (one `--gpu vulkan`, one `--gpu cpu`) and gives each request to a free one | the +37% measured above, for two concurrent requests | a second copy of the weights; no protocol change, no engine change |
| **B. engine-side pool** | request ids in the protocol plus a thread per engine inside `strata --serve` | the same +37%, and only one process to manage | engine + client protocol change, contract tests to extend, still a second model copy |

Route A is the cheaper one and it is exactly what the two-process measurement simulated: each request still runs on one engine
with the unchanged protocol, and the pool only decides *which* engine. Route B is worth it only if the product wants a single
engine process to own both backends (for example to share the tokeniser and the MTP head, or to keep one stdio pipe).

For the packs this machine can hold: 27B (14.5 GiB) and 35B-A3B (18.4 GiB) fit twice, tiny fits many times, and the 54-62 GiB
packs do not fit once - so the pool has to be a per-model decision with a memory guard, not a global switch.

### Route A, first piece: the pool rule (tested)

pps/host/server/native-pool.mjs decides how many workers a model gets and on which backend, as a pure function so it can be
tested without an engine (pps/host/tests/native-pool.test.mjs, 6 cases):

* one worker on the GPU unless the second copy of the weights fits beside 1 GiB of system room, in which case one worker per
  backend (Vulkan + CPU);
* the 54-62 GiB packs never get a second worker, and the reason string says why (the page can show it instead of the setting
  silently doing nothing);
* the setting off, an install with one backend, or unknown sizes all fall back to one worker.

The plumbing - starting the second engine and routing a request to whichever worker is free - is the next piece; it needs no
protocol change because each request still goes to one engine.

> **Correction (2026-11-18, `PERFORMANCE.md` §4).** `native-pool.mjs` is not in this tree at this commit, and the rule
> above is wrong in one place: a 54-62 GiB pack *can* have a second worker, because a pack loaded with `--load-mode mmap` is one
> page cache shared by both processes, not two copies (measured: GPU alone 9.14, CPU alone 8.10, both at once 9.34 + 3.40 =
> 12.74, +39%). What already exists here is the rest of route A, in Python: `serve/replicas.py` has `ReplicaEngine`, `Router`,
> `replica_cfg`, `gpu_groups`, and the server has `--replicas N` and per-slot replica routing. Its partition dimension is the GPU
> *card*, so on a one-card machine `replicas` yields nothing. The delta is one controlled extension of `gpu_groups` /
> `replica_cfg` to admit a replica with no cards that runs `--gpu cpu` - not a new module, and not a protocol change.

### Item (9b): what a layer split needs (a failed first attempt, recorded)

The first attempt at a layer split was `--cpu-layers N`: put the first N blocks' tensors on the CPU buffer type through the same
`tensor_buft_overrides` mechanism as `--cpu-moe`. It does not work, and the failure is informative:

```
[strata-vk] first 8 block(s) on the CPU backend
Assertion failed: !isnan(x), file ggml-cpu/ops.cpp
```

The CPU backend is handed activations that still live in Vulkan buffers, so it reads them as host memory and the first thing it
computes is NaN. Moving *weights* across the boundary is not enough - the scheduler also has to copy the activations, and it only
does that for devices it knows about.

The correct mechanism is llama.cpp's own **layer split between devices**: name both `{Vulkan0, CPU}` in
`llama_model_params::devices`, set `LLAMA_SPLIT_MODE_LAYER`, and give a `tensor_split` proportion. The scheduler then owns the
copies and the cross-device boundaries. That is the prerequisite for any (9b) measurement, and it is a small change to
`plan_devices`/`load_model` rather than a new placement rule.

The attempt was reverted; the tree is back at the committed state (`--cpu-layers` does not exist) and 27B decode is 7.49 tok/s.

### `--split-cpu <share>`: the layer split, measured (item 9b's prerequisite)

The engine now names both devices and lets the scheduler own the boundary
(`llama_model_params::devices = {Vulkan0, CPU}`, `tensor_split = {1-share, share}`), which is the mechanism the failed
`--cpu-layers` attempt needed. Measured on the 27B (64 blocks, decode 8 tokens):

| CPU share | decode | prompt |
| --- | --- | --- |
| 0 (GPU only) | 7.44 tok/s | 489 ms |
| 0.25 | **crashes** | - |
| 0.5 | 6.08 tok/s | 1 558 ms |
| 0.75 | 5.95 tok/s | 2 069 ms |

So a **serial** 50/50 split costs 18% on decode (and 3.2x on prefill, which crosses the boundary with a full batch). Two things
follow:

* the split works and is cheap enough that a pipeline has something to recover: with 67 ms of GPU work and 67 ms of CPU work per
  token (both halves of a 134 ms token), a perfectly overlapped pipeline would be bounded by `max(67, 67)` plus the boundary
  cost, i.e. ~+38% for a single stream and up to ~2x aggregate for two;
* `--split-cpu 0.25` **crashes** (`Assertion failed: scale > 0.0f` in ggml-cpu) while 0.5 and 0.75 run, so the layer-split path is
  not yet reliable at every share. Any (9b) work has to start by making the split stable for all shares, or by only admitting the
  shares that are known to work.

### Both split routes corrupt the GPU/CPU boundary, so (9b) starts here

Two mechanisms were tried and both fail the same way:

| route | result |
| --- | --- |
| `--cpu-layers N` (tensor_buft_overrides) | `Assertion failed: !isnan(x)` in ggml-cpu |
| device split, `devices = {Vulkan0, CPU}`, `tensor_split` | works at 0.5/0.75 (`5.1-6.1` against 7.47 tok/s), **`Assertion failed: scale > 0.0f` at 0.25** |
| offload count, `mp.n_gpu_layers = N`, one device | `Assertion failed: !isnan(x)` at 0.25, 0.5 and 0.75 |

The asserts are both "the CPU backend was handed an activation it could not have produced": `!isnan(x)` in a CPU op and
`scale = 1/sqrt(mean+eps)` collapsing to 0 in `ggml-cpu`'s RMS_NORM because `mean` was inf. So the boundary itself is unreliable in
this tree, and it is not a matter of which knob asks for the split.

Where to look: the Vulkan backend must refuse an op whose inputs are not in a Vulkan buffer, so the scheduler inserts the copy
(`ggml-vulkan.cpp` already has the predicate at `ggml_backend_vk_buffer_type_name` / the buffer check near line 13732, and
`ggml_backend_vk_device_supports_op` is where it has to be applied). Until that holds for every op, a GPU/CPU layer split will
keep producing garbage at some share.

That makes (9b)'s first task a **correctness fix in the backend boundary**, not an executor: with the split stable, the numbers
above already say what a pipeline has to beat (a serial 50/50 split is 18% behind a pure GPU run, and the pipeline bound is
~+38% for one stream). The engine-side `--split-cpu` flag stays as the measurement knob (it works at 0.5 and 0.75), and it is
documented as unusable at 0.25.

### The Vulkan-side boundary check does not fix the 0.25 crash

`ggml_backend_vk_device_supports_op` was made to refuse nodes whose sources are not Vulkan-resident (the predicate the backend
already has is `ggml_backend_buffer_is_vk`), on the theory that the scheduler would then insert the cross-backend copy. Measured
after the change:

| CPU share | before | after |
| --- | --- | --- |
| 0 | 7.44 | 7.46 |
| 0.25 | crash (`scale > 0.0f`) | **crash (`scale > 0.0f`)** |
| 0.5 | 6.08 | 5.94 |
| 0.75 | 5.95 | 5.72 |

Nothing changed, so the node that breaks is not a Vulkan node holding a host buffer: the failure is on the **CPU** side (an
RMS_NORM whose input is inf), which means an op *assigned to the CPU* is reading a Vulkan-resident activation. The symmetric check
- the CPU backend refusing non-host sources - is the next place to look, and that lives in `ggml-cpu` (`ggml-cpu/ops.cpp` /
`ggml-cpu.c`), which this overlay deliberately does not carry.

The Vulkan-side change was reverted (it was unproven and can push nodes to the slower backend). Recorded conclusion: (9b)'s
prerequisite is a **CPU-backend boundary check**, and the decision to touch `ggml-cpu` (a file outside the overlay's 47) belongs
to the owner of the port, not to a silent edit.

### The CPU-side boundary check (overlay grew by one file) and what it settled

The owner's decision was to allow the CPU half of the boundary rule, so the overlay now also carries
`ggml/src/ggml-cpu/ggml-cpu.cpp` (48 files, 0 mismatches) with: a node whose source is not in a host buffer is refused by the CPU
backend, so the scheduler inserts the copy. That is the symmetric rule to the Vulkan one.

It did **not** fix the 0.25 share: 27B decode is still 7.42 at share 0, still crashes at 0.25, and 0.5/0.75 measure 5.82/5.75. Both
halves of the rule are now in place and the failing node is neither a Vulkan node with a host source nor a CPU node with a device
source - so the remaining defect is elsewhere in this tree's GPU/CPU split handling. Nothing regressed: 35B-A3B decode 40.16 tok/s,
MUL_MAT oracle 38/38 in both precision modes.

### Item (9b) is not worth building - measured, not assumed

With the split measurable at 0.5, the pipeline's bound can be computed from the serial numbers. On 35B-A3B:

| configuration | tok/s | ms/token |
| --- | --- | --- |
| GPU only | 34.8 | 28.8 |
| CPU only | 35.6 | 28.1 |
| serial 50/50 split | 23.2 | 43.2 |

Each half of the token is ~14 ms, and the serial split costs 43.2 instead of ~28, i.e. **~15 ms of cross-device copy per token**. A
two-stage pipeline's steady-state period is `max(stage) + copy` ~ 29 ms, which is 34.5 tok/s aggregate - **no better than the GPU
alone**, and far below the 53.8 tok/s that two engines already reach (item 9a). The 27B says the same thing more mildly: a pipeline
bound of ~+34% against 9a's measured +37%, with worse single-request latency because every token still crosses both stages.

So the answer to "is 9b good enough to replace 9a" is **no**, and (9a) stays. The boundary rule stays too - it is a correctness
fix, and a layer split is still the only way to run a pack that does not fit in device memory at all - but no executor is built on
top of it.

### The 0.25 share crash is flash attention, and there is a workaround

Localised by elimination on the 27B:

| run | result |
| --- | --- |
| `--split-cpu 0.25`, default policy (FA on at 32k context, 64 layers) | crash: `scale > 0.0f` in ggml-cpu |
| `--split-cpu 0.25 --flash-attn off` | **runs, 6.17 tok/s** |
| `--split-cpu 0.5`, default policy | runs, 5.80 tok/s |
| `--split-cpu 0.75`, default policy | runs, 5.72-5.95 tok/s |
| `--split-cpu 0.75 --flash-attn off` | runs but **0.83 tok/s** (non-FA attention on 48 CPU layers) |

So the split and flash attention do not mix at some shares, and the failure is an inf activation reaching the CPU side; with FA
off every share runs. Forcing FA off whenever a split is active was tried and reverted: it makes 0.5 only 3% slower (5.74 against
5.94) but takes 0.75 from 5.72 down to 0.83, so the workaround belongs to the person doing a specific experiment, not to the
default policy.

Practical state for item (9b): a stable split for experiments is **0.5 with the default policy (5.80 tok/s)** or **0.25 with
`--flash-attn off` (6.17)**. Against GPU-only 7.51 and CPU-only 7.40, that is ~15-38% of per-token copy overhead, which is why the
pipeline bound stays under the two-engine aggregate.

### (9b) retried with a stable split: still not worth building, now with per-share numbers

With the split stable (0.5 under the default policy, 0.25 with `--flash-attn off`), the per-share costs on the 27B are:

| CPU share | decode | ms/token | expected from uniform layers | boundary cost |
| --- | --- | --- | --- | --- |
| 0 | 7.51 | 133 | - | - |
| 0.25 | 6.17 | 162 | 0.75x133 + 0.25x135 = 134 | **~28 ms** |
| 0.5 | 5.80 | 172 | 134 | **~38 ms** |
| 0.75 | 5.72 | 175 | 134 | **~41 ms** |

(CPU-only is 7.40 tok/s = 135 ms, so the two halves are almost the same speed and the "expected" column is what a split would cost
with no boundary overhead.)

The overhead is 28-41 ms per token - a quarter of a token's time - and it barely depends on the share, which points at the
scheduler bouncing nodes (and its syncs) across the two backends rather than at one bulk copy.

A two-stage pipeline's steady-state period is `max(stage) + boundary` = 67 + 35 = ~102 ms, i.e. **9.8 tok/s aggregate** for the
27B. Two engines (9a) reach ~14.9 tok/s aggregate there (7.51 + 7.40, and 53.8 against 39.3 on the MoE, which is the same 37%
shape). So (9b) is behind (9a) on every pack *and* needs a new executor, while (9a) needs none. The retry confirms the earlier
decision: **not built**, and the split stays a measurement/correctness tool rather than a performance path.

