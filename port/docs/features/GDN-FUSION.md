# Fusing the gated-delta-net step (item 12)

The 35B-A3B pack is a hybrid: of its 40 blocks, 30 are linear-attention (gated delta net, GDN) layers and 10 are full
attention, followed by a sparse mixture-of-experts feed-forward. After upstream's matmul-epilogue fusion the decode issues
**331 dispatches per token**, and only a few dozen of those are matmuls - the rest is the GDN chain. This note records the exact
sequence (extracted from the graph the backend is given, with `GGML_VK_SYNC_LOGGER=1`), the plan to fold it into one kernel, and
every attempt that was measured.

> **Status: closed as not profitable.** Six kernel shapes and two graph-level changes were built and measured; each came out at
> parity or slower, and the barrier work later explained why (the per-op time attributed to those small ops is mostly waiting on
> dependencies, not work that can be moved). Every experiment has been **removed from the tree**; only the conclusions below
> remain, which is where the evidence lives. The default path is the upstream one: two `L2_NORM` nodes per GDN layer.


## What one GDN layer actually runs

Node names and ops from the graph dump of a single layer:

| # | op | name |
| --- | --- | --- |
| 1 | RMS_NORM | norm-0 |
| 2 | MUL | attn_norm-0 |
| 7 | SCALE | cache_r_l0 (view) |
| 8 | MUL_MAT | node_14 (q/k/v projection) |
| 16 | SCALE | cache_s_l0 (view) |
| 22 | CONCAT | conv_input-0 |
| 24 | MUL_MAT | node_35 |
| 27 | CPY | cache_r_l0 (copy of the conv window) |
| 28 | SSM_CONV | conv_output_raw-0 |
| 29 | UNARY | conv_output_silu-0 |
| 30 | ADD | node_37 |
| 32 | UNARY | a_softplus-0 |
| 33 | MUL | gate-0 |
| 34 | MUL_MAT | node_41 |
| 41 | L2_NORM | q_conv_predelta-0 |
| 42 | L2_NORM | k_conv_predelta-0 |
| 43 | UNARY | beta_sigmoid-0 |
| 44 | MUL_MAT | z-0 |
| 45 | GATED_DELTA_NET | node_46 |
| 48 | RMS_NORM | norm-0 |
| 49 | MUL | node_49 |
| 50 | UNARY | node_52 |
| 51 | MUL | node_53 |
| 53 | MUL_MAT | linear_attn_out-0 |
| 54 | ADD | attn_residual-0 |

`GATED_DELTA_NET` is already a single fused op (it is this fork's/upstream's whole delta-rule step over the state tensor), and
the two `L2_NORM`s plus the unaries are the pieces around it. Counting the dispatches per token that belong to this chain (from
the perf logger's dispatch counts): `CONCAT` 30, `CPY` 30, `GATED_DELTA_NET` 30, `L2_NORM` 60, `SIGMOID` 30, `SILU` 30,
`CLAMP` 40, `ADD` 33, plus the scattered `MUL`/`SCALE`/`CONT` remainder - roughly 300 of the 331.

## Why this is the largest remaining lever

The MoE decode is 26 ms per token at ~39.7 tok/s and the matmuls account for most of the *work*; the GDN chain is 30 layers x
~10 small kernels, each with a dispatch latency and a full pass over a 2-4 KiB activation. `GGML_VK_DISABLE_FUSION=1` shows how
much dispatch overhead matters here (35.72 against 39.7 tok/s, 1424 against 331 dispatches), so shrinking 331 to ~60 should be
worth a comparable or larger share.

## What is already fused, and what is actually left

The dispatch counts per token (perf logger, 35B-A3B decode) separate what still exists from what does not:

| op | dispatches per token | belongs to |
| --- | --- | --- |
| GLU | 80 | MoE swiglu (already one op) |
| L2_NORM | 60 | GDN, two per layer |
| CLAMP | 40 | MoE router weights |
| ADD | 33 | residuals |
| CONCAT | 30 | GDN conv window |
| CPY | 30 | GDN conv window copy |
| GATED_DELTA_NET | 30 | GDN core (already one op) |
| SIGMOID | 30 | GDN `beta_sigmoid` |
| ROPE / CONT / SCALE / GET_ROWS | 10 / 10 / 4 / 2 | attention and router |

`SILU` does not appear at all, and it is 60 per token with `GGML_VK_DISABLE_FUSION=1`: **the conv+silu fusion already runs**
(`ggml_vk_can_fuse_ssm_conv`, `num_additional_fused_ops == 1`). So the earlier "fold SSM_CONV+SILU" stage is already done
upstream, and the remaining GDN target is ~180 dispatches per token: the conv-window pair, the two normalisations, the beta
sigmoid and the delta-net core.

## Plan, in stages (corrected)

1. **Stage 1 - the conv-window kernel (largest single piece).** `CONCAT` + `CPY` + `SSM_CONV` (+ the already-fused silu) become
   one kernel that reads the previous window from the per-layer cache, writes the shifted window back and computes the
   convolution: ~60 dispatches per token removed plus two full copies of the window, and it is the piece whose shader shape is
   clearest (a sliding window over `n_conv` taps, no delta-rule state).
2. **Stage 2 - the normalisation front end.** One kernel that applies the conv-output scaling, the two `L2_NORM`s and the beta
   `SIGMOID` to the projected q/k/v: another ~90 dispatches per token, self-contained, no state.
3. **Stage 3 - fold the preparation into the delta-rule kernel.** With stages 1-2 done the per-layer GDN is three kernels; if
   `GATED_DELTA_NET` still dominates, its input preparation moves inside it, which needs a new graph op rather than a backend
   fusion because the delta-rule kernel already owns the state tensor.

Each stage keeps the same verification triple: `test-backend-ops` in both precision modes where an oracle case exists,
CPU-vs-Vulkan token equality on all five packs, and the bench suite for the number. The CPU backend must implement anything a
stage fuses, or the token-equality gate loses its reference.

## The exact geometry (Qwen3.6-35B-A3B, `qwen35moe`)

Read straight out of the pack with `port/tools/gguf-dims.py` (a dependency-free GGUF header reader, kept because every later
stage needs these numbers):

| metadata | value |
| --- | --- |
| `ssm.inner_size` | 4096 |
| `ssm.state_size` | 128 |
| `ssm.conv_kernel` | 4 |
| `ssm.group_count` | 16 |
| `ssm.time_step_rank` | 32 |
| `n_embd` | 2048 |

and the per-layer tensors: `attn_qkv.weight [2048, 8192]` (q 4096 + k 2048 + v 2048), `ssm_conv1d.weight [4, 8192]`,
`ssm_a [32]`, `ssm_dt.bias [32]`, `ssm_norm.weight [128]`, `ssm_out.weight [4096, 2048]`.

So the conv window is `[8192 channels, 4 taps]` of f32 per GDN layer, and the per-token state update is a 3-column shift of
`8192 x 3` f32 (96 KiB per layer, 30 layers - the `CPY` in the graph is that shift, and it is pure data movement).

## Stage 1 implementation spec (conv window) - with the measured layout

The layout was read out of the running graph (a temporary dump of the node shapes inside the node loop, since removed), and it
is not what the op names suggest:

```
CONCAT dst conv_input-0            [67, 8192, 1, 1]  nb[4, 268, ...]      <- dim0 contiguous, 3 cached + 64 new columns
   src0 conv_states_reshaped-0     [ 3, 8192, 1, 1]  nb[4, 12, ...]       <- the cache view, 3 taps
   src1 qkv_mixed_transposed-0     [64, 8192, 1, 1]  nb[32768, 4, ...]    <- the projection, transposed (dim0 = token)
CPY  dst cache_r_l0 (view)         [24576, 1, 1, 1]  nb[4, ...]           <- 3 x 8192 floats = 96 KiB
   src0 conv_input-0 (view)        [ 3, 8192, 1, 1]
SSM_CONV over conv_input (silu fused) -> [8192, 64]
```

So the cache is **three** columns (not four), the convolutions slide over `3 + n_tokens` columns of a materialised concat, and
the `CPY` copies the last three columns back - 96 KiB per layer per chunk, plus the 2.2 MB concat write and re-read that the
conv then does.

The fused kernel therefore is:

* bindings: `x` = `qkv_mixed_transposed` (dim0 = token with stride `nb[0]`, dim1 = channel stride `nb[1]` = 4 bytes, read
  strided, no materialisation), `w` = the conv kernel `[4, 8192]`, `cache` = the 3-column window (read and written),
  `dst` = `[8192, n_tokens]`;
* per channel `c` and token `t`: `y = sum_{i=0..3} ext[t + i][c] * w[i][c]`, where `ext` is the virtual `cache ++ x` buffer,
  then `dst[c][t] = silu(y)`;
* one workgroup also writes the new window back (`cache[0..2][c] = ext[n_tokens .. n_tokens+2][c]`, i.e. the last three
  columns of `x`), which removes the `CPY`;
* backend: look ahead at the `CONCAT` node for `{CONCAT, CPY, SSM_CONV, UNARY}` with the shapes above,
  `num_additional_fused_ops = 3`, one pipeline created next to `ssm_conv`; gated by `STRATA_VK_GDN_CONV=1` until it is proven;
* verification: the existing path stays as the reference, the CPU keeps running the four ops, and the five-pack token equality
  gate must stay identical with the gate on before it can become the default.

## Status

The fused kernel and its pipeline are in the tree (`ssm_conv_window.comp`, the generator entry, the device pipeline and the push
constants) but the path **cannot fire**, and the reason is structural: the graph's node order is

    22 CONCAT   23 CPY   24 MUL_MAT   25 VIEW   ...   28 SSM_CONV   29 UNARY(silu)

so the four nodes are not contiguous. The backend's fusion mechanism consumes a contiguous run of *following* nodes
(`num_additional_fused_ops`), which is why a diagnostic run printed `no match at 22: op sequence (ops CONCAT CPY MUL_MAT VIEW)` for
every layer. A four-node conv-window fusion is therefore not expressible at the CONCAT node; the wiring was reverted and the shader
stays inert.

What remains expressible is a **CONCAT + CPY** fusion: those two *are* adjacent (22, 23), and folding the state copy into the concat
removes 30 dispatches per token (~0.3%). It needs a much simpler kernel than the one written - concatenate into the destination and
write the newest three columns to the cache - and it would be the only piece of stage 1 that fits this graph.

Stages 2 and 3 (the normalisation front end and the delta-rule input fold) are unaffected by this: they change the *graph* (a new
op) rather than fusing the existing node order, so they can still be done.

### CONCAT + CPY: built, wired, gated, and one predicate away

The two-node version is in the tree: `concat_window.comp` (concat plus a fourth binding that writes the trailing three columns into the
cache), a generator entry, a four-binding pipeline, `ggml_vk_can_fuse_concat_window`, the decision branch and the dispatch in the
`GGML_OP_CONCAT` case, all behind `STRATA_VK_GDN_CONV`.

It does not fire yet. A probe in the helper printed 11 447 hits, so the decision chain *is* evaluated for every node - the predicate
itself rejects the real `CONCAT`/`CPY` pair. The checks it makes (window width 3, `concat->ne[0] == 3 + x->ne[0]`,
`cpy->src[0]->view_src == concat`, cache element count `3 * channels`, contiguous cache) are all true of the dumped graph, so one of
them is reading a field differently than assumed - most likely `view_src` on the CPY source or the element count of the cache view.
Next step is a reason print per failed check, then the fix.

The gate is off by default, so nothing runs differently: 35B-A3B decode is 39.87 tok/s and the dispatch counts are unchanged
(3 183 total, cpy 270) with the gate either way.

### CONCAT + CPY: what the dump showed, and where it stands

The predicate now matches the shapes the graph actually carries, learned from a field dump of the first CONCAT nodes:

```
concat ne[67,8192] nb[4,268] type f32 | win ne[3,8192] type f32 | x ne[64,8192] nb[32768,4]
next CPY | cpy->src[0] view_src != concat (it points at the reshape below) | cpy->src[1] flat 24576 = 3*8192
```

Three predicate versions were tried: `view_src == concat` (wrong: the copy views a reshape), a data-pointer containment check (the
view's data is not resolved yet when the decision runs), and finally a shape-only check (`cpy->src[0]` is 3 columns over the same
channels, `cpy->src[1]` is a flat `3 * channels` f32 buffer). With the shape-only version the fusion still does not fire and the
dispatch counts stay at 3 183 with 270 CPYs, so the predicate is no longer the only suspect: the next probe has to print at the
decision branch (taken or not, and with which index) and at the `GGML_OP_CONCAT` case (`num_additional_fused_ops` as seen there).
One of the two is not where this fusion is being decided - the `SSM_CONV` fusions that work live in the same chain, but their
dispatch sits in a different case, and the ordering of the decision chain relative to the dispatch switch is now the thing to
verify rather than assume.

The kernel and wiring stay in the tree behind `STRATA_VK_GDN_CONV`: default runs are unchanged (decode 35-40 tok/s across runs,
dispatch counts identical with the gate on or off).

### Why CONCAT + CPY also cannot be fused: the adjacent CPY is a different copy

A field dump of the first CONCAT nodes with the gate on settled it:

```
[cw2] 22 c ne[67,8192] nb[4] ty0 | win ne[3,8192] ty0 | x ne[64]
           nxt CPY s0 ne[524288,0] nb[4,2097152] ty0 | s1 ne[524288,0] nb[4] ty0
```

The CPY that follows the CONCAT copies **524 288 elements** (2 MiB), not the 24 576-element window - it is the conv-state slab copy,
not the three-column state update. The window copy is the CPY at index 27, after the MUL_MAT at 24. So the two nodes this fusion
needed are not adjacent either, and a two-node fusion cannot express it.

The implementation was therefore removed again (shader, generator entry, pipeline, helper, decision branch and the case change), and
this lane is closed: every fusion shape the GDN graph offers is either non-adjacent (CONCAT/CPY/SSM_CONV/UNARY) or already done
upstream (conv + silu). What remains of the GDN work is stages 2 and 3, which change the *graph* (a new op) rather than fuse its
order.

### Stage 2 plan, with the exact sites

The front end is two `L2_NORM`s plus the beta sigmoid, built in `src/models/qwen35moe.cpp`:

```
438: q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);
439: k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);
372: cb(beta, "beta_sigmoid", il);      // the sigmoid on the beta path
```

A ggml op has a single destination, so the fused op has to write one buffer and let the graph view it:

1. **ggml op** `GGML_OP_GDN_NORM_PAIR`: `src[0]` = q, `src[1]` = k, `src[2]` = beta, `op_params` = the eps; `dst` is
   `[2, d_head, n_head]` (q in slot 0, k in slot 1, both normalised), and a second dst for the sigmoid is avoided by keeping the
   sigmoid as it is (it is one dispatch per layer, and the pair already removes two);
2. **CPU reference** in `ggml-cpu/ops.cpp` (already in the overlay): the same l2 normalisation the two existing ops do, so the
   token-equality gate keeps a reference;
3. **graph change**: `qwen35moe.cpp` builds the pair and views slot 0/1; views cost nothing (no dispatch);
4. **Vulkan kernel**: one dispatch over `[2, d_head, n_head]`, one workgroup per (head, half);
5. **verification**: token equality on the five packs with the op active, and the bench to see the ~90-dispatch drop per token
   (two `L2_NORM`s and one `SIGMOID` per GDN layer, times 30 layers).

The `qwen35moe.cpp` edit is a file outside the current overlay (which is ggml + shaders + ggml-cpu), so the overlay grows by one
model file; that is the same trade as `ggml-cpu/ggml-cpu.cpp` and the user has allowed the overlay to grow when a fix needs it.

#### Step 1 done: the op exists and the CPU reference runs

GGML_OP_GDN_NORM_PAIR is in ggml.h/ggml.c (enum, name and symbol tables, the count assertions moved from 99 to 100, the
constructor ggml_gdn_norm_pair(ctx, q, k, eps) producing [2, d, heads]), its CPU reference is in ggml-cpu/ops.cpp
(ggml_compute_forward_gdn_norm_pair) with the dispatch in ggml-cpu/ggml-cpu.c. It is not wired into a model yet, so nothing
runs differently: 35B-A3B decode 38-40 tok/s and the MUL_MAT oracle 38/38 after the change, and the overlay carries the five new
files (ggml.h, ggml.c, ggml-cpu/ops.h, ops.cpp, ggml-cpu.c).

#### Step 2 done: the pair is in the graph and its CPU path is token-identical

`qwen35moe.cpp` builds the pair when `STRATA_GDN_PAIR` is set (the default stays the two `L2_NORM`s until the Vulkan kernel exists),
with the two halves taken as `ggml_view_4d` of the `[2*d, heads, tokens, seqs]` destination. The op is 4-D because the GDN q/k carry
four dimensions and a `[2, ...]` destination would need five.

Two things the CPU reference needed: the destination is `[2*d, ...]` with q in the first half and k in the second (a ggml tensor has
four dims, so the pair lives on dim 0), and `ggml-cpu.c` has a second switch for the task split where the new op had to be listed
next to `GGML_OP_L2_NORM` or it aborts with "op not implemented".

Verified: 35B-A3B with `STRATA_GDN_PAIR=1` produces the **same four tokens** as the default path, and decode is 10.4 tok/s against
39.1 because the op currently runs on the CPU (the Vulkan backend does not support it yet, so every layer round-trips to the host).
That gap is the size of the prize: the Vulkan kernel is what turns 30 host round-trips per token back into one dispatch each.

#### Step 3: the kernel exists but its first dispatch faults; the op stays on the CPU

`gdn_norm_pair.comp` (one workgroup per row, 512 threads, shared reduction, q then k), its generator entry, push constants,
pipeline creation, dispatch case and pipeline selection are all in the tree. With the op declared supported by Vulkan the process
dies on the first token with **exit code 0xC0000005 (access violation)** and no message; with the declaration removed the op runs on
the CPU and the tokens are identical to the default path (9.8 tok/s, the cost of 30 host round-trips).

The next step is to let the validation layers report the actual error
(`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`, or the loader's debug output) rather than guessing at the shader - every
parameter this port can inspect (push-constant size, three bindings, grid = rows, contiguous f32, destination layout `[2*d, ...]`)
matches what the kernel expects, which is the same situation the direct-graph cooperative bug ended in.

#### The fault is in the dispatch layer, not the shader

Two kernels were tried with the op declared supported:

* the original: 512 threads per row, shared-memory reduction, `barrier()` between phases;
* a minimal one: one thread per row, no shared memory, no barriers, every index provably inside its buffer
  (`row < n_rows`, `i < d`, destination `2*d*n_rows` elements).

Both die on the first token with **0xC0000005**, and under `VK_LAYER_KHRONOS_validation` (with `VK_LAYER_PATH`,
`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`) the run "succeeds" with an impossible `tg 1 tokens in 0.0 ms` - the submit is
accepted by the layer and never completes. So the shader body is not the problem: the fault is in how this op's dispatch reaches the
device, the same class as the direct-graph cooperative bug, and both need a GPU capture to settle.

The op stays on the CPU (identical tokens, 9.8 tok/s) and the kernel stays in the tree with its `supports_op` declaration removed. GDN
stage 2 is therefore implemented and verified on the reference path and **not** accelerated; the ~90-dispatch saving waits on that
dispatch fault.

#### The dispatch fault was a missing descriptor-set reservation, and the measured pair is not profitable

Root cause found: a custom dispatch must call `ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1)` before
`ggml_vk_dispatch_pipeline`. Without it the descriptor set array is not grown for the new pipeline and the process dies with
0xC0000005 on the first token - which is exactly what both kernel variants did, no matter how simple the shader was. With the call
added (and `ggml_vk_sync_buffers` not even needed), the pair kernel runs and its tokens are identical to the default path.

The measurement then settles the economics:

| variant | decode | tokens |
| --- | --- | --- |
| default (two `L2_NORM`s) | 39.9 tok/s | reference |
| pair, one thread per row | 36.7 tok/s | identical |
| pair, 512 threads per row with a shared reduction | 38.3 tok/s | identical |

The fusion removes 60 dispatches per token (30 GDN layers x 2), which at 2.6 us each is **0.08%** of a 200 ms token - and the
per-row kernel is less efficient than the two `l2_norm` dispatches it replaces, so the net is negative. The pair therefore stays
behind `STRATA_GDN_PAIR` (opt-in) and off by default: it is correct, it is measured, and it is not worth switching on.

The reusable result is the dispatch rule above: any future custom op in this fork needs the descriptor-set reservation, and the
absence of it looks exactly like a driver fault (silent 0xC0000005, or an impossible "0.0 ms" under the validation layer).

### Stage 3 (delta-rule input fold): closed by arithmetic and by the stage-2 precedent

The MoE decode runs **398 dispatches per token** (3 183 for 8 tokens), which at the measured 2.6 us per dispatch is **1.03 ms of a
25.4 ms token (4%)**. The GDN front end is the part stage 3 could fold: one `SIGMOID` per GDN layer, i.e. 30 dispatches per token,
or **78 us (0.3%)** at best - and it would need another new op, a CPU reference and a kernel.

Stage 2 already tested that hypothesis at a larger scale: fusing the two `L2_NORM`s (60 dispatches) into one kernel *lost* 4%,
because a fused kernel whose shape is worse than the dispatches it replaces costs more than the dispatches save. Stage 3's ceiling is
five times smaller than stage 2's, so the same outcome is certain.

Closed: not built. The general lesson from stages 1-3 is that dispatch-count arithmetic overstates the gain in this model - a fusion
only pays when the fused kernel is at least as efficient per element as the operations it replaces, and none of the GDN candidates
were.

### The reshaped kernel: correct, at parity, and still not worth turning on

The evaluation said the upside should be 4% on 35B-A3B (half of the 1.98 ms/token the two `L2_NORM`s cost) and 1.7% on 27B, so
the kernel was rebuilt for these tiny rows: one workgroup per (row, slot) with 128 threads matching the 128-wide rows,
`subgroupAdd` plus a single cross-subgroup add in shared memory instead of a 512-entry tree, and `2 * n_rows` workgroups so the
decode case has 32 blocks instead of 16.

| run | decode | tokens |
| --- | --- | --- |
| 35B-A3B default | 39.0-40.4 tok/s | reference |
| 35B-A3B pair, reshaped | 36.9-39.0 tok/s | identical |
| 27B default | 7.50 tok/s | reference |
| 27B pair, reshaped | 7.49 tok/s | identical |

Four kernel shapes have now been measured (512-thread shared tree, one thread per row, 128-thread subgroup, and the same with the
grid fixed): every one is at parity with the two `l2_norm` dispatches or worse. The dispatch saving (34 per token) is real, but the
fused kernel's cost grows by the same amount, so the fusion nets out at zero.

The likeliest reason, and the thing an attempt would have to fix, is the destination: the pair writes a fresh `[2*d, heads, tokens,
seqs]` tensor that the graph allocates for each layer, while the two `L2_NORM`s write into `q_conv`/`k_conv`, buffers the model
already owns and reuses. Until the pair can write into memory the graph keeps alive across tokens, the fusion cannot bank the
dispatch it saves. That is a graph-level change (a persistent pair buffer, or writing through views of the existing q/k
allocations), not a kernel change, and it is not in this fork.

Closed: the path stays behind `STRATA_GDN_PAIR`, off by default, correct and at parity. The measured bar (35B-A3B at least
41 tok/s) was not met.

#### The graph-level change was tried as well: in-place writing is slower

The theory the parity result pointed at was the destination: the pair wrote a freshly allocated `[2*d, heads, tokens, seqs]` tensor
each layer, while the two `L2_NORM`s write into `q_conv`/`k_conv`, buffers the model owns and reuses. So the op was changed to write
**in place** - q normalised into a view of itself and k into itself, no scratch tensor at all, one binding pair, the same kernel
shape - which is exactly the graph-level fix.

| run | decode | tokens |
| --- | --- | --- |
| 27B default | 7.46 tok/s | reference |
| 27B in-place pair | **7.17 tok/s** | identical |
| 35B-A3B default | 40.57 tok/s | reference |

It is slower, not faster, and the tokens stay identical. So the allocation was not the cost either: writing the model's own buffers
made the dispatch worse, most likely because a read-write buffer forces a barrier the read-only version did not need.

The in-place change was reverted; the tree keeps the parity version behind `STRATA_GDN_PAIR`. The conclusion across five kernel
shapes and now a graph-level change is that this fusion cannot bank its dispatch saving in this graph: every cost model pointed at
4%, and every implementation measured zero or negative.

### Fusing the normalisation into the delta kernel: no gain either, and it changes the tokens

This was the last shape worth trying, because it is the only one that does not add a kernel: `gated_delta_net.comp` already loads
both rows into registers (lines 133-137) and already carries `reduce_partial()`, a per-row reduction that works in every compiled
path. The change: an `eps` field in the push constants (0 means "q and k arrive normalised", so the default path is untouched), and
inside the token loop, when `eps > 0`, two extra row reductions and an in-register scale. The graph hands over raw q/k when
`STRATA_GDN_FUSED_NORM` is set, with the epsilon travelling in `op_params[1]` of the delta node.

| run | decode | tokens |
| --- | --- | --- |
| 35B-A3B, gate off | 39.33 tok/s | `11,353,2688,4313,310,958` |
| 35B-A3B, gate on | 37.82 / 39.90 tok/s | `11,353,2688,1608,279,2614` |

Two results, both negative:

* **no speed-up**, although 68 dispatches per token disappear. That contradicts the per-dispatch cost measured with the perf logger
  (29.3 us x 67.5 = 1.98 ms per token) and it invalidates that arithmetic: those per-op times include the gaps between dispatches,
  so the L2_NORM "share" of a token is mostly time the GPU spends waiting, not work that moving it elsewhere can reclaim. Every
  attempt in this document that tried to bank that "share" measured zero for exactly this reason;
* **the tokens differ** from the fourth token on. The fused reduction sums a 128-wide row with `reduce_partial`'s order, while
  `l2_norm` uses a 512-thread tree, so the f32 rounding differs by ~1e-7 per element and the recurrence amplifies it.

Both experiments stay behind their flags (`STRATA_GDN_PAIR`, `STRATA_GDN_FUSED_NORM`), off by default, with the default path
bit-identical to before: the epsilon defaults to 0 and the two `L2_NORM`s stay in the graph.

## What was removed from the tree, and what the default path is

Every experiment described above has been deleted from the overlay, so the fork carries no inert GDN code: the
`GGML_OP_GDN_NORM_PAIR` op (enum, name and symbol tables, the op-count assertion back to 99, constructor, CPU reference, dispatch
cases and the `[2*d, ...]` destination), its `gdn_norm_pair.comp` shader with the generator entry, pipeline, push constants, type
field, dispatch case, pipeline selection and `supports_op` entry, the `STRATA_GDN_PAIR` wiring in `qwen35moe.cpp`, the
`STRATA_GDN_FUSED_NORM` path in `delta-net-base.cpp` and the `eps` field plus normalisation block in `gated_delta_net.comp`. The
overlay is back to 47 files, none of them model-side, and the default graph is the upstream one: two `L2_NORM` nodes per GDN layer.

The measurements above are the reason nothing replaced them: 35B-A3B decode stayed at 37-40 tok/s in every variant against a
39-40 tok/s default, 27B stayed at 7.2-7.5 against 7.5, and the only shapes that produced different tokens were the ones with a
different reduction order. The barrier investigation (`ADRENO.md`) explained the pattern: the per-op time of these small ops is
dominated by waiting on dependencies, so removing the dispatches moves the wait instead of removing it.

