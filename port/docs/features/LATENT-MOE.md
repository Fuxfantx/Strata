# Latent-MoE packs (Edge0 145B / 120B): what they are and what this port needs

*状态：**规格（施工中）**。本页记录从公开资料读到的结构与本分支要补的部位；凡是"未验证"的都写明。 / Status: a work-in-progress specification. Facts read from public documentation are separated from what this port has actually run.*

## What the models are

Edge0 145B and Edge0 120B are **derived from Qwen3.8-Flash-Next** (the family this port already loads as `qwen4exp`).
Their difference from the base model is concentrated in the expert block:

| Item | Edge0 145B | Edge0 120B |
| --- | --- | --- |
| Layers | 48, hybrid attention (full every 4) + sparse indexer | same |
| Experts per layer | 512 routed + 1 shared, top-10 routed | same |
| Expert SwiGLU width | 640 | 640 |
| **Latent width ℓ** | **1792** | **1280** |
| Expert pool | 84.6 B parameters at width ℓ | smaller |
| Expert format (Q3_K folder) | Q3_K gate/up, IQ4_NL down, column scales | same |
| D/U projections | per layer, **8 routing groups**, Q8_0 or Q4_0 | same |
| PLE | 51.2 B n-gram table, fp8, memory-mapped | same |
| MTP head | 1.26 B draft, 512 experts | same |
| Build size (Q3_K) | 40.81 GiB | 27.98 GiB |

One layer, in order: attention reads and writes the full 2560 width; **D** projects the token down to ℓ *for its routing
group*; the router picks 10 of 512 experts; each expert runs ℓ → 640 → ℓ; a shared expert runs alongside; **U** projects
the combined result back to 2560. Only the experts work at width ℓ, so an expert is a fixed record:

- ℓ = 1792: gate/up Q3_K (2 × 1792 × 640 at 3.4375 bpw) + down IQ4_NL (640 × 1792 at 4.5 bpw) ≈ **1 630 720 B** per expert,
  which matches the published record size.

The published folder layout (from the Edge0 launch template) is a **directory pack**, not one GGUF:

```
model-skeleton.gguf     dense weights: attention, hyper-connections, norms, lm_head
experts.bin             the latent experts, fixed-size records, indexed by (layer, expert)
labels.bin              each expert's routing-group label (0..7), which selects its D/U pair
the D/U file          the per-layer, per-group D and U projections, Q8 or Q4 (the launch template names it)
ple-fp8.gguf            the n-gram embedding table
profile.bin             expert residency order (which experts to keep on the device first)
mtp/                    the MTP draft head (gguf) and its draft vocabulary
tokenizer/              the usual tokenizer directory for the server
```

## The file formats, read from the reference implementation

These were read from the fork's own headers (`include/strata/core/latent_dense.hpp`,
`include/strata/kernels/cpu/native_expert.hpp`, `include/strata/kernels/latent.hpp`), not inferred.

**The D/U file** (`latent_dense_q8.bin` or `latent_dense_q4.bin`) is one flat file, mmap-able, layer by layer:

```
layer l:  D  [G * ELL, H]   then   U  [G * H, ELL]
```

with `G = 8` groups, `ELL` = the latent width (1792 or 1280), `H = 2560`. Each row is quantised on its own:

| Format | Row of 32 values | `D` row bytes (H = 2560) | `U` row bytes (ELL = 1792) |
| --- | --- | --- | --- |
| Q8_0 (`fmt 8`, the default) | 34 B | **2 720 B** | **1 904 B** |
| Q4_0 (`fmt 4`) | 18 B (`{f16 d; 16 B nibbles}`, element j in the low nibble, j+16 in the high one, value `(code - 8) * d`) | 1 440 B | 1 008 B |

So one layer's D/U is `G * ELL * d_row + G * H * u_row` bytes, and the file is `48 *` that. The format is not in the
file: it comes from `STRATA_LATENT_DENSE_FMT` and is cross-checked against the file's size.

**An expert** in a Q3_K pack is a *native* blob: the three GGUF slices back to back,

```
[ gate rows | up rows | down rows ]
```

each row in its quantised layout, with the per-layer formats described by `native_experts.txt` beside the pack. The
published record size (1 630 720 B for `ELL = 1792`, 640-wide SwiGLU) matches gate/up in Q3_K plus down in IQ4_NL.
Q3_K folders also carry `scales/l<layer>.f32`: the gate/up rows are stored as `A * diag(s)`, so the activation must be
multiplied by the inverse scales **before** it is quantised.

**How the groups reach the experts.** The D projection writes not one latent row per token but `G` of them
(`token * G + group`). An expert of group `g` reads the row of its own group, so the reference rewrites the grouped
matvec's activation row indices to `(token) * G + labels[expert]` instead of `token`. The U stage then projects each
routed entry separately: `y[r] = yl[r] @ U[labels[expert]]`, H values per entry, which are weighted and summed.

## The route this port takes: repack into ggml tensors, then use existing ops

A latent block can be expressed with the ops the Vulkan backend already has, if the pack is repacked so that the
group-selected pieces look like ordinary expert tensors. That is the route taken here, because it needs no new
quantised kernel and no new number format:

| Pack piece | Repacked as | The op that consumes it |
| --- | --- | --- |
| D, per layer, `[G*ELL, H]` in the file | `[H, ELL*G]` Q8_0/Q4_0 (`latent_d`) | one `MUL_MAT`, then a reshape to `[ELL, G, n_tokens]` |
| U, per layer, `[G*H, ELL]` in the file | `[ELL, H, G]` Q8_0/Q4_0 (`latent_u`) | `MUL_MAT_ID` with the per-entry group label as the id |
| gate/up of the 512 experts, `[n_ff | n_ff] x ELL` | `[ELL, n_ff, n_expert]` Q3_K, two tensors | `MUL_MAT_ID` + `SWIGLU` |
| down, `[ELL x n_ff]` | `[n_ff, ELL, n_expert]` IQ4_NL | `MUL_MAT_ID` |
| `labels.bin`, `[n_layers * n_expert]` int32 | `[n_expert]` I32 per layer (`latent_labels`) | `GET_ROWS` on the label table to build the gather index |

Why the shapes line up: for a routed expert `e` of group `g`, the activation is the latent row of group `g`, and
`MUL_MAT_ID` already multiplies a *duplicated* activation per (token, expert slot) — the same duplication the
reference implementation gets by rewriting its row indices to `token * G + label[expert]`. So the graph is:

1. `xl = MUL_MAT(D, cur)` → `[ELL, G, n_tokens]`;
2. router picks `n_used` experts per token (the existing `build_moe_ffn` code);
3. `idx = GATHER(labels, ids)`, then `xg = GET_ROWS(xl, idx)` → the latent row of each entry's group;
4. `a = MUL_MAT_ID(gate/up, xg, ids)`, `SWIGLU`, `y = MUL_MAT_ID(down, ...)` → `[ELL, n_tokens * n_used]`;
5. `u = MUL_MAT_ID(latent_u, y, idx_group)` → `[H, n_tokens * n_used]`, weighted and summed over the slots;
6. the shared expert and everything outside the expert pool stay at width `H` and are untouched.

Two consequences worth stating plainly:

- **The repack is a real cost**: it writes one GGUF holding the same weights (about 40 GB for the Q3_K folder), so it is
  a conversion step, not a loader trick. The front end has to run it once per pack and keep the result beside the pack.
- **The arithmetic is unchanged**: every multiply is the same quantised dot product on the same weights; only the
  tensor layout and the row indexing change. So `test-backend-ops` can check the pieces, and a text comparison against
  the reference path remains the acceptance test.

## What this port needs, in order

1. **A loader for the directory pack.** Today `strata-vk` takes one `--native` GGUF. A latent pack needs
   `--pack <dir>` with the parts above, and a model family that knows which tensors are "dense" and which live in
   `experts.bin`.
2. **The latent expert graph.** For each layer: `D[group]` (2560 → ℓ, Q8_0/Q4_0), the 10 routed experts
   (`Q3_K` gate/up → SwiGLU → `IQ4_NL` down, at width ℓ), the shared expert, `U[group]` (ℓ → 2560). The group label
   comes from `labels.bin` and is per expert, not per token.
3. **Kernels.** The Vulkan backend already has `Q3_K` and `IQ4_NL` dequantisation and matvec/mat-mat entry points, so
   the work is in the *grouped* dispatch: one matvec per selected expert, with an l-width activation, plus the D/U
   matmuls whose group differs per expert. Nothing here needs a new number format.
4. **PLE.** A 51.2 B fp8 table read through a memory mapping; the pack is too large to copy, so the loader has to map
   it and the kernel has to read it directly.
5. **MTP.** One draft head (1.26 B, 512 experts) for speculative decode; the port already grafts an MTP/NextN head for
   its other families, so this is a loader plus a graph-composition question.
6. **Vision.** A separate mmproj file; the port's `strata-vision` path already handles mmproj + `.sve`, so this is
   wiring rather than new code.
7. **Front end.** `setup.py` / `strata` / `serve/server.py` / the app need to recognise a latent pack as a model kind,
   choose the D/U format, and place the parts; downloads have to pick a source (Hugging Face or ModelScope).

## What the released pack actually contains (read from the files, 2026-11-25)

The ModelScope folder `Edge0/Edge0-145B-q3k` was listed and its small files and the head of its skeleton were read
(the weights were **not** downloaded):

| File | Bytes | What it is |
| --- | ---: | --- |
| `model-skeleton.gguf` | 5 991 569 248 | **a `qwen4exp` GGUF** - the family this port already loads |
| `experts.bin` | 40 076 574 720 | 48 x 512 native blobs of 1 630 720 B |
| `ple-fp8.gguf` | 51 200 246 112 | the per-layer n-gram embedding table (FP8) |
| `latent_dense_q8.bin` | 3 743 416 320 | 48 layers x (8 x 1792 x 2720 + 8 x 2560 x 1904) = exactly this |
| `latent_dense_q4.bin` | 1 981 808 640 | 48 layers x (8 x 1792 x 1440 + 8 x 2560 x 1008) |
| `dense.bin`, `index.txt` | 1 538 625 024 / 117 464 | the pack index (v3) and the dense tensor pool |
| `labels.bin` | 98 304 | 48 x 512 int32 group labels |
| `profile.bin` | 196 632 | expert residency ranking |
| `mmproj-BF16.gguf` | 907 543 008 | the vision encoder |
| `mtp/` | 116 MB dense + 708 MB experts + draft vocab | the MTP draft head |
| `native_experts.txt` | 1 463 | header `v6: latent A-only, Q3_K gu + shared colscale (expert_embd 1792, expert_ff 640)`, then `layer gu_type down_type offset blob` with **`11 20`** = `Q3_K` / `IQ4_NL` |

The skeleton's metadata confirms the architecture key and the parameters:

```
general.architecture        = qwen4exp
qwen4exp.block_count        = 48
qwen4exp.embedding_length   = 2560
qwen4exp.expert_count       = 512
qwen4exp.expert_used_count  = 10
qwen4exp.expert_feed_forward_length = 640
qwen4exp.expert_shared_feed_forward_length = 640
qwen4exp.attention.recurrent_layers = 48 values   (hybrid: full attention every 4)
qwen4exp.hyper_connection.count = 4, .low_rank = 320
qwen4exp.ple.*               (layers, ngram_size 3, heads_per_ngram 8, head_offsets, head_vocab_sizes, ...)
```

**Two consequences that change the plan for the better:**

1. This is not a new model family. The skeleton loads through the existing `qwen4exp` path, which this port already
   supports, including its PLE keys (the overlay carries `src/llama-ple-disk.*` and `qwen4exp.cpp` has PLE code).
2. The skeleton has **no routed expert tensors** (`ffn_*_exps` are absent from its 1 079 tensors) and no latent-dimension
   key. So the work is: (a) let the loader accept the missing expert tensors and take them from the mapped pack, and
   (b) add the latent block (D, grouped A, U) to the graph for this pack only, with the latent geometry taken from
   `native_experts.txt` and the D/U file sizes.

## What this port needs, in order

- **Run here**: nothing of this feature yet. This page is the specification step.
- **Not runnable on this machine as-is**: the Q3_K folder is 40.81 GiB of weights and this laptop has 47.6 GB of unified
  memory with no discrete VRAM, so a full model cannot be resident; the published numbers come from 24 GB and 60 GiB
  NVIDIA cards. Any end-to-end claim from this port would therefore be on a synthetic fixture, not on the pack.
- **Consequence for the plan**: the loader, the graph and the kernels can be built and validated against small fixtures
  and against the existing `test-backend-ops` oracle; the throughput numbers can only be *stated as untested here*.

## References

- `github.com/Edge0-AI/Edge0`, `edge0-strata/EDGE0.md` (launch contract, pack parts, formats) and
  `docs/models/edge0-145b.md` (architecture, sizes, published speed).
- The reference implementation's fused prefill kernels are CUDA (`lat_q3k_gu.cu`, `lat_q3k_down.cu`,
  `lat_q3k_swiglu.cu`, `lat_q3k_fused.cu`); this port needs the same *arithmetic* through Vulkan, not the same source.

### The repack tool, and what a dry run says

port/tools/edge0-repack.py --pack DIR --dry-run writes nothing and prints the tensor table.  On the released 145B
Q3_K pack it reports **288 tensors** (48 layers x 6) and **43 820 089 344 bytes (40.81 GiB)** of tensor data, which is
exactly the published size of the Q3_K build - a check that the per-layer arithmetic, the expert blob split and the D/U
row layout all agree with the pack as shipped.

One detail worth keeping: every piece of the pack is already in the byte order the target tensors want (the D rows are
(group, latent)-major, the U rows (group, embd)-major, an expert blob is [gate | up | down]), so the repack is a
copy of byte ranges with new tensor headers.  It does not dequantise or reorder anything.

### Why the block becomes one op and not a chain of `MUL_MAT_ID`s

The row the latent A stage reads is `group[expert] + G * token`. `MUL_MAT_ID` takes one index tensor, and `ggml` has no
`arange`, so the composite index cannot be built from graph tensors without a host-side constant that changes with the
batch size. Two ways out exist; this port takes the second:

1. keep the chain of ops and add a per-token constant input tensor for the `G * token` part - small, but it means every
   graph edit has to keep that tensor in step, and the gather then costs its own dispatch per layer;
2. **one op for the latent block** (`GGML_OP_LATENT_MOE`), whose parameters name the D/U tensors, the three expert
   tensors and the label tensor, with the composite index computed inside the kernel where the token stride is known.

The second is chosen because it is also the shape the CPU reference and the Vulkan kernel can share, and because it
keeps the number of dispatches per layer at one instead of five. The work order is:

- **CPU first**: a scalar reference over the repacked tensors, so the block's arithmetic (D -> group A -> SwiGLU ->
  down -> U -> weighted sum) can be checked against the reference implementation's own numbers before any shader exists;
- **then Vulkan**: the same block as one kernel family, with the Q3_K and IQ4_NL dot products the backend already has;
- **lazy residency** rides along: the op is where the expert cache is consulted, so the "hot experts stay, the rest
  stream" behaviour has a single place to live (see the URAM note below).

### URAM: strong lazy load and the hot-expert cache

This machine has no separate VRAM: the 47.6 GB of unified memory holds both the weights and the KV cache, and the
Q3_K build alone is 40.81 GiB. The only workable policy is to **map** `experts.bin` and keep a bounded, ranked working
set resident:

- `profile.bin` (196 632 B) is the pack's own expert ranking, so the residency set is "the first N experts in profile
  order", not an arbitrary prefix;
- the op reads a miss from the mapping and touches the page, so the OS page cache does the streaming; nothing is copied
  into a second buffer;
- the D/U file is small (3.74 GB in Q8) and is always resident;
- the PLE table (51.2 GB) is only ever read by row, through the disk reader that already exists in the overlay
  (`src/llama-ple-disk.*`), so it is never mapped whole.

### The residency knob, and what a 36 GiB budget buys on this machine

`port/tools/edge0-pack.py --plan` reports the split for a memory budget.  On the released Q3_K pack with a 36 GiB
budget, a 262 144-token context and a q4_0 KV cache it prints:

```
fixed     : 13.76 GiB
  skeleton weights          5.58 GiB
  D/U (q8)                  3.49 GiB
  KV cache, 12 full layers, 262144 ctx, q4      1.69 GiB
  recurrent state, 36 layers           0.004 GiB
  compute buffers           3.00 GiB (given)
experts   : 22.24 GiB -> 14645 of 24576 blobs resident (59.6%), about 305 of 512 per layer,
            15.08 GiB streamed per full pass
per token : 10 routed slots -> about 4.0 expert misses, 6.3 MiB streamed at most
```

Three things this makes explicit:

- **the KV figure is small because only 12 of the 48 layers keep a KV cache** (full attention every fourth layer); the
  other 36 keep a fixed recurrent state, which costs about 4 MiB in total.  A q4_0 256k cache for this geometry is
  therefore about 1.7 GiB, not the tens of GiB a dense model of the same context would need;
- **what is left goes to experts**, and at 1 630 720 B per blob a 36 GiB budget holds about 60% of the expert set;
- **the PLE table is never resident**: it is mapped and read by row, so its 47.68 GiB does not enter the budget.

The KV and compute figures are formulas over the documented geometry, not measurements on this machine, and the expert
count is what the budget allows before fragmentation.  Those two caveats are printed by the tool as well.

`--simulate-heat` compares the pack's static ranking with an online heat policy over a Zipf access model.  It is a
deliberately simple simulation (cold start, resident set recomputed on every draw) and on it the static ranking wins:
92.2% against 90.2% over 20 000 draws.  Read that as "the static ranking is a strong prior, and an online policy has to
beat it before it is worth the bookkeeping", not as a verdict - a real online policy would warm up over thousands of
tokens and would only re-rank occasionally, which this simulation does not model.

## The companion is not the shipped route

Measured on this machine: the repack writes at about 69 s per 0.85 GiB, which is roughly **55 minutes for the full
40.81 GiB** of the released pack.  A conversion step that long cannot be part of a setup workflow, so the companion is
kept as a **debugging and benchmarking tool** and the shipped route binds the pack's own files:

- the loader already keeps a table of files (`llama_model_loader::files`) and addresses every weight as a
  **(file index, byte offset)** pair; `apply_sidecar()` uses exactly that to serve a tensor from another file;
- so the latent tensors can be **registered against `experts.bin`, `latent_dense_q8.bin` and `labels.bin` directly**,
  with offsets computed from `native_experts.txt` and the D/U arithmetic. No bytes are copied, nothing is rewritten, and
  the pack stays exactly as downloaded;
- the repack then only exists to make a small, inspectable GGUF for tests (`--layers 1` takes about 70 s) and for the
  probe, not to run the model.

That also settles the memory question: a mapped weight costs nothing until it is touched, which is the same property the
URAM residency policy relies on.

### Binding the pack's own files: what fits and what does not

The loader addresses every weight as a **(file index, byte offset)** pair, so a tensor can be served from any mapped
file with no copy.  Checking that against the pack's layout gives one clean case and one that needed a change:

- `latent_d`, `latent_u` and `latent_labels` are **contiguous per layer** in their files, so they bind directly;
- the **three expert sections are not**.  `experts.bin` stores each expert as one blob `[gate | up | down]`, so a layer's
  gates are 512 runs of `n_ff * row` bytes separated by the up and down sections.  A GGUF tensor is one contiguous
  range, so "the layer's gate tensor" cannot be expressed over this file at all - which is exactly why the repack existed.

The resolution keeps the zero-copy property: the op takes **one byte view of the whole blob per layer** plus the two
quantisation types and the three section sizes, and computes the section offsets itself.  It has to know that layout
anyway (the lazy-residency policy is per expert blob), so nothing is added; what is removed is the copy.

Concretely, the loader registers four tensors per layer - `latent_d`, `latent_u`, `latent_labels` and `experts_blob`
(an `I8` view of the layer's `512 * blob` bytes) - and the op carries `gu_type`, `dn_type`, `latent`, `n_ff`,
`n_expert` and `blob` in its parameters.  A companion GGUF remains usable for tests: the probe can present the same
weights as three typed tensors, and the op accepts either form.

One caveat worth writing down rather than discovering later: a byte-view tensor bypasses ggml's per-type row size
arithmetic, so the op has to validate that `blob == 2*n_ff*row(gu) + latent*row(dn)` and refuse to run otherwise.

## The per-layer embedding table at a lower precision

`ple-fp8.gguf` is named for FP8, but the tensor inside it is **int8** (`per_layer_token_embd.weight`, type 24 in the
built library's numbering): 320 001 536 rows of 160 values, one signed byte each, 47.68 GiB in total.  That matters for
any conversion plan, because int8 is the reference the error has to be measured against.

`port/tools/ple-quant-eval.py` samples rows from the file **as downloaded** (a partial file is fine, rows are
independent) and reports what each storage scheme would cost and lose.  On 48 rows sampled while the download was at
73%:

| scheme | bytes/row | % of the file | relative L2 | 1-cos |
| --- | ---: | ---: | ---: | ---: |
| the file as it is (int8) | 160 | 100.0 | - | - |
| Q8_0 | 170 | 106.2 | 0.0034 | 0.00001 |
| **Q4_0** | **90** | **56.2** | **0.0621** | **0.00191** |
| GSQ-4 (one scale per 32 values) | 90 | 56.2 | 0.0621 | 0.00191 |
| affine 4-bit, one scale and zero per row | 84 | 52.5 | 0.0534 | 0.00142 |
| k-means 4-bit, 16 centroids per row | 112 | 70.0 | 0.0341 | 0.00059 |

What the numbers say, in the terms the choice is actually made in:

- **Q4_0 halves the table** (90 bytes per row against 160) at about 6% relative L2 on the row vector, and its cosine
  error is 0.2%.  Q8_0 is *larger* than the file (170 bytes), so it is not a candidate at all;
- **GSQ-4 with a per-32 scale lands on exactly the same error as Q4_0**, which is expected: the two differ only in the
  sign/zero handling of the codes, and these rows are zero-mean enough that it makes no difference here;
- **a per-row affine code is slightly better than Q4_0 at fewer bytes** (84 against 90), because one scale and one zero
  per row describe 160 values well;
- **k-means per row buys about 1.8x lower error for 24% more bytes than Q4_0** (112 against 90); it is the accurate end
  of the 4-bit options, not the cheap end.

Two limits worth stating: 48 rows is a sample, and the error is measured against the int8 decode of the same rows, so it
does not include the error the file already carries.  Whether a 6% row error costs text quality is a question for a
text comparison, not for this table.

## It runs, and what it costs today

The released 145B Q3_K pack now loads and generates on this machine with no companion file: the expert blobs, the D/U
pair and the labels are read from the pack's own files, the per-layer table is read from `ple-fp8.gguf` row by row, and
the latent block runs in the scalar CPU reference.  Measured on one run, prompt 9707, 8 tokens:

```
41315,232842,144604,35844,112013,96466,100022,70313,
0.1 tok/s decode (steady 0.1 over 7), first step 20441 ms, prompt 20420 ms,
cpu 140531 ms of 144810 ms wall
```

Read the numbers as what they are:

- **about 20 s per token, and 97% of the wall clock is CPU time** - the scalar latent block, not the GPU.  The Vulkan
  kernels do the dense part; the expert block is the whole cost.
- **the first token id repeats across runs** (41315 for this prompt), which is the only correctness signal available
  here: no reference implementation of this pack runs on this machine to compare logits against.
- a single-step run printed `597.1 tok/s (steady over 1)`.  That figure is degenerate - one step, measured when the
  pipeline had nothing left to do - and is not a throughput.  Seven steps give 0.1 tok/s, and that is the number.

So the next piece of work is not tuning but moving the latent block off the scalar path.  Two routes, with the costs
this file can state up front:

1. **stage the selected experts per token**: the routed set is 10 experts per layer, 1 630 720 B each, so 780 MiB per
   token across 48 layers; at the memory bandwidth this machine shows that is tens of milliseconds, and it reuses the
   quantised matvec kernels the Vulkan backend already has;
2. **a latent kernel family** that reads the blob in place (a host-visible, device-local buffer on a unified-memory
   part), which removes the staging copy but costs new shaders to write and validate.

Route 1 is the one that can be measured soon; route 2 is the one that is right in the end.  Both are stated here as
plans, not results.

## The Vulkan kernel: interface, staging contract, validation plan

The CPU op is the correctness baseline only; the shipped path is a Vulkan kernel.  What follows is the interface the
backend needs, written down before the shader so the pieces can be built and checked separately.

### Why the block cannot be expressed with the backend's existing ops

Two attempts at a graph-only version fail on the same point: the row an expert reads is
`group[expert] + n_groups * token`.  `ggml_get_rows` takes one index tensor per token and there is no integer `add` in
the graph that both backends implement for I32, so the composite index cannot be built on the device; and the pack's
experts are one blob `[gate | up | down]` per expert, which cannot be three standard tensors over that file.  Both
facts point at one kernel that knows the layout, which is also what the reference implementation does.

### The interface

    LATENT_MOE(cur, d, u, experts, labels, ids, weights) -> out

    cur      [n_embd, n_tokens]          the block's activation
    d        [n_embd, latent*n_groups]   per-group down-projection rows, Q8_0 or Q4_0, on the device
    u        [latent, n_embd, n_groups]  per-group up-projection rows, same format, on the device
    experts  the selected experts' rows for this dispatch (see the staging contract)
    labels   [n_expert] int32            group of each expert (device, small)
    ids      [n_used, n_tokens] int32    the routed experts (device, produced by the router)
    weights  [1, n_used, n_tokens] f32   their normalised weights (device)

`d` and `u` are small enough to stay on the device (3.7 GiB for the released pack at Q8_0), so they are ordinary
device tensors and the kernel reads them in place.

### The staging contract for the experts

`experts.bin` is 40.8 GiB and cannot be resident, so the routed set is staged: per layer, per token, the kernel reads
`n_used` expert blobs, which is 10 x 1 630 720 B = 15.6 MiB.  The host copies exactly those blobs into a device-visible
staging buffer before the dispatch, and the kernel addresses slot `s` at `s * blob`.  Two properties make this the right
first step:

- the copy is bounded and predictable (about 750 MiB per token across 48 layers), and it disappears for the resident
  majority once the hot-expert cache holds them on the device;
- the kernel's addressing is then identical to the CPU reference's (`slot * blob + section offset`), so a mismatch is a
  kernel bug and not a layout question.

The resident set is what the memory budget knob sizes (`--expert-cache`): blobs that are resident are read in place, the
rest are staged for the token that needs them.  The kernel does not have to know which is which - both are device
addresses - so the cache can land later without touching the shader.

### Validation plan, in the order it has to happen

1. a fixture whose weights are Q8_0 (D/U) and Q3_K/IQ4_NL (experts), small enough to check every value by hand;
2. kernel vs the CPU reference on that fixture, value for value, with the reference's own tolerance;
3. kernel vs reference on one real layer (`--layers 1` companion, about 70 s to make);
4. only then the end-to-end token, compared with the CPU baseline's first-token ids for the same prompt.

Steps 1-2 are the gate: a kernel that disagrees on a fixture will not be debugged through a 40 GiB model.

### Two details the first shader draft settled

Writing the kernel's skeleton answered two questions that the graph-only attempts kept tripping over, so they are
recorded here rather than rediscovered:

- **the staged blob is addressed in bytes, not words.**  A Q8_0 block is 34 bytes - `{half d; int8 q[32]}` - which is
  eight and a half `uint` words, so a shader that indexes the staging buffer as `uint[]` has to compute the byte offset
  and then divide, on every access.  The staging buffer is therefore declared as a byte array and only the scale is
  read as a word;
- **`xl` needs a token stride.**  The D projection writes `xl[g][r]` per token, so the buffer is
  `[n_groups * latent, n_tokens]` and every read is `xl[(g * latent + r) * n_tokens + t]`.  My first draft omitted the
  stride, which is the same class of mistake the two CPU attempts made (assuming one token), so it is written down.

Neither is a design change; both are the kind of detail that costs a debugging session on a 40 GiB model if it is
discovered there instead of here.

### How this kernel gets landed: in pieces that each compile

Two attempts at writing the whole kernel in one pass produced drafts I would not commit, and the reason is worth
writing down: a shader of this size cannot be validated until the pipeline, the meta entry, the dispatch and the
staging buffer all exist, so a single-pass attempt is a large change with no way to test any part of it.  The kernel
therefore lands in three pieces, each of which is compiled and checked before the next:

1. **plumbing, with a trivial shader**: `latent_moe_copy.comp` reads `xl` and writes it to `y` unchanged.  It exercises
   the meta entry, the pipeline variant, the descriptor set, the parameter block and the dispatch, and its result is
   checkable in isolation (y must equal xl).  Nothing about the arithmetic is involved, so a failure here is a wiring
   failure and nothing else.
2. **the expert half**: gate/up/down for one slot, with `shared` arrays sized from specialization constants - the idiom
   the existing shaders use (`layout(constant_id = 1) const uint N = 128; shared float shmem[ROWS * N];`), not
   `extern shared`, so the host does not have to pass a shared-memory size at pipeline creation.
3. **the U half and the weighted sum**: `out[t][r] = sum_s w_s * dot(U[g_s][r], y_s)`, one invocation per output
   element, which needs no reduction and no atomics.

Step 1 is the piece to land next: it is small, it compiles on its own, and it turns every later failure into an
arithmetic question instead of a wiring question.

### The six places a new op touches the backend

Read off `ATTN_MERGE`, which is the same shape of addition (one fork-specific op, one shader, one dispatch):

| file | what to add |
| --- | --- |
| `vulkan-shaders/latent_moe_copy.comp` | the shader source |
| `vulkan-shaders/vulkan-shaders-gen.cpp` | `string_to_spv("latent_moe_copy_f32", "latent_moe_copy.comp", {})`, and the file named in the source list near line 461 |
| `ggml-vulkan-shader-meta.h` | `{ "latent_moe_copy", VK_SHADER_MASK_DERIVE, true }` |
| `ggml-vulkan-types.h` | `vk_pipeline pipeline_latent_moe_copy_f32;` |
| `ggml-vulkan.cpp` (pipeline) | `ggml_vk_create_pipeline(device, device->pipeline_latent_moe_copy_f32, "latent_moe_copy_f32", latent_moe_copy_f32_len, latent_moe_copy_f32_data, "main", 2, sizeof(vk_op_push_constants), {256, 1, 1}, { 256 }, 1)` |
| `ggml-vulkan.cpp` (dispatch) | the `ggml_vk_latent_moe_copy` function, a `case GGML_OP_LATENT_MOE:` in the graph-build switch (near line 14705) and one in `supports_op` (near 18473) |

Two notes that save a session each:

- **the copy probe uses push constants, not the parameter block.**  The conventions prefer the parameter block for a
  new shader, and the real kernel will take that path; the probe exists to prove the wiring, and `attn_merge` is a
  complete push-constant template to copy.  The switch happens with the real kernel, not with the probe.
- **`supports_op` must say no while the probe is in place.**  A probe that only copies must not be scheduled into a
  real model: `supports_op` returns true only for a tensor whose name says it is the probe, so the engine's fallback to
  the CPU keeps working during the three landing steps.

### 已验证的 Vulkan 结果

四条路径都在 `ggml_backend_vk` 上跑通，并与 `port/tools/latent-ref.py`（其 Q3_K 解码由
`port/tools/q3k-check.py` 对齐到 ggml 的 `to_float`，误差 2.4e-7）以及 CPU 参考逐值对照：

| 用例 | 规模 | 结果 |
| --- | --- | --- |
| typed，真实 layer 0（Q3_K gate/up + IQ4_NL down） | latent 1792, n_ff 640, n_embd 2560 | max abs 2.19e-05（量级 2.4e4）|
| typed，小型夹具（F32 专家 + Q8_0 的 D/U） | latent 32, n_ff 8 | max abs 0.0039（量级 2.4e4）|
| blob，小型 blob 夹具（与发布包同布局） | latent 256, n_ff 64, 4 专家, blob 23,296 B | max abs 13（量级 3.0e7）|
| blob，同一夹具，CPU 参考 | 同上 | max abs 5.5（量级 3.0e7）|

### 定位这些问题时用到的方法

调试一个"数值全错"的 GPU 内核，靠反复上设备二分很慢（每轮数分钟）。更快的是**把 shader 的表达式逐行转写成
Python，与已对齐的解码器在同一份字节上比对**：差异在几十毫秒内就能看出来，而且能精确指出是哪一个偏移、哪一处位序。
本功能里 Q3_K 的三处错误（量化字节应始于 32 字节掩码之后、块不按 4 字节对齐、scale 的取字索引）都是这样找到的，
之后再用设备确认。

还有两条经验值得记下：判据必须是"对照参考实现"，不能是"两次设备运行互相比较"；同一个表达式同时存在编译期和运行时
两种写法时，必须一次性统一，否则会出现"typed 对、blob 错"这种很难定位的状态。

### 设备侧放置的实测边界（145B Q3_K）

`--latent-device` 把专家 blob 放进图形设备可见的内存，在 145B 包上实测到两处硬边界：

1. **设备分配（device-local）**：第一个 1 GiB 量级的分配就失败（`Device memory allocation of size 1009160064
   failed`），所以 40.8 GiB 的 blob 不可能作为设备分配存在；
2. **设备可见的宿主内存（host-visible，经外部内存导入）**：模型能完整加载，但随后分配 297 MiB 的 compute buffer
   失败——因为 40.8 GiB 的 blob 加上 5.6 GiB 的骨架与 3.5 GiB 的 D/U 已经超过本机 47.6 GiB 的物理内存。

结论：**全量常驻在这台机器上不可行**，这与"强 Lazy Load + 热专家缓存"的设想一致——blob 只能有一部分常驻，其余按需读取。
因此 `--latent-device` 的正确形态是带**预算参数**（常驻子集大小），而不是全量。默认路径（宿主内存、按行读取）保持不变。

顺带修掉的两处结构性问题（都会影响任何"把大张量放到设备上"的尝试）：

- 图 arena 的余量按"没有额外输入"预留，latent 块的 scratch 输入正好把它撑破（差 368 字节）；
- 每个 buffer type 的模型上下文按**文件声明的**张量数分配，而一个 pack 还会**创建**张量（每层 D/U、labels、blob），
  把 latent 张量放到设备上时正好差一个槽位。

### 流式读取的实测代价

既然 40.8 GiB 无法常驻设备可见内存（见上一节），可行的形态是**每 token 把选中的专家从文件流进设备**。用
`port/tools/expert-stream-bench.py` 按真实访问模式实测（48 层 × 每层 10 个专家 × 1.63 MB，随机专家号，三次取平均，
在本机 NVMe 上、Python 直接读）：

```
每个 token 读取 746 MiB，耗时 3.5 s
整包 37.32 GiB 顺序读一遍需要 179.3 s
```

对照当前 **21.7 s/token**（8 token 实测，98.7% 时间在 CPU），流式读取占约 16% 的时间预算，且可以与 GPU 计算重叠；
而若计算本身从 CPU 转到 GPU，这个比例会变得更突出，因此实现上应尽量让读取与计算重叠，而不是串行。

这个数字只说明可行性，不代表已经实现的性能：它是 Python 读文件的读数，换成 C++ 侧带预读的读取会不同。
