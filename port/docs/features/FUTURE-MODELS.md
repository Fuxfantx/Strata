# Models this port plans to take on

## Hy4-Flash-Compact

The pack is at `C:\Users\Fuxfantx\C-Res\Qwen\Hy4-Flash-Compact` on the development machine.  The work starts **after
the repin to b12722 is finished and the gates are green again**, because it would otherwise be built on a tree that does
not compile.

Order of work:

1. Read the pack's `README.md` first, and anything beside it (`model.json`, a native manifest, an index).  The pack
   decides the shape of the work; the README is the author's own statement of what the files contain.
2. Look through `PrismML-Eng/llama.cpp` and `halo-box/strix-llama.cpp` for facilities this port needs and does not
   have.  Two things to look for in particular, because the next quantisation step depends on them: the layout and
   kernels behind **PTQ1_0**, and whatever those forks do for the KV cache and the MoE dispatch that this port still
   does the slow way.  Nothing is adopted without being measured or validated in this tree.
3. Integrate, then tune.  Same rules as everything else here: a new feature is opt-in, the default path stays
   byte-identical, and every claim comes with a measurement and the hardware it was taken on.

What is *not* known yet, and must not be assumed: whether this pack fits the existing latent-MoE layout, whether it needs
new operators, and how large it is.  The README answers the first two better than any guess from a file listing.

## Hy4-Flash-Compact: what the pack actually contains (read on 2026-11-26)

The README is accurate about the shape of the release, and the inspector confirms it. Facts taken from
`tools/inspect_gguf.py` on file 00001 (not from the README alone):

- architecture key is **`hy_v4_flash`**, 43 blocks (`block_count` 43 with `leading_dense_block_count` 1), embedding 3072,
  vocab 120832, 512 experts with top-16, one shared expert, expert FFN 1536, latent dimension 1536;
- MLA: `attention.key_length` 576, `value_length` 512, `kv_lora_rank` 512, `q_lora_rank` 1024,
  `key_length_mla`/`value_length_mla` 256;
- **an indexer** (`attention.indexer.head_count` 16, `key_length` 128, `top_k` 2048, per-layer `types`) — the same idea as
  the DeepSeek-V4 indexer this tree already implements;
- **hyper-connections** (`hyper_connection.count` 4, epsilon 1e-6, magnitude 2.0) — again the DSV4 family of operators;
- `swiglu_clamp_exp` per layer, `expert_gating_func` 2, `expert_weights_scale` 3.998, `expert_weights_norm` 1;
- 1661 tensors in the backbone file, split by type: F32 716 (0.28 GiB), **CSQ8 478** (2.42), **CSQ4 126** (12.02),
  **VQ4_128 123** (5.88), **VQ4_64 123** (13.51), Q8_0 95 (0.26);
- the rotation contract is `prism.hadamard.*` (block 512, 849 names, explicit signs, inverse for `token_embd.weight`);
- codebooks live in the file (`prism.csq8.codebook`, `prism.csq4.codebook`, `prism.vq4_128/64.codebook`) and the expert
  split maps in `hy4f.expert_split.*` (hot/warm offsets and ids, 42 layers).

So the work is: four new backbone block types (144–147), three Engram types (148–150) in the second file, an
`hy_v4_flash` architecture that is a close relative of the DeepSeek-V4 path already here (MLA + indexer +
hyper-connections + MTP `nextn` tensors), split expert groups instead of one expert stack, and the Engram table lookup.

Order of work from here:

1. Types first: implement dequantisation for CSQ8/CSQ4/VQ4_128/VQ4_64 from the codebooks and block layouts, verified
   against a reference we write from the pack's own description (block sizes 196 / 132 / 116 / 100 bytes, block 512).
2. Then the architecture: reuse `deepseek4`'s graph where it matches (MLA, indexer, hyper-connections) and add the
   differences (split expert groups, the Hadamard pre-rotation, Engram).
3. Then the Engram table with its index and tier dispatch, streamed rather than resident.
4. Tune last, with the same rules as everything else here.

## The four backbone block types, from the pack's own encoder

`tools/csharp/tiercodec/Program.cs` is the encoder that produced the file, so it is the reference for the layout rather
than the README prose. Block = **512 weights**; every block is `index bytes + a float32 scale last`
(`*(float*)(o + ib) = d`).

| Type | id (file) | block bytes | index bytes | bits | codebook |
| --- | ---: | ---: | ---: | ---: | --- |
| CSQ8 | 144 | 196 | 192 | 3 | 8 scalar entries (`prism.csq8.codebook`) |
| CSQ4 | 145 | 132 | 128 | 2 | 4 scalar entries (`prism.csq4.codebook`) |
| VQ4_128 | 146 | 116 | 112 | 7 | 128 vectors of 4 (`prism.vq4_128.codebook`) |
| VQ4_64 | 147 | 100 | 96 | 6 | 64 vectors of 4 (`prism.vq4_64.codebook`) |

- Scalar types: `x[i] = d * codebook[idx[i]]`.
- Vector types: 128 groups of 4 weights, `x[4g + j] = d * codebook4[idx[g]][j]`.
- Bit packing: **LSB first into little-endian 32-bit words** (`PackBits`); CSQ4 packs four indices per byte as
  `(a&3) | ((b&3) << 2) | ((c&3) << 4) | ((e&3) << 6)`, which is the same LSB-first order.
- Storage order inside a block is the **rotated** order: the encoder applies `row[i] *= signs[i]` and then an in-place
  FWHT over each 512-weight block scaled by `1/sqrt(512)` **before** quantising. So dequantising gives the rotated
  weights, and inference must apply the matching transform on the input side (see `prism.hadamard.*`, and the inverse
  order for `token_embd.weight`).

### A decision this forces: the type ids collide with ours

The file uses 144 CSQ8, 145 CSQ4, 146 VQ4_128, 147 VQ4_64, 148/149/150 Engram. This tree already owns 144/145/146 for
`TQ2_T`/`TQK6`/`TQK7` (the comment in `ggml.h` says those ids are the contract with GGUF files already made), and
`GGML_TYPE_COUNT` is 147.

Plan: keep the existing ids untouched and give the new types fresh ids from 147 upwards (CSQ8 147, CSQ4 148, VQ4_128 149,
VQ4_64 150, Engram 151/152/153), with a **loader remap for the `hy_v4_flash` architecture only**: file ids 144..150 map to
our ids when reading this pack. Nothing else in the tree changes meaning.

### The four backbone types are in the tree and it builds

- `ggml.h`: `GGML_TYPE_CSQ8 = 147`, `CSQ4 = 148`, `VQ4_128 = 149`, `VQ4_64 = 150`, and placeholders for the three
  Engram types (151..153); `GGML_TYPE_COUNT` is now 154.  The trellis ids 144..146 are untouched.
- `ggml-common.h`: `QK_HY4 = 512` with `block_csq8` (196 B), `block_csq4` (132 B), `block_vq4_128` (116 B),
  `block_vq4_64` (100 B), each with a `static_assert` on the size.
- `ggml-quants.h/.c`: `dequantize_row_csq8/csq4/vq4_128/vq4_64` plus `ggml_hy4_set_codebooks`, which the loader calls
  once with the four codebooks (8, 4, 128*4 and 64*4 floats).  Indices are unpacked LSB first out of little-endian
  32-bit words, which is what the pack's encoder writes for all four types (the CSQ4 byte formula is the same order).
- `ggml.c`: traits entries with `blck_size = QK_HY4`, the block size, `is_quantized` and `to_float`.

`cmake --build build --target ggml` links cleanly with these changes.

Still to do for this stage: a probe that decodes a real block through `to_float` and compares it byte-for-byte against
`port/tools/hy4-dequant-ref.py`, and the loader side (id remap for `hy_v4_flash`, codebook extraction from the KVs).

### Verification: the C dequantisers agree with the pack reference

`port/probe/hy4-probe.cpp` decodes one **real block** of each type through `ggml_get_type_traits(...)->to_float` and
compares it value for value with `port/tools/hy4-dequant-ref.py`, which was written from the pack's own encoder. The
blocks and expected values come from `hy4-dequant-ref.py --dump <dir>`.

```
csq8       512 values  max abs diff 0.000e+00  first -0.049038 (ref -0.049038)
csq4       512 values  max abs diff 0.000e+00  first -0.007795 (ref -0.007795)
vq4_128    512 values  max abs diff 9.313e-10  first  0.041675 (ref  0.041675)
vq4_64     512 values  max abs diff 9.313e-10  first  0.013893 (ref  0.013893)
all four types match the reference
```

The two `9.3e-10` rows are float rounding (the reference prints nine significant digits), so the unpacking matches the
encoder's packing exactly.

Next for this stage: the loader — remap the pack's file ids 144..150 onto this tree's ids, read the four codebooks from
the key-values, call `ggml_hy4_set_codebooks`, and expose the tensors to a graph (still no compute kernels, so a model
cannot run yet).

### The reader now accepts the pack's header (and stops on the Engram types)

`ggml/src/gguf.cpp` (now carried in the overlay) detects `general.architecture == "hy_v4_flash"` after the key-values and
remaps the file's tensor type ids before any size arithmetic:

- 144..150 -> this tree's CSQ8/CSQ4/VQ4_128/VQ4_64/ENGRAM_Q4B12/ENGRAM_Q6B12/ENGRAM_Q2B12 (147..153);
- id 4 (UINT32 in the writing format, kept as "removed" here) -> `GGML_TYPE_I32`.

With that, the backbone file gets past the first failure, which used to be
`tensor 'output_hc_base.weight' has offset …, expected …` — the offsets now line up, so the block sizes we registered
match what the file was written with. `hy4-probe` still matches the reference after the change.

The next stop is expected and is the next piece of work:

```
gguf_init_from_reader: tensor 'engram_warm.weight' of type 152 ((null)) has 96 elements per row, not a multiple of
block size (0)
```

i.e. the three Engram types exist in the enum but have no traits yet. Their layouts, from the pack's own
`TYPE_LAYOUT` and README: 96 values per block as **eight 12-value sub-blocks**, an 8-bit logarithmic scale per
sub-block, then 2/4/6-bit codes — 32 / 56 / 80 bytes per block respectively, which is exactly `8 scales + 96*bits/8`.
The `engramcodec` encoder is the reference for the scale formula and the code semantics.

### The three Engram types are in, and the pack's header now parses end to end

From `tools/csharp/engramcodec/Program.cs` (the encoder), 96 values per block as **eight 12-value sub-blocks**, codes
first and one 8-bit logarithmic scale per sub-block after them:

| type | id | block bytes | codes | centre / divisor |
| --- | ---: | ---: | --- | --- |
| ENGRAM_Q2B12 | 151 | 32 | 24 bytes, 4 x 2-bit per byte | `round(v*1.5 + 1.5)`, dequant `(q - 1.5)/1.5` |
| ENGRAM_Q4B12 | 152 | 56 | 48 bytes, 2 x 4-bit per byte | `round(v*7.5 + 7.5)`, dequant `(q - 7.5)/7.5` |
| ENGRAM_Q6B12 | 153 | 80 | 72 bytes, 12 x 6-bit LSB-first over 9-byte groups | `round(v*31.5 + 31.5)`, dequant `(q - 31.5)/31.5` |

The scale code is `round((log(s) - lo) * 255/(hi-lo))` and the table is `exp(lo + q*(hi-lo)/255)`, with `lo`/`hi` from
`hy4f.engram.scale_lo` / `scale_hi`; `ggml_hy4_set_engram_range(lo, hi)` builds it.

With those registered, the reader gets through **all 1661 tensors** of the backbone file. The next stop is llama's
loader, not the format:

```
llama_model_load: error loading model: missing key split.no in GGUF split …
```

The pack is not a llama-style split: there are no `split.*` keys, and the second file is the Engram table rather than a
shard of the same model. So the `hy_v4_flash` loader has to ignore the `-00001-of-00002` naming for the split logic and
treat 00001 as the whole backbone. That is part of the architecture work (stage 2), which is the next piece.

### Stage 2 started: the architecture is registered and the loader reaches the model factory

- `llama-arch.h/.cpp`: `LLM_ARCH_HY_V4_FLASH` with the name `"hy_v4_flash"`, and it joins the DeepSeek-V4 family's
  capability groups (`llm_arch_is_hybrid`, `supports_rs_rollback`, `supports_mixed_batch`), because the pack is the same
  kind of model: sparse attention + MoE with hyper-connections.
- `llama-model-loader.cpp`: a `hy_v4_flash` pack is **not** a gguf-split. 00001 is the whole backbone and 00002 is the
  Engram table, so the name-based split path (which exists for packs assembled outside `gguf-split`) has to skip this
  architecture; otherwise the loader opens the Engram file as a shard and dies on its missing `split.no`.

What the loader says now:

```
llama_model_load: error loading model: unsupported model architecture: 'hy_v4_flash'
```

That is the expected next wall: there is no model class yet. The remaining stage-2 work is the model itself — tensor-name
table (`load_arch`), weight loading (`load_tensors`) with the split expert groups and the Engram table, and the graph
(MLA + indexer + hyper-connections + split-expert MoE + Engram + MTP), modelled on `deepseek4` where it matches.

### The architecture is a DeepSeek-V4.1-Flash variant, so the work is an adaptation, not a new model

`hy_v4_flash`'s 63 distinct tensor suffixes are the DeepSeek-V4.1 names — `attn_q_a`, `attn_q_b`, `attn_kv_a_mqa`,
`attn_k_b`, `attn_v_b`, `attn_sinks`, `exp_probs_b.bias`, `ffn_gate_inp`, `ffn_*_shexp`, `indexer.*`, `hc_attn_*`,
`hc_ffn_*`, `nextn.*`, `output_hc_*` — and this tree already implements that family in `src/models/deepseek41.cpp`,
**including Engram** (`LLM_TENSOR_ENGRAM_KV/GATE`, the n-gram hash, and one lazily read Engram table).

What differs, and is therefore the actual work:

| Difference | Detail |
| --- | --- |
| Split experts | `ffn_{gate,up,down}_exps_{hot,warm,cold}` instead of one `_exps` stack per layer, with the global-to-local map in `hy4f.expert_split.*` |
| Block types | CSQ8/CSQ4/VQ4_128/VQ4_64 for the backbone, Engram 2/4/6-bit for the table (all now in ggml) |
| Hadamard rotation | `prism.hadamard.*`: 849 names, 512-weight blocks, signs, inverse order for `token_embd.weight` |
| Tiered Engram | `engram_low/high/hot/warm` tables plus a 660M-row `engram_index` (top 2 bits tier, low 30 bits row) instead of one table |
| Extra tensors | `post_attention_norm`, `post_ffw_norm`, `ffn_routed_{down,up}`, `indexer_compressor_*`, `engram_lookup_norm`, `engram_pre_gamma`, `engram_hc_scale`, `engram_kv_b` |

So stage 2 becomes: bring up a `hy_v4_flash` model class modelled on `deepseek41`, add the split-expert loading and
dispatch, add the tiered Engram lookup, and apply the Hadamard contract on the input side. That is still a large piece of
work, but it is a delta on working code rather than a new architecture from scratch.

### The tree already implements the Hadamard contract, and the loader says so

Bringing `hy_v4_flash` up as a `llama_model_deepseek41` subclass (declared at the end of `models.h`, one factory case, no
new .cpp yet) gets the loader to the first real difference:

```
error loading model hyperparameters: prism.hadamard: arch 'hy_v4_flash' is not verified to apply the activation
transform to all folded weights
```

That message comes from `llama-model.cpp`, which **already** reads `prism.hadamard.*` — version, block size, transform,
axis, sign mode, weight names, sign widths, sign values, the inverse list, and an optional GDN flag — and refuses an
architecture whose graph might skip the transform for a folded weight. The contract it implements is the same one this
pack documents: `W*x = W_f*(H*(D*x))`, i.e. apply the signs then the normalized block Hadamard to the matmul input.

So the Hadamard piece is **not** new work: what remains is to add `hy_v4_flash` to that verified list once its graph is
known to apply the transform to every folded weight (the split expert names `ffn_*_exps_hot/warm/cold` begin with the
kinds the check already knows). Until then the loader refuses on purpose, which is the right behaviour.

Next: the model's own `load_arch_hparams` / `load_arch_tensors` overrides — split expert groups, the tiered Engram
tensors, and the extra norms — then the graph.

### Next wall: the foldable-weight list, one graph path at a time

Adding `LLM_ARCH_HY_V4_FLASH` to the verified architectures gets past the architecture check and straight into the
per-weight one:

```
prism.hadamard: weight 'blk.0.attn_kv_a_mqa.weight' is not on a verified Hadamard-aware matmul
```

That is the check doing its job. It matches each folded name against an exact list of kinds
(`attn_q`, `attn_k`, `attn_v`, `attn_qkv`, `attn_gate`, `attn_output`, `ffn_gate`, `ffn_up`, `ffn_down`,
`ffn_*_exps`, `ffn_*_shexp`, `ssm_out`, plus `output.weight`), and refuses anything it does not recognise — because a
folded weight is only correct if the graph multiplies it through `build_lora_mm`/`build_lora_mm_id`, which is where the
signs and the block Hadamard are applied to the input.

The pack folds 849 names, and its MLA and Engram tensors use names this tree has never folded
(`attn_kv_a_mqa`, `attn_k_b`, `attn_v_b`, `attn_q_a`, `attn_q_b`, `indexer.*`, `engram_*`, `nextn.*`, `hc_*`,
`output_hc_*`). So the honest path is: for each name group in `prism.hadamard.weight_names`, read how the deepseek41
graph builds that matmul, and only add the kind once the path is known to go through the Hadamard-aware builder. Anything
that does not is a real bug in the pack's assumptions or a graph path to change, not a list entry to add.

### The folded names settle which MLA the pack uses: DeepSeek-V2/V3 naming inside a V4-family model

The 849 folded names fall into 25 groups. Checked against how the graph multiplies them:

| group | count | kind | graph path |
| --- | ---: | --- | --- |
| `attn_gate`, `attn_output`, `ffn_gate/up/down` (dense), `ffn_*_exps_hot/warm/cold`, `ffn_*_shexp`, `output` | 1–43 each | already in the foldable list | `build_lora_mm` / `build_lora_mm_id` |
| `attn_q_a`, `attn_q_b`, `engram_kv` | 43 / 43 / 4 | new kinds, but the graph already uses `build_lora_mm` for them | ✓ safe to add |
| `attn_kv_a_mqa`, `attn_v_b` | 43 / 43 | **not present in deepseek41's tensor table at all** | needs the V2/V3 MLA path |
| `ffn_routed_down`, `ffn_routed_up` | 42 / 42 | new tensors (latent-MoE projections) | needs its own path |
| `eh_proj` (MTP) | 1 | new for this graph | needs its own path |

The two MLA groups are the important discovery: `attn_kv_a_mqa` and `attn_v_b` are the **DeepSeek-V2/V3** MLA names, which
this tree implements in `deepseek2.cpp`, while `deepseek41` uses its own `attn_kv.weight` shape. So `hy_v4_flash` is a
V4-family model (indexer, hyper-connections, Engram, `nextn`) whose attention is the **older MLA decomposition**:
`attn_q_a`, `attn_q_b`, `attn_kv_a_mqa`, `attn_k_b`, `attn_v_b`, `attn_gate`, `attn_sinks`.

That is why `LLM_TENSOR_ATTN_KV_A_MQA`, `LLM_TENSOR_ATTN_K_B` and `LLM_TENSOR_ATTN_V_B` existing in the enum matters: the
`hy4flash` model needs a tensor table that mixes `deepseek2`'s attention tensors with `deepseek41`'s indexer /
hyper-connection / Engram ones. Next step is exactly that table, then the attention builder to go with it.

### The tensor table the model needs, with the shapes read from the file

The pack's layer-1 shapes match the V2/V3 MLA formulas exactly, so the tensor table is not guesswork:

| pack tensor | shape | meaning |
| --- | --- | --- |
| `attn_q_a.weight` | (3072, 1024) | `{n_embd, q_lora_rank = 1024}` |
| `attn_q_b.weight` | (1024, 8192) | `{q_lora_rank, n_head * n_embd_head_k_mla}` with 32 x 256 |
| `attn_kv_a_mqa.weight` | (3072, 576) | `{n_embd, kv_lora_rank(512) + qk_rope(64)}` |
| `attn_k_b.weight` | (192, 512, 32) | `{n_embd_head_qk_nope, kv_lora_rank, n_head}` |
| `attn_v_b.weight` | (512, 256, 32) | `{kv_lora_rank, n_embd_head_v_mla, n_head}` |
| `attn_gate.weight` | (3072, 8192) | `{n_embd, n_head * n_embd_head_v_mla}` |
| `attn_sinks.weight` | (32,) | one sink per head |
| `ffn_gate_inp.weight` | (3072, 512) | router over 512 experts |
| `ffn_gate_exps_hot.weight` | (1536, 1536, 160) | latent MoE: 1536 latent, 1536 intermediate, 160 hot experts |
| `ffn_routed_up.weight` | (1536, 3072) | latent to expert space |
| `ffn_routed_down.weight` | (3072, 1536) | expert space back to latent |

with `hot/warm/cold` carrying 160 / 96 / 256 experts per layer and the global-to-local map in `hy4f.expert_split.*`,
and layer 42 (the MTP head) using 512 hot experts.

That is the specification `load_arch_tensors` follows: V2-style MLA tensors, V4-family indexer and hyper-connection
tensors, split expert stacks, `ffn_routed_{up,down}` around the latent MoE, and the Engram group
(`engram_kv`, `engram_kv_b`, `engram_gate`, `engram_lookup_norm`, `engram_pre_gamma`, `engram_hc_scale`).

### The order the remaining work has to follow

`hy4flash` now has its own `load_arch_hparams` override (it calls deepseek41's and then reads the MLA ranks from
`attention.q_lora_rank`, `kv_lora_rank`, `key_length_mla`, `value_length_mla`). It builds and links.

What it shows is the ordering constraint: the Hadamard per-weight check runs **inside** the base hyperparameter load, so
it refuses `blk.0.attn_kv_a_mqa.weight` before any tensor table is reached. That check exists to guarantee that every
folded weight is multiplied through `build_lora_mm`, so it cannot be satisfied by adding names first. The sequence has to
be:

1. the `hy4` layer structure and tensor table (attn_q_a/q_b, attn_kv_a_mqa, attn_k_b, attn_v_b, split experts, routed
   projections, Engram group);
2. the attention and MoE graph on top of them, all multiplications through `build_lora_mm`/`build_lora_mm_id`;
3. then the four MLA kinds join the foldable list, because by then the claim is true.

Until step 2 exists, the loader refusing is the correct behaviour and not a blocker to work around.

### What the tensor table needs, learned by trying it (and backing out cleanly)

A first attempt at `load_arch_tensors` (the V2 attention group: `attn_norm`, `attn_q_a_norm`, `attn_kv_a_norm`,
`wq_a`, `wq_b`, `wkv_a_mqa`, `wk_b`, `wv_b`, `wo`, `attn_sinks`) does not compile yet, for details that are worth
writing down instead of rediscovering:

- the shared `llama_layer` struct in `llama-model.h` **already has** `wkv_a_mqa`, `wk_b`, `wv_b`, `wq_a`, `wq_b`,
  `attn_q_a_norm`, `attn_kv_a_norm`, `attn_sinks`, `ffn_gate_inp` and the `ffn_*_shexp` fields, so the table can fill
  them without touching that struct;
- `hparams.n_rot` is a method, and the vocabulary size comes from the loader (`LLM_KV_VOCAB_SIZE` /
  `LLM_KV_TOKENIZER_LIST` as `deepseek2` does), not from `hparams` — both mistakes are in the reverted attempt;
- the split expert stacks have no dedicated fields, so they need either new fields or a per-class vector plus name
  lookup for the graph.

The incomplete attempt was reverted rather than committed, and the build is green at this commit.

### Bring-up progress: the attention table compiles, and the hparams are the next mismatch

- `hy4flash::load_arch_tensors` now creates the **V2 attention group** per layer (`attn_norm`, `attn_q_a_norm`,
  `attn_kv_a_norm`, `wq_a`, `wq_b`, `wkv_a_mqa`, `wk_b`, `wv_b`, `wo`, `attn_sinks`) and the library builds.
- The Hadamard per-weight check stands between the hparams and any tensor, so bring-up uses an explicit, documented
  switch: `STRATA_HY4_BRINGUP=1` adds the pack's not-yet-verified kinds (the four MLA names, the three routed/tiered
  expert names, `nextn.eh_proj`, `engram_kv`) **for the run only**. It is a diagnostic, not a way to ship a folded
  weight on an unverified path, and the kinds join the real list when the graph that multiplies them exists.
- With that, the pack's folded names all pass and the loader moves on to a plain hyperparameter mismatch:

```
key not found in model: hy_v4_flash.attention.output_group_count
```

i.e. `deepseek41`'s hparams read keys this pack does not carry. Fixing that is the next step (read the pack's own keys,
and make the base ones optional where the pack does not use them), after which the tensor table is actually exercised.

### The hparams have to be read from the pack's own keys, not deepseek4's

The pack's structural keys and what `deepseek4::load_arch_hparams` asks for do not line up: the pack has
`attention.q_lora_rank`, `kv_lora_rank`, `key_length_mla`, `value_length_mla`, `attention.sliding_window`,
`expert_feed_forward_length` (scalar), `expert_shared_count`, `expert_weights_scale`, `expert_weights_norm`,
`swiglu_clamp_exp`, `expert_gating_func`, `hyper_connection.{count,epsilon,magnitude}`, `attention.indexer.{head_count,
key_length,top_k,types,kpool,kpool_select_tail}`, `ple.{layers,layer_multipliers}`, `embedding_length_per_layer_input`,
`engram.{pad_token_id,token_map}` and `expert_latent_length`, but it has **none** of `attention.output_group_count`,
`attention.output_lora_rank`, `attention.compress_rope_freq_base`, `hyper_connection.sinkhorn_iterations`,
`hash_layer_count`, `attention.compress_ratios` or `swiglu_clamp_shexp`, which is what the first mismatch
(`hy_v4_flash.attention.output_group_count`) was about.

So `hy4flash::load_arch_hparams` must read the pack's keys directly rather than delegating: the generic part of the
loader already covers block count, context length, embedding width and head counts, and everything else comes from the
list above. That replaces the delegating override with an explicit one, which is also what makes the tensor table's
shapes provable.

### The attention is per-layer: head count and flavour both vary

`hy4flash::load_arch_hparams` now reads the pack's keys directly and the loader reaches the **tensor dimension checks**,
which is the first time the pack's shapes are validated against the declared table. The first mismatch is informative:

```
check_tensor_dims: tensor 'blk.2.attn_q_b.weight' has wrong shape; expected 1024, 8192, got 1024, 16384
```

Reading the per-layer keys explains it:

| key | value |
| --- | --- |
| `attention.head_count` | **array of 43**: `[32, 32, 64, 64, 64, 32, 64, 64, 64, 32, …]` |
| `attention.indexer.types` | array of 42 of 0/1 (two attention flavours) |
| `attention.sliding_window_pattern` | array of 43 of 0/1 (full vs sliding window) |
| `attention.head_count_kv` | 1 (single latent KV head) |
| `attention.key_length` / `value_length` | 576 / 512 |
| `attention.key_length_mla` / `value_length_mla` | 256 / 256 |
| `expert_latent_length` | 1536 |

So layer 1 has 32 heads (`q_b` = 1024 x 32x256 = 8192) and layer 2 has 64 (1024 x 64x256 = 16384). The tensor table has to
take `n_head` **per layer** and pick the MLA or the plain shape from `indexer.types`, exactly as the DeepSeek-V4 path does.
That is the next correction to `load_arch_tensors`.

### The attention tensor group now matches the pack exactly

With the per-layer head count (`hparams.n_head(i)` from the `attention.head_count` array) and the MLA head width for
every layer, the declared attention tensors agree with the file:

```
done_getting_tensors: wrong number of tensors; expected 1661, got 420
```

420 is 10 tensors over 42 layers (the optional sinks are absent on layer 0), so the shapes are right and the loader is
only saying that the rest of the table is still missing. Every MLA layer uses the 256-wide MLA heads: 32 heads give
`1024 x 8192` and 64 give `1024 x 16384`, which is exactly what the file holds, so no per-layer flavour switch is needed
for these tensors.

Next groups, in the order the file lists them: router (`ffn_gate_inp`, `exp_probs_b.bias`), the split expert stacks,
`ffn_routed_{up,down}`, the shared expert, the hyper-connection tensors, the indexer, the Engram group, then output and
the `nextn` head.

### The remaining layer tensors, with the shapes read from the file (layer 1)

| tensor | shape |
| --- | --- |
| `ffn_norm.weight` | (3072,) |
| `post_attention_norm.weight`, `post_ffw_norm.weight` | (3072,) each |
| `ffn_gate_inp.weight` | (3072, 512) |
| `exp_probs_b.bias` | (512,) |
| `ffn_{gate,up}_exps_{hot,warm,cold}.weight` | (1536, 1536, 160 / 96 / 256) |
| `ffn_down_exps_{hot,warm,cold}.weight` | (1536, 1536, 160 / 96 / 256) |
| `ffn_up_shexp.weight` | (3072, 1536) |
| `ffn_down_shexp.weight` | (1536, 3072) |
| `ffn_routed_up.weight` | (1536, 3072) |
| `ffn_routed_down.weight` | (3072, 1536) |
| `hc_attn_base.weight` / `hc_ffn_base.weight` | (8,) each |
| `hc_attn_fn.weight` / `hc_ffn_fn.weight` | (12288, 8) each |
| `hc_attn_scale.weight` / `hc_ffn_scale.weight` | (2,) each |
| `indexer.attn_k.weight` | (3072, 128) |
| `indexer.attn_q_b.weight` | (1024, 2048) |
| `indexer.k_norm.weight` | (128,) |
| `indexer.proj.weight` | (3072, 16) |
| `indexer_compressor_ape.weight` | (128, 4) |
| `indexer_compressor_gate.weight` | (128, 128) |
| `indexer_compressor_norm.weight` | (128,) |
| `indexer_compressor_value.weight` | (128, 128) |

Two of these need care before they go into the table: the hyper-connection widths are 4 lines of `n_embd` (12288 = 4 x
3072), which is the "four residual streams" of the architecture, and the tier counts are per layer (160/96/256 on the
trunk, 512 on the MTP layer).  The tensor-name enum also has to be checked for each name before the table is extended, so
that a wrong enum constant does not stand in for a missing one.

### `create_tensor` takes an `LLM_TN_IMPL`, not a name string

Extending the table with the remaining groups ran into the builder's shape: `create_tensor` is
`create_tensor(const LLM_TN_IMPL & tn, std::initializer_list<int64_t> ne, int flags)`, and `tn(...)` produces that
`LLM_TN_IMPL` from a tensor-name enum. So a tensor whose name has no enum entry — `post_attention_norm`, `post_ffw_norm`,
`exp_probs_b.bias`, `indexer_compressor_gate/value`, the three tiered `ffn_*_exps_{hot,warm,cold}` names — cannot be
declared by string as it stands.

Four of those enums do exist (`HC_*`, `INDEXER_*` except the compressor's gate/value, `FFN_ROUTED_*`, `FFN_EXP_PROBS_B`,
`ATTN_GATE`), so the missing ones are: `POST_ATTENTION_NORM`, `POST_FFW_NORM`, `INDEXER_COMPRESSOR_GATE`,
`INDEXER_COMPRESSOR_VALUE`, and the nine tiered expert names.

Next: add those enum entries (name table plus mapping) rather than working around the builder, then extend the table.
The incomplete extension was reverted; the build is green at this commit.

### The trunk tensor table is complete: 1576 of the file's 1661 tensors, all shapes matching

Details that the file settled while building the table:

- `create_tensor` takes an `LLM_TN_IMPL`, so the pack's names need enum entries; thirteen were added
  (`POST_ATTENTION_NORM`, `POST_FFW_NORM`, `INDEXER_COMPRESSOR_GATE/VALUE`, and the nine
  `FFN_{GATE,UP,DOWN}_EXPS_{HOT,WARM,COLD}`), each with a name row, a tensor-info row and the right op kind.
- The hyper-connection widths are **`2 * hc_mult`** for the base tensors and `4 * n_embd` by `2 * hc_mult` for the
  per-stream functions (the file's `(8,)` and `(12288, 8)`), not `4 * n_embd`.
- Layer 0 is the **dense lead layer**: a plain feed-forward of `feed_forward_length` (13824), no router and no expert
  stacks.
- Every **trunk** layer uses 160 / 96 / 256 experts per tier; the 512-expert layer is the MTP layer, which sits outside
  the trunk loop, so the tier counts do not vary inside it.

With those, the loader reports

```
done_getting_tensors: wrong number of tensors; expected 1661, got 1576
```

i.e. every declared shape agrees with the file and only the remaining groups are missing: the Engram group, `token_embd`,
`output`, `output_hc_*`, and the MTP layer including its `nextn.*` tensors.

### 1579 of 1661, and the shortfall counted

`token_embd`, `output_norm` and `output` are declared, which brings the table to 1579. `tn(...)` for a tensor whose name
has no layer index must be called **without** the layer argument — passing `0` makes the loader reject it with
"input/output layer tensor token_embd.weight used with a layer number".

The 82 tensors still missing are the four Engram groups (`engram_kv`, `engram_kv_b`, `engram_gate`, `engram_lookup_norm`,
`engram_pre_gamma`, `engram_hc_scale` on layers 1, 11, 21 and 31 = 24), the `output_hc_{base,fn,scale}` trio, and the
whole MTP layer (`n_layer` is 42, so layer 42's attention, router, 512-expert hot stack, shared expert, routed
projections, norms and `nextn.*` are all outside the loop). The exact list is what the next pass should diff against the
file rather than estimate.

### 1610 of 1661: the MTP layer is in the table

`block_count` is 43 while the trunk is 42 layers, so the loop runs over `n_layer_all` and the last block is the MTP
layer: same attention, norms and hyper-connections, a **512-expert hot stack only** (no warm or cold tiers), its own
router, shared expert, routed projections and indexer.

That brings the table to 1610 and leaves 51 tensors, now a precise list rather than an estimate:

| group | count |
| --- | ---: |
| Engram (`engram_kv`, `engram_kv_b`, `engram_gate`, `engram_lookup_norm`, `engram_pre_gamma`, `engram_hc_scale` on layers 1, 11, 21, 31) | 24 |
| `output_hc_{base,fn,scale}` | 3 |
| `nextn.{eh_proj,enorm,hnorm,shared_head_norm}` | 4 |
| the remaining per-layer names the file has on the MTP block (an extra indexer pair and similar) | 20 |

Four of the Engram names have no enum entry yet (`engram_kv_b`, `engram_lookup_norm`, `engram_pre_gamma`,
`engram_hc_scale`), so they need the same enum + name + info treatment as the thirteen added earlier before the group can
be declared.

### 1637 of 1661: the Engram group and the output hyper-connections are in

Seven more names needed enum, name and info rows (`ENGRAM_KV_B`, `ENGRAM_LOOKUP_NORM`, `ENGRAM_PRE_GAMMA`,
`ENGRAM_HC_SCALE`, `OUTPUT_HC_BASE/FN/SCALE`), and the file corrected two widths: `engram_gate` and `engram_hc_scale`
are `{n_embd, hc_mult}` wide (3072 x 4), and the output hyper-connections are `{hc_mult}`, `{4*n_embd, hc_mult}` and
`{hc_mult/4}` (4, 12288 x 4 and 1).

The Engram group is declared on the layers listed in `ple.layers`, which this pack gives as `[1, 11, 21, 31]`, so the
table follows the file instead of assuming every fourth layer.

What is left is 24 tensors: the four `nextn.*` projections on the MTP block and the per-layer names that only that block
carries.

### 1648 of 1661, with thirteen names still unaccounted for

The indexer and its compressor turn out to exist on **every** block including the dense lead layer, and the MTP block
carries four `nextn` projections (`eh_proj` is `2*n_embd` by `n_embd`). Declaring those brings the table to 1648.

The file has 1661, so thirteen names are still not created even though a name-by-name model of the table says otherwise.
Finding which ones needs the loader to report names rather than counts: the next step is a small probe that lists the
file's tensor names next to the ones the model created, instead of guessing again.

### The file's per-block tensor counts say where the indexer lives

Counting the file's 1661 tensors per block gives five distinct sizes, and they explain the structure:

| tensors per block | blocks | what it is |
| ---: | ---: | --- |
| 31 | 1 | the dense lead layer: attention, norms, hyper-connections, plain feed-forward, indexer |
| 36 | 29 | a plain MoE block **without** an indexer |
| 42 | 2 | a MoE block without an indexer that carries the Engram group (36 + 6) |
| 44 | 9 | a MoE block **with** the indexer and its compressor |
| 50 | 2 | Engram block with the indexer (44 + 6) |

plus six global tensors (`token_embd`, `output_norm`, `output`, `output_hc_{base,fn,scale}`).

So the indexer is present on **eleven** blocks, not all of them, which is why declaring it everywhere with
`TENSOR_NOT_REQUIRED` is right and why the created count is not simply the declared count. The thirteen remaining
tensors need the name-level comparison the next step will do; the per-block counts above narrow it to the optional group
rather than the trunk.

### The whole pack loads: 1661 of 1661 tensors

The thirteen were the **indexer query projection** (`indexer.attn_q_b`, `q_lora_rank` by `indexer_n_head * indexer_head_size`,
i.e. 1024 x 2048) on the thirteen blocks that carry an indexer — it was the one name the table never declared.

A diagnostic now makes this kind of question answerable instead of guessed: with `STRATA_HY4_TRACE=1`,
`done_getting_tensors` prints every file tensor the model never asked for, by name. That is how the thirteen were
found in one run.

With the indexer projection declared, the backbone file loads completely:

```
[strata-vk] model loaded in 2.3 s: hy_v4_flash, 42 layers, n_embd 3072, n_ctx_train 1048576, template yes
```

That is the first time the model class is good enough to load real weights. The next wall is one step later, in the
context:

```
Assertion failed: status == LLAMA_MEMORY_STATUS_SUCCESS, llama-kv-cache-dsv4.cpp:2309
```

i.e. the KV cache class the architecture is routed to does not fit this model's cache shape yet — the next piece is the
KV cache for `hy_v4_flash`, then the graph.

### Context creation: two findings

1. The DeepSeek-V4 KV cache refuses a small context (`n_ctx` 256 or 1024 assert in
   `llama-kv-cache-dsv4.cpp:2309`), while **2048 gets through cache creation** — the cache has a size floor (the
   compressor and window need room), so bring-up uses 2048 or more.
2. With the cache created, the graph asserts `hparams.swa_type != LLAMA_SWA_TYPE_NONE` ("DSV4 expects SWA raw cache"), so
   the architecture must declare its sliding-window arrangement. The pack gives `attention.sliding_window = 512` and a
   per-block `sliding_window_pattern` of 43 zeros and ones.

`hy4flash` now sets `swa_type = LLAMA_SWA_TYPE_STANDARD` and `set_swa_pattern(0)` the way `deepseek4` does, and marks the
MTP block as SWA. That gets past the graph assert but the cache still fails to prepare, so the next step is to honour the
**per-block** pattern (the pack alternates full and windowed blocks) instead of the uniform one, and to confirm `n_swa`
reaches the cache from `attention.sliding_window`.

### The KV failure is before the cache prepare, not in it

A trace inside `llama_kv_cache_dsv4::init_batch`'s context factory prints both prepare sizes. With
`STRATA_HY4_TRACE=1` the file prints **nothing** before the assertion, so `make_context` is never called: the failure is
in one of the two checks that run first —

- `hparams_raw.ple_n_heads > 0 && !dsv41_shared_history_ok(...)`, or
- the ubatch loop breaking because `balloc.get_n_used() < balloc.get_n_tokens()` or a split came back empty.

So the next step is not the cache sizing at all: it is to print which of those two returns early, and for the first one to
confirm whether `ple_n_heads` is non-zero for this pack (it would mean the Engram history check is running, and the pack's
single-sequence shape should satisfy it).

### The raw cache is fine; the compressed caches are what is missing

With the `init_batch` traces in place the assertion **moves**: it is now `llama-kv-cache-dsv4.cpp:2327`, the compressed
cache accessor, instead of `:2309`, the raw one. So the raw and SWA caches prepare correctly and what fails is
`ctx_csa` / `ctx_hca`, the compressed caches.

Those exist only when the model declares compression ratios. `deepseek4` reads `attention.compress_ratios` for them, and
this pack has no such key — its indexer keys are `attention.indexer.{types,kpool,kpool_select_tail,top_k,head_count,
key_length}`. So the next step is to derive the compression configuration from the pack's own indexer keys (or confirm it
uses none and the graph must take the uncompressed path), rather than leaving `dsv4_compress_ratios` empty by accident.

### The compressed caches exist only for layers that declare a ratio — and this pack declares none

Reading the DeepSeek-V4 cache shows the rule: a layer gets a compressed cache only when
`hparams.dsv4_compress_ratios[il] != 0`, and that array is filled from `attention.compress_ratios`. The pack has no such
key, so every entry is zero and **no layer is compressed** — which is consistent with its attention tensors being the
plain V2/V3 MLA set (`attn_kv_a_mqa`, `attn_k_b`, `attn_v_b`) rather than a compressed-KV arrangement.

That changes the shape of the remaining work: the inherited `deepseek41` graph is the **compressed / cross-layer-KV-shared**
graph, and it is the wrong graph for this model. `hy4_flash` needs its own attention built on the uncompressed MLA path
(what `deepseek2` does) plus the hyper-connections and Engram pieces from `deepseek41` — not a tweak of the V4.1 graph.

So the order from here is: the KV cache takes the uncompressed configuration, then the attention graph, then MoE with the
split expert tiers, then Engram, then the Hadamard fold list. None of those steps needs the compressed-cache machinery.

### Correction: this tree already implements the Hy4 family as `hy_v4`

While looking for where the KV cache class is chosen, `create_memory` turned up a case this tree already has:

```
case LLM_ARCH_HY_V4:   // "hy_v4", models/hy-v4.cpp, struct llama_model_hy_v4
```

and `hy-v4.cpp` carries the very tensors that were re-derived from the pack by hand:

| `hy-v4.cpp` | this pack |
| --- | --- |
| `wkv_a_mqa` = `{n_embd, kv_lora_rank + n_embd_rope}` | `attn_kv_a_mqa` (3072, 576) ✓ |
| `hc_attn_fn` = `{hc * n_embd, 2 * hc}` | (12288, 8) with hc = 4 ✓ |
| `hc_attn_base` = `{2 * hc}` | (8,) ✓ |
| `hc_attn_scale` = `{2}` | (2,) ✓ |

So the pack is a **`hy_v4` variant** (a "flash" one), and the correct base for `hy4flash` is `llama_model_hy_v4`, not
`deepseek41`: the MLA decomposition, the hyper-connections and presumably the split expert stacks and Engram already
exist there. The table built so far is not wasted — it is the independent check that the pack's shapes match `hy_v4`'s —
but the graph and cache should come from `hy_v4`, and `create_memory` should route `hy_v4_flash` next to `hy_v4`
(the DSA-iswa cache) rather than into the DeepSeek-V4 compressed path.

Next step: make `hy4flash` derive from `llama_model_hy_v4` and lower it to overrides for the differences the pack
actually has.

### Rebased on `hy_v4`: the loader now runs `hy_v4`'s own path

`hy4flash` derives from `llama_model_hy_v4` and delegates both loaders, is registered next to `hy_v4` in the model
factory, the rope type and the memory factory, and the whole thing builds. The differences then show up in order:

1. `blk.0.indexer.k_norm.bias` — the pack has the weight but no bias for this norm, so `hy_v4`'s creation of it is now
   `TENSOR_NOT_REQUIRED` (safe for the existing `hy_v4` models too, which simply keep theirs).
2. `blk.1.ffn_gate_exps.weight` — `hy_v4` expects **one** expert stack per layer, and this pack has the split
   `ffn_gate_exps_{hot,warm,cold}` stacks. That is the genuine difference to implement: the tensor table creates the
   three tiers and the MoE graph dispatches experts across them, with the global-to-local map from
   `hy4f.expert_split.*`.

So the rebase removed the cache and graph questions entirely and left one substantive difference (the split expert
stacks) plus whatever falls out after it.

### Split expert stacks work; the head count is the next per-layer difference

The expert stacks are now a **tier concept in `hy_v4`**, not a copy of its table: the base class exposes
`expert_tiers()`, `expert_tier_enums()`, `expert_tier_count()` and `expert_in_embd()`, whose defaults reproduce the
single stack every existing pack has, and `hy4flash` overrides them with the three hot / warm / cold stacks
(160 / 96 / 256, 512 on the MTP block) reading a 1536-wide latent. `hy4flash` also declares the `ffn_routed_{up,down}`
projections around them.

With that the expert tensors load and the next difference is a shape:

```
check_tensor_dims: tensor 'blk.2.attn_sinks.weight' has wrong shape; expected 32, got 64
```

`hy_v4` sizes the attention tensors from one head count, while this pack carries `attention.head_count` as a **per-block
array** (32 heads on the MLA blocks, 64 on the others). So the table needs `hparams.n_head(i)` inside the loop, which is
also what makes `wq_b`, `wk_b`, `wv_b` and `wo` come out right on the 64-head blocks.

### The table now covers the MTP block, and the differences are per-block flags

Three more hooks made the pack's table fit without special-casing `hy4flash` in the base:

- `tensor_layers()` — how many blocks the table covers (the trunk by default, trunk plus MTP for this pack);
- an expert tier whose `expert_tier_count()` returns 0 is skipped, which is how the MTP block gets a **hot stack only**;
- `is_indexer_full()` accepts the MTP block (it carries the same indexer arrangement as the last trunk block) instead of
  aborting on `il == n_layer`.

`hy4flash` also declares the groups `hy_v4` does not have: the four indexer compressor tensors, the two extra norms, the
Engram group on `ple.layers`, the output hyper-connections and the `ffn_routed_{up,down}` projections.

The next per-block difference: the MTP block has **no hyper-connections**, so `blk.42.hc_attn_fn.weight` is not found.
That needs a block-level flag (the same shape as `tensor_layers()`), not an optional tensor for every `hy_v4` pack.

### The pack loads through `hy_v4` with the variant's extras: 1661 of 1661

With the per-block hyper-connection flag and the MTP block's `nextn` projections declared, and with the duplicate output
hyper-connections removed (`hy_v4`'s `HC_HEAD` tensors already carry the `output_hc_*` names), the backbone loads
completely through the family's own loader:

```
[strata-vk] model loaded in 45.1 s: hy_v4_flash, 42 layers, n_embd 3072, n_ctx_train 1048576, template yes
```

That is the second milestone: the first was the hand-built table reaching 1661 with `deepseek41` as a stand-in, this one
is the **real** model path — `hy_v4`'s hparams, its tensor table with the variant's hooks, its KV cache class and its
graph — agreeing with the file.

The next wall is inside graph construction:

```
ggml.c:3619: GGML_ASSERT(as->ne[0] == b->ne[0]) failed
```

a shape mismatch between `hy_v4`'s graph and this pack's geometry. The likely candidates are the ones the table already
knows differ: the expert stacks read the 1536-wide latent instead of `n_embd`, and the head count varies per block, so
the MoE and attention builders need the same per-block/variant inputs the table got.

### The latent MoE projection is wired, and the router is the next input mismatch

`hy_v4`'s MoE now projects into the expert width and back out **only when the pack has the tensors**: it looks up
`blk.N.ffn_routed_{up,down}` by name and, when they exist, runs `build_lora_mm` in and out around `build_moe_ffn`. A pack
without them runs exactly the old path, so existing `hy_v4` models are unaffected.

That fixes the `mul_mat_id` assertion (`as->ne[0] == b->ne[0]`, the expert stacks read the latent width) and the next one
is a plain `ggml_can_mul_mat`:

```
ggml.c:3483: GGML_ASSERT(ggml_can_mul_mat(a, b)) failed
```

which is the **router**: `ffn_gate_inp` is `{n_embd, n_expert}` (3072 x 512), so it must see `cur`, while the expert
stacks must see the projected latent. `build_moe_ffn` takes one input for both, so this variant needs the routing done on
`cur` and the experts on the projected tensor — a small change to the builder's interface rather than another hook.

### The MoE builder takes a separate expert input

`build_moe_ffn` gained `cur_experts_in` (default empty) on both overloads: the router still reads `cur`, and when the
parameter is set the **expert stacks** read it instead. Every existing call site passes nothing, so it behaves exactly as
before; `hy_v4` passes the projected latent when the pack has `ffn_routed_up`.

That removes the router/stack width conflict. The `ggml_can_mul_mat` assertion at `ggml.c:3483` that remains is therefore
a **different** matmul, not the MoE: the candidates are the Engram path (`engram_kv` is 1536 by 15360 while its input is
n_embd wide) and the attention pieces whose per-block head count the table already honours. The next step is to identify
that matmul by its operands rather than by elimination.

### The shape trace names the operands, and the projection direction was inverted

A `STRATA_GGML_SHAPE_TRACE=1` diagnostic now logs both operands of a `ggml_mul_mat` before the assertion fires, which
answered the question in one run instead of by elimination:

```
ggml_mul_mat: cannot multiply 'blk.1.ffn_routed_up.weight' (1536,3072,1,1) by ' (reshaped)' (3072,1,1,1)
```

That also showed the projection was wired the wrong way round: the names describe the **residual direction**, so
`ffn_routed_down` (3072 x 1536) maps n_embd into the latent and `ffn_routed_up` (1536 x 3072) maps the latent back out.
The graph now projects with `down` on the way in and `up` on the way out.

The remaining assertion is the expert multiply again (`mul_mat_id`, `as->ne[0] == b->ne[0]`), so one more expert-path
matmul still reads the wrong input — the trace names it on the next run.

### The routed tensors are found, so the failing expert call is another path

A trace in the MoE builder shows both projections resolved:

```
graph: il 1 routed_in 00000080022D47C0 routed_out 00000080022D4650
```

and `cur_x` is set from `cur_experts_in` at the top of the builder, yet the assertion still reports the hot stack
multiplied by a 3072-wide `b`. So the call that fails is **not** the one the builder now feeds with the projected latent:
the remaining candidates are the expert-cache lookup path (`moe_cache->build_lookup`, which is handed `cur`) and any
other place that multiplies an expert stack directly.

Next: trace the `moe_cache` lookup the same way, since a pack whose experts sit on the device may still take it.

### The latent MoE is now in effect, and the next wall is a view bounds check

Two real bugs were behind the stubborn expert multiply:

1. the **compact** `build_moe_ffn` overload accepted `cur_experts_in` but did not forward it, so the full builder always
   saw an empty parameter;
2. the expert-cache branch passed `cur` instead of the projected input, and its lambda parameter was named `cur`, which
   shadowed it.

With both fixed the builder reports what it should:

```
build_moe_ffn: il 1 cur 3072 cur_x 1536 (experts in 000000802FE766F0)
```

i.e. the router sees 3072 and the expert stacks see the 1536-wide latent. The next wall is a different class of problem:

```
ggml.c:1966: GGML_ASSERT(view_src == NULL || view_size == 0 || view_size + view_offs <= ggml_nbytes(view_src))
```

a view that runs past the end of its source tensor, which is a **shape** mismatch elsewhere in the graph (a tensor built
with the wrong width, most likely one of the attention or Engram pieces that still assume the base `hy_v4` geometry).

### The builder hard-codes `n_embd` in three places the latent variant needs

The MoE builder assumed `n_embd` wherever it touched the expert input:

1. `cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens)` — now reshapes the **projected** input at its own width;
2. the `ggml_repeat_4d(...)` that spreads the weighted input over the selected experts — same width;
3. and a per-expert view of the weighted tensor still slices `n_embd` from a 1536-wide result, which is the assertion
   that remains:

```
view '' (3072,1,1,1) of 12288 bytes at 92160 overruns 'ffn_moe_weighted-1' (1536,16,1,1) of 98304 bytes
```

Each of those is the same mistake in a different place: the width of the expert input is a property of the model, not a
constant. The remaining slice is the next one to take from `cur_x->ne[0]`.

### The MoE builds now; the attention wants the per-block head count

With the last `n_embd` slice taken from the tensor's own width (`experts->ne[0]`), the MoE graph builds and the wall moves
to the attention:

```
ggml_mul_mat: cannot multiply 'blk.2.attn_k_b.weight' (192,512,64,1) by ' (view) (permuted)' (192,1,32,1)
```

`attn_k_b` on block 2 has **64** heads, and the graph builds it with 32 — the same per-block head count the tensor table
already honours via `hparams.n_head(i)`, but the graph still reads the single default. So the attention builder needs the
per-block head count too, in the k/v up-projection and in whatever reshape folds the heads back.

This is the last of the "constant where the model has a per-block value" family that has driven the last few rounds: width
of the expert input, hyper-connections per block, head count per block.

### The graph builds; the remaining wall is the Hadamard fold list, one weight at a time

With the per-block head count in both MLA builders, the graph builds through attention and MoE and stops at the
Hadamard policy check rather than a shape error:

```
failed to initialize the context: Hadamard-folded weight 'blk.0.attn_q_a.weight' is consumed without its activation
transform; this graph's matmul path does not support it
```

That is the check working as intended: the pack folds an activation transform into 849 weights, and each of them may only
be multiplied through the Hadamard-aware builder (`build_lora_mm` / `build_lora_mm_id`). `hy_v4` used raw
`ggml_mul_mat` for the MLA projections, so nine of those calls now go through `build_lora_mm`; the check then moves to the
next folded name (`attn_q_b`), which is exactly the one-at-a-time list the tensor table already produced.

What remains is mechanical but must be done name by name, in the order the check reports them, and each step is only
correct when the graph path really does apply the transform.

### The graph builds and the first token computes; the CPU row lookup is the new wall

After routing the folded weights through `build_lora_mm` (`wq_b`, `output`, and the `v_mla` multiplies inside
`build_attn`), and after making `cur_x` follow the reshaped MoE input, the graph builds and evaluation starts. It stops in
the **CPU** backend:

```
ggml-cpu/ops.cpp:5820: GGML_ABORT("fatal error")   // ggml_compute_forward_get_rows, default case
```

`get_rows` on CPU has cases for f32/f16/bf16/i32 and the quantised types it knows, and no case for the Hy4 block types
(CSQ8 for `token_embd`). So the embedding lookup needs a generic path: dequantise the selected rows with the type's own
`to_float` when there is no specialised kernel. That is a small, self-contained addition and it is the next step.

The picture is worth recording: shapes, per-block values, latent widths and the fold list are all resolved; what remains is
support for the new block types in the operations the graph actually calls, starting with `get_rows`.

### `get_rows` needs a case for the new block types (and where it has to go)

The dispatcher in `ggml-cpu/ops.cpp` ends with

```
        default:
            {
                GGML_ABORT("fatal error");
            }
```

which is what fires for CSQ8 `token_embd`. The generic path has to sit in **that** switch, inside
`ggml_compute_forward_get_rows`, where `ne01`, `nb01`, `ne10`, `ne11`, `nb10..nb12`, `nb1..nb3` and the thread split are in
scope — dequantise the selected row with the type's own `to_float` and copy it into `dst`.

An attempt this round patched the first `default: GGML_ABORT` in the file, which is a different function, and did not
compile; it was reverted and both copies are back to their previous content. The next attempt anchors on the
`ggml_compute_forward_get_rows` body instead.

### The `get_rows` case needs a precise anchor, and a broad replace is not one

Two attempts this round failed the same way: `default: GGML_ABORT("fatal error")` appears in several functions of
`ggml-cpu/ops.cpp`, and matching it by text landed the new case in `ggml_compute_forward_get_rows_back`'s neighbour
instead of the `get_rows` dispatcher. A follow-up attempt that replaced stride expressions **by text** also hit unrelated
functions (`dup_from_q`, `get_rows_q`), which is worse than not trying: it quietly changes working code.

Both were reverted, the file is byte-identical to what it was, the build is green and the encoding check is clean.

What the next attempt must do, and this is the important part:

- anchor by **line**: the dispatcher's `default:` is at the end of `ggml_compute_forward_get_rows`, immediately before the
  closing braces of the switch and the function, and the block to replace is exactly

  ```
          default:
              {
                  GGML_ABORT("fatal error");
              }
      }
  }
  ```

  with `} break;` as the replacement's tail;
- keep the new code **self-contained** — inside that switch the only locals are `src0`, `params` and `dst`, so read
  `dst->src[1]` and the strides from the tensors themselves rather than from the helper locals the row kernels have;
- verify with a diff of the whole file, not just a successful build, because a broad replace can compile and still be
  wrong.

### The generic `get_rows` case is in, at the right place this time

Anchored on the **definition** (`void ggml_compute_forward_get_rows(...) {`) and then on the four-line
`default: { GGML_ABORT("fatal error"); }` immediately after it, the new case lands at line 5817 — the dispatcher's own
default, which is where the row strides and the thread split live. The whole-file diff shows a **single hunk**, which is
the check that the earlier attempts failed.

The case is self-contained: it takes `src1` from `dst->src[1]`, reads the strides from `src0`/`src1`/`dst` themselves,
dequantises the selected row with the type's own `to_float`, and copies it into `dst`.

The `GGML_ABORT("fatal error")` on `token_embd` is gone. Evaluation now stops with an access violation instead, i.e. a
step further in, and the next round starts from that.

### The generic row lookup works; the crash is later in the graph

The trace confirms the new case does its job on the real embedding table:

```
ggml_compute_forward_get_rows: generic rows type csq8 ne00 3072 ne01 120832 nr 1
```

3072 values per row, 120832 rows, one row requested — exactly the vocabulary table. So the `token_embd` lookup no longer
aborts and no longer mis-indexes; the access violation is **after** it, in a later node of the same graph.

An earlier detail worth keeping: the first version of the case used `dst->ne[…]` for the row arithmetic; the row kernels
take those from **src1** (the ids tensor), and using the destination's shape produced garbage indices. That is fixed.

Next: instrument the op sequence (the CPU-only run reaches "context ready" and then builds the graph) to find which node
overruns.

### The op trace names the crash: a `MUL_MAT` on a weight whose type the CPU has no dot product for

With `STRATA_GGML_OP_TRACE=1` the CPU dispatcher prints each node it runs. The last lines before the access violation are

```
cpu op MUL   'node_43' [3072,1,1,1]
cpu op MUL_MAT 'node_45' [512,6,1,1]      <- router, 512 experts, 6 tokens
cpu op MUL_MAT 'node_47' [1024,1,1,1]     <- q_lora projection
```

so attention, rope, `SET_ROWS` and the router all run, and the crash is in the 1024-wide matmul — a weight in one of the new
block types. The CPU `mul_mat` dispatches on the weight type through `vec_dot`; the Hy4 types have none, so the call goes
through a null pointer rather than producing a wrong number.

That is the next real gap, and the same shape as the `get_rows` one: a **generic fallback** that dequantises the weight
block with the type's `to_float` and multiplies in f32, so every type that has a converter can be used even without a
specialised kernel.

### The generic matmul fallback works, and the next assertion is the split-expert routing itself

`ggml_compute_forward_mul_mat` now has a fallback for a type that has a converter but no dot product: it dequantises the
weight row and multiplies in f32 (correctness first; not a fast path). With it the graph runs past the projection that
crashed and reaches the expert gather:

```
Assertion failed: i02 == -1 || (i02 >= 0 && i02 < n_as), ggml-cpu.c:1781
```

`n_as` is the expert count of the stack it was handed, and the ids come from a router over **512** experts while
`layer.ffn_*_exps` holds the **hot** stack of 160. So the real remaining work is the one the pack's design implies:

- the router scores 512 global experts;
- those are split into hot (160) / warm (96) / cold (256) stacks with the `hy4f.expert_split.*` map;
- a selected expert has to be sent to the stack that owns it, which means either gathering per tier and summing, or
  building a per-layer view over the three stacks and letting one gather see all 512.

Anything else — widening the hot stack, or clamping ids — would compute the wrong answer, so the routing is where the next
effort goes.

### The tier maps are per block and explicit, and their id arrays are narrower than int32

The split metadata is exactly what the routing needs, and it is explicit:

```
hy4f.expert_split.layers        42
hy4f.expert_split.hot_offsets   [43]  0, 160, 320, ...        (per block)
hy4f.expert_split.hot_ids       [7072] 488, 279, 308, ...
warm_offsets                    [43]  0, 96, 192, ...
warm_ids                        [3936]
cold_offsets                    [43]  0, 256, 512, ...
cold_ids                        [10496]
```

So `"<tier>_ids[<tier>_offsets[il] + j]"` is the **global** expert id of local expert `j`, and 41 blocks x 160 plus the
MTP block's 512 give exactly the 7072 hot ids. That is the map a tier dispatch needs.

`llama_model_loader::get_arr` refuses these arrays — they are stored in a **narrower integer type** than its supported
int32/uint32/uint64 group — so `hy4flash` reads them in a `try` and logs when it cannot, leaving the map empty rather than
failing the load. Reading them through `gguf_get_arr_data` with their real type is the next small step, and then the graph
can perform the three-way dispatch:

- per tier, select the ids that belong to it and rewrite them to local indices (via a 512-entry table per tier);
- zero the weight of the experts that belong to another tier;
- run `mul_mat_id` against that tier's stack and add the three results.

That is the correct routing: no widening of a stack and no clamping of ids, both of which would produce wrong numbers.

### Reading the tier id arrays with their own type

`hy4flash` now reads `hy4f.expert_split.*` through the gguf metadata rather than `get_arr`: a small helper switches on the
array's real element type (`UINT8/INT8` through `UINT64/INT64`) and widens into `int32_t`, which is what the loader's
typed helpers refused. The build is green and the model still loads.

What is not yet confirmed is the **coverage check** — that every global expert of a block appears in exactly one tier — because
the log line that reports it did not appear in the run output (the run still reaches the expert-gather assertion first).
The next step is to surface that count (a visible log level, and running before the graph is built) so the map is verified
before any dispatch code depends on it.

### The tier map is verified: every expert of every block has exactly one tier

With the coverage log at a visible level:

```
load_arch_hparams: expert tiers read for 42 block(s), 0 expert(s) without a tier
```

Forty-two blocks read, and **not one** global expert left unmapped. Since the map stores one slot per global id, that also
means no block has two tiers claiming the same expert — a duplicate would have been overwritten silently, but the counts
(hot 160 + warm 96 + cold 256 = 512 per block, and 41 x 160 + 512 = the 7072 hot ids the file carries) only add up if the
three tiers partition each block.

So the routing data is correct and the next piece is the dispatch itself. The plan, with the tensors the graph can build
from host data (no new file tensors, so the load count does not change):

- per block and tier, a `[512]` **local index** table (global id -> index within that tier) and a `[512]` **mask**;
- for each tier: `ids_t = get_rows(local_table_t, topk_ids)`, `w_t = w * get_rows(mask_t, topk_ids)`;
- `out_t = build_lora_mm_id(stack_t, x, ids_t)`, scaled by `w_t`;
- the block's expert output is the sum of the three.

That keeps the ids inside each stack's range, which is what the current assertion (`i02 < n_as`) demands, and it computes
exactly what the pack's split means.

### The dispatch tables are built and checked

`llama_model_hy4flash` now produces, per block and tier, the two flat tables the graph will feed:

- `tier_local_table(il, tier)` — for each global expert id, its index inside that tier (0 when it is elsewhere);
- `tier_mask_table(il, tier)` — 1.0 for the experts in that tier, 0.0 otherwise.

A load-time check walks all of them:

```
load_arch_hparams: expert tiers read for 42 block(s), 0 expert(s) without a tier, 0 empty tier(s)
```

Forty-two blocks, nothing unmapped, and **no tier empty** — which is the property the dispatch relies on, since an empty
tier would mean the stack is never read while the mask still routes weights to it.

With the tables in place the remaining step is the graph itself: three `build_lora_mm_id` calls per block with the local ids
and the masked weights, summed. That is a change to the MoE builder rather than to the model, and it is what the next round
does.

### The tier dispatch, as the builder needs it

Reading the existing MoE builder gives the exact sequence the split path has to reproduce, once per tier, sharing everything
that does not depend on the stack:

1. `logits = build_lora_mm(gate_inp, cur)` — the router reads `cur` (n_embd), as it should;
2. gating per `hparams.expert_gating_func`; `selection_probs = probs + exp_probs_b` when the bias is there;
3. one top-k over `selection_probs` -> `ids` `[n_expert_used, n_tokens]`, and `weights = get_rows(probs, ids)`;
4. per tier: `ids_t = get_rows(local_table_t, ids)`, `mask_t = get_rows(mask_table_t, ids)`,
   `w_t = weights * mask_t`;
5. per tier: `up = build_lora_mm_id(up_t, cur_x, ids_t)`, `gate = build_lora_mm_id(gate_t, cur_x, ids_t)`,
   `act = swiglu_split(gate, up)` (with the per-block clamp when `swiglu_clamp_exp` is set — the existing builder shows both
   branches), `down = build_lora_mm_id(down_t, act, ids_t)`, `out += down * w_t`;
6. the block's MoE output is the sum over the tiers, and the shared expert is added exactly as it is now.

Two details from the existing code matter for correctness: the expert multiplies read `cur_x` (the projected latent) while
the router reads `cur`, and the weights are applied **after** the down projection, not before.

This is a self-contained addition to `llama-graph.cpp` — one function that takes the three stacks and the two tables per
tier — and it is what the next round implements. The model side is already done and checked.

### The dispatch needs one input class for the constant tables

The two tier tables are constant for the model but they still have to reach the graph as tensors, and a graph tensor can
only be filled through an `llm_graph_input_i` (the tensors are allocated after the graph is built, so a plain memcpy at
build time would write into nothing).

The shape of that class, following `llm_graph_input_embd`'s pattern:

- it holds, per tier, one I32 tensor for the local index table and one F32 tensor for the mask, plus the host copies of the
  same tables;
- `set_input` copies each host row into its tensor with `ggml_backend_tensor_set`, the same call the embedding input uses;
- the graph registers it, and the builder uses the tensors for the two `get_rows` lookups per tier.

registered with `res->add_input(...)` in the graph that builds the MoE, with the tensors created by `ggml_new_tensor_1d` so
the builder can use them for the `get_rows` lookups.

With that in place the tier builder itself is the sequence already written down: router on `cur`, one top-k, then per tier
`get_rows` for the local ids and the mask, `up`/`gate`/`down` against that tier's stack with `cur_x`, and the masked weights
applied after `down`.

### Wiring the dispatch: the compact overload is the one the call site uses

The attempt to pass the tier parameters from `hy_v4`'s MoE call compiled down to one error worth recording: the call site
resolves to the **compact** `build_moe_ffn` overload (the one with 19 parameters), while the tier parameters had only been
added to the full one. Extending the compact overload then collided with the full declaration, because the two share the
same trailing text in the header and a text match cannot tell them apart.

The attempt was reverted; the tree is byte-identical to the last green commit and builds.

Next time, the edit has to touch **both** overloads in a way that distinguishes them:

- in the header, the compact declaration ends the parameter list after `cur_experts_in`, while the full one continues with
  the tier pointers — so the compact declaration gets the tier pointers as its own trailing parameters and the full one
  keeps the ones it already has;
- the compact definition forwards all five to the full definition;
- only then can the call site pass them.

An alternative that avoids the duplication: pass the tiers through a small struct by pointer, one optional parameter on
both overloads, which removes the long parameter lists entirely.

### It runs end to end; the numbers are still wrong

The model now completes a forward pass (21903 nodes, logits over the 120832-token vocabulary, exit 0), and the engine's
own admission check did **not** report an empty logits row — so the row is present and the sampler simply picks token 0 from
it. The output `0,0,0` therefore means "wrong numbers", not "no numbers".

What the graph still does not use, and what the next pass has to wire in, in the order the model defines it:

| loaded but unused | why it matters |
| --- | --- |
| the **Engram group** on the four `ple.layers` blocks | those blocks feed a table lookup into the residual stream; without it their output is simply a different function |
| `post_attention_norm`, `post_ffw_norm` | the pack carries a second norm after attention and after the feed-forward |
| `engram_pre_gamma`, `engram_hc_scale`, `engram_lookup_norm` | the scale and norm of the Engram path |

`hy_v4`'s graph predates all three (a plain `hy_v4` pack has none of them), so this is the same "the model is a variant"
work as before, and it is now visible as a numerical error rather than a crash — which is the easier kind to chase: each
missing piece can be added and checked against a layer's output instead of against a segfault.

### What the Engram path needs, and why validation is by output

The pack ships two files and only the first is opened today:

| file | contents |
| --- | --- |
| `…-00001-of-00002.gguf` | the whole backbone, 1661 tensors, loads completely |
| `…-00002-of-00002.gguf` | the Engram tiered tables and the 660M-row index the lookup needs |

The backbone's four Engram blocks reference that lookup, so the Engram path cannot be finished without opening the second
file — which the loader currently never does. That is the next structural piece: open it as a companion (not as a split,
which the loader already knows to avoid for this architecture), read `engram_index` plus the four tier tables, and let the
block look a row up by its n-gram hash.

There is also no reference implementation in the pack — only the dequantisers and assemblers — so there are no reference
logits to compare against. Validation therefore has to be behavioural: the numbers are right when the model produces
coherent text, and the tool for finding a divergence is a layer-by-layer print of hidden-state norms rather than a
comparison with a stored reference.

That makes the order of work clear:

1. open the second file and load the Engram tables and index;
2. implement the lookup in the four blocks (hash, gather, lookup norm, pre-gamma, gate, hyper-connection scale);
3. add `post_ffw_norm`;
4. print hidden-state norms per block on a fixed prompt and look for the first block whose norm departs from a smooth
   progression — that is where a wiring error shows up.

### The companion file, read

Opening `…-00002-of-00002.gguf` gives the Engram layout as plain key-values and five tensors:

| key | value |
| --- | --- |
| `hy_v4_flash.ple.layers` | `[1, 11, 21, 31]` |
| `hy_v4_flash.ple.ngram_size` / `heads_per_ngram` | 3 / 8 |
| `hy_v4_flash.ple.head_offsets` | 9 entries, the row range of each of the 8 heads |
| `hy_v4_flash.ple.head_vocab_sizes` | 8 entries, about 10.3M rows each |
| `hy4f.engram.hot/warm/high/low_count` | 13.2M / 15.1M / 123.7M / 508.0M rows |
| `hy4f.engram.{hot,warm,high,low}_tensor` | `engram_hot.weight`, …, one tier table each |
| `hy4f.engram.index_tensor` | `engram_index`, the row index |
| `hy4f.engram.scale_lo/hi` | -4.605… / 6.105…, the log-scale range the block types use |
| `hy4f.engram.index_tier_bits` / `index_local_bits` | 2 / 30 |

So a lookup is: hash the 3-gram for each of the eight heads inside that head's own row range, read the global index,
split it into `tier` (top two bits) and `local row` (low thirty), and gather the 96-value row from that tier's table with
the scale range above.

The one piece that is **not** in the pack is the hash itself — `ngram_size` and the head sizes describe its shape, but the
exact mixing is in the model's reference implementation, not in these files. That is what to confirm before writing the
lookup, because a wrong hash silently produces plausible but wrong rows.

### The Engram hash is already in the tree

`src/models/deepseek41.cpp` implements it, and it reads exactly the keys this pack carries:

```
n_gram = hparams.ple_ngram_size                 3
n_head = hparams.ple_heads_per_ngram            8
mult   = hparams.ple_layer_multipliers[e*n_gram + s]
primes = hparams.ple_head_vocab_sizes[e*n_cols + col]
offs   = hparams.ple_head_offsets[e*n_cols + col]

rolling = toks[0]*mult[0];  for o in 1..n_gram: rolling ^= toks[o]*mult[o]
out[col] = rolling % primes[col] + offs[col]
```

So the hash is not something to reverse-engineer: the tree already has it, and the pack's keys are the same ones it reads.
What is missing is narrower than it looked:

1. `ple.head_offsets` and `ple.head_vocab_sizes` live in the **companion file**, so it has to be opened at load and those two
   arrays read from it (`ple.layers`, `ple.ngram_size`, `ple.heads_per_ngram` and `ple.layer_multipliers` are in the
   backbone already);
2. the four Engram blocks need the lookup wired in — hash, read the row index, gather from the tier table, apply the
   lookup norm and the pre-gamma, then gate into the residual streams;
3. the tier tables themselves are in the companion file too, so the same open serves both.

That makes the next step concrete and bounded: open the companion, read five arrays and five tensors, and reuse
`engram_hash`'s arithmetic for the row indices.

### The companion path has to be plumbed before the lookup can open it

Reading the companion needs its path, and neither place that would know it can hand it over today:

- `llama_model_loader` keeps the open files as `llama_files`, and `llama_file` exposes size, seek, read and `file_id`, but
  **not** the path it was opened with;
- the engine knows the path it passed in, but there is no field on `llama_model_params` for a companion and the model
  cannot reach the engine.

So the next step is plumbing rather than lookup code, and there are two clean shapes for it:

1. give `llama_model_params` an optional companion path — explicit, and the engine already tokenises the file name, so it
   can derive the sibling (`…-00002-of-00002.gguf`) from the backbone path it was given;
2. give `llama_file` a `path()` accessor and let the loader derive the sibling itself — fewer moving parts at the API, and
   the derivation lives next to the split logic that already parses these names.

The second is the smaller change and keeps the knowledge in one place, so it is the one to try first.

Once the path is available, the lookup is the shape already written down: hash the three tokens per head with the
`engram_hash` arithmetic, read `engram_index`, split the tier and the local row, gather the 96-value row from that tier's
table with `engram.scale_lo/hi`, then the lookup norm, the pre-gamma, the gate and the hyper-connection scale.

### The PLE arrays say the index space is 64 columns over about 660M rows

Reading the two arrays from the companion and cross-checking them against the same keys in Python gives a consistent
picture:

```
head_offsets      64 entries, 0, 10312007, 20624028, … , last 649693215   (monotonic)
head_vocab_sizes  64 entries, 10312007, 10312021, 10312039, …
```

Sixty-four columns is exactly `4 Engram layers x 16`, and 16 is `2 x heads_per_ngram` — which is what `engram_hash` builds
with `col = (o-1)*n_head + h` for `o` in `1..n_gram-1` and `h` in `0..7`. So the column index in the hash is the same
`n_cols = 16` per layer that the loader's other Engram reader uses, and the last offset plus the last head's size lands on
about 660M, matching `engram_index`'s length.

That fixes the arithmetic the lookup needs: for a token at position `i`, hash each of the 16 columns, add its offset, and
the result is a row of `engram_index`; the top two bits of that entry choose the tier and the low thirty give the row
inside that tier's table.

### The companion's tensors, with types and offsets

Reading the companion's tensor table gives exactly the five tensors the layout keys named, and their types confirm the
pack's `FP8-hot+Q6-warm+Q4-high+Q2-low` scheme in terms of the ids this tree already implements:

| tensor | dims | file type | this tree | bytes |
| --- | --- | --- | --- | --- |
| `engram_index` | 660,006,416 | 4 (uint32 in the writing format) | `I32` | 2.6 GB |
| `engram_hot.weight` | 96 x 13,200,128 | 43 | `F8_E4M3` | 1.3 GB |
| `engram_warm.weight` | 96 x 15,116,216 | 149 | `ENGRAM_Q6B12` | 1.2 GB |
| `engram_high.weight` | 96 x 123,653,086 | 148 | `ENGRAM_Q4B12` | 6.9 GB |
| `engram_low.weight` | 96 x 508,036,986 | 150 | `ENGRAM_Q2B12` | 16.3 GB |

That is 28.3 GB of table plus the index, which is why the block's design is a **row lookup** rather than a load: only the
rows a token needs are read. It also means the three Engram block types verified earlier against the reference are exactly
the ones this path uses, and the hot table is plain FP8 with no scale to apply.

The lookup is therefore: hash to a column, add its offset, read one 32-bit entry of `engram_index`, split tier from row,
then read 96 values of that tier's table at that row and dequantise them (FP8 directly, the other three with
`engram.scale_lo/hi`).

### What moving the Engram lookup into the model needs

The probe proves the whole chain, so moving it into the model is a transcription plus the wiring the graph needs. The
pieces that have to exist before the code compiles:

- the four PLE values are **not** in `hparams` today (`ple_ngram_size`, `ple_heads_per_ngram`, `ple_layer_multipliers`,
  and the pad id); `hy_v4` never reads them, so `hy4flash` has to read them from the backbone (multipliers and layer list)
  and from the companion (the two head arrays);
- the companion has to stay open, with the five tensors' data offsets and per-tier row sizes resolved once at load;
- a 64-bit seek helper, since every one of those offsets passes 2^31 — the probe's bug is the same one to avoid here;
- a graph input to upload the gathered rows, exactly like the expert-tier tables use.

An attempt this round started on that transcription without the helpers in place and did not compile, so it was reverted;
the tree builds at the last green commit. The next pass starts from the list above rather than from the function body.

### Two details the Engram move has to get right, learned by trying

Building the resolver surfaced two things that only show up when the code is actually compiled:

1. **`get_arr` has no `int64_t` instantiation.** `ple.layer_multipliers` holds values like 81853575799317, which do not fit
   an `int32_t`, so it has to be read through the gguf metadata (`gguf_get_arr_data` with the array's own type) exactly as
   the tier id arrays are — not through the loader's typed helper.
2. **the resolver runs from a `const` method**, so everything it fills has to be `mutable`: the PLE parameters, the head
   arrays and the resolved table offsets.

Neither is hard, but both are the kind of thing that costs a compile cycle each when discovered one at a time. The attempt
was reverted; the tree builds at the last green commit, and the next pass starts with these two points and the earlier
list.

### The pack's own guidance names the path: the existing Engram gate/KV

The README's runtime checklist settles two open questions:

> **Engram Row Lookup**: read `engram_index[row]`, select the Tier Table, decode 2/4/6-Bit or read FP8, then apply the
> 96-Dimension Lookup Norm and the **existing Engram Gate/KV Path**.

and separately, the architecture list says the Engram embeddings are "8 Hash Heads per Bigram/Trigram Order, 96 Dimensions
per Head" at layers 1/11/21/31 — which matches the sixteen columns per layer the hash produces (2 orders x 8 heads).

So the block's Engram path is the one this tree already implements for DeepSeek-V4.1 in `deepseek41.cpp`
(`build_engram`), fed by this pack's row lookup instead of a recomputed table. That also explains the shapes:

- `engram_kv` is 1536 by **15360**, and 15360 = **5 x 3072**, which lines up with `engram_kv_b` being 3072 by **5**:
  the five-slot bias belongs to the same projection;
- `engram_lookup_norm` is 96 by 16 — one norm per head per column, applied to the looked-up row before the projection;
- `engram_gate` and `engram_hc_scale` are 3072 by 4, the hyper-connection width of the four residual streams.

The next step is therefore not to invent a path but to read `deepseek41`'s `build_engram` and feed it this pack's rows: the
lookup norm over the 96-dimension row, the gate, and the write into the four streams.

### The Engram block, as the existing builder writes it

`deepseek41::graph::build_engram` is the path the README points at, and it explains every shape at once:

```
kv    = build_lora_mm(engram_kv, rows)          // rows are [96*16, nt] -> [15360, nt]
key   = view(kv, n_embd, hc, nt)                // [3072, 4, nt]   the four residual streams
value = view(kv, n_embd, 1, nt, hc*n_embd)      // [3072, 1, nt]   the fifth slot
dot   = sum_rows(mul(rms_norm(x), rms_norm(key))) / sqrt(n_embd)
gate  = sigmoid(sqrt(max(|dot|, 1e-6)) * sign(dot))
x     = x + repeat(value) * gate
```

So `15360 = (hc + 1) x n_embd = 5 x 3072`, which is exactly the observation made last round, and it is why
`engram_kv_b` is 3072 by 5: the projection produces one value slot plus one slot per residual stream.

Two details matter for the port:

- `x` here is the four-stream residual **[n_embd, hc, nt]**, so the block's Engram write happens on the streams, not on a
  collapsed vector;
- the gate is a *signed square root of a dot product*, not a plain sigmoid of a projection — it is a similarity between the
  current stream and the looked-up key.

Porting it into `hy_v4`'s block loop is therefore a transcription that fetches `engram_kv` and `engram_gate` by name and
uses the rows tensor the new input already uploads. That is the next edit; nothing about it is undecided any more.

### Where the batch path breaks: the second decode step, not the prefill

Three runs with the same prompt narrow it down:

| run | result |
| --- | --- |
| `--prompt "The capital of France is" --n 1` | exit 0 |
| same prompt, `--n 2`, `--n 4`, `--n 8` | the RMS_NORM assertion (`scale > 0.0f`) |
| `--tokens 9707 --n 3` | exit 0, and the decoded text is 克力对社会 whisk |

So a multi-token **prefill** is fine (the `--n 1` run does one), and a multi-step **decode from a one-token
prompt** is fine (the `--tokens` run does three). What breaks is decoding a step **after a multi-token prefill** — that is,
the first attention that reads a populated cache and writes one new row.

That focuses the next check on the KV path rather than on the projections: the cache's per-layer layout against the
per-block head counts (`attention.head_count` is an array in this pack, 32 for some blocks and 64 for others), and the
`attn_sinks` handling, which only matters once there is history to attend to.

### The sparse-attention path is active, so that is where the inf lives

Two facts from the pack settle what has been running:

`
hy_v4_flash.attention.indexer.top_k      2048        -> is_dsa is true
hy_v4_flash.attention.head_count         [32, 32, 64, 64, 64, 32, 64, ...]
hy_v4_flash.attention.indexer.types      [1, 1, 0, 0, 0, 1, 0, 0, 0, 1, ...]
`

	op_k is non-zero, so every block goes through uild_attention_dsa, and a trace placed in the plain MLA builder never
prints - confirming that path is the one in use. The head count is indeed per block with a repeating pattern of two
thirty-two-head blocks followed by three sixty-four-head ones, and indexer.types marks which blocks carry a full indexer
rather than sharing the previous one's top-k.

That narrows the multi-token failure to the sparse path: the top-k of 2048 against a prefill of a handful of tokens, and
the kpool reduction that follows it, are the two places where a short sequence and a large k meet.

### Step one works, step two does not - so look at where the cache row is written

The top-k is clamped (`n_top_k = min(score->ne[0], indexer_top_k)`), so a large k against a short prompt is not the fault.
What the runs actually say is finer than "multi-token breaks it":

| run | cached positions when it ends |
| --- | --- |
| five-token prompt, one token out | prefill 5, decode at 5, fine |
| same prompt, two tokens out | prefill 5, decode at 5, then decode at 6 |
| one-token prompt, three tokens out | decode at 0, 1, 2, fine |

So a decode at position 5 after a prefill is fine and a decode at 6 is not - which is a very specific difference. The most
likely explanation is that the first decode wrote its row at the wrong slot, so the second decode reads a row that was
never written and gets whatever the buffer held - an infinite or NaN value, which is what the RMS_NORM assertion reports.

That points at the cache's write index rather than at the attention maths: the position a decode writes to, and the
positions the following decode reads. `llama-kv-cache-dsv4.cpp` carries the index computation for both.

### The infinite value is the stream itself, by layer thirty-one

Two things were fixed and one moved:

- the Engram gate is a quantised weight, and a quantised tensor has no binary-op kernel, so it has to be cast to f32 before
  the multiply - without that the gate's raw block bytes are read as floats. With the cast in, `engram_out` is finite
  (sum of absolute values about 843323 over 12288 entries, so a mean magnitude near 68).
- the indexer score reporting `inf` was a false alarm: with the classification in place it reads one finite entry and 255
  at minus infinity, which is the causal mask for the first row of a five-token prefill, exactly as intended.

What remains is that the assertion still fires, and the op trace puts it at block **31** inside the Engram block: a REPEAT
to [3072, 4, 1] followed by an RMS_NORM of the same shape. Since the projection feeding it is finite, the tensor that has
gone infinite is the residual stream that the norm is about to normalise - the stream grows by an Engram write whose
magnitude is around 68 per block at four blocks, and something in that growth is unbounded.

The next measurement is the stream's magnitude at each block boundary, which says whether it climbs steadily (a scaling
error) or jumps once (a specific write).

### The stream jumps six hundred fold at the first Engram block

Measuring the four-stream magnitude at every block boundary (`hc_attn_out-N`) gives a clear shape:

| boundary | sum of absolute values |
| --- | --- |
| after block 0 | 1711 |
| after block 1 | 1.036e6 |
| after block 2 .. 7 | about 1.05e6 |
| after block 36 .. 41 | about 9.24e6 |

Every one of those is finite, and the magnitudes are cumulative sums over 12288 or 61440 entries, so the per-element
magnitude is around 150 throughout. What stands out is the **six hundred fold jump between block 0 and block 1** - and
block 1 is the first Engram block. So the Engram write is contributing of order a thousand times more than the stream it
writes into, which is not plausible for a gated addition whose gate is a sigmoid.

That suggests the Engram value itself is too large rather than the stream diverging slowly: `engram_kv` reads a sum of
absolute values near 2.0e6 over 15360 entries, so a mean magnitude near 130, against a stream whose own elements are of
order 150 - close enough that a gate of one doubles the stream every Engram block, and there are four of them.

Two candidates, both checkable: the lookup rows are being scaled wrongly before the projection, or the projection is
missing a normalisation the reference applies (the pack's checklist mentions a 96-dimension lookup norm, which the current
write does not use).

### The lookup norm, and why it was not simply dropped in

Reshaping the gathered rows to [96, 16, nt], applying an RMS norm over the ninety-six values and scaling by
`engram_lookup_norm.weight` is the natural reading of "96-Dimension Lookup Norm". Inserting exactly that compiles, but the
graph then fails in the allocator (`buffer_id >= 0`), which is the scheduler saying a tensor it was handed has no buffer
assignment - the shape of that failure is a tensor that is not reachable from the graph's outputs, not a numeric error.

So it was reverted rather than left half-wired, and the next attempt has to keep the input's identity intact: the rows
tensor is a graph input whose `set_input` writes into it, and replacing the reference with a derived tensor can leave the
input itself unused. Building the normalised copy as a new tensor while leaving the input wired, or applying the norm
inside the gather on the host before upload, both avoid that.

### The failure moves with the number of tokens, so something grows per step

With the lookup norm in, the failure boundary is between three and four generated tokens, and it always lands on block 31's
Engram norm. That assertion fires when `1/sqrt(mean + eps)` is not positive, which happens when a squared value overflows -
so the tensor reaching it is not infinite yet, it is around 1e19. The stream measured 8.2e4 at block 41 for a two-token
run, so between two and four steps something multiplies the magnitudes by about fifteen orders.

Growth per step of that size, confined to a late block, points at the routed experts rather than at the attention: the tier
dispatch sums three tiers of sixteen experts each, and if each tier applies its own normalisation of the router weights
then one block contributes three times what it should, which compounds by 3^layer over forty-one blocks. That is the first
thing to check - the weights a tier applies have to be the ones the router computed for the whole expert set, not a
renormalisation of that tier's own sixteen.

### A second prompt trips a bounds check in the row gather

Decoding the eight-token answer to the English prompt gives text that starts sensibly and then drifts:
\富人区的 yticetic rant rant rador ranter maid\. A Chinese prompt (\法国的首都是\) instead trips
\GGML_ASSERT(i01 >= 0 && i01 < ne01)\ in the CPU get_rows path, which is the bounds check on a gathered row index - so
some lookup is being handed an id beyond its table, and the token count of that prompt is what brings it out.

That is worth separating from the previous failure: the Engram index entries are now bounds-checked against their tier, so
this id is coming from a different gather. The candidates are the tier local-id and mask tables, whose rows equal the
expert count, and the token embedding, whose rows equal the vocabulary size.

### The shared expert is in the pack and not in the graph

The logits are not degenerate: a Chinese prompt's admission row reads top-1 8.267 and top-2 7.997, a healthy shape, and
the answer it decodes to is wrong rather than random. So the numbers are plausible and the model is missing something,
which is the harder kind of error.

Block 1's FFN tensors include the shared expert:

```
blk.1.ffn_{up,gate,down}_shexp.weight
blk.1.ffn_{up,gate,down}_exps_{hot,warm,cold}.weight
blk.1.ffn_gate_inp.weight
blk.1.ffn_routed_{down,up}.weight
```

while the graph's MoE call passes only the routed stacks (`ffn_up_exps`, `ffn_gate_exps`, `ffn_down_exps`) and the router -
there is no argument for the shared expert, so those three tensors are loaded and never used. The pack's own architecture
list agrees they matter ("512 Routed Experts per MoE Layer, Top-16 Routing, plus one Shared Expert").

That is a missing addend in every MoE block, which is exactly the shape of a plausible-but-wrong result, and it is the next
thing to wire: the shared expert's up/gate/down path evaluated on the same input and summed with the routed output.

### The sliding window is in the pack and nowhere in the model

Three keys describe the hybrid attention, and none of them is read:

```
hy_v4_flash.attention.sliding_window          512
hy_v4_flash.attention.sliding_window_pattern  [0, 0, 1, 1, 1, 0, 1, 1, 1, 0, ...]
hy_v4_flash.rope.freq_base_swa                10000.0      (against freq_base 1e7 for the others)
```

`hy_v4.cpp` never mentions `n_swa`, `sliding_window` or a second rope base - a search finds nothing - while
`deepseek41.cpp` does use `hparams.n_swa`, so the hyperparameter machinery is there and simply unused for this family.
The pattern says that after the first two blocks every fourth one is full attention and the other three are windowed, and
those windowed blocks use a rope base **three orders of magnitude smaller** than the full ones.

That is the largest remaining structural difference between what the pack describes and what the graph builds: three
quarters of the attention layers attend to the whole context with a rope base they were not trained with. It would leave
logits healthy in shape and wrong in value, which is exactly the symptom, and it explains why adding Engram and
feed-forward terms one at a time changed the tokens without improving the text.

The next step is therefore not another missing addend but the window itself: read the three keys, select the base per
block, and mask the windowed blocks to the last 512 positions.

### The window cache is built after all - so why does the mask not change?

Reading the memory construction for this family corrects the previous conclusion. The branch that a pack with an indexer
takes does pass the window through:

```
res = new llama_kv_cache_dsa(..., cparams.n_seq_max, hparams.n_swa, hparams.swa_type, nullptr, filter_lid, nullptr);
```

so `n_swa` and `swa_type` reach the cache, and the two changes made this round - reading the per-block pattern and setting
`swa_type` to the standard window - are both on that path. The tokens changed when the pattern was read (the rope base per
block began to differ) and did not change when `swa_type` was set, which suggests the mask is not consuming the window even
though the cache is given it.

The next check is therefore inside `llama_kv_cache_dsa`: where it builds the attention mask, whether `n_swa` is consulted
per layer through `is_swa`, and whether the window is only applied when the cache holds a separate windowed buffer - which
the "nullptr, nullptr" halves of the older constructor suggest it might.

### The hybrid cache exists and needs more than a constructor swap

There are two caches for this family, not one:

| cache | window |
| --- | --- |
| `llama_kv_cache_dsa` | takes `n_swa` and `swa_type` but builds its three inner caches with `0, LLAMA_SWA_TYPE_NONE`, so no windowed buffer exists |
| `llama_kv_cache_dsa_iswa` | the pair, and it is what another arch in this tree already uses |

So the right structure is the second one, and the plain cache can never mask a window no matter what the model declares.
Switching this family to it, though, is not a constructor swap: with the plain filters and a null MLA filter the context
fails to initialise (`invalid unordered_map key`), which means the hybrid cache expects the layer filters to describe which
layers own an MLA slot and which own an indexer slot, and an empty/incorrect mapping is rejected rather than defaulted.

That change was reverted to keep the tree green. The next attempt needs the two filters spelled out for this pack: the MLA
filter covering the trunk layers (excluding the MTP block when one is present) and the indexer filter covering exactly the
blocks whose `indexer.types` entry is one - which the model already computes as `is_indexer_full`.

### The hybrid cache needs its paired graph input, and the tree already has one

With both filters supplied the context still fails (`invalid unordered_map<K, T> key`), but the log now says the hybrid
cache was chosen and it built a full-size window buffer. The failure is one layer up: the graph's attention input is built
for the plain DSA context, and the memory context handed to it is the hybrid pair, so the lookup that keys layers to a cache
finds a type it does not know.

The tree already carries what is needed:

`llm_graph_input_attn_k_dsa_iswa` holds two inputs and the hybrid context: one for the full-attention and indexer half, one
for the windowed half, plus the context itself.

so the remaining work is on the graph side: build that paired input when the memory is hybrid, and dispatch each block to
`get_dsa()` or `get_swa()` according to `hparams.is_swa(il)`. That is the last structural piece of the window, and it is
bounded - one input class, one dispatch, no new kernels.

### The prompt has to be a chat turn

`tokenizer.json` records `bos` 120000, `eos` 120025 and `add_bos_token` false, and the pack's chat template wraps a turn in
its own special tokens:

```
<|hy_start:opensource|>  120000
<|hy_middle:opensource|> 120001
<|hy_end:opensource|>    120025
```

so bare text is out of distribution for this checkpoint - which matters for how the output is judged. A five-token turn
(`120000 113685 2757 3764 120001`, that is the start marker, the four prompt tokens, then the middle marker) produces eight
tokens whose bytes read as `noteyayster "御zingerb escapes`, a longer and more varied continuation than the bare prompt gave.
It is still not a good answer, so the remaining error is numerical rather than a formatting mistake, but every later quality
check should use the turn markers: without them the comparison measures the wrong thing.

### What has been checked against the pack, and what has not

Each entry below was compared against the file itself rather than assumed:

| piece | pack | graph |
| --- | --- | --- |
| expert counts | 512 used 16 shared 1 | read |
| router | scale 3.998, norm on, gating 2 | printed at load, matches |
| hyper-connections | count 4, epsilon 1e-6, magnitude 2.0 | read |
| `hc_attn_fn` | 12288 x 8, i.e. n_embd*hc by 2*hc | table and builder use both halves |
| `hc_attn_scale` / `hc_attn_base` | 2 and 8 | two scales, two base halves of hc |
| `attn_sinks` | 32 per block, matching that block's head count | passed to `build_attn` |
| head tensors | `output_hc_fn` 12288 x 4, `output_hc_base` 4, `output_hc_scale` 1 | table matches |
| Engram | tiers, index, lookup norm, bias, pre-gamma, stream scale | all in the write |
| attention | window 512 with a per-block pattern, rope base 10000 windowed | window half now dispatches |
| Hadamard | 849 folded weights | verified by the graph walker at load |

What has **not** been checked against an independent implementation is the arithmetic of the hyper-connection fold itself:
the two gates, the reduce and the write back into the four streams. The pack carries no reference for it, and the shapes
and constants agree, so the remaining error is most likely in that formula - the multiply order in the reduce, or which
half of the eight mixes belongs to the pre gate and which to the post gate.

### Two samples on the hyper-connection question

The halves were decided on one turn; a second one was run to check the decision was not a fluke. Both prompts are full chat
turns, and the same weights produced:

| turn | halves as implemented before | halves the other way |
| --- | --- | --- |
| `法国的首都是` | `noteyayster "御zing` | `illardails机械校对adingilles feel restoresayana` |
| a second turn | `correspondinglyordurdyman ded一扫 Tk quit` | `tis mates Project vag mechanical` |

The other assignment produces a Chinese phrase in the first case and ordinary English words in the second, where the
earlier one produced neither. Two samples are still not proof - the text is not right either way - but they are consistent,
which is what justifies making that reading the default and keeping the other one switch-able.

The same method applies to the remaining unchecked detail: whether the reduce multiplies the gate per stream before summing,
which is what the code does now, or after. That is one switch and two runs away.

### A regression check that does not need a second model

The Engram row probe still returns the same values it did before the graph work: scale code 235 reading as 193.406708 and
the first four values 113.588066, -82.888588, 144.287537, 113.588066, matching a hand decode of the same eighty bytes. That
covers the type converters and the file offsets after every change made since - the block types, the reader, the windowed
attention and the hyper-connection reading all run above it and none of them touched it.

The other probe, hy4-probe, needs the block dump the reference tool writes; that directory is not in the tree any more, so
it reports a missing codebook rather than a mismatch. Regenerating the dump is how to bring that check back.

### The four-type dequant gate, restored and passing

The block dump the probe needs is one command away, and with it regenerated the gate reads:

```
csq8     512 values  max abs diff 0.000e+00  nmse 0.000e+00
csq4     512 values  max abs diff 0.000e+00  nmse 0.000e+00
vq4_128  512 values  max abs diff 9.313e-10  nmse 6.882e-18
vq4_64   512 values  max abs diff 9.313e-10  nmse 3.566e-17
all four types match the reference
```

So the two codebook types are exact and the two vector-quantised ones differ only in the last float bit, which is the
codebook's stored precision rather than an index or scale mistake. To rerun it: dump with the reference tool into a
directory, then point the probe at that directory.

The dump covers one real block of each type from the backbone, so it exercises the block layout, the codebook indexing and
the scale application together - which is what a change to the reader or the converters would break first.

### The logits gap is a better instrument than the text

Two admission rows recorded along the way:

| prompt | top-1 | top-2 | gap |
| --- | --- | --- | --- |
| French capital turn | 8.267 | 7.997 | 0.27 |
| another turn | 8.090 | 8.010 | 0.08 |

A model that has learned the continuation puts one token well clear of the rest - a French capital answer should sit several
logits above its runner-up. A gap of a tenth of a logit means many tokens are near-equally likely, which is what a partly
wrong forward pass looks like: not random, but flattened.

That makes the gap a cheap numeric score for the experiments that have no reference: run the same turn under two readings of
an unverified formula and keep the one with the larger gap and the larger top-1. It is faster to read than the text and less
noisy than judging whether a fragment of Chinese is meaningful.

The reading of the hyper-connection halves was chosen by text; re-checking it against the gap is the first thing to do with
this instrument.

### The gap agrees with the hyper-connection choice

The same turn, the same weights, the two readings:

| reading | top-1 | runner-up | gap |
| --- | --- | --- | --- |
| halves swapped (default now) | 7.971 | 7.894 | 0.077 |
| halves as before | 7.402 | 7.353 | 0.049 |

The swapped reading is ahead on both numbers, which is what the text comparison had said and what an independent instrument
should confirm. It also shows how far the forward pass still is from a model that knows the answer: a confident continuation
would sit several logits clear of its runner-up, and both readings are within a tenth.

So the instrument works, the choice stands, and the remaining error is elsewhere.

### Choosing a copy that is already on disk

The app now accepts this family from the local-model importer, which the objective asks for by name since the pack is not
downloaded by Strata. Four things were in the way, each found by running the importer against the real file rather than by
reading it:

| what | why it failed | now |
| --- | --- | --- |
| architecture set | neither hy_v4 nor hy_v4_flash was listed | both listed |
| shard detection | the names end -00001-of-00002, but the pack is one backbone plus a companion | companion families skip the shard logic, and a file marked strata.companion is treated the same way |
| block-size table | the reader knew no block type past the standard ones | the pack's 144 to 150 and its fp8 are in the table |
| head count | stored per block, so the reader reports a count rather than a value | accepted, with the lengths taken from the explicit key and value lengths |

With those in place, inspecting the real pack returns hy_v4_flash, one file, a 1M context and one warning. Two probes under
port/probe keep that checkable: one calls the importer, one prints the metadata the importer validates.

### On Vulkan the context crashes while being created

The same pack that runs on the CPU stops on the graphics backend, right after the hybrid cache announces its window buffer:

```
creating a 1024-token context (batch 1024, 18 threads, kv f16) ...
llama_kv_cache_dsa_iswa: using full-size SWA cache
(the process stops here)
```

Nothing in the log blames a type or a shape, and the model itself loads (10.9 s, hy_v4_flash, 42 layers, 1M training
context, template present). So the fault is in constructing the paired cache on that backend, not in the pack or in the
graph - the graph has never been reached.

That is the next thing to look at for this family on the machine it targets, and it is a different question from the numeric
one the CPU path still has: one is a crash before the first token, the other a plausible but wrong answer.

### The Vulkan crash is in the graph reserve, not in the cache

Unbuffered markers settled where it happens. The paired cache completes - its constructor prints enter, filters, both inner
caches and done - and `llama_context` reaches `resolve_fused_ops`, whose first line writes `fused` before calling it. The
next marker never appears, and neither does the one the engine writes when `llama_init_from_model` returns.

So the fault is inside `resolve_fused_ops`, which reserves a probe graph and then a worst-case one: the Hy4 graph is being
built and allocated on the Vulkan backend for the first time, and that is where the process dies. The cache construction,
which earlier looked guilty, is innocent.

Two consequences for the next attempt:

- the CPU path builds the same graph, so the difference is in what the backend supports rather than in the graph's shape -
  the pack's own block types have no Vulkan kernels, and the scheduler has to find a host fallback for each of them;
- the instrumentation was reverted, but the sequence to reproduce it is recorded here: unbuffered markers at the cache
  constructor, at `resolve_fused_ops` and before the two reserves name the stage without a debugger.

### No operation is ever checked, so the crash is in the reserve's allocation

The Vulkan path already prints every MUL_MAT type it is asked about under `STRATA_VK_OP_TRACE`, and a run on this pack
prints **none** before dying. So the process never reaches the per-operation support check; the crash is earlier in the
reserve, in the scheduler's allocation of the graph's buffers.

That rules out the avenue the last round proposed. The question is no longer "does the backend support this block type" but
"does reserving a graph of this shape allocate something it cannot", and the difference matters because the first would be
answered by a support table while the second is answered in the allocator.

The pack's buffers are large and its tier tables are separately allocated device buffers, which is the kind of input an
allocator path can meet for the first time here.

### The support checks do run; the crash is after them

A trace at the top of the backend's support check corrects the previous reading. It runs **2510** times on this pack before
the process dies, beginning with the embedding gather and the first projections:

```
[sup] GET_ROWS a=csq8 b=i32
[sup] MUL_MAT a=csq8 b=f32
[sup] MUL      a=f32 b=f32
```

and ending on plain `MUL` checks of f32 tensors. The earlier conclusion that no operation was ever checked came from a trace
that only covers one op - it printed nothing because nothing had reached that op yet - so the correction matters: the
backend does answer for the pack's own block types all through the graph.

What dies is therefore after the checks, in the reserve's allocation, which is where the previous round had already pointed.
The trace was removed after the run; the flag it used was temporary and is not part of the tree.

### Neither backend reaches the scheduler's reserve

Markers at the top of `ggml_backend_sched_reserve` print nothing on **either** backend, and the CPU path runs to
completion, so that function is simply not on this path: the probe allocates through `split_only`, where
`llama_context::graph_reserve` calls `ggml_backend_sched_split_graph` directly and never `ggml_backend_sched_reserve`.

Two consequences:

- the crash on Vulkan is in the graph's build or in `split_graph`, not in the reserve that follows it - which also explains
  why no allocation-time marker appeared;
- the previous round's conclusion ("after the support checks") still holds, but the step after them is `split_graph`, and
  that is the next place to look.

Forcing the plain DSA cache instead of the paired one does not change it, so the paired cache is not involved either.

### The crash is inside the graph build, and the MoE cache is the likely reason

A marker before `model.build_graph` prints on Vulkan and the one after it never does, so the process dies while the graph is
being built - not in the split, not in an allocation, and not in the cache construction. The CPU builds the same graph from
the same code, so what differs is something the graph builder sees only on a device backend.

The most likely such thing is the expert cache. It exists only when there is a device to hold experts, and `build_moe_ffn`
asks it to resolve the routed stacks and the selected experts:

```
moe_cache->build_lookup(*this, { il, up_exps, gate_exps, down_exps, ... }, selected_experts, cur_x)
```

For this pack those routed stacks are **empty** - the experts live in the three tier stacks instead - so the lookup is
walking tensors that do not exist. On the CPU the cache is null and the call is skipped, which is why the same graph is
fine there.

The fix to try next is a guard: when the routed stacks are absent, treat the layer as uncached and leave the cache out of it.

### Vulkan runs the pack, and it does not agree with the CPU

With the Engram weight copied through its backend instead of read through a host pointer, the graphics backend builds the
whole graph (all 42 blocks and the head) and generates:

| backend | top-1 | runner-up | tokens |
| --- | --- | --- | --- |
| CPU | 7.971 (7992) | 7.894 | 7992,86129,... |
| Vulkan | 9.047 (273) | 8.753 | 273,33418,... |

Same weights, same prompt, same KV type - and different tokens. That is a correctness question rather than a performance one,
and it matters more than the speed: two backends of one engine have to agree on the first token before either can be
trusted, and the earlier CPU-side tuning work assumed the CPU was right.

The suspects are the places where the two paths genuinely differ rather than the arithmetic itself: flash attention being
on for one and off for the other, and the block types that have no Vulkan kernels and therefore run as a host fallback
inside a device-scheduled graph. Both are checkable by pinning the option and re-running.

The speed also says something: 0.3 tok/s with 309 s of CPU time against 21 s of wall time means the work is mostly on the
host, so the pack's block types are falling back rather than running on the device.

### Pinning flash attention splits the two backends further, not closer

Same turn, `--flash-attn off` on both:

| backend | top-1 | runner-up |
| --- | --- | --- |
| CPU | 8.500 (5102) | 8.459 |
| Vulkan | 9.047 (273) | 8.753 |

Two facts come out of that:

- the CPU path **does** move when flash attention is turned off (7.971 before, 8.500 now), so the flag is live there;
- the Vulkan path does **not** move at all - exactly the same top-1 and the same runner-up as with it on - so on that backend
  the flag does not reach whatever computes attention, or the attention it computes is already the same either way.

So the divergence is not flash attention: pinning it separates the CPU from itself and leaves the device where it was. The
next variables worth pinning are the two the device path differs on by construction - the windowed half of the cache and the
indexer - since both change which keys an attention sees rather than how it adds them.

### The two backends agree to half a percent, and the flat distribution turns that into a different token

Comparing the last hidden state instead of the sampled token settles it:

| backend | sum of absolute values of `result_norm` |
| --- | --- |
| CPU | 2862.12 |
| Vulkan | 2874.40 |

That is a difference of about 0.4 percent, which is what two accumulation orders of the same arithmetic produce. It is enough
to flip the argmax only because the distribution is nearly flat - the gap between the top two logits was 0.077 in one run -
so a tenth of a logit of numerical noise decides which token comes first.

So the backend question and the numeric question separate cleanly:

- the device path is **not** computing something different in kind; it is the same arithmetic with a different accumulation
  order, and it ends up where the host ends up within half a percent;
- what makes the visible disagreement look dramatic is the unresolved numeric error, which flattens the distribution until
  small differences decide the answer.

That retires the backend as a suspect for the wrong answers: fixing the remaining numeric error would also make the two
backends agree on tokens.

### Ablating the Engram write needs the input built conditionally

The obvious next experiment - run the turn with the Engram write switched off and compare the logits gap - does not work as
a one-line switch: skipping the write leaves the graph input that feeds it unreferenced, and the run dies rather than
comparing. So the ablation has to be done where the input is created: build the rows tensor and add the input only when the
write will use it.

That is worth recording because it is a general shape in this graph: an input that exists for one consumer cannot be
disabled downstream of the consumer. Any future ablation of the Engram path, or of the expert tiers, has the same
constraint and needs the same treatment.

### Switching the Engram path off sharpens the distribution, which says the write is wrong

With the ablation moved to where the input is created, the comparison runs:

| | top-1 | runner-up | gap |
| --- | --- | --- | --- |
| Engram on | 7.971 | 7.894 | 0.077 |
| Engram off | 9.099 | 8.806 | 0.293 |

Removing the Engram contribution raises the peak and widens the gap fourfold. A block the model was trained with should not
make its predictions less confident, so the write is adding something wrong - not merely something small. That is the
clearest numeric signal obtained so far, and it is a better target than the search for missing terms: the term is present and
its shape is wrong.

The candidates, in the order the code would produce them: the gate uses the root of the dot product rather than the dot
itself, the value slab is the fifth projection rather than a separately projected value, and the lookup weight may belong on
the row before the projection rather than on the projected key.

### The position of the Engram write needs the write extracted, not duplicated

Whether the write belongs before the block's attention or after its feed-forward is the last guess left in this path. Trying
it by duplicating the block - one guarded copy at each end - compiles but crashes on the late reading, which is what a
duplicate that shares tensor names and a static switch with its twin tends to do.

So the experiment needs the write extracted into a helper that takes the streams and returns the updated ones, with the call
site chosen by a flag. That is a small refactor rather than a copy, and it is the next step for this path.

Everything the write contains has now been measured: the gate, the slab split, the lookup norm and the pre-scale. The
position is the only piece still resting on a reading of the pack's wording rather than on a number.

### Sharper, and still not coherent

Twelve tokens from the same turn now decode as

```
??152Lytasya cooperayana幕富人区的ayanaailscope
```

against the earlier eight that read `noteyayster "御zing`. The distribution is much sharper - the peak went from 7.971 to
10.490 and the gap from 0.077 to 1.520 - and two Chinese phrases appear ("富人区的", "幕"), but the text is still fragments
with a repetition (`105666` twice), which is what a model whose attention is partly wrong produces.

So the Engram path is now measured rather than guessed and the result improved, and the content is still wrong: the
remaining error is elsewhere. In the order the evidence points:

- the hyper-connection formula, which was settled on two samples and one gap measurement rather than on a reference;
- the attention itself - the MLA absorption through `wk_b`, the value projection through `wv_b`, and the sinks, none of
  which has been compared against anything;
- the output head: the hyper-connection collapse, the final norm, and the Hadamard rotation that the graph verifier accepts
  but that no number has checked.

The instrument for all three is the one built this session: a switch, two runs, and the peak and gap to compare.

## State of the Hy4 adaptation

Where this pack stands, what each claim rests on, and what is still open. Kept here rather than in a summary so it moves
with the code.

### Working, with the evidence

| what | evidence |
| --- | --- |
| loads completely | 1661 of 1661 tensors created, coverage log reports zero experts without a tier |
| runs on the CPU | generates, 42 layers, the graph built every step |
| runs on Vulkan | the same pack, the same graph, exit 0; the earlier crash was a host pointer read on a device tensor |
| backends agree | last hidden state 2862.12 against 2874.40, 0.4 percent apart |
| does not need a bring-up flag | the folded weights are on the verified list and the graph walker accepts them |
| can be chosen from disk in the app | the importer reports `hy_v4_flash`, one file, 1M context; a test covers it |

### Verified by measurement rather than by reading

Each of these was settled by a switch, two runs and the logits peak and gap:

| piece | effect of getting it wrong |
| --- | --- |
| Engram gate | the signed root instead of the similarity: gap 1.520 to 0.606 |
| Engram slab split | value from the last slab: 1.520 to 0.606 |
| Engram lookup norm | leaving it out: 1.520 to 0.436 |
| Engram pre-scale | leaving it out: 1.520 to 0.174 |
| Engram write position | after the feed-forward: 1.520 to 0.074 |
| router scale | one instead of the pack's 3.998: 1.520 to 0.149 |
| head collapse | a plain mean instead of the gate: 1.520 to 1.286 |

The peak and gap on the reference turn went from 7.971/0.077 to **10.490/1.520** over that work.

### Open

| what | why it is open |
| --- | --- |
| coherence | the text is still fragments with occasional Chinese; something outside the settled list is wrong |
| MLA attention | the compressed-K/V absorption, the value projection and the sinks have not been compared against anything |
| MTP | the nextn block loads but nothing uses it |
| expert residency | the tier stacks are plain tensors, so the expert cache and the residency budget do not reach them |
| Q2 path | this pack has no Q2 variant; the objective's Q2 requirement applies to the other family |

The instrument for the open items is the same one: a switch, two runs, and the peak and gap to compare.

### The attention tensors are per head, and the shapes line up

Block 0's attention is worth reading once in full, because it settles whether the absorption is a plain matrix product or a
batched per-head one:

```
attn_q_a      (3072, 1024)      attn_q_b        (1024, 8192)
attn_kv_a_mqa (3072, 576)       attn_k_b        (192, 512, 32)
attn_gate     (3072, 8192)      attn_v_b        (512, 256, 32)
attn_output   (8192, 3072)      attn_sinks      (32,)
```

`attn_k_b` and `attn_v_b` carry a third dimension of 32, the block's head count, so both are **per head**: the query is
permuted to `[192, tokens, heads]`, multiplied by a `[192, 512, heads]` weight and permuted back, which is a batched product
and not a matrix one. The widths then chain correctly - 512 absorbed plus 64 rope gives the 576 the score is taken over,
`attn_v_b` turns that 512 into a 256-wide value per head, and `attn_output` maps 8192 back to 3072.

So the builder's permutations are doing the right thing, and the K and V projections are not a place a plain
`build_lora_mm` would go unnoticed: a two-dimensional reading of these weights would have failed the shape check at load.

### The terminal rejects the pack for two checkout limits, not because of the pack

`strata launch --plan --model-path` refuses this family with three sentences. They come from the same pre-flight check that
guards the model folder, and the reasons behind them are limits rather than faults:

| sentence | the check behind it | the limit |
| --- | --- | --- |
| its header is damaged | a metadata array larger than the reader allows | `MAX_ARRAY_ITEMS` is 500000, and the pack's rotation sign arrays are bigger than that |
| it stores weights in a format Strata does not read | a block type outside the table | `GGML_TYPES` lists 50 entries; the pack's 144 to 150 are not among them |
| this is not a model size Strata knows | a size it has no reading for | the family is not in the size table |

None of the three says the file is wrong. The three edits are therefore in the checker rather than in the pack: allow the
large arrays for the keys that legitimately carry them, add the block types to the table, and record the family. The app's
importer was already taught the architecture and the block sizes this session, so the two paths disagree today and this is
the terminal side of that disagreement.

### The terminal's block table already uses the ids this pack needs

The pre-flight check takes its type names and block geometry from one table, and that table already has entries for 144 to
150 - the trellis family, at 128-value blocks of 34, 26, 30, 28, 42, 50 and 58 bytes. This pack writes 144 to 150 too, but
means its own types: 512-value blocks of 196, 132, 116 and 100 bytes for the backbone and 96-value blocks of 56, 80 and 32
for the Engram tables.

So adding rows for this family would not be enough - the same id would carry two geometries, and the byte-count check would
reject whichever family it was not written for. The ids are per file only because the writer chose them; the reader has to
disambiguate by architecture, which is exactly what this engine already does: `gguf.cpp` remaps 144 to 150 for
`general.architecture` `hy_v4_flash` and leaves them alone otherwise.

The terminal therefore needs the same arch-aware reading the engine has, in one shared helper, rather than a second copy of
the mapping in the pre-flight check. That is the shape of the fix and the reason it is a small refactor rather than three
table rows.

### The terminal has one objection left, and it is the companion as a shard

After the block types and the architecture list, the backbone of this pack passes the terminal's check and the only thing
still refused is the companion file. It is refused for reasons that name a model: no tokenizer, no known size. That is the
same confusion the app's importer had, and the same answer applies - a file that declares a companion architecture and
carries no tokenizer is a companion rather than a shard, and belongs in the list beside the model the way a picture reader
does rather than inside the shard set.

The rule is checkable on the file itself, which is why it can be applied before the model is judged: `tokenizer.ggml.tokens`
is either there or not, and a companion never has it. The engine already opens these files itself, so the check's job is only
to leave them alone.

### Skipping the companion is in place but an earlier pass still names it

The rule now sits where the metadata is read: a file that declares the family and carries no tokenizer is left beside the model
and named in the notes rather than judged as a shard. The terminal still refuses the companion, which means an **earlier**
pass is inspecting it first and recording the failure - the per-file loop only runs when nothing has failed yet, so its skip
cannot help.

So the next step is to find the first place the shard set is inspected and apply the same classification there; the rule itself
is settled.

### The launch layer wants a catalogue name, and this family has none

plan_import now reports no issues for the pack and names the companion in its notes, but strata launch still refuses it
because the step after the plan needs the plan to name a **catalogue** family and model - the same pair the download flow
would produce. A family the engine implements but the catalogue does not list has no such pair, so the launch stops even
though nothing is wrong with the files.

The fix belongs there rather than in the plan: when the plan has no issues and the architecture is one the engine
implements, the launch should take the architecture as the identity and say that it is a family Strata serves without a
catalogue entry. That is a small branch, but it needs care because the family and model names are used for presets and
paths later, so they have to be chosen rather than left empty.

### Checking the terminal changes without the test suite

`pytest` is not installed in either Python on this machine, so the terminal tests cannot be run here. What was checked instead,
after six edits across four files:

- all four changed modules compile;
- the architecture-aware tables behave as intended at the boundary that matters: file type 4 is a 32-bit integer for
  `hy_v4_flash` and stays `Q4_2` for every other architecture, so the trellis and Flash-Next families read exactly as before;
- `plan_import` reports no issues for the pack and names the companion in its notes, and `strata launch --plan` prints a plan.

That is weaker than the suite and should be said plainly: the changes are exercised on this pack and on the shape of the old
tables, not across the terminal's whole behaviour.

### What the suite could and could not run here

The terminal suite is now installed and runs, with two limits this machine imposes:

| what | result |
| --- | --- |
| `test_models`, `test_import_projector`, `test_feed_model_plan_drafter` | 23 of 23 pass |
| the full run | stops in `test_launch_tui_walk`, which drives a real terminal through an asyncio event loop and waits on a completion port forever here |
| the screen tests | need `textual` to render; they were not part of the focused run |

So the modules this session changed are covered by passing tests, and the parts that hang are the interactive walk tests rather
than anything the changes touch. That distinction matters: a suite that cannot finish should not be reported as failing code,
and a focused pass should not be reported as a full pass.

### The inverse rotation is enforced, not assumed

The embedding table is the one weight the pack says to rotate in the opposite order, and it is worth knowing that this is
checked rather than trusted: the graph walker treats a row lookup on an inverse weight as a case of its own and refuses a
graph that reads such a table without the inverse transform. This pack loads, so the inverse path is wired - one of the few
places where a load-time check stands in for a measurement.

That leaves the list of things covered by neither a check nor an experiment short: the MTP block's participation and the
expert residency path are the two, and both are on the objective's list rather than in the model's arithmetic.

### The text judgement was sound, and the main path is 42 layers

Two things were checked to make sure the earlier conclusions rest on the right reading.

**The detokenisation.** The engine has a mode that reads a token stream back as text with its own vocabulary, without loading
the weights. It renders `27984,12395,68460,61490,5102,15709` as `??152Lytasya cooper` - the same fragments a hand decode of the
same ids produced. So the judgement that the text is wrong does not come from a bad byte table in a script; it is the model's
own vocabulary saying so.

**The MTP block is not in the main path.** The load line says 43 blocks, which is the whole file including the nextn block, and
that is only the tensor table's count: `n_layer()` returns `n_layer_all - n_layer_nextn`, which is 42, and the graph loops over
`n_layer()`. So the draft block is loaded and not used, which is the state the objective describes, and not a layer that is
silently contributing.

Both were worth confirming: one protects the conclusion that the numbers are wrong, the other removes the suspicion that a
43rd layer is quietly part of the forward pass.

### The turn format is not the reason either

The template's rendering for one user message is

```
start marker, "user", middle marker, the text, end marker, start marker, "assistant", middle marker
```

and the role words inside the model's own vocabulary are `3717` for "user" and `611, 10372` for "assistant". A turn built
exactly that way - eleven ids including the four markers - gives a peak of 9.837 with a gap of 1.314, and the eight tokens it
produces decode as `yg-Ly暇 PassageapaMentapa` through the engine's own detokeniser.

So the earlier prompts were indeed not formatted the way the model expects, and formatting it properly changes the answer
without making it coherent. That removes the last non-numeric explanation from the list: what remains is arithmetic, in a place
no switch has reached.

Worth noting for the next attempt: the tokenizer accepts the special markers as **text** but splits them into seven ordinary
tokens rather than the single id each, so a prompt written with them is not the same prompt. The ids have to be given
explicitly.

### The indexer is dormant at this context, and the scale has now been bracketed

Two more reads and one more measurement on the same formatted turn.

The sparse indexer is real in this pack - `attention.indexer.top_k` is 2048, its head count 16, its key length 128 - but 2048
exceeds the eleven tokens of the prompt, so a selection of the top 2048 out of eleven is all of them. The indexer's tensors
load and the graph does not compute them, which is the right state for this context and would stop being right in a long one.
Worth writing down before the context work starts.

The scale has now been bracketed around the value in use: with the compressed width it is one over the square root of 576, and
multiplying that by 1.5 gives a peak of 9.508 against 9.837 with the same token on top. So the scale is not a free parameter
that happens to be near the right answer - the value in use is the best of the three tested, which closes that line.

### Nothing loads and is unused except what should

A coverage dump now exists: with `STRATA_HY4_COVERAGE` set, the graph prints every weight that appears as a node source once
per build, and that set can be compared with the file's tensor list. On this pack the comparison reads

| | count |
| --- | --- |
| tensors in the file | 1661 |
| read by the forward pass | 1573 |
| never read | 88 |

and the 88 fall into three groups, each with a reason: the 36 of block 42, which is the draft block and not part of the forward
pass; the indexer's compressor tensors in the other blocks, which belong to the sparse path dormant at an eleven-token context;
and the four Engram lookup norms, which the graph does not reference because the lookup is normalised host-side while the rows
are gathered. None is an orphan.

The first count taken said two groups rather than three, because block 42's tensors were being counted under their own names -
`blk.42.indexer_compressor_ape` and the rest - rather than as one block. The numbers were right and the description was not.

That is the answer to "is something loaded and ignored" for this pack: no. It also means the remaining error is not a missing
term in the forward pass, which narrows the problem to how a term is computed rather than whether it is there.

### What "MTP usable" will take, from the block's own shapes

Block 42 is a full block - attention, routed experts over the hot tier, shared expert, both post norms - plus four tensors no
trunk block has:

| tensor | shape | what the shape says it does |
| --- | --- | --- |
| `nextn.enorm` | 3072 | norm on the next token's embedding |
| `nextn.hnorm` | 3072 | norm on the trunk's last hidden state |
| `nextn.eh_proj` | 6144 by 3072 | projects the concatenation of those two, hidden first, into the block's width |
| `nextn.shared_head_norm` | 3072 | norm before the shared head |

So a draft step is: take the trunk's last hidden state and the embedding of the token whose successor is wanted, norm each,
concatenate in that order, project, run block 42 over the cache with the draft position, norm, and read the shared head - the
same `output` weight the trunk's logits come from, which is why it is called shared.

That is a second graph, not a flag: the draft token needs its own cache entry and its own attention over everything the trunk
has written. It is the next piece of work, and the shapes above are what make the recipe checkable before any code.

The block's presence is already correct: it loads, and coverage shows its 36 tensors are read by nothing, which is exactly the
state "loaded but not used" that this section describes.

### The two remaining pieces, with the lines they touch

Both of the objectives still open on this family meet existing code at a named place, which is worth writing down before the
work rather than during it.

**The draft step** needs a second graph. Nothing in `hy4flash`'s tensor table changes - block 42 already loads - and the recipe
above names the operations. What is missing is a graph type that writes the draft token's cache entry and attends over
everything the trunk wrote, and that is a new builder beside `llama_model_hy_v4::graph`, not a flag inside it.

**The expert cache** meets a guard this session put there: in `llama-graph.cpp` the lookup is skipped when the layer is served
from tiers - `moe_cache == nullptr || tiered_experts` - because the tier path sums its own weighted experts. Letting the hot
tier be paged therefore means teaching the lookup that a stack can be split across several tensors, one per tier, with the
router's ids remapped per tier. The remap table already exists (the tier's `local` table, built at load), so the work is in the
cache rather than in the graph.

Neither is a workaround: one is the draft path the pack ships, the other is the residency the objective asks for. Both are
larger than a switch, which is why they are written here as scopes rather than done at the end of a session.

### The router's selection bias is applied, and the count says which blocks have one

Forty-two tensors named `exp_probs_b.bias` sit in the file, one per MoE block, 512 values each - a bias on the router's
selection. The graph reads **forty-one** of them, which is the right number rather than an off-by-one: block 0 is the dense
block and has no MoE to bias, and block 42's belongs to the draft block that nothing reads yet. So the bias is being applied,
and the coverage dump distinguishes "not read because it does not apply" from "not read because it was forgotten" by itself.

That closes a hypothesis worth closing: a per-block router bias that a builder quietly ignored would move the expert selection
in every block and produce exactly the kind of plausible-but-wrong output this pack shows.

### The half split, re-measured where the prompt is right

The split of the eight hyper-connection mixes - which half feeds the pre gate and which the post gate - was chosen early in this
work, on a turn that was not formatted the way the model expects. Re-measured on a properly formatted turn: the reading in use
gives a peak of 9.837 and a gap of 1.314, the other half 7.795 and 0.627.

So the early choice holds, and it now holds on a prompt the model would actually be given rather than on a rough one. Worth
doing because a decision measured on a bad input is a weaker decision than it looks, even when it turns out the same.

### The Engram gate, re-measured the same way

The gate reading was chosen on the rough prompt, so it gets the same re-measurement as the half split: on a formatted turn the
plain similarity gives a peak of 9.837 and a gap of 1.314, its signed root 9.705 and 1.187. The choice holds, with a narrower
margin than before and the same token on top - which is what a real effect looks like when the input stops being rough, rather
than a measurement artefact that would have disappeared.

### The lookup norm on the new baseline

With the slab split re-chosen, the lookup norm is re-measured against the new baseline rather than the old one: leaving it in
gives a peak of 10.712 and a gap of 2.675, taking it out 7.917 and 0.253 - a gap ten times smaller, and a different token on
top. It is the strongest single item in this path, and now that is a statement about the formatted turn.

The pre-scale is the last of the five to re-measure; the write's position was already checked and neither reading moved the
top token there.

### The pre-scale on the new baseline

Fourth of the five to be re-measured against the baseline the slab change produced: with the pre-scale the turn gives a peak of
10.712 and a gap of 2.675, without it 9.107 and 1.269 - the same token on top and half the gap. So it earns its place, and this
one does not flip the way the slab split did.

Of the five, then: the gate and the lookup norm hold by a wide margin, the pre-scale holds by half, the slab split flipped, and
the write's position has still to be re-checked. That is a more useful description than "all five are verified", which the
earlier work could not honestly claim once the prompt turned out to be wrong.

### The five, re-measured on the formatted turn: the final tally

| item | with it | without it | verdict |
| --- | --- | --- | --- |
| lookup norm | 10.712 / 2.675 | 7.917 / 0.253 | holds, ten times the gap |
| gate from the similarity | 10.712 / 2.675 | 9.705 / 1.187 | holds |
| pre-scale | 10.712 / 2.675 | 9.107 / 1.269 | holds, half the gap |
| value from the last slab | 10.712 / 2.675 | 9.837 / 1.314 | **flipped** from the first slab |
| write at the block's start | 10.712 / 2.675 | 9.242 / 0.628 | holds, four times the gap |

Every one of the five was measured twice - once on the rough prompt that this session started with, once on a turn formatted
the way the template says. Four kept their answer and one did not, which is worth knowing in both directions: the work was not
wasted, and "measured once" was not the same as "measured".

The peak and gap on the formatted turn are now **10.712 and 2.675**, against 7.971 and 0.077 where the session started - and the
generated tokens are still fragments, so none of this is the whole answer.

### The two extra norms, on the formatted turn

Re-measured against the new baseline: with the norms after attention and after the feed-forward the turn gives a peak of 10.712
and a gap of 2.675; with neither, 8.808 and 0.026 - the gap all but disappears and both tokens on top change. That is the
strongest reading in the whole session, and unlike the slab split it does not flip.

The pattern across the re-measurements so far is that the structural items - the norms, the lookup norm, the write's position -
hold by a wide margin and the arithmetic choices that had two plausible readings (the gate, the slab split) sit closer, with one
of them turning out the other way. That is what a correct structure with one wrong detail would look like, and it points the
next measurement at the details rather than at the shape of the block.

## V2.1: what the quantiser's note asks of this build

The note from the quantization side arrived as a UTF-16 file, which is why it read as mojibake in a terminal; decoded, it says
the pack this session has been working with is **V1** and a **V2.1** is being produced. Its requirements, and where this build
already stands:

| requirement | this build |
| --- | --- |
| block type ids stay 144 to 150 with the same layout | already how the table is written; the tree's own ids are the remap |
| do not hardcode the per-tier expert counts (V1 160/96/256, V2 160/160/192, MTP 512) | the loader reads `hy4f.expert_split.*_offsets` and `*_ids` and derives every tier from them, so a different split needs no code change |
| Engram row counts change with the corpus's measured heat | `rows[t]` comes from each table's own shape, so it follows the file |
| new key `hy4f.layout_version`, `A` compatible and `B` not | **added this round**: a pack whose version is not `A` is refused with a sentence saying which version it is |
| `general.alignment` becomes 4096 | the reader takes the file's alignment, so this is a file-side change only |

The note also asks two questions of this side. The answer to the first is that a **CPU `vec_dot` for the CSQ, VQ and Engram
types does not exist yet** - the types have `to_float`, and the CPU path dequantises rows through it, which is why this pack
runs at a fraction of a token per second. That is now the highest-value piece of engine work rather than MTP or the frontend.
The answer to the second is that the meaning of `hy4f.expert_split` will not change: it stays "a per-block partition read from
the file", because that is what lets V2.1 arrive without a loader change.

### The order this puts the work in

1. verify what the V2.1 pack changes against this build (the table above, on the real file when it lands);
2. `vec_dot` for the five custom types, CPU first, then Vulkan kernels under the shader conventions;
3. Edge0's dense selection and PLE work, which the same quantization side is preparing;
4. MTP and the remaining frontend wiring.

That order is different from the one this session was following: numeric work on V1 continues to matter only where it also
applies to V2.1, since V2.1 replaces the pack.

### Where a `vec_dot` for these types goes

The five custom block types are **not in the CPU type table at all** - `type_traits_cpu` in `ggml-cpu/ggml-cpu.c` has no entry
for them, which is exactly why every multiplication of a Hy4 weight dequantises row by row through `to_float`. The table is the
place; a modern entry looks like

```
[GGML_TYPE_Q2_0] = {
    .from_float   = quantize_row_q2_0,
    .vec_dot      = ggml_vec_dot_q2_0_q8_0,
    .vec_dot_type = GGML_TYPE_Q8_0,
    .nrows        = 1,
},
```

and for this family `.vec_dot_type` is `GGML_TYPE_F32`, because the other side of these products is the residual stream in f32
rather than a quantised activation. The dot itself is a short loop: unpack the indices with the existing `hy4_unpack`, read the
codebook the pack set through `ggml_hy4_set_codebooks`, and accumulate `d * sum(codebook[i] * y)` - the same arithmetic the
dequantiser already does, without the intermediate row.

So the work is: four `ggml_vec_dot_*` functions beside the four `dequantize_row_*` in `ggml-quants.c`, four declarations, and
four entries in the table. The Engram types do not need one - they are only ever read by row lookups, which already have their
fallback.

It is worth saying plainly what this changes: it is the difference between reading a 512-value block through a 512-float
temporary for every product and reading it once. The measured effect will be shown on the same turn the rest of this work used,
and nothing about the arithmetic should change - a `vec_dot` that disagrees with the dequantiser is a bug, not a faster path.

### The dot products are written, and turning them on trips an assertion

Four `ggml_vec_dot_*` functions now sit beside the four dequantisers, each the same arithmetic without the row temporary, and
the type table has an entry shape ready for them: `vec_dot` set, `vec_dot_type` `F32`, no `from_float` because this build never
quantises one of these types from floats.

**They are not registered.** Doing so makes the run die in the first MoE block:

```
build_moe_ffn: il 1 tiers 3 up0 ... local0 ...
Assertion failed: i02 == -1 || (i02 >= 0 && i02 < n_as)
```

So the failure is not in the arithmetic - it is in how the expert path passes ids once a type has a dot product of its own. The
likely shape of it: the tier stacks are per-tier tables and the ids handed to `mul_mat_id` are the tier-local ones, which is what
the dequantising fallback tolerates and the direct path may not, because `n_as` there is the stack's row count while the id it
receives is compared against a different bound.

Until that is understood the functions stay unregistered, which is the honest state: written, not proven, and not in the way of
anything. The next step is to run one multiplication through the function and one through the dequantiser on the same block and
compare, then to read what `mul_mat_id` does with `n_as` and the ids once `vec_dot` exists.

### What the id assertion is about, and what is left to look at

`n_as` in that check is `src0->ne[2]` - the number of matrices in the stack - and the id comes straight from the ids tensor. For
a tier stack that means the bound is the **tier's** row count, while the ids we hand over are tier-local indices with `-1` for
the experts that are not in that tier; `-1` is allowed by the check, so the crash means some positive id is larger than its own
tier.

Three places can produce that and they are distinguishable:

- the `local` table's contents, if an entry was filled with a global id where a local one was meant;
- the per-block tier sizes, if a stack's `ne[2]` does not match what the `local` table was built for - the trunk's hot tier and
  the MTP block's 512 are not the same size;
- the shape of the table tensor itself, if a block's stack was loaded with a different `ne[2]` than the split says.

The check now prints the id, the bound, the stack's `ne[2]`, its type and the ids' `ne[1]` when it would fail, so the next run
that registers the dot products names the block and the number instead of only asserting.

Nothing else was changed: the fallback still runs and the reference turn is unchanged at 10.712 against 8.037.

### The expert partition was one block early

The id assertion that blocked the dot products turned out to be a real bug in plain sight. The split arrays carry **42**
partitions and the file has **43** blocks, because block 0 is the dense block and has no experts. So entry `i` describes
block `i+1`, and reading entry `i` as block `i` puts every block's partition one block early. The last entry gives it away:
it is `(512, 0, 0)`, the MTP block's hot tier, while block 41 is an ordinary 160/96/256 block.

| block | split (hot, warm, cold) | its stacks' shape |
| --- | --- | --- |
| 1 | 160, 96, 256 | 160, 96, 256 |
| 41 | 160, 96, 256 | 160, 96, 256 |
| 41 read with the old indexing | **512, 0, 0** | 160, 96, 256 -> local ids up to 511 |

That is why an id of 170 met a stack of 160 rows. With the offset applied, the same turn gives

```
top 1261 (8.340), then 12565 (7.590) -> 标 micros botÿ更好的一分
```

against `yg-` fragments before. The peak is lower than the 10.712 the wrong partition produced, and that is expected: a
sharper distribution over the wrong experts is not a better answer. The text is the evidence that matters, and it went from
fragments to words.

This is the first fault in this session that was a plain indexing error rather than a choice between readings, and it was found
by following the assertion rather than by another switch.

### The dot products are exact, and the path around them is not

The probe now compares each dot product with its own dequantiser on a fixed row, and all four agree to within a few times
1e-8:

```
csq8      dot product -0.311706 (dequantised -0.311706)  diff 2.0e-08
csq4      dot product  0.742862 (dequantised  0.742863)  diff 3.9e-07
vq4_128   dot product -0.006783 (dequantised -0.006783)  diff 3.0e-08
vq4_64    dot product -0.369360 (dequantised -0.369360)  diff 6.8e-08
```

So the arithmetic is not the problem, and registering the types still turns the answer into a repeated token (`220` six times)
while leaving them unregistered gives `标 micros botÿ更好的一分`. The difference is therefore in what surrounds the product - most
likely the shape the expert product returns, since the dequantising path and the direct one build their result differently and
everything after depends on that layout.

Both states are in the tree: the functions and the probe comparison stay, the registration is out. The next step is to print the
shape of one expert product on each path and compare, which is a smaller question than it was.

### The gate, after the expert partition was corrected

The partition fix changes which experts every block reads, so the choices measured before it are worth a third look. The Engram
gate was the weakest of them: on the corrected baseline the plain similarity gives a peak of 8.340 and a gap of 0.750, the
signed root 8.310 and a gap of 0.006 - the two tokens swap places and the gap all but vanishes. So it holds, and holds more
clearly than it did on the wrong partition, where the margin was one part in ten.

The lesson from the last three baselines is that a margin is only as good as the answers it was measured on: this one grew when
the experts became right, the slab split flipped when the prompt became right, and neither could have been predicted from the
number alone.

### The slab split, third baseline: the margin and the text disagree

On the corrected partition the first slab measures better and the last slab writes better:

| reading | peak / gap | eight tokens |
| --- | --- | --- |
| value from the last slab | 8.340 / 0.750 | `标 micros botÿ更好的一分 Jensen feeder` |
| value from the first slab | 9.191 / **1.401** | `标重复的ya_--` |

The peak and the gap favour the first slab by every number, and the text favours the last one by the only measure that reaches
the user. The last slab stays the default.

This is the first time in this work that the gap has pointed the wrong way, and it is worth stating plainly: a wider gap means
the model is more sure of its own next token, not that the token is more often right. For a choice that only changes a small
part of the computation, a sharper distribution can be a confident mistake. Decisions that reach the text should be settled on
the text, with the gap as a second opinion rather than the first.

### The other per-block arrays, and why only one of them needed the offset

After the expert partition turned out to be one block early, every other per-block array in the file was worth checking against
its length:

| key | entries | origin |
| --- | --- | --- |
| `expert_split.*_offsets` | 43 offsets, 42 partitions | blocks 1..42, because block 0 is dense |
| `attention.head_count` | 43 | blocks 0..42 |
| `attention.sliding_window_pattern` | 43 | blocks 0..42 |
| `swiglu_clamp_exp` | 43 | blocks 0..42 |
| `attention.indexer.types` | 42 | blocks 0..41 |

Only the first needed the offset. The indexer's array is short by the **last** block rather than the first - block 42 is the draft
block, its indexer is never consulted - and the contents say so: the positions where it holds 1 are 0, 1, 5, 9 and so on, which
are exactly the positions where `sliding_window_pattern` holds 0, a full-attention block. Two arrays that index the same blocks
line up at offset zero; the one that indexes blocks 1..42 does not.

So the shift is not a general rule about this file. Each array declares its origin by its length and its contents, and the
expert split was the one that lied about it.

### The head count really does vary by block, and the shapes follow it

`attention.head_count` is `[32, 32, 64, 64, 64, 32, ...]`, and the tensors agree with it rather than with a single value:

| block | `attn_q_b` | `attn_k_b` | `attn_v_b` | `attn_output` |
| --- | --- | --- | --- | --- |
| 0 | 1024 by 8192 | 192 by 512 by **32** | 512 by 256 by 32 | 8192 by 3072 |
| 2 | 1024 by **16384** | 192 by 512 by **64** | 512 by 256 by **64** | **16384** by 3072 |
| 5 | 1024 by 8192 | 192 by 512 by 32 | 512 by 256 by 32 | 8192 by 3072 |
| 6 | 1024 by **16384** | 192 by 512 by **64** | 512 by 256 by 64 | **16384** by 3072 |

So a block either has 32 heads or 64, and every width that depends on it doubles with it - the query projection's output, both
per-head K and V factors, and the attention output projection's input. The builder reads the count per block, which is the only
way this adds up; a single global value would have failed the shape check on the first 64-head block.

That is the last shape-level check in the attention worth writing down. Everything that varies between blocks - heads, sliding
window, indexer type, expert partition - is now known to vary and is read per block, with the one offset that the expert split
needed recorded above.

### 576 and 256 are both right, and they measure different things

Two key lengths sit in the file and they look contradictory until the projections are read together:

- `attention.key_length` is **576**, the width the score is taken over: 512 absorbed plus 64 rope;
- `attention.key_length_mla` is **256**, the width of one query head: **192** non-rope plus 64 rope.

`attn_q_b` produces exactly that: 8192 for 32 heads and 16384 for 64, which is 256 per head in both cases. `attn_k_b` carries
the **192**, not the 256 - it absorbs the non-rope part of the query into the 512-wide latent, which is where the 576 comes from.
So the numbers chain: 192 + 64 = 256 per query head, 512 + 64 = 576 per score, and the scale is one over the square root of the
score width, which is the reading that measured best.

The hyper-connection constants are read from the file as well rather than assumed: `hyper_connection.epsilon` is 1e-6 and
`hc_magnitude` is loaded beside the count, and both are used where the block builds its gates.

Nothing changed this round; these are checks that close two questions the shapes raised earlier.

### The bisection says dense, not expert

Registering **only CSQ8** - the type the attention and the dense block are made of - already turns the answer into
`220,549,220,220,220,220`, with 0.7 tokens per second instead of 0.4. So the expert path is not implicated: the difference shows
up as soon as one type takes the direct product in an ordinary matrix multiply.

That narrows the question to the two things a `vec_dot` has to agree with ggml about, neither of which the probe can see because
it calls the function directly:

- **the result stride `bs`** - a caller may space its results, so writing `s[r]` is only right when `bs` is the size of a float;
- **the row stride and contiguity of the other operand** - the direct path is only valid when the elements of a row are adjacent
  in memory, and a view can break that in a way the dequantising path tolerates.

The functions stay unregistered and the pack still answers `标 micros botÿ更好的一分`; the next step is to print `bs`, `bx`, `by`
and `nrc` once from the dense path and compare them with what the function assumes.

### The trace says the call convention is not the problem

With `STRATA_DOT_TRACE` set, the first calls to a dot product print their arguments:

```
[dot] type 0 ne00 12288 ne01 8 nr 1 bs 0 bx 0 by 0
```

`bs`, `bx` and `by` are all zero and `nr` is one, which is the case the four new functions are written for; the type shown is F32
because that is the first product in the graph, the hyper-connection mix. So the calling convention is not what breaks when a Hy4
type is registered, and the two candidates recorded last round are down to one: it is not `bs` or `bx` or `by`, it is somewhere
else in the branch that a native `vec_dot` selects.

The most likely such place is the conversion buffer: ggml decides what to convert `src1` into from `vec_dot_type`, and for these
types that decision changes from "nothing to convert" to "convert to F32" - a case that a f32 activation never exercises.

The trace stays behind the variable, and the registration stays out.

### The conversion buffer is a no-op here, and every row is a whole number of blocks

Two more candidates, both checked and both dismissed.

The conversion buffer turns out not to be involved: `wdata` is `src1->data` whenever `src1->type` is already `vec_dot_type`, and
the row stride used for the second operand is `ggml_row_size(vec_dot_type, ne10)`, which for F32 is simply `ne10` floats. So the
new types take the same route through `mul_mat` that F32 and F16 already take, and those work.

And no tensor of these types has a row whose width is not a whole number of 512-value blocks - the check over the whole file
returns zero - so the kernel is not silently truncating a partial block.

That leaves one difference worth naming: with the dot product absent, these products were being computed by the fallback written
earlier in this work, and with it present they are computed by ggml's own kernel. The two disagree, and the probe says the
arithmetic agrees, so the difference is in **which rows each of them multiplies** - the fallback was written to handle a row at a
time and may have been doing something the kernel does not, or the reverse.

The way to settle that is one product, both ways, on the same input: instrument the kernel to compute one row through the
fallback and through the dot, and print both. That is the next round's work.

### One product, both ways: the first sample is inconclusive because it is nearly zero

The kernel can now compute one product twice - through the dot and through the dequantiser - and print both, behind
STRATA_DOT_CHECK and limited to the Hy4 type under test. The first three rows it reports:

`
vec_dot  0.000001  dequant  0.000001
vec_dot -0.000004  dequant -0.000004
vec_dot -0.000002  dequant -0.000004
`

Two agree, the third differs by half of a number that is two millionths in size - the signature of cancellation rather than of a
wrong row, since a quantised row against a vector whose entries are around 1e-5 has no significant digits left. The check samples
the first product, which is the smallest in the graph, so it proves nothing either way.

The next version samples an up-projection or the output head and reports the largest difference over a row range. That is the last
thing between the registration and the 75 percent it measured.

### A large product disagrees on its first two rows and agrees on the rest

Sampling eight rows of a big CSQ8 product - `ne00 1024`, `ne01 8192`, the query up-projection - gives

```
row 0  vec_dot -0.002807   dequant -0.002386
row 1  vec_dot  0.003764   dequant -0.002386
row 2  vec_dot -0.001763   dequant -0.001763
...
row 7  vec_dot -0.001486   dequant -0.001486
```

Two things stand out. The first two rows disagree, and the second of them reports the **same** dequantised value as the first,
which cannot be right for two different rows of the same tensor. Rows 2 to 7 agree exactly.

A repeated value from a loop that advances a pointer per row points at the instrument before it points at the kernel: either the
row offset used by the check and the row offset used by the call are not the same quantity, or the first rows are being read before
the tensor is in the state the loop assumes. Rows 2 to 7 agreeing is the reassuring half - a wrong kernel would disagree
everywhere, not in two rows and then not.

So the next step is to print `ir0`, `iir0` and the byte offset beside the two values, and to confirm the check is looking at the
same row the call is. If it still disagrees after that, the kernel is the suspect; if it agrees, the discrepancy seen in the model
output is somewhere else again, and the instrument will have been worth its noise.

### With the row index printed, every sampled row agrees

Adding ir0, iir0 and the byte offset to the check fixes the disagreement: the eight rows now pair the same row in both
columns and agree to the last digit.

`
ir0 0    off 0       vec_dot -0.002807   dequant -0.002807
ir0 64   off 25088   vec_dot  0.003764   dequant  0.003764
ir0 576  off 225792  vec_dot -0.000245   dequant -0.000245
`

The earlier mismatch was the instrument: its counter was a plain static int touched from several threads, so the two numbers on
one line could come from different rows. One line of output about where the check was looking was worth more than the previous
three rounds of suspicion.

So the kernel computes these products correctly, and the arithmetic, the calling convention, the conversion buffer, the block
boundaries and the row mapping are all now dismissed by evidence. Whatever registering the type changes in the model output is
**not** in this product - the next place to look is what else the engine does around a multiplication that a native dot product
might cause to be skipped.

### The next test is a node-by-node comparison, and the trace has to be made to emit one

With every property of the product itself dismissed, the question is what changes **around** a multiplication when the type gains a
dot product. The cheapest way to answer it is to compare the two runs' graphs node by node: the first node whose value differs is
the one that names the cause.

`STRATA_HY4_TRACE` does not currently give that - it prints build-time markers, not evaluated values, and the run reports a single
line. The engine does have a dump of node results behind its evaluation trace, so the next step is to turn it on for the first
block in both configurations and compare the names in order. That turns "somewhere outside the product" into a named node.

Until then the registration stays out, the pack answers `标 micros botÿ更好的一分`, and the 75 percent that registering measured is
banked rather than collected.

### The first differing node is the attention's own product

The engine's evaluation dump prints a checksum per named node, and the same dump taken in both configurations can be compared in
order. Of the 2520 nodes the pattern `attn` matches, the first three are identical and the fourth is not:

| node | without the dot product | with it |
| --- | --- | --- |
| `attn_norm-0` | 0.00803676181 | 0.00803676181 |
| **`attn_kqv-0`** | **1.51521365** | **1.18083755** |

So the divergence begins exactly at the attention's output and nothing before it moves. Everything inside that edge is the
attention's own projections, which narrows this to the products `attn_q_a`, `attn_q_b` and `attn_kv_a_mqa` - and the one that was
sampled and agreed, `attn_q_b` with `ne00 1024`, is therefore not the one that differs.

The next sample is `attn_q_a`, whose row length is 3072 rather than 1024, and the projection into the cache, `attn_kv_a_mqa`,
whose output is 576 wide - a width that is not a multiple of the 512-value block, which the row-width check over the file did not
examine because it looked at the weights' input width rather than their output width. That is the shape most likely to expose a
difference between "one row at a time" and "one block at a time", and it is where the next check goes.

### What the draft step needs beyond the recipe: one input tensor

An attempt to wire the draft step's first half into the graph stopped on a detail worth writing down, because it is the only
thing between the recipe and working code: **the graph has no token tensor to look the next embedding up from**.

`build_inp_embd` returns the embedding, not the ids; the ids live inside the input struct it builds and are consumed there. So a
draft step that needs `get_rows(tok_embd, <one id>)` has to carry its own one-element `I32` input, registered on the result the
same way the tier tables are, with `set_input` filling it from the last token of the batch. That is a small, well-defined piece of
plumbing - an input struct with one tensor and one `set_input` - and then the projection chain, `hnorm` of the trunk's last hidden
state, `enorm` of that embedding, the concatenation hidden first and `eh_proj`, can be built as written.

The recipe and the shapes are already recorded above, so this note is only about the missing input. Nothing else in the draft step
needs engine support.

The edit was reverted rather than left half-wired; the tree is at the state where everything builds and the reference turn is
unchanged.

### Wiring the draft projection runs into an unallocated tensor

Building the projection with the new input gets as far as graph construction and then stops:

```
build_moe_ffn: il 1 tiers 3 ...
GGML_ASSERT(buf != NULL && "tensor buffer not set")
```

So one of the three `blk.42.nextn.*` tensors has no buffer when the graph is built. That is a different failure from the last one:
the input is now fine, and what is missing is that these tensors exist in the model and are **not part of any allocation the
scheduler plans**, because nothing used them before. A tensor that is created and loaded is not automatically given a place to
live; the scheduler gives it one when a graph needs it, and a graph that mentions it for the first time at the very end of the
build is past the point where the planner looked.

The fix is either to create these tensors before the plan is made - the usual way is to reference them during the build's first
pass - or to make the allocation cover them explicitly. Which of the two is right depends on how the scheduler in this tree
decides, and that is the next thing to read.

The edit is reverted; the tree builds and the reference turn is unchanged.

### The draft chain builds and evaluates; the token side is what crashed

Split into two halves, the two failures separate cleanly:

- the projection - `hnorm` of the trunk's last hidden state, the concatenation, `eh_proj` - **builds and runs** when it is made an
  output (`res->t_embd`), and the generated token is unchanged at `1261`, so nothing on the logits path is disturbed;
- the day's earlier crash came from the other half, the `get_rows(tok_embd, <one id>)` with the new one-element input.

So the geometry is confirmed by execution and only the token input is still wrong. The likely reason is ordering: the input tensor is
created and marked with `ggml_set_input`, then its owning object is moved into the result, and the allocator may be seeing the
tensor without a buffer - the same shape of failure as the earlier one, now with a specific suspect rather than a general one.

The probe stands behind `STRATA_HY4_MTP`, off by default, and with the flag off the reference turn is bit-identical to before.

### The token half resists three explanations

Wiring the lookup into the same probe crashes the run, and three explanations have now been tried and each of them failed to change
that:

- **the input itself** - the one-element class exists and is filled from the batch like the tier tables, so it is not simply absent;
- **the inverse transform** - the embedding table stores rotated rows and a second reader has to restore the primal basis the way
  the embedding input does, so that was applied exactly as `build_inp_embd` applies it, and the crash is unchanged;
- **the output assignment** - the same draft chain with the lookup removed runs fine as an output, so making it an output is not
  what breaks.

What is left is inside `ggml_get_rows` on this table with an input tensor created at the very end of the build, or in the ordering
between creating that input and the allocator taking its picture of the graph. That is where the next look goes; the token half is
reverted so the tree stays at the committed half-probe, which builds and runs.

Worth saying plainly: this is the third round on the draft step and it is not working yet. The recipe, the shapes and the projection
are settled and verified by execution - the lookup is the piece that is not, and it is one node, not a design.

### There is already an MTP graph in this tree, in the closest family to this one

Three rounds were spent building the draft step into the **decoder** graph, which was the wrong place. The framework has a graph type
for it - `LLM_GRAPH_TYPE_DECODER_MTP` - and `qwen4exp` implements one; `deepseek41` explicitly refuses it in a sentence. qwen4exp is
also the closest family here: same hyper-connections, same `n_layer_nextn == 1` convention.

What its `graph_mtp` does, and what this pack's draft step therefore needs:

| step | qwen4exp | for this pack |
| --- | --- | --- |
| select the graph | `build_arch_graph` returns `graph_mtp` for that type | same, one branch in `hy-v4.cpp` |
| the trunk's hidden state | an **input** (`h`) from `build_inp_embd_state`, filled by the caller | same - this is what the last three attempts were trying to build inline |
| the next token | an **input** `tokens`, then `get_rows(tok_embd, tokens)` | same |
| the block | `il = hparams.n_layer()`, the nextn block | same, block 42 |
| the cache | `build_inp_mem_hybrid()` plus an indexer pool, with the recurrent copy expanded by hand | hy4's DSA/SWA cache, same shape of call |
| the projection | `eh_proj` of `hnorm`/`enorm` | `blk.42.nextn.*`, as recorded |

So the draft step is **not** something to be spliced into the decoder graph at all: it is its own graph, selected by a type the
framework already carries, and there is a working example of it one file away. The three failed attempts share one root cause -
building it in the wrong graph - and the fix is to write `graph_mtp` for `hy_v4_flash` against this reference rather than to keep
patching the decoder.

That is the next piece of work, and it is now a matter of following a blueprint instead of inventing one.

### The reference corrects two details of the recorded recipe

Reading `qwen4exp`'s draft graph end to end fixes two things this document had guessed:

- **the concatenation order is embedding first, hidden second** - `ggml_concat(e_norm, h_norm, 0)`. The earlier note said hidden
  first. The reference is the authority, and the width agrees either way, so only the reference could settle it;
- **the hidden state that goes in is the collapsed one**, n_embd wide, not the hyper-connected streams: the projection's input is
  6144 for this pack, two times 3072, and 3072 is `n_embd`. qwen4exp repeats the normed embedding out to its stream count because
  its projection is wider; this pack's is not.

The rest of the structure, in the order the reference does it:

1. inputs: the tokens, and the hidden state from the previous step (`t_h_nextn`), both as graph inputs;
2. `hnorm` of the hidden, `enorm` of the token embedding, the concatenation, `eh_proj`;
3. the block at `il = hparams.n_layer()` - for this pack block 42 - built with the same attention and feed-forward helpers the
   decoder uses, against its own cache input;
4. the block's residual flattened and published as `t_h_nextn` for the following draft step;
5. the head: `shared_head_norm` and the shared `output` weight, published as `t_logits`.

So the recipe is now a paragraph of code with a reference implementation of every step, and the two guesses in the older note have
been replaced by what the working one does.

### The scaffold is already there and empty, and one refactor turns it on

`models.h` already declares `graph_mtp` for this model and already has `build_arch_graph` overridden - and both do nothing useful
yet: `build_arch_graph` returns `graph` whatever the requested type is, and `graph_mtp` has no constructor anywhere, so selecting it
would not link.

The reason is structural and now visible: `graph_mtp` is declared to derive from `llm_graph_context`, while every helper the draft
step needs - attention over a cache input, the block's feed-forward, the head - are methods of `graph`. qwen4exp solves this the
other way round: its `graph_mtp` derives from `graph` and passes a `no_build` tag so the base constructor does not run the decoder,
then builds the draft by hand.

So the one refactor that turns the scaffold on is the one qwen4exp already uses:

1. split `llama_model_hy_v4::graph`'s constructor body into a `build()` method and add a `graph(params, no_build{})` constructor that
   only initialises the context;
2. make `graph_mtp` derive from `graph` and call that tag, then build the draft with the same helpers the decoder uses;
3. select it in `build_arch_graph` for `LLM_GRAPH_TYPE_DECODER_MTP`, and keep returning `graph` for every other type.

That is a mechanical change with a working example, and it replaces the three attempts that tried to build the same thing inside the
decoder, where none of it belongs. The recipe itself is recorded above and corrected against the reference.

### The last prerequisite is one more extraction

The draft graph now has its base class and its constructor slot, and reading the decoder's block to write its body shows the same
pattern that worked for the Engram write: the block is inline in the layer loop, so a second graph that needs one block - block 42 -
cannot call it.

What the block needs from its surroundings, from reading the loop:

| piece | where it comes from in the decoder |
| --- | --- |
| the streams | `inpL`, four of them, n_embd wide |
| the hyper-connection gates | `build_hc_pre` on the block's own `hc_*` tensors |
| attention | `build_attention` or `build_attention_dsa`, chosen by `hparams.is_swa(il)` |
| the norms | `build_norm` with the block's `attn_norm`, `ffn_norm`, and the two post norms |
| the feed-forward | the dense path or the tiered MoE, by `il` |
| the Engram write | the closure built before the loop |

So the extraction is a method `build_block(int il, ggml_tensor * inpL)` with those six things either taken from the model by block
index or passed in, and the loop becomes a call. That is mechanical and testable by the reference turn staying at 1261.

Once it exists, `graph_mtp` is: inputs, the two norms and the projection, one call to `build_block(n_layer(), ...)`, the shared head.
Nothing else.

## State of the adaptation, second pass

The first table was written before the expert partition turned out to be one block early, before the dot products existed and before
the draft graph's scaffold was understood. This one replaces it.

### Working, with the evidence

| what | evidence |
| --- | --- |
| loads completely | every tensor the file carries is created and matched |
| runs on the CPU | 42 layers, block 42 excluded, a plan per step |
| runs on Vulkan | the same pack, exit 0, backends agree to 0.4 percent in the hidden state |
| reads the right experts | the split's offset is applied; the reference turn changed from fragments to `标 micros botÿ更好的一分` |
| reads every tensor it should | 1573 of 1661 in the forward pass; the 88 unread are block 42, the dormant indexer compressor and four host-side norms |
| the terminal accepts it | `strata launch --plan` prints a plan; the Control and Terminal tests pass |
| a V2.1 pack will not be mis-read | `hy4f.layout_version` other than `A` is refused with a sentence |

### Measured, and how strongly

| item | verdict |
| --- | --- |
| lookup norm | ten times the gap without it; strongest in the path |
| two post norms | the gap all but disappears without them |
| slab split | the numbers prefer the first slab, the text prefers the last; the text wins |
| gate from the similarity | holds, more clearly on the corrected partition |
| pre-scale, write position, gate nonlinearity, sinks, router scale and normalisation, head collapse, stream input, attention scale | each measured twice, each confirmed |

### Open

| what | state |
| --- | --- |
| coherence | still fragments; the remaining error is in a computation, not a missing term |
| `vec_dot` registration | the dot agrees with the dequantiser row for row; registering it makes the answer degenerate, and the bisection puts that in the dense path |
| the draft step | scaffold, recipe, shapes and base class in place; the block needs to be callable for one block first |
| expert residency | the tier stacks bypass the cache by construction; the remap tables already exist |
| long context | the indexer is dormant at these lengths and would matter at a thousand tokens and beyond |

The instrument for all of it is the same: a switch, two runs, and a comparison - of the logits peak and gap, of the generated text, or
of the per-node checksums, depending on which question is being asked.

### The draft graph's body compiles; the probe that would run it crashes

The body is written and the tree builds. To run it, a context has to ask for `LLAMA_CONTEXT_TYPE_MTP`, so a probe was added beside
the engine's existing throwaway warmup context: build one with that type, decode a single token, print the best logit.

It reaches the point where the draft context is created, prints the cache's own line, and then dies with an access violation -
before any of the draft graph's own markers, so the failure is in creating or first using the context, not inside the projection.

That is consistent with what the graph needs and nothing supplies yet: the trunk's hidden state. `build_inp_embd_state` reads an
input that the caller fills, and a context that has never been given one may be reading a null tensor rather than a zero one.

So the order of work is now: give the draft context a caller that supplies the trunk's hidden state and the draft token, then the
probe has something real to measure. The probe as written was reverted rather than left in the engine, since a crash behind an
environment variable is not a feature.

### A quality gate that needs no interpretation: `1+1=`

Every measurement so far compared two readings against each other, which says which is better but not whether either is good. One
prompt does say that, and it has a right answer a person can check in a second:

one start marker, the word user, a middle marker, the text 1 + 1 =, an end marker, then a start marker, the word assistant
and a middle marker. Its ids are `120000,3717,120001,16,10,16,28,120025,120000,611,10372,120001`. A working model continues
with `2`.

whose ids are `120000,3717,120001,16,10,16,28,120025,120000,611,10372,120001`. A working model continues with `2`.

This one does not. It produces ` freilt ambi重复的 Villa阿`, and with the Engram path switched off `antonurbsurbsikuturbsucer` -
both wrong, the second with a repeating fragment. So neither the Engram write nor its absence makes this model answer a sum, which
means the error is upstream of every choice this session has been comparing: those choices sit after the model has already gone
wrong.

That is worth stating as the summary of the current state: the adaptation reads the pack correctly, runs on two backends, uses the
right experts, and produces plausible-looking distributions - and it cannot add one and one. The next work is not another switch; it
is to find the computation that is wrong everywhere rather than in one place.

This gate belongs in the record because it is the cheapest test in it: one prompt, one expected token, no interpretation.

### The global suspects so far are applied, which moves the search on

With a gate that says whether any of this works, the candidates are now the things that affect **every** token rather than one term.
The first two checked are both applied correctly:

- `hyper_connection.magnitude` is read from the file and multiplied into the block's post gate, and the head's gate too;
- `hyper_connection.epsilon` is added to both gates, and the pre gate's sigmoid is the one the pack describes.

So a missing global scale is not the explanation, and neither is a missing epsilon. What remains as a global candidate is the **expert
lattice**: every block routes to sixteen of five hundred and twelve experts and adds a shared one, and if the routing or the latent
projection is wrong then every block is wrong in the same way - which is what a model that cannot add one and one looks like.

The next experiment is therefore to run the same gate with the routed experts bypassed and only the shared expert contributing. If the
answer changes at all, the lattice is where to look; if nothing changes, the error is in attention, the hyper-connections or the
embedding, and the lattice is exonerated.

### The routed projection is not swapped

The names read backwards but the arithmetic is right: `ffn_routed_down` is 3072 by 1536, so it takes the block input and compresses it
to the latent, and `ffn_routed_up` is 1536 by 3072, so it expands the expert output back. The block uses each for its own direction,
and the shared expert is computed on the block input rather than the latent, which is what the widths require.

So the neighbourhood of the double-add is clean. What is left to read there is the placement of the two post norms relative to the
hyper-connection write, and that is a question about order rather than about a missing term.

### Three more candidates, none of them the answer

Against the 1+1 gate, three changes that each affect every token:

| change | result |
| --- | --- |
| the stream reduce as a weighted average instead of a weighted sum | dostMechanical邻机械monaryturned |
| the windowed attention split switched off | readsurbsquber次级ernet |
| the Engram write switched off | antonurbsurbsikuturbsucer |

All wrong, and every one of them a different wrong. So the error is not in the reduce's normalisation, not in the sliding-window pairing
and not in the Engram write - and with the shared expert's double add already fixed, the block itself has no remaining candidate that a
switch can reach.

That leaves the attention's own arithmetic and the embedding-to-head path, which no switch has touched. Between those two the attention is
the larger and the less standard: an absorbed MLA with a 192-wide non-rope query, a 512-wide latent, per-head K and V factors and sinks.
The next thing worth doing is not another experiment but a read of that path against what the shapes say it should be - which is how the
double add was found.

### Reading the attention path, and one correction to an earlier claim

The assembly reads consistently with the shapes at every step:

| step | width |
| --- | --- |
| `wq_a`, norm, `wq_b` | 8192 for 32 heads, 16384 for 64 - 256 per head either way |
| split into non-rope and rope | 192 + 64 = 256 |
| absorption through `wk_b` | 192 into 512, per head |
| query for the score | 512 absorbed plus 64 rope = 576 |
| the cache's latent | `n_lora_kv` = 512, and its rope part 64 |
| value | the latent itself, 512, projected per head by `wv_b` to 256 |

The one correction: the scale is one over the square root of **256**, not 576. `n_embd_head_k_mla()` is the per-head query width, and that
is what the scale divides - the earlier note in this file said 576 and was wrong, though the number that matters was measured with the
value the code uses, so no decision rested on the mistake.

Nothing in the path is a second double add: the gate is applied once, between the attention output and `wo`, and the sinks are passed in
as an argument rather than added. So the attention's assembly is not where the error is, which leaves its **internals** - the absorbed
score, the softmax, the sinks - and the embedding-to-head path.

### The hyper-connection write is wired the way its own comment says

`build_hc_post` loops the streams and computes `residual_i + x * post_i` in f32 and casts back once, which is what the comment above it
describes. The four streams start as the same embedding and diverge only through their own gates, which is the point of independent
hyper-connections, and the reduce that feeds each sublayer mixes them with the pre gates. Nothing there is a missing term or a doubled one.

So the block, its constants, its wiring and the attention's assembly are all read and all consistent, and the gate still fails. The next
step is not another read of the same code but a **unit test of one block**: a single token, one block, a hand-checked expectation of what
the output should be from the tensors it reads. That turns read-and-compare-to-intent into compare-to-a-number, which is what has been
missing from this half of the investigation.

### The embedding path, node by node, against the reference

The graph dump names every node and its sources, so the embedding can be compared with the reference implementation step by step:

```
node_0          GET_ROWS   [3072,1,1,1]   token_embd.weight, inp_tokens
embd            MUL        [3072,1,1,1]   (the rotated lookup) prism.hadamard.signs.3072
hc_init         REPEAT     [3072,4,1,1]   embd
```

Two things this settles. The lookup is rotated and **then** multiplied by the signs, which is exactly the order `qwen4exp`'s token
embedding uses - so the inverse transform is applied the way the working reference applies it, not merely applied. And the four streams
start as that same embedding repeated, which is what the reduction and the gates assume.

So the embedding-to-head path is not where the error is either, and it is now checked the same way the block and the attention were: by
reading, against a reference that works. What is left is the attention's **interior** - the absorbed score, the softmax over it and the
sinks - which is the one piece of this graph that has no counterpart in the reference family and no switch that isolates it.

### The bare continuation is wrong too, so it is not the chat format

The same question without any markers - the four tokens for 1 + 1 = and nothing else - gives alphpol一则avax. So the model does not
continue a plainly worded sum either, and the turn markers are not what stands between this pack and a right answer. That closes the last
non-arithmetic explanation.

The graph dump now covers the embedding, the block, the attention's assembly and the head, and each matches the reference that works or
the shapes that constrain it. What has never been compared against anything is the **rotation itself**: every folded weight is multiplied
through a hadamard transform whose packing is read from the pack, and the only check on it so far is our own graph walker, which was built
from the same reading. If that reading is wrong, every folded weight is wrong in the same way, which is exactly the signature this gate
keeps showing.

A unit test of one rotation is the next thing worth doing: take a small folded weight, apply the transform the way the loader sets it up,
and compare against a hand computation from the pack's own sign values and widths. It is bounded, it needs no engine run, and it is the one
piece of the path that has no independent check.

### The rotation's own specification, read at last

The pack states the transform in six keys, and this file had never recorded them:

| key | value |
| --- | --- |
| `prism.hadamard.transform` | `normalized-sylvester-walsh-hadamard` |
| `prism.hadamard.block_size` | **512** |
| `prism.hadamard.axis` | `input-last-dimension` |
| `prism.hadamard.sign_mode` | `explicit` |
| `prism.hadamard.sign_widths` | 512, 1024, 1536, 3072, 6144, 8192, and two more |
| `prism.hadamard.inverse_weight_names` | `token_embd.weight` alone |
| `prism.hadamard.weight_names` | 849 folded weights |

So the transform is a normalised Walsh-Hadamard over each 512-value block of the **input** axis, with an explicit sign vector per width -
which is exactly the shape of the implementation: the signs are applied to the activation, then the transform, and the block size the
pack names and the block size of the custom types are the same 512.

The inverse pairing also checks out on paper: for a forward transform of `WHT(signs * x)`, the inverse is `signs * WHT^-1(y)`, because a
diagonal sign matrix is its own inverse. The embedding path does exactly that - the transform first, the signs after - and the forward
paths do signs first, transform after. The two orders are each other's inverses, which is what makes the embedding a consistent reader of
the same tables.

Nothing changed this round; what is recorded is the specification the implementation has been matching without it being written down.

### Where this stands, honestly

The gate is a fair question and the pack fails it in every configuration tried. A summary of the search, so the next attempt does not
repeat it:

**read and found consistent** - the block's structure, its constants, the routed projection's direction, the attention's assembly and its
per-block head widths, the hyper-connection write and reduce, the embedding's rotation order against the reference, the rotation's own
specification from the pack's six keys, and the inverse pairing on paper.

**found and fixed** - the expert partition read one block early; the shared expert added twice per MoE block; the terminal's block-type and
architecture limits; the layout-version gate.

**tried and wrong, each a different wrong** - Engram on and off, the reduce as sum and as mean, the windowed split on and off, the attention
gate on and off, flash attention on and off, the chat format and the bare text, the head collapse gated and averaged.

**never compared against anything** - the attention's interior: the absorbed score, the softmax over it, and the sinks. It is the only part
of the graph with no counterpart in the reference family and no switch that isolates it.

That is the shape of the remaining problem, and it is worth saying plainly: the adaptation reads the pack correctly, runs on two backends,
serves through the terminal and produces plausible distributions, and it cannot add one and one. Everything a person can check on the way
is in place; what is missing is a comparison for the one piece that has none.

### The attention call's arguments land in the right slots

Checked against the declaration rather than by eye: the parameters are wo, wo_b, wo_s, the query, key and value, kq_b, the sinks and then
v_mla before the scale, and the call passes nullptr for the three output-side ones, the three tensors, nullptr for kq_b, the sinks, wv_b,
the scale and the block. So the sinks and the value projection are not swapped, and their shapes agree with what the declaration asks for:
the sinks are n_head wide (32 or 64 by block, matching the pack's head counts) and wv_b is 512 by 256 by n_head, which is
n_embd_head_v_mla by n_embd_head_v by n_head_v.

That leaves the interior - the absorbed score, the softmax that the sinks join, and the weighted sum - as the only unverified piece, and it
is inside the framework's builder rather than in this model's code.

### The attention's interior, read

The framework's builder does, in order: the softmax with the scale and the mask, then the sinks added to the normaliser, then the value
matmul, then the value projection, then a permute and a reshape that put the heads back in order.

| step | what it works on |
| --- | --- |
| `soft_max_ext(kq, mask, kq_scale)` | the absorbed score, 576 wide against 576 |
| `soft_max_add_sinks(kq, sinks)` | the per-head sink logits |
| `mul_mat(v, kq)` | the 512-wide latent value, shared across heads and repeated by the matmul |
| `build_lora_mm(v_mla, kqv)` | per head, 512 into 256 |
| permute and `cont_2d` | heads into the order the output projection expects |

Each of those is the operation the MLA description calls for, and the value is shared rather than per head, which is what passing the latent
as the value means. So the interior is not where the error is either - which leaves the **expert gating**, the one quantity in this graph
whose meaning is a number: the pack says `expert_gating_func` is 2, and what 2 means has to be read from the enum rather than assumed.
That is the next thing to check, and it is small.

### The gating number, and what a zero means

The pack writes expert_gating_func = 2, and the tree forwards it to a switch with four cases: softmax, sigmoid, raw weights and the
square root of a softplus. Which number is which is not written where the switch is, so the value was made overridable for a run and two
of the candidates tried:

| value | result |
| --- | --- |
| 2, the pack's own | fails the gate |
| 1 | fails the gate, with different text |
| 0 | aborts in the switch's default branch, so 0 is not softmax |

So the enum's numbering has to be read from its definition rather than assumed, and the override stays for that. What the abort already
rules out is the simplest guess: whatever this enum starts at, it is not zero, which means the pack's 2 may well mean something other
than the raw-weight case it lands on today - and if it does, every expert weight in every block has been wrong, which is the shape this
gate keeps reporting.

### Correction: the gating enum, read from its definition

The enum is in `llama-hparams.h`: 0 none, 1 softmax, **2 sigmoid**, 3 raw weights, 4 the
square root of a softplus. So the pack's 2 is sigmoid, which is what the code has been passing all along; the note above that said 2 was
the raw-weight case came from the Vulkan shader's own defines, which number the same cases from zero for the GPU side and are translated
in C++ before they reach it. The gating is right, and the gate still fails.

### The one thing never verified is the pack itself

Every check in this session compares the code against a reading of the format, and some of them compare it against a reference
implementation in the same tree - but all of those readings and references were written from the same source: this document's
understanding of what the pack means. A pack that is itself wrong, in a way the format description does not describe, would pass every
one of those checks and still produce garbage.

The quantiser's fork was compared for exactly that reason: `hy-v4.cpp` and `hy4flash.cpp` in `Hy4-Works/engine/Strata-vk` differ from this
tree only by the fixes made since the copy was taken - the tier offset, the layout gate, the draft graph, the gate switch - so their side
has not solved anything this side has not, and the comparison adds nothing.

What would settle it is a **reference continuation from another implementation** on the same pack: one prompt, one answer, from a build
that is not this one. If that answer is also wrong, the pack is the problem and no amount of reading this tree will find it. If it is
right, this tree's difference from that build is the bug, and it is a finite diff rather than a search.

That is the next thing to ask for, and it is cheap: the prompt ids are recorded above and the answer is four tokens long.

### The codebooks are handed over in the right order

One more way the pack could be read wrongly without any check noticing: the four codebook arrays could be passed to the quantiser in the
wrong order, in which case every tensor of one type would decode with another type's table, and the probe - which sets them correctly
itself - would still pass while the model produced garbage.

They are not: the loader reads prism.csq8.codebook into c8, prism.csq4.codebook into c4, prism.vq4_128.codebook into 128 and
prism.vq4_64.codebook into 64, and calls ggml_hy4_set_codebooks(c8, c4, v128, v64) - the same order the probe uses and the same
order the signature declares. The widths are checked against what each table must hold (8, 4, 128 times 4 and 64 times 4 floats) before
the call.

So that too is excluded, and the list of things that could be wrong without any local check noticing is down to the pack's own contents.

## Where each objective item stands

**V2.1 pack** - the loader side is ready: the split is read from the file so a different tier layout needs no code, the Engram row counts
come from each table's shape, and a layout version other than A is refused. Nothing more can be done until a V2.1 file exists.

**The custom types' `vec_dot`** - the four functions are written and agree with their dequantisers to a few times 1e-8, and the call
convention is confirmed by trace. Registering them makes the answer degenerate, and the bisection puts that in the dense path rather than in
anything the probe can see. Parked with both states in the tree.

**Edge0** - not started. Its dense selection is a choice between pack variants, and the terminal currently carries a fixed 5 GB estimate for
the dense part rather than a per-variant one, so the work begins with the catalogue rather than with the engine.

**MTP** - the draft graph's body is written against the qwen4exp reference and builds; running it needs a caller that supplies the trunk's
hidden state and the draft token, which nothing does yet.

**Expert residency** - not started. The tier stacks bypass the cache by construction and the remap tables that a cache-aware path would need
already exist.

**Coherence** - blocked on something that no local check can see, with two ways forward recorded: a reference continuation from another
implementation, or one tensor from the BF16 source to check the pack against.

### The wider terminal run, and the one failure that is the machine

Running the terminal suite beyond the focused set gives 44 passed, 7 subtests passed and one failure:
`test_archive_gaps.py::HipPackageForAnotherCard::test_a_package_without_the_cards_arch_is_not_chosen_and_one_with_it_is`. Its message is
not about the code under test: this machine is windows-arm64, and the HIP path refuses that combination outright - "hip is not a supported
combination on windows-arm64" - before it ever reaches the architecture check the test exercises. So the failure is the platform rather than
the change, and it would pass on an x86_64 Windows host.

The two interactive walk tests are omitted for the same kind of reason: they drive a real terminal through an asyncio loop and wait on a
completion port here. The focused set for the modules this session changed remains 23 of 23.

### What caching the hot tier would take, in calls

The cache's interface turns out to fit a tier better than the guard suggests: `build_lookup` takes **one layer's** up, gate and down tensors
and the selected experts, and returns slots for the rows it holds plus a host fallback for the misses. A tier **is** such a stack - the hot
tier is 160 or 512 rows of up, gate and down - so the call shape is:

1. `llama_moe_cache_layer{ il, tiers->up[0], tiers->gate[0], tiers->down[0], nullptr, weight_before_ffn }`, with `selected_experts`
   replaced by that tier's **local** ids, which is what the tier's own tables already produce;
2. `build_lookup` for that layer, and `build_experts` with the same up/gate/down and a build callback for the hot stack - the same callback
   the uncached path uses, only with the tier's tensors;
3. the warm and cold tiers unchanged and uncached, since their masks are zero for the hot ids and the sum is over tiers.

Two things make this safe to try and one makes it necessary to try behind a switch:

- safe: the ids are already local to the tier, so the cache's id-to-row mapping means the same thing there as for a whole stack;
- safe: a model whose layers are all tiered never takes the old path, so the cache's per-layer slot map has one owner per layer;
- necessary: if the mapping were wrong the rows would silently come from the wrong experts, which is a wrong answer rather than a crash -
  so it goes behind a variable and the reference gate decides, rather than a default.

That is the next piece of engine work, and it is now a matter of two calls rather than a redesign of the cache.

### The memory-budget knob is wired end to end

Checked rather than assumed, because the objective lists it as a deliverable: the terminal documents --expert-cache in its parameter
table, its adoption path passes it to the engine, its drafting model converts a slot count into MiB, and the engine reads
expert_cache_mib into the loader, computes the cache size from the RAM available when the flag is uto, and hands that to llama's MoE
cache. So the knob reaches the thing it is supposed to move.

What it does not reach is a layer whose experts are tiered, and that boundary now says so at runtime - which makes the pair of facts
consistent: the budget works where the cache works, and the places it does not apply are announced rather than implied.

### The draft step's caller already has an API

The missing caller turns out to be two calls the tree already has. A decoder context can be asked to publish its last hidden state -
set_embeddings_nextn(true) before the decode and get_embeddings_nextn() after it - and a context whose type is MTP consumes a state through
the batch: the graph's h input is filled from ubatch->embd_state, which a batch extension sets, and asserts exactly that when it is missing.
That assert is the crash the probe saw two rounds ago, in the right place.

So the flow for a first real draft is: decode the trunk with embeddings_nextn on, take the state, then in an MTP context decode one token with
both its id and that state. The draft graph written against qwen4exp already reads both, and block 42's expert partition is now kept.

That is the next piece of work, and it is now a sequence of named calls rather than a design question.

### The two contexts are meant to be linked on the device, not through the host

The batch API settles how the draft step is fed, and it is better than copying a hidden state through host memory:
`llama_set_ctx_state_src(mtp_ctx, trunk_ctx)` points the draft context at the trunk's nextn buffer, and then a batch entry can say that its
state is row r of that buffer with `llama_batch_ext_set_embd_state_row`, while its token id comes from the ordinary token call. The draft
reads the trunk's own device buffer, so nothing is downloaded and re-uploaded at all.

That also explains the earlier probe's access violation: it had none of those, and the graph asserts a state embedding when the batch
carries none.

So the caller is three calls and a batch, and it is fully specified now: create the MTP context, link it to the trunk, decode the trunk
with nextn embeddings on, then decode one token through a batch that names its state row. The draft logits come out of the usual getter.

### The draft context crashes while it is being built, before any graph

With the calls in place - trunk with nextn embeddings on, the state linked, one token whose state is row zero - the run dies while the MTP
context is being **created**, not while it runs: the last line printed is the KV cache's own message from the new context, and none of the
graph's markers appear. So this is a third distinct failure, after "no caller" and "no state in the batch", and it happens before the draft
graph exists at all.

That points at what a second context on the same model has to set up rather than at the draft graph: memory parameters, the nextn buffer's
size, or something the MTP path expects the caller to have configured first. The probe was reverted rather than left behind, since a crash
behind a variable is not a feature, and the next look is at the context's constructor for that type rather than at the model.

### Where the draft context's construction stops, and what to read next

The last line before the crash comes from the new context's own KV cache, so the failure is inside **memory creation for an MTP context**, not
in the model and not in the graph. Two things there are worth reading before another attempt:

- the model's memory for this pack is a DSA-plus-windowed pair; a second context built with the MTP type asks for one of those again, and the
  pairing has its own invariants about sizes and about which layers it covers;
- supports_state_row() gates the whole state-row path on the context type being MTP **and** on a source context being linked whose nextn
  buffer has the width the graph expects - and the width comes from the model's nextn block, so an unset n_layer_nextn would make that
  check fail rather than the constructor crash.

So the next look is at the memory the constructor creates for that type, with the layer count and the window sizes printed once, rather than
at the draft graph.

### The draft context dies inside its own construction, after memory

With the probe printing each step, the sequence is unambiguous: the trunk context is created, the trunk decode returns 0, the draft context
starts to be created, the new context's ISWA cache prints its line - and nothing returns. So the crash is inside \llama_init_from_model\ for the
MTP type, after memory, and before the context exists to be used.

That is as far as reading this tree can go without the next fact: which allocation after the cache is the one that fails. The candidates in
that window are the nextn buffer (whose size depends on the model's nextn block count), and whatever the context sets up for reading a source
context's rows. The probe was reverted again rather than left behind.

### Two of the construction window's candidates are guarded, and what is left

The nextn device buffer is not allocated in this flow: the allocation returns early unless embeddings_nextn_dev is set, and the context sets
that false for this path. And the source-context pointer is read through a guard rather than directly - supports_state_row() returns false
while no source is linked, and the one place that dereferences the pointer checks it first.

So neither of the two candidates in the window between the cache and the return explains a crash there, and guessing further is what this
session has learned to stop doing. The instrument that would answer it is the library's own log level: raising it should print the last step
the constructor reaches before it dies, and that is one environment variable rather than another probe.

### The library log prints nothing past the cache line

Raising the library's log level for that run adds no line between the draft context's cache message and the crash: the last thing printed is
still that message, from the constructor's memory step. So the failing step is one the library does not log, which is why several rounds of
guessing have gone nowhere.

At this point the instrument that would answer it is a stack trace rather than another print or another probe: the crash is an access
violation inside a constructor, and a debugger would name the frame in one run. That is a tool this session has not used, and it is the next
step for this particular question rather than any more reading of the source.

The probe was reverted again; the tree is at the state where the decoder works and the draft graph builds but is never entered.

### Asking the draft context not to publish a state does not help either

The draft context was created with embeddings_nextn off, on the theory that publishing a state it never produces was the trigger. It crashes
the same way. So the trigger is not that flag, and the two flags the probe has tried - publishing nextn embeddings and setting the context
type - are both needed for the flow and neither is the cause on its own.

Three attempts have now been reverted and the tree stays where the decoder works. For this one question the next instrument has to be a stack
trace; printing has taken it as far as it goes.

### The stack is capturable, the symbols are not there yet

A vectored exception handler around the draft context's creation prints the faulting address and sixteen frames as module-relative offsets,
so the crash is now a stack rather than a guess. The frames cannot be named, because this build produces no PDB: llvm-symbolizer resolves
every offset to question marks, which matches a release build with no debug information.

The next step is therefore a build with symbols - RelWithDebInfo, or the two flags that produce a PDB - and then the same probe and the same
symbolizer. That is a build-configuration change rather than a code change, and it is the only thing standing between the stack and the
function that faults.

The probe and the handler were reverted; the tree is unchanged.

### A symbol build is blocked on a compiler bug, not on the tree

Configuring a RelWithDebInfo build of the same engine in its own directory gets as far as compiling and then stops in ggml:

`
FAILED: ggml-cpu/arch/arm/repack.cpp.obj
fatal error: error in backend: unknown codeview register Z3_Z4
`

That is clang-cl's CodeView writer failing on a file full of ARM intrinsics with debug information on - a toolchain defect, not something
this tree can fix by changing code. The options for the next attempt are to keep debug info off for that one file, to try /Z7 instead of /Zi,
or to resolve the captured offsets against a linker map instead of a PDB.

The scratch build directories were removed so the tree stays as it was.

### The map resolves addresses, and the stack backtrace was the handler's own

Linking with /MAP produces a symbol map the size of the binary's code, and looking up the captured offsets in it works: each address lands
inside a named function with an offset. So a stack can be read without a PDB, which closes the toolchain question.

The capture itself was wrong, though: CaptureStackBackTrace called from inside a vectored exception handler returns the **handler's** stack,
not the faulting thread's - which is why every frame resolved to the engine's own load function and the C runtime. The faulting state is in
the exception's ContextRecord, so the fix is to read the program counter and walk the frame chain from there, then resolve those addresses
with the same map.

That is a small, mechanical change to the probe, and the tool that resolves whatever it prints now works.

### The fault is in ggml-base, called from llama

With the handler reading the exception's own context and naming the module of every frame, the crash resolves to a place rather than a
region:

`
pc   ggml-base.dll+0x26ec
lr   ggml-base.dll+0x7ce8
f0   llama.dll+0x961ec
f1   llama.dll+0x226e6c
...
f7   strata-vk.exe  (the probe)
`

So the access violation happens inside ggml-base, at a very low offset - an early function in that module - called from llama's context
construction. That is consistent with the draft context allocating or touching something ggml-base owns before it exists, and it rules out
the model code, the graph and the Vulkan backend, none of which appear in the chain.

Naming the exact function needs symbols the map does not carry: it lists public symbols, and an early static in ggml-base is not one. That is
the same wall as before, but now with the module and the offset in hand - and a function that begins within the first ten kilobytes of
ggml-base is a short list to read by hand.

### The map for that module carries no code symbols

Two facts from the map itself correct the previous round's arithmetic: ggml-base's preferred base is 0x180000000 rather than 0x140000000, and
its map has no function symbols at all - only the image base and four guard entries. So the earlier resolution against it was measuring the
wrong place, and the offset 0x26ec cannot be named from this map.

What the stack still says is useful: the fault is inside ggml-base, very early in the module, called from llama's context construction. Naming
it needs a map that includes internal symbols, which the linker only writes with debug information - and that is the same wall the
RelWithDebInfo attempt hit, though now narrowed to one module.

An alternative that avoids the toolchain entirely: the same call can be found by reading llama's context constructor for the ggml-base calls
it makes, since the frame below the fault is in llama and the function is one of the first in ggml-base - a short list either way.

### The fix moved the fault into our own constructor, which is progress

Re-running with the state read from the input moves the access violation: it is no longer in ggml-base but at
llama_model_hy_v4::graph_mtp::graph_mtp **+0x2b4**, a symbol the map names because this one is in llama.dll. So the null tensor that
ggml_is_contiguous_0 was choking on is gone, and a second one remains inside the draft graph's own constructor.

The candidates there are the tensors it touches in order: the block's nextn weights (created, and present in the file), the token embedding,
the token input we make, and the state input the framework fills - which is the one that can be null, since it depends on the model exposing
an output width and a nextn block. An assertion on that tensor before use, with its width printed, would name it in one run.

The probe and the handler were reverted again; what stayed is the fix to the state source.

### The draft constructor gets as far as its attention inputs

Step logs inside graph_mtp narrow the second fault to one statement: the constructor prints that it began, the two widths and the block count,
the state input's address - which is not null - and then that the attention inputs are next, and stops. So the crash is in building those,
which for this model takes the DSA branch because the pack sets an indexer top-k, and then the windowed pairing when the memory is the
hybrid kind.

The two facts printed also close a question from earlier rounds: n_embd_inp and n_embd_out are both 3072 and n_layer is 42 with nextn 1, so
the widths and the block split were never the problem.

The logs stay - they are one line each and only ever run for a draft context - and the probe was reverted again. The next step is to print the
two booleans that choose the branch and try the plain attention input if the paired one is what faults.

### The draft takes the wrong attention-input path, and the reference shows which one

The branch log answers it: the draft context has a memory, the pack sets an indexer top-k, and the memory is the paired kind, so the graph
calls build_attn_inp_k_dsa_iswa - which asks the cache for block 42's rows, and block 42 is not one of the cache's layers. That is where it
stops.

The reference family's draft graph does something else, and reading it is what this should have done first: it calls build_inp_mem_hybrid,
takes the hybrid context from that input, and then walks its own attention with the per-layer helper, allocating the recurrent input even
though the draft has no recurrent layer. In other words the draft builds its **own** memory input rather than borrowing the decoder's
per-layer builder, which is what makes block 42 addressable at all.

So the next change to this graph is structural and small: replace the borrowed attention input with the hybrid-input build the reference uses,
and index the draft block the way that helper expects.

### The attention input builder completes, and the fault is past it

Logging the paired builder's own steps shows that all of them finish - the DSA input, the windowed object, its key indices and its mask -
for every decoder layer and for the draft one. So the builder is not where the draft dies, which corrects the previous round's reading: the
last log line is the mask, and the next statement in the frame is the object's construction, whose own log I placed at the wrong one of that
builder's two call sites and therefore never saw.

That is the state of this thread: the draft graph's first two crashes are fixed (the null state and the mismatch is gone), the attention input
for the draft is built, and what remains is one statement - the combined input's construction or its registration - whose fault has not been
named yet. The instrumentation was reverted; what stays is the fix and the logs inside the model's own constructor.

### Two more nulls, and one of them is the draft block's own weights

The tensor log finally printed, and it names the next layer of the problem: the token input and the state input are both non-null after the
move is fixed, and the draft block's own weights - eh_proj, enorm, hnorm - are all null. So block 42 exists in the file, the tensor table
creates its four nextn tensors, and the layer the graph looks at has none of them.

Two fixes went in for that: the token input is captured before the input object is handed to the result (reading it afterwards was a null
dereference, which is what the last crash was), and the state was already captured. What remains is why the draft layer's tensors are empty -
whether the layer array stops at the decoder's block count, or the table's names for block 42 do not match the file's.

The cheapest way to tell them apart is the loader's own report: it prints how many tensors it read and which ones it could not place, and a
draft block whose four tensors were never created would show up there rather than in the graph.

### The draft block's weights are in the file, in the table, and empty in the layer

Three facts that should not coexist: the pack carries all four `blk.42.nextn.*` tensors with the shapes the table declares, the table's loop
for the draft block runs from the decoder's layer count to the file's block count, and the layer array is sized to that same count - yet the
graph sees three of the four as null while the fourth, the shared head norm, is not.

So one of the three assumptions is wrong in a way that a reading cannot settle: the loop may not reach 42, the names may not be the ones the
loader looks for on this block, or the layer the graph holds is not the one the table filled. The cheapest way to separate them is a log
inside the table's loop for the draft block, printing the name it asks for and whether the loader answered - three lines, one run.

That is the next step; everything else in the draft constructor is now either fixed or known non-null.

### The residue is the draft block's missing hyper-connections

The new fault resolves completely, because every frame is in llama and ggml-base names itself: ggml_mul_mat, called from build_hc_pre, called
from graph_mtp. So the block's hyper-connection pre-mix is being handed a null weight - and the reason is in the tensor table: the
hyper-connection tensors are created only when block_has_hc(i) is true, and that override returns false for the draft block, which is what
the comment beside it says (a pack may leave the hyper-connections off a trailing block).

What does not follow that rule is the block builder, which calls build_hc_pre unconditionally. So the last piece of the draft path is a branch
the block builder is missing: a layer without hyper-connections must take a plain pre-norm residual path instead of the pre-mix and the
post-mix, exactly as its tensors say it should.

That is now a small, bounded change with the evidence behind it: the weights are absent by design, and the builder is the only place that has
not been told.

### The draft context now says why it fails

The failure stopped being a crash and became a message: llama_init_from_model reports that it could not initialise the context because of an
invalid unordered_map key. That is a lookup with at() on a key that is not there, thrown during the initialisation that follows the draft
graph's build - which is progress of a specific kind, because the reason is now stated rather than inferred.

The first candidate was the per-layer device lookup, which does use at() and has no entry for a block past the decoder's layers; it is now
guarded to answer with the output device for such a layer. That did not change the message, so the lookup is elsewhere in the same
initialisation - the remaining maps there are keyed by tensors rather than layers.

### The draft block asks the cache for a slot it does not have

Catching the C++ exception instead of the access violation resolves it: the throw comes from llama_kv_cache::cpy_k, called from the draft
graph. So the draft block asks its context's cache to copy a key row for a layer the cache has no slot for - block 42, one past the layers the
cache was built for - and the lookup throws instead of returning.

That is not a defect in the lookup; it is the shape of the thing: a draft block needs its own memory, which is exactly what the reference
family's draft graph builds - it calls build_inp_mem_hybrid on the draft context rather than borrowing the decoder's per-layer inputs. Two
guards went in along the way and stay, since they are right regardless: the per-layer device lookup answers with the output device for a layer
past the decoder's, and the attention-rotation table skips a width it has no entry for rather than throwing out of a constructor.

So the next piece is the draft context's memory: it must cover the draft block, and the graph must build its attention inputs from that.

### The cache leaves the draft block out on purpose

Reading the filter settles it: the layered cache's MLA slot is built with the predicate that a layer belongs to it only below the decoder's
layer count, and the comment beside it says why - the trunk, excluding an MTP block. So a draft block has no cache slot **by design**, and
asking that cache for one throws; the throw was a symptom, not a bug in the lookup.

That means the draft graph must not borrow the decoder's attention inputs at all. What it should use instead is the draft context's own
memory, which is what the reference family's draft graph builds - and whether this fork offers the equivalent for the paired cache is the next
thing to read: the alternatives are a memory builder for the draft context, or a cache-less attention input for the single draft token.

### The draft block's mixture has one tier, and now says so

The next null was the draft block's own experts: the pack gives it a hot stack and nothing else, while the tier loop walked all three and
handed build_lora_mm_id null weights for the two it does not have. A tier whose three stacks are absent is now skipped rather than
dereferenced, which is the same reading the split itself gives - the block's partition is all hot.

With that the draft graph gets past its mixture and stops on a reshape whose element count does not match, which is the next thing to name:
the assertion is ggml's own, so the graph is now well past the point where a tensor was missing.

### The reshape that fails is not the rotation's

The assertion after the mixture is ggml_reshape_2d element-count, and the rotation's input builder is ruled out first: it only reshapes when
the weight is in the rotation table and its tiling factor is above one, and a tier stack is a slice of a name the table lists as a whole, so
the builder returns the input untouched for it. The remaining reshapes on the draft path are the two the block's plain path introduces - the
pre step widening the single stream to the sublayer's input and the post step narrowing it back - and both should agree with the block's own
tensor widths.

The instrument that names it is one line per reshape: print the tensor's four extents and the target's before each call. That is what the
last several rounds did rather than reason about it, and it worked every time.

### The draft's state travels on the host, and the batch now reaches evaluation

Two facts from the last probe: the trunk does publish a nextn state row as 3072 floats on the host, and the batch that names a device row
instead is not the one this configuration can use - the trunk keeps the row on the host, so the batch must carry it. With the host form the
evaluation is no longer refused as an invalid batch: it builds the draft graph a second time, for the run rather than for the reservation, and
stops inside the paired attention input's build.

So the shape of what is left is narrow: the same builder that reserves fine at context creation is what the evaluation trips over, with a real
batch behind it rather than an empty one.

### The plain residual now adds two shapes that agree, and a buffer is still missing

The reservation pass builds the draft graph with the batch's token count, not one, so the plain post step was adding a flat sublayer output
to a residual carrying a stream axis. Both sides are now given the same four-dimensional shape. The buffer assertion that follows is not that:
it survives the fix, and it says a node reached the scheduler with no buffer at all.

The draft block's weights are created by the tensor table and the cache now has a slot for the block, so the candidates are the tensors the
graph builds for it - the rotation inputs, the cache inputs, the embedding rows. Naming them is the next step, and the cheapest way is to
print, right after loading, which of the draft block's tensors has no buffer; a table that says all of them do would move the question to the
graph's own tensors instead.

### Every draft weight has a buffer, so the missing one is the graph's

The check is direct: eh_proj, enorm, hnorm, the shared head norm, the embedding table and the output weight all report a buffer. So the
assertion is not about a weight - it is a tensor the draft graph builds, and the candidates are the ones the graph creates for itself: the
token input, the state input, the cache's key indices and mask, and the rotation inputs the embedding read applies.

That narrows it to a handful and they differ in one way worth using: the input objects are the ones the framework fills from the batch, and the
others are built by the model. Printing which of them has no buffer at the point the graph is handed over would name it, and the graph is
reachable from the model's side of the call.

### The buffer assertion only appears on the path that inverts the embedding

Turning the inverse transform off with a switch makes the assertion disappear - but it also makes the context fail to be created, because the
same verification that demanded the transform refuses the read without it. So the experiment bounds the fault without isolating it: the
assertion is somewhere on the path the inverse transform takes, and it cannot be measured off that path while the verifier is in the way.

The next read is therefore the inverse transform itself: what tensors it builds, whether any of them is a view of one the scheduler has to
place, and how the decoder's own use of the same helper differs from this one. The helper is shared, so the difference is in the arguments
rather than in the helper.

### The inverse transform is shape-safe, so the assertion is about placing it

Read in full, the helper takes the rotation's width as the row length, reshapes the input to that width, multiplies, and reshapes the result
back to the input's own four extents. With a full-width rotation and a two-dimensional embedding row that is a plain reshape and a plain
matmul - nothing in it can leave a tensor without a buffer.

What it does do is mark the product with a hint, and that hint is what the scheduler acts on when it places the node. So the difference
between the draft graph and the decoder's use of the same helper is not the arithmetic but the placement: the decoder's graph was the one the
context's fused-op probes ran over, and the draft graph is not. Reading how the hint is placed, and whether a backend that cannot place it
leaves the node unallocated rather than falling back, is the next step.

### The assertion is a buffer type asked of nothing

The frame is ggml_backend_buffer_get_type, and it asserts on a null buffer: something asked a tensor for its buffer and then asked that
buffer for its type. The same failure appears on both backends, so it is not a placement decision either - it is a tensor that reached the
scheduler with no buffer at all, and the scheduler then queried it anyway.

That is the same sentence as two rounds ago, now with a precise frame: the question is which tensor. The answer is one log line in the
scheduler's leaf handling, printing the tensor's name when its buffer is null, and it is the next thing to add.

### The null buffer is asked for while the batch writes into the graph

Naming the frames moves the fault out of the scheduler and into the context: the query comes from llama_context::decode(const
llama_batch_ext&), so it happens while the batch's inputs are being written into the freshly built graph, not while nodes are being placed.
That is a different question from the one the last rounds asked, and it fits the state input being a plain graph input rather than a weight.

So the next read is decode's input-copy path: which tensor it asks for a buffer type, and why that one has none at that point - a graph input
written before its split was allocated, or an input the graph does not list.

### The state-source guard is right, and the assertion is somewhere else in decode

The support check did ask a buffer-less source for its buffer type, and it now answers no instead: a source that keeps its rows on the host
has no device rows to offer. That guard is right and stays, but it did not remove the assertion, so the query that fails is a different one in
the same call - decode writes several inputs and only one of them is the state.

The next instrument is the same stack print with the tensor's name attached: the copy path knows which input it is writing, so naming it there
costs one line and ends the guessing.

### The state-row check is called often and never sees a source

Logging its inputs settles it: supports_state_row runs many times during the draft's evaluation and every call has no source context at all,
so it returns before it can ask any buffer for its type. The assertion therefore comes from a different query in the same call, and the next
step is to repeat the stack print now that the surrounding code has changed - the offset that named decode last time will name it again, and
the two guards added since then have narrowed what can be left.

### The query is in decode and not in either support check

Repeating the stack print after the two guards gives the same caller at a slightly later offset - decode, called straight from the engine's
probe, with nothing between them. Both support checks in decode are excluded now: one returns immediately for a model that is not the draft
architecture, and the other was logged returning on a null source every time it ran. So the query that fails is a third one inside decode's own
body, and the next step is to list every place decode asks a buffer for its type rather than assuming which of the two known ones it is.

### The cache rotation guard is right and is not the one

The rotation setup does ask a cache tensor's buffer for its type without checking it, and a layer whose tensors were created after the buffers
were sized has exactly such a tensor; that guard now skips it. The assertion survives it, so the failing query is another one.

At this point the only efficient instrument left is to mark the two places decoder asks this question in llama-context and see whether either
marker prints: if neither does, the call is inlined and the answer needs a build without inlining rather than more reading.

### Accounting for memory is not where it should crash either

The memory bookkeeping walks the buffers it holds, and a buffer that failed to allocate is held as null so the failure surfaces where it
happened; the walk now skips it rather than asking a null buffer for its type. The assertion survives that too, so it is a third guard in a
row that is right in itself and does not move this particular failure.

That is the signal to stop guessing callers: the offsets say decode, decode is large, and the next useful step is line-level information for
one translation unit. Compiling llama-context.cpp with /Z7 on its own - the object that failed before was a ggml-cpu file, not this one - and
linking with debug info would let the same offset resolve to a line instead of a function.

### Line-level information, and the line it gives

Debug information for one translation unit works: with /Z7 on llama-context.cpp alone and a /DEBUG link, the build produces a six-megabyte
PDB and llvm-symbolizer resolves the offset that used to be a function name into a line. The line is the call to process_ubatch inside decode,
which is the statement that builds and runs the graph - so the null-buffer query happens somewhere inside that, not in decode's own body.

The instrument that fits this is now cheap: process_ubatch has four stages - build, allocate, write the inputs, compute - and a marker before
each names the stage in one run. With line information in place, the next marker should be the last one needed for this assertion.

### It is the cache input write, and it is the rotation tensors

The context's own trace names the stage in one run: the draft graph is built (201 nodes), its buffers are allocated, and then the input
writes begin - embedding, positions, and then the paired cache input, where it stops. So the null buffer belongs to something the cache input
writes, and the only tensors there that the draft block touched are the attention rotations: the rotation table is built over the layers the
cache knows, and the block we added to the cache after that point has rotation tensors with no buffer behind them.

That also explains why skipping them in the rotation setup did not help: skipping the write leaves the tensor exactly as unbacked as before.
The fix belongs one level up - the block's rotation tensors must either be backed like every other layer's or not created at all.

### The windowed cache input is where it stops, and its tensor is the one without a buffer

Two markers inside the paired input settle the order: the DSA half completes and the windowed half does not. That half writes the cache's key
indices into the layer's own key tensor, so the tensor it needs is the cache tensor of the block we added - and that tensor has no buffer,
because the cache sizes its buffers from the layers it knew when the context was created, before the block joined it.

The fix therefore belongs in the cache's own sizing rather than in the input write: the block must be part of the layer set the cache allocates
for, exactly as the filter change already made it part of the set it serves.

### The cache never creates a layer for the draft block

The layer loop runs over every block the file has, but it skips any block the model says has no KV - and the draft block is one of those, by
the same reckoning that keeps it out of the decoder's cache. The filter change therefore never had anything to act on: the block is dropped
one step earlier. Letting a block past the decoder through that check is a behaviour change without evidence behind it and was reverted.

What the next attempt needs is a table rather than a guess: after the cache is built, print every layer it holds with the index, the two
tensors and whether each has a buffer. That says in one run whether the draft block is present at all, and if it is not, the change belongs
where the layer set is decided rather than where the tensors are allocated.

### The key-index tensor has no buffer when the windowed half writes it

The write that stops is the cache's own set_input_k_idxs, and its first assertion is that the destination is a host buffer. So the tensor
exists - the write was reached - and has no buffer behind it, which means the scheduler never allocated it. Combined with the count the trace
prints at that point (four inputs in the draft graph against the decoder's dozens), the likely shape of the fault is that the windowed input
object's tensors were not part of the graph's input list when it was built, so nothing allocated them.

That is a different question from the cache's layer set, and it can be answered the same way: print the input list, with each tensor's name and
whether it has a buffer, right before the writes begin.

### Guarding the write moved it to the next write, with a clearer message

The plain cache input wrote its two inputs without checking either, while the paired form checks both; it now checks both too and logs when it
skips. The refusal moved from the key-index write to ggml_backend_tensor_set's own assertion that a tensor has a buffer - so the same class of
fault, one tensor later, now with a message that says exactly what is wrong instead of a null dereference deep inside a copy.

Which fixes the plane rather than the symptom: several of the draft graph's input tensors are reaching the write stage with no buffer, so the
question is why the scheduler left them unallocated rather than which write notices first.

### The paired input was the wrong form for a block that is not windowed

The last crash was a shape of the same mistake as the cache filter: the paired attention input builds two halves, and a block that is not
windowed never uses the windowed one, so those tensors are not part of the graph and nothing allocates them - then the input write asks for
their buffers. The draft now picks the input by the block's own windowing, exactly as the decoder does per layer.

With that, the draft context no longer crashes: the run completes, and the draft context comes back null without a message. That is the state
to work from - the crash is gone and what remains is a construction that refuses for a reason it has not stated.

### The refusal is about the context type, and it says nothing

Two contexts are asked for in the same configuration, one after the other: the draft one (type MTP) comes back null and the ordinary one comes
back a real pointer. So the model, the memory and the size are all fine; what is refused is the MTP type itself, and it is refused without a
message - which is why several rounds of looking for an error line found none.

That narrows the next read to the constructor's own handling of that type: what it requires that the others do not, and where it can return
without saying why.

### The two ways it can refuse, and neither one logged

Reading the entry point narrows a silent null to two branches: a check that warns and returns when the requested type is MTP but the model has
no MTP layers, and a try/catch around the constructor that logs the exception and returns. Neither message appears in the run, and the same
entry point builds an ordinary context from the same parameters a moment later, so the parameters themselves are fine.

Two readings remain: the check is firing and its warning is being dropped before it reaches the log, or the constructor is throwing something
the catch does not handle. Marking both branches with a direct write to stderr - not through the log - separates them in one run.

### The exception is cpy_k asking for the draft block's slot

Marking the refusal branches gives the exception and the map resolves every frame: the throw is llama_kv_cache::cpy_k, called from
graph_mtp while the context is being created, and the lookup it throws on is the layer map - the draft block has no entry in it, because the
cache skipped that block before the filter ever ran.

So the one change that is actually needed is to let a cache hold layers the model does not list: the draft context should ask for a cache
that covers its own block, and the check that drops the block belongs to the model's layer set rather than to the cache's loop. That is a
small constructor change and it is now the only thing between this path and a draft that runs.

### The filter is now allowed to name a layer the model does not

The cache admitted a layer only when the model gave it KV, and asked the filter afterwards; the filter is the place a context says which
layers it wants, so it now comes first and can admit a layer the model leaves out. That is the shape the draft block needs, and it changes
nothing for a cache with no filter.

The draft context still comes back null, so either the filter the draft context passes is not the one that was changed, or the throw is from
the other cache in the pair. Printing the layer count each cache ends up with, for the draft context alone, says which.

### One cache has the draft block and the other does not

The counts settle it. The trunk's three caches hold twelve, thirteen and thirty layers; the draft context's hold thirteen, thirteen and thirty
- so the first two gained the block and the third did not. The third is the windowed one, and a draft block that is windowed reads and writes
through it, which is where the lookup throws.

So the same change has to reach the windowed half of the pair: its layer set is decided by a different filter from the one that was changed,
and that filter is the next thing to read - it is the second, smaller version of the same mistake rather than a new one.

### It is the value write, not the key write

Two functions sit next to each other in the cache - one writes keys, the next writes values - and each asks the layer map for its slot. Only
the first was instrumented, and it never reported a miss, while the exception still comes from the second: so the block is missing from the
cache that receives **values**, and the key path is satisfied by a cache that has it. The two writes go to different caches when the layer is
windowed, which is the same split the layer counts showed.

That makes the change concrete rather than exploratory: the cache that takes the values has to know the draft block, and it is one filter away
- the same one that was changed for the other half of the pair.

### The miss is layer 42 in a cache of thirty, and it is the key write

Guarding every lookup at once names it: the miss is the draft block, in a cache that holds thirty layers - the windowed one. So the block is
windowed, the draft graph takes the paired input for it, and the half that writes is the windowed cache, which does not have the block. The
same change that made the other two caches grow has to reach this one, and there are two reasons it might not have: the filter the windowed
half receives may not be the one that was changed, or the check before the filter still wins in the path this cache is built through.

### The filter does say yes; the cache that matters is built somewhere else

The filter reports its decision: for the trunk it rejects the draft block, and for the draft context it keeps it - twice, which is the two
caches that consult it. So the filter is not the reason the thirty-layer cache lacks the block; that cache is created through a path that does
not ask this filter, or asks a different one, and it is the one the draft block's keys are written to.

The next instrument is the same decision log placed in the other filter - the one for the indexer layers - so both are compared on the same run
rather than one at a time.

### The draft block is not windowed, and the cache it writes is the big one

One line inside the block settles the branch: for the draft block the sliding flag is zero, the DSA flag is one, the windowed input is null
and the DSA input is not. So the attention writes through the DSA cache - and that is the cache with thirty layers, not the thirteen that
grew by one. The counts were read in the wrong order: the cache that gained the block is the windowed one, and the cache that takes the
block's keys is the other.

So the filter change landed on the half that did not need it, and the half that does needs the same treatment in its own filter: the one
built from the DSA predicate rather than the sliding one.

### Both halves now decide correctly, so the remaining refusal is a new one

Logged at the point where the pair chains its filters, the draft block is accepted by the half that writes it and rejected by the other:
the DSA half keeps it, the windowed half does not, and the trunk rejects both. That is exactly the partition the block needs, and it means the
lookup that threw earlier should no longer throw - the layer is now in the cache it writes to.

The current run still ends with the draft context coming back null, so what stops it is something after that. The next capture is the same
refusal marker once more, on the current build: the exception it prints will name the new obstacle rather than the one just removed.

### The filter accepts it and the cache still does not have it

Two measurements that contradict each other: the chained filter logs a keep for the draft block on the half that writes it, and the write
still reports no slot for that block in a cache of thirty layers. So either the cache that is written is not the one built by that filter, or
the filter's answer is not what the layer loop acts on.

The next measurement is inside the cache's own construction: print the layer count and whether it contains the block, at the end of the
constructor, for the draft context only. That separates "built without it" from "built with it and asked incorrectly".

### The pair does hold the block; the write goes to the other half

Read at the end of the pair's constructor, the numbers are unambiguous: the attention half holds twelve layers on the trunk and thirteen on
the draft, and the block is in it; the indexer half has it too. So the cache that should receive the draft block's keys does have it, and the
thirty-layer cache that reports the miss is the **windowed** half - which the block does not belong to at all, and which correctly refused it.

So the write is being routed to the windowed half for a block that is not windowed. That is a routing question inside the attention build, not a
layer-set question, and it is the last one on this path: the graph chooses an input for the block, and the input's half decides where the keys
go.

### The write is never reached, so the throw comes from earlier in the block

A print placed at the cache write itself never runs for the draft block. So whatever throws during the reserve is not the block's attention
write - that code is never reached - and the earlier attribution to cpy_k was the nearest named symbol, not the call. The draft graph stops
somewhere before the write, in the same build that reaches the block's mixture and its norms.

The next step is a marker per stage of the block for the draft layer alone, which is the same instrument that has worked at every other step of
this path.

### The block stops inside its indexer

Two markers inside the block and one inside the DSA attention name the stage: the hyper-connection pre step completes, the attention function is
entered, and the last thing reached is the line before the indexer build - which runs for this block because the pack marks it as an
indexer-owning layer. So the draft graph does not fail in its attention write at all; it fails in the indexer, one step earlier, and that is
the call whose nearest symbol was cpy_k.

The indexer writes keys of its own into a cache of its own, and the block's membership there is the next thing to check - the same question as
before, for the third cache.

### The indexer writes into the lid cache, and that is the one that must hold the block

The function that stops writes its key through the indexer cache - the lid half of the pair - and then reads that same cache back for the
score. So the cache that must hold the draft block is the lid one; the constructor log said that half holds thirteen layers with the block in
it, which means either the pointer the write uses is not that half, or the half the draft context builds is not the one this constructor made.

The next measurement is at that call: print which half the input points at, and its layer count, before the write. That is one line and it ends
the last ambiguity on this path.

### The indexer runs, finds a cache, and that cache is not the one the constructor reported

The pointer log shows the draft block entering the indexer build and a lid cache behind it, so the call is real and the cache is not null. What
it does not settle is identity: the constructor log counted thirteen layers with the block, and the write reports thirty without it. Two caches
can satisfy both statements, and the one the indexer receives is evidently the second.

The next measurement is the pair itself: print the lid pointer at construction and the lid pointer the input carries, in the same run. Same
number means one cache and a stale count; different numbers name the cache that has to learn about the block.

### The indexer uses a lid cache that none of the constructed pairs owns

Printed before and after the constructor, the difference is stark. The draft context's pair holds the block in both halves: twelve plus one
layers in the attention half and thirteen in the indexer half, both reporting the block present. The indexer build then asks for its lid cache
and gets a pointer that matches none of the four pairs constructed in the run.

So the cache the indexer writes is a fifth object: created later, owning no draft block, and never seen by the constructor log. That is the last
thing to explain on this path - where that object comes from and why the input carries it.

### The paired builder is the right one, and it exposes the unused half

The pointer evidence says the DSA-only input is built against a paired memory, so the context it carries is not the one the graph needs. Asked
for the paired input instead, the draft gets past that point and stops on a write into a tensor with no buffer - the windowed half's tensors,
which a block that is not windowed never uses and therefore nothing allocates. So the choice was between a correct context and an unbuffered
half, and the real fix is both: the paired builder for the context, and its unused half left alone.

The guard that does that already exists for the plain input; the DSA half needs the same one, and then the write has nothing left to trip over.

### Guarding every write in both inputs leaves the assertion standing

With the DSA half guarded like the plain one, no input write can run against an unbacked tensor, and the assertion is unchanged. So it is not
the input writes: the same assertion is produced by the copies the scheduler makes while running the graph, and what it objects to is a node
whose tensor has no buffer at compute time.

That moves the question from the inputs to the graph's own nodes - the tensors the model builds for the draft block, the likely one being a
result the graph writes but nothing consumes.

### The tensor with no buffer is the expert-tier table

Named directly, the write that trips is into expert_tier_local: one row per expert, an input the mixture of experts reads to look up each
expert's tier. It is created by the MoE builder and written like any other input, so the question is registration rather than content - an
input the graph builds for a layer it also uses, but which the scheduler never allocated.

That is a small, checkable thing: whether the tier input is added to the graph's input list on the path the draft block takes, or only on the
one the decoder takes.

### The draft step runs

The tier input was the last thing standing: it writes one table per tier, and the guards it had covered null tensors but not unallocated ones -
the draft block's tiers are resolved at build time, so its tables are not part of the graph and nothing allocates them. With the same check
the other inputs in that file already use, the draft graph evaluates end to end: the probe reports the trunk publishing its state row, the
draft consuming it, and a top token out of the shared head.

What that number means is limited, and it is worth saying plainly: the trunk's own answer is still wrong, so the draft's is a measurement of the
mechanism rather than of the model. What is established is the path - the state, the projection, the block, the head - which is what the
objective asked to confirm.

So the MTP item of this work is done: the draft graph builds, runs, and produces a token through the shared head, and the plumbing that feeds it
is the plumbing the decoder uses.

## Wiring the draft into the engine

The mechanism is confirmed; what remains is to use it. The engine already carries `--mtp <file>` for a **separate** draft model and
`--npu-mtp` for where such a model runs, but this pack's draft block lives inside the pack itself, at the index the decoder's layer count names -
so it needs its own switch rather than that one.

The shape of the work, in the order it pays off:

1. a switch that is set when the model reports a nextn block, so the difference is discoverable rather than automatic;
2. after a decode, ask the trunk context for its state row, run one draft step through the second context, and record the token it proposes
   next to the token the trunk actually produced;
3. an acceptance count over a run - proposed versus confirmed - which is the number that says whether the draft is worth its cost, and the
   only honest measure until the trunk's own answers are right.

Two things make this cheaper than it sounds: the probe the last rounds used already performs steps two and three by hand, and the state transfer
is the device-side link the framework provides, so the engine does not need to move hidden states through the host.

What it must not do is change what the decoder produces. Speculative decoding is only sound if the trunk's output is the output, so the draft
step has to be a measurement first and a speedup second.

## What the existing types already have on the CPU, and what the new ones do not

Asked whether the work here is limited to the new types, the answer from the type table is no: everything the existing packs use is already
accelerated, and only the block types this port added are not.

| type | CPU dot kernel | converts the other side to |
| --- | --- | --- |
| Q2_0 | yes | Q8_0 |
| Q3_K | yes | Q8_K |
| Q4_0 | yes | Q8_0 |
| Q4_K, Q5_K, Q6_K | yes | Q8_K |
| Q8_0 | yes | Q8_0 |
| IQ2 family | yes | Q8_K |
| F8_E4M3 | yes | F32 |
| F32, F16 | native, widened to F32 | - |
| **the Hy4 block types** | **none** | - |

So a Q3_K or Q4_0 path - what Edge0 asks for - is not a place this port needs to work: those are the fastest paths in the backend already. Two
extras sit on top of them and are worth knowing: a tiled K-quant matmul that takes over when the batch is large enough (its gate lists Q2_K
through Q6_K plus the IQ family, so a Q3_K batch is covered), and a per-expert variant of the same for a mixture-of-experts matmul.

The Hy4 block types have neither a dot kernel nor an entry in the type table; they are usable because two fallbacks were added to this tree -
one in the dense matmul and one in the mixture's - which dequantise a row and multiply in F32. Those are the wall this objective names, and they
are also why the model runs at all: without them a block type would abort instead of being slow.

### A kernel that is the fallback still changes the output

The most direct experiment available: replace the registered dot's body with exactly what the fallback does - dequantise each block with the
same routine and dot in f32 - and register that. It cannot be numerically different from the path it replaces, and the output still changes:
the top token moves and its logit rises by most of a point. So the difference is not arithmetic in the kernel at all; it is what the framework
does differently once a type has a kernel.

That leaves two candidates and they are separable: either the fallback reads the type in a way the kernel does not (its own dequantisation
versus the row routine), or the framework routes a type with a kernel through a path with different inputs. The probe that already compares
kernel and dequantiser can be pointed at the fallback's own output to settle which.

### The kernel is used, and it still disagrees with the fallback

A kernel that returns zeros drives every logit to zero, so the framework really does route this type through the registered kernel. A kernel
whose body is the fallback's own arithmetic (dequantise each block, dot in f32) still produces a different answer from the fallback. Both cannot
be right, and the fallback's code is a correct dequantise-and-dot in double, so the difference has to be in the arguments: what the caller passes
as the weight row, the activation row, and the destination.

The kernel-argument fix went in for the row stride - the caller advances both sides when it asks for several rows - and changed nothing, because
this type is asked for one row at a time today. So the next comparison has to be made inside one run rather than across two: compute the same
product both ways with the debug hook that already exists and print both numbers, on the args the framework itself passes.

### The apparent disagreement was the diagnostic, and the scatter is where to look next

At full precision the comparison shows a handful of calls that differ - and one of them repeats the previous call's exact value, which is the
tell: the check keeps its scratch in a static buffer shared by every thread, so the number it prints is often another thread's row. The calls
that agree do so to a part in a million-million, which is the precision of a double dot over dequantised data.

So the kernel and the fallback agree on the arguments the framework passes, and the difference in the model's output must come after the dot:
the kernel branch scatters the per-row results from a thread-local array into the destination, and that scatter is the one thing the fallback
does not share. That is the next thing to read, with the check's buffer made per-thread so the instrument stops lying.

## Layout B, digested from the quantiser's handoff

The next pack is layout **B**, and it is now the release format: the A backbone has been moved aside for local comparison and the shipped
directory contains only B. That reverses what this build assumed - the loader refuses anything but A - so the gate and the reading it guards
both change.

**What B is.** Same key/values as A except `hy4f.layout_version = "B"` and `general.alignment = 4096`. The four block types keep their codes
and lose their per-block `f32` scale: the codes are a contiguous, 16-byte-aligned plane of 192, 128, 112 and 96 bytes per 512 weights, and each
quantised tensor `X` gains a companion `X.scale` of type F32 with dimensions `(nb, rows)` where `nb = X.dims[0] / 512` and
`rows = prod(X.dims[1:])`, laid out row-major as `scale[row*nb + block]`. File type ids 151 to 154 name those four, and the loader maps them to
internal ids above the ones already in use.

**What the quantiser verified.** Codes and scales are byte-identical to A's for all 850 quantised tensors, and a full decode gives the same
logits to `0.0` with the same top token. So B is a **repacking**: if this build reads it correctly, its output must equal A's exactly.

**The trap they hit, which applies to us.** Their first scale indexing derived the row from the `k` a call passes, and some calls pass half a
row (512 values); the row then came out wrong. They fixed it by recording each tensor's full row geometry when the scale is bound and mapping
`(row, block-in-row)` from that. The rule stated in their note is worth repeating: **a B kernel must not assume `k` equals a full row.**

**The five things they hand back.**

1. Their changes live only in `third_party/llama.cpp`; the overlay is stale and the next apply would overwrite them. Porting them into
   `port/llama-overlay-vanilla/` is the first job, and it is a diff between two trees rather than a rewrite.
2. `llama-perplexity` crashes on this model in both layouts, after the context reports the draft block being rejected by both cache filters,
   with a `tensor not allocated` assertion in a tensor read - their guess is that a context with the draft block switched off still has the
   cache refer to that block's tensors. That is a real bug on the generic path and it is ours to fix.
3. The engine's own `--ppl` crashes at chunk 32 and above (chunk 8 and 16 are fine), which points at a micro-batch boundary.
4. On the CPU both packs produce repeated tokens through `--prompt` and through synthetic ids, and they ask for this build's gate - the
   template, the BOS/EOG handling and the parameters - to be the standard entry for gating and perplexity. That matches what this record has
   been saying for a while: the CPU path is what the gate must measure, and it is currently the only path where the answer is known to be wrong.
5. Their split offsets carry a leading duplicate zero (44 entries) where the reference has 43; they offer to change the writer. This build
   already reads the partition from the file, so both forms have to work here - the 44-entry form must be detected rather than assumed.

**What this means for the objective.** "Refuse layout B" was right while B did not exist. With B the release, the work is to read it, and the
gate is that its output matches A's - which is a much stronger check than the coherence gate, because it compares two files rather than two
opinions.

## One layout only: v2.1 digested

The quantiser has settled the format question. There is now **one** layout, the one this record called B, and the released pack carries **no
layout field at all** - the key was removed from the file header, with the data section untouched, and both the reader and the model are to stop
looking for it. The old inline-scale layout is being deleted, along with the packs that use it, in a few hours.

**The contract, in one place.** Quantised tensors use file ids 151 to 154 for the four block types - 192, 128, 112 and 96 bytes of codes per
512 weights, 16-byte aligned - and every one of them has an `X.scale` companion of type F32 with dimensions `(nb, rows)` where
`nb = X.dims[0]/512` and `rows = prod(X.dims[1:])`, indexed as `scale[row*nb + block]`. Engram rows keep ids 43, 148, 149 and 150 in the second
file, alignment is 4096, and a pack is 2511 tensors.

**What their fork now does, which this tree must match.** File ids 151-154 map to the four types and 148-150 to the three Engram row types;
**144-147 are no longer mapped**, so an old pack fails at the shape check instead of being read as something it is not. The loader's layout flag
defaults to true, every quantised tensor binds its scale once its data is in place, and the tensor count always excludes `*.scale`. The model
performs no layout check at all.

**What is left as dead code on purpose.** The inline-scale types at 147 to 150, their structs, and their dequantisers and dot kernels are now
unreachable. They can go, but only after confirming nothing else refers to the structs; the `_B` suffix on the names is a naming question with
two answers, the cheap one being a rename with the ids kept.

**The verification being asked of us, and its two parts.** First, byte equality: a script compares the old pack's code fields and scales against
the new pack's, tensor by tensor, and expects zero mismatches. Second, end-to-end: the same prompt through both packs, dumping logits, expecting
the two dumps to be **identical** - the quantiser measured a difference of exactly zero and the same top token, and confirmed the engine is
deterministic by running one pack twice.

**The two bugs that come with it, and neither is about layout.** `llama-perplexity` crashes on load on this model in either layout, after the
context reports the draft block being rejected by both cache filters, on a tensor read of a tensor that was never allocated - the guess being
that a context with the draft block switched off still has the cache referring to that block's tensors. And the engine's own `--ppl` crashes at
chunk 32 and above while 8 and 16 are fine, which points at a micro-batch boundary.

**The entry point question they ask us.** On the CPU, both packs produce repeated tokens through `--prompt` and through synthetic ids - the old
one no worse than the new one - so they want this build's gate, its template, its BOS/EOG handling and its parameters, to be the standard way to
gate and to measure perplexity. This record has been saying the same thing for a while: the CPU path is where the answer is known to be wrong,
so it is the only honest place to measure.

**Priority, as this record sees it.** The old pack disappears in hours, so the exact comparison that proves the new reader is correct has to
happen first, while both packs still exist. Everything else - dead-code removal, naming, perplexity, tuning - is worth doing after that comparison
either passes or names a step in this port.
