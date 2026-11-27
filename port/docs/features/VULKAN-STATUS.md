# The Vulkan backend, per model family

This page says one thing once, so later reports can quote it instead of re-deriving it: for each of the three model
families, which parts of the Vulkan backend are present, and what is missing.

## How support is graded on this page

The backend decides per operation. If no pipeline exists for a weight type, the operation is not refused - it is placed
on the CPU instead, and the transfer is paid. So "no support" here means "silently runs on the CPU", not "fails".

| level | meaning |
|---|---|
| **L0** | no pipeline for the type. Every matmul on it runs on the CPU. |
| **L1** | a dequantise plus a generic mat-vec (`mul_mat_vec` through a `dequant_<type>` shader). |
| **L2** | a quantised integer-dot matmul (`matmul_<type>_q8_1`), including the subgroup variant and the per-expert `mul_mat_id` variants. This is the fast path. |

The type whitelist is in `ggml-vulkan.cpp`: `non_lut_quant_types`, the LUT family and the trellis family. A type that is
in none of those lists has no pipeline.

## Qwen3.8-Flash-Next, Q2_0 weights

**L2.** Complete.

- `dequant_q2_0` shader present.
- `mul_mat_vec` pipelines for Q2_0 in the f32, f16 and q8_1 variants.
- `mul_mat_vec_id` pipelines for Q2_0, so the per-expert path is covered.
- Quantised matmul pipelines `matmul_q2_0_q8_1`, plus `matmul_id_subgroup_q2_0_q8_1` and `matmul_id_q2_0_q8_1`.

There is no `mul_mat_vec_q2_0.comp` file, and none is needed: the pipeline is built from the dequant shader plus the
generic mat-vec, and the integer-dot path is the one that matters for speed.

## Edge0, Q3_K and the dense layers

**L2 for the standard types.** Q3_K has `mul_mat_vec_q3_k` and is in the quantised matmul list. Q4_0 and Q8_0 are both
in `non_lut_quant_types` and both have quantised matmul pipelines.

**The open question is the PLE tensor type, and it is a pack-side decision, not an engine one.** The objective asks for
the PLE tensors at Q4_0 quality through affine and GSQ-RCO. If the pack writes them as Q4_0, they are L2 like the rest.
If the pack writes them as a type of their own, they are L0 until that type has a shader - the same gap Hy4 has now.
Nothing in this tree defines such a type today.

## Hy4, the codes-only block types and the Engram rows

**L0.** There is no Vulkan support at all.

`CSQ8`, `CSQ4`, `VQ4_128`, `VQ4_64`, `ENGRAM_Q4B12`, `ENGRAM_Q6B12` and `ENGRAM_Q2B12` appear **zero** times in the
Vulkan backend: no dequant shader, no mat-vec pipeline, no quantised matmul, no per-expert variant. On a machine with a
working Vulkan device, every matmul on those tensors would be placed on the CPU, and the tensors would be copied for it.

So the CPU path this port runs today is not a temporary workaround. It is the only path that exists. Closing the gap
means writing the shaders - the dequantise side and the quantised matmul side, per the shader conventions - which is the
performance item the objective already names.

## What was measured, and what was read

Nothing on this page was measured on this machine. The Vulkan backend cannot load here at all: the system's
`vulkan-1.dll` is x64 and this is an ARM64 host, so the loader finds no library. Every statement above comes from
reading `ggml-vulkan.cpp` and the shader directory, and from the pipeline-creation calls for each type.

A measurement would need a machine that loads the backend, and would then have to name the device, the driver, and
whether the op ran on the GPU or fell back - the `STRATA_VK_OP_TRACE` marker in the support check prints the type and
shape of each matmul it is asked about, which is the cheapest way to see a fallback happen.
