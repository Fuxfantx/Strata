// Declarative shader metadata: every shader states, next to its source, what the backend needs to know about it.
//
// write_mask      bit i set means binding i is written by the shader.  VK_SHADER_MASK_DERIVE means "read it from the
//                 SPIR-V NonWritable decoration", which is what older shaders rely on.
// param_block     the shader takes its scalars from the per-dispatch parameter block instead of push constants, so the
//                 host must write a slot, bind it and not record push constants.
//
// Rule for new shaders: declare an entry here, use the parameter block, and mask out divisions and modulo that the
// host can replace with magic multiply-shift values.
#pragma once
#include <cstdint>

#define VK_SHADER_MASK_DERIVE 0xFFFFFFFFu

struct vk_shader_meta {
    const char * name;          // pipeline name prefix; entries are matched by prefix, longest wins
    uint32_t     write_mask;    // roles, or VK_SHADER_MASK_DERIVE
    bool         param_block;   // scalars come from the parameter block
};

inline const vk_shader_meta g_vk_shader_meta[] = {
    { "add1",                    1u << 2,                 true  },
    // The set_rows variants are built from copy_to_quant.comp, which now reads the parameter block, so their
    // pipelines have to be created as block users too.
    { "set_rows_",               1u << 2,                 true  },
    // Attention matvec variants: the reduced push-constant structs they carry are written into the block too.
    { "mul_mat_vec_nc",  1u << 2,                 true  },
    { "mul_mat_vec_p021", 1u << 2,                true  },
    { "im2col",          1u << 1,                 true  },
    // Elementwise and bookkeeping kernels the port adds; they write only their destination binding.
    { "add2",            1u << 0,                 true  },
    { "dup_rows",        1u << 1,                 true  },
    { "repack_q2_0",     1u << 1,                 true  },
    { "touch",           1u << 0,                 true  },
    // The staged latent MoE variants share one source: they write the output (9) and the scratch block (10).
    { "latent_moe_q",    (1u << 9) | (1u << 10),  true  },
    // Last three sources with their own block.  Their variants all come from one source, so the family prefixes cover
    // every variant and are listed here before the generated block at the end of the array.
    { "conv2d_dw",                (1u << 2), true },
    { "gated_delta_net",          (1u << 2), true },
    { "lightning_indexer",        (1u << 2), true },

    // Small self contained kernels with one struct each.
    { "silu_flat",                (1u << 2), true },
    { "dsv4_hc_comb",             (1u << 2), true },
    { "row_max",                  (1u << 2), true },
    { "count_experts",            (1u << 2), true },
    { "quantize_q8_1",            (1u << 2), true },
    { "attn_merge",               (1u << 2), true },
    { "latent_moe_copy",          (1u << 1), true },
    // DERIVE: the shader writes binding 9 (out) and binding 10 (the scratch); the NonWritable decoration is the exact
    // evidence of which bindings it writes, and an explicit mask here was wrong once already
    { "latent_moe_f32q8",         VK_SHADER_MASK_DERIVE, true },

    // Dense and quantised matmul: matmul.comp and mul_mmq.comp pass vk_mat_mat_push_constants.  The expert variants
    // (matmul_id, matmul_id_subgroup) pass another struct, so the prefix is listed first and keeps push constants.
    { "matmul_id",                VK_SHADER_MASK_DERIVE, false },
    { "matmul",                   (1u << 2), true },

    // Flash attention: one struct of 128 bytes shared by every variant of flash_attn.comp and its coopmat forms.
    // The split-k reducer, the mask optimiser, the decode phases and the prefill kernel pass other structs and start
    // with fa_, so this prefix does not reach them.
    { "flash_attn_",              (1u << 2), true },

    // GLU and rope families read shared heads that are now parameter block capable.  The rope variants pass rope_params,
    // so each prefix is listed explicitly and nothing is caught by a shorter one by accident.
    { "swiglu",                   (1u << 2), true },
    { "geglu",                    (1u << 2), true },
    { "reglu",                    (1u << 2), true },
    { "rope_multi",               (1u << 2), true },
    { "rope_neox",                (1u << 2), true },
    { "rope_norm",                (1u << 2), true },
    { "rope_vision",              (1u << 2), true },
    // Session added fusions whose structs are single and self contained.
    { "multi_add",                (1u << 2), true },
    { "topk_moe",                 (1u << 2), true },
    { "dsv4_hc_pre",              (1u << 2), true },
    { "dsv4_hc_post",             (1u << 2), true },
    { "gated_mul",                (1u << 2), true },
    { "fwht",                     (1u << 2), true },

    // soft_max family: A read, D written, scalars from the parameter block.  The backward and the large variants are
    // listed first: one passes a different struct, the other still reads push constants.
    { "soft_max_back",            VK_SHADER_MASK_DERIVE, false },
    { "soft_max_large",           VK_SHADER_MASK_DERIVE, false },
    { "soft_max",                 (1u << 2), true },

    // copy.comp: A read, D written, scalars from the parameter block.
    { "cpy_",                     (1u << 2), true },
    { "contig_cpy_32_vec4",       (1u << 2), true },
    { "contig_cpy_",              (1u << 2), true },
    { "copy_transpose_",          (1u << 2), true },
    { "copy_from_quant_",         (1u << 2), true },
    { "copy_quant_same_",         (1u << 2), true },
    { "repeat_back_",             (1u << 2), true },
    { "repeat_",                  (1u << 2), true },
    { "diag_",                    (1u << 2), true },
    { "tri_",                     (1u << 2), true },
    { "roll_",                    (1u << 2), true },
    { "log_",                     (1u << 2), true },
    { "pad_reflect_1d_",          (1u << 2), true },

    // The _mul unary fusions take the binary struct, so they stay on push constants.
    { "gelu_mul_",                VK_SHADER_MASK_DERIVE, true },
    { "silu_mul_",                VK_SHADER_MASK_DERIVE, true },
    { "sigmoid_mul_",             VK_SHADER_MASK_DERIVE, true },
    { "softplus_mul_",            VK_SHADER_MASK_DERIVE, true },

    // unary.comp single input ops: A read, D written, scalars from the parameter block.
    { "abs_",                     (1u << 2), true },
    { "ceil_",                    (1u << 2), true },
    { "clamp_",                   (1u << 2), true },
    { "cos_",                     (1u << 2), true },
    { "elu_",                     (1u << 2), true },
    { "exp_",                     (1u << 2), true },
    { "expm1_",                   (1u << 2), true },
    { "floor_",                   (1u << 2), true },
    { "gelu_",                    (1u << 2), true },
    { "gelu_erf_",                (1u << 2), true },
    { "gelu_quick_",              (1u << 2), true },
    { "hardsigmoid_",             (1u << 2), true },
    { "hardswish_",               (1u << 2), true },
    { "leaky_relu_",              (1u << 2), true },
    { "neg_",                     (1u << 2), true },
    { "relu_",                    (1u << 2), true },
    { "round_",                   (1u << 2), true },
    { "sgn_",                     (1u << 2), true },
    { "sigmoid_",                 (1u << 2), true },
    { "silu_",                    (1u << 2), true },
    { "sin_",                     (1u << 2), true },
    { "softplus_",                (1u << 2), true },
    { "sqr_",                     (1u << 2), true },
    { "sqrt_",                    (1u << 2), true },
    { "step_",                    (1u << 2), true },
    { "tanh_",                    (1u << 2), true },
    { "trunc_",                   (1u << 2), true },
    { "xielu_",                   (1u << 2), true },

    // Unary head families that opt in through PARAM_BLOCK_ENABLED: A read, D written.
    { "scale_",                   (1u << 2), true },
    { "l2_norm",                  (1u << 2), true },

    // Unary head families (shared generic_unary_head.glsl): A read, D written.

    // migrated to the parameter block
    { "ssm_conv_f32",             VK_SHADER_MASK_DERIVE, true },
    { "ssm_conv_silu_f32",        VK_SHADER_MASK_DERIVE, true },
    { "ssm_conv_bias_silu_f32",   VK_SHADER_MASK_DERIVE, true },
    { "ssm_conv_nc4_f32",         VK_SHADER_MASK_DERIVE, true },
    { "ssm_conv_nc4_silu_f32",    VK_SHADER_MASK_DERIVE, true },
    { "ssm_conv_nc4_bias_silu_f32", VK_SHADER_MASK_DERIVE, true },

    // Exceptions first: these names share a prefix with a migrated family but pass a different struct, so they keep
    // push constants.  Matching stops at the first entry, so the longer name must come first.
    { "add_id_",            VK_SHADER_MASK_DERIVE, false },
    { "get_rows_quant_",    VK_SHADER_MASK_DERIVE, false },
    { "get_rows_back_",     VK_SHADER_MASK_DERIVE, true },
    { "mul_mat",            VK_SHADER_MASK_DERIVE, false },

    // 64 wide cooperative matrix matmul.  Roles are declared rather than derived: A and B are read (bindings 0, 1) and
    // D is written (binding 2), so bit 2 is set.  The tile sizes are #defines, so every divide in the body folds to a
    // shift at compile time.

    // Elementwise and norm families, one shared struct each, all on the parameter block.
    { "add_",               VK_SHADER_MASK_DERIVE, true },
    { "sub_",               VK_SHADER_MASK_DERIVE, true },
    { "mul_",               VK_SHADER_MASK_DERIVE, true },
    { "div_",               VK_SHADER_MASK_DERIVE, true },
    { "acc_",               VK_SHADER_MASK_DERIVE, true },
    { "solve_tri_",         VK_SHADER_MASK_DERIVE, true },
    { "get_rows_",          VK_SHADER_MASK_DERIVE, true },
    { "rms_norm_",          VK_SHADER_MASK_DERIVE, true },

    // Remaining families, generated by port/tools/declare-families.py: param_block is true only where every
    // source of the family opts in through PARAM_BLOCK_ENABLED.  Longer names first.

    { "topk_radix_select_f32_subgroup", VK_SHADER_MASK_DERIVE, false },
    { "gated_delta_net_f32_nocluster", VK_SHADER_MASK_DERIVE, false },
    { "gated_delta_net_f32_lanes16",   VK_SHADER_MASK_DERIVE, false },
    { "gated_delta_net_f32_lanes32",   VK_SHADER_MASK_DERIVE, false },
    { "gated_delta_net_f32_lanes64",   VK_SHADER_MASK_DERIVE, false },
    { "fa_sparse_compact_subgroup",    VK_SHADER_MASK_DERIVE, false },
    { "gated_delta_net_f32_lanes2",    VK_SHADER_MASK_DERIVE, false },
    { "gated_delta_net_f32_lanes4",    VK_SHADER_MASK_DERIVE, false },
    { "gated_delta_net_f32_lanes8",    VK_SHADER_MASK_DERIVE, false },
    { "gated_delta_net_f32_shmem",     VK_SHADER_MASK_DERIVE, false },
    { "cross_entropy_loss_back",       VK_SHADER_MASK_DERIVE, false },
    { "ssm_conv_nc4_bias_silu",        VK_SHADER_MASK_DERIVE, false },
    { "concat_transpose_i16",          VK_SHADER_MASK_DERIVE, true },
    { "concat_transpose_i8",           VK_SHADER_MASK_DERIVE, true },
    { "conv2d_dw_cwhn_f16",            VK_SHADER_MASK_DERIVE, false },
    { "conv2d_dw_whcn_f16",            VK_SHADER_MASK_DERIVE, false },
    { "cross_entropy_loss",            VK_SHADER_MASK_DERIVE, false },
    { "ssm_conv_bias_silu",            VK_SHADER_MASK_DERIVE, false },
    { "timestep_embedding",            VK_SHADER_MASK_DERIVE, false },
    { "conv_transpose_1d",             VK_SHADER_MASK_DERIVE, false },
    { "cumsum_multipass1",             VK_SHADER_MASK_DERIVE, false },
    { "cumsum_multipass2",             VK_SHADER_MASK_DERIVE, false },
    { "fa_sparse_compact",             VK_SHADER_MASK_DERIVE, false },
    { "fa_split_k_reduce",             VK_SHADER_MASK_DERIVE, false },
    { "gated_linear_attn",             VK_SHADER_MASK_DERIVE, false },
    { "lightning_indexer",             VK_SHADER_MASK_DERIVE, false },
    { "ssm_conv_nc4_silu",             VK_SHADER_MASK_DERIVE, false },
    { "ssm_scan_subgroup",             VK_SHADER_MASK_DERIVE, false },
    { "topk_radix_select",             VK_SHADER_MASK_DERIVE, false },
    { "concat_transpose",              VK_SHADER_MASK_DERIVE, true },
    { "topk_nary_search",              VK_SHADER_MASK_DERIVE, false },
    { "gated_delta_net",               VK_SHADER_MASK_DERIVE, false },
    { "pool2d_subgroup",               VK_SHADER_MASK_DERIVE, false },
    { "conv2d_dw_cwhn",                VK_SHADER_MASK_DERIVE, false },
    { "conv2d_dw_whcn",                VK_SHADER_MASK_DERIVE, false },
    { "opt_step_adamw",                VK_SHADER_MASK_DERIVE, false },
    { "pad_reflect_1d",                (1u << 2), true },
    { "split_k_reduce",                VK_SHADER_MASK_DERIVE, false },
    { "argsort_large",                 VK_SHADER_MASK_DERIVE, false },
    { "fa_decode_ph1",                 VK_SHADER_MASK_DERIVE, false },
    { "fa_decode_ph2",                 VK_SHADER_MASK_DERIVE, false },
    { "ssm_conv_silu",                 VK_SHADER_MASK_DERIVE, false },
    { "opt_step_sgd",                  VK_SHADER_MASK_DERIVE, false },
    { "ssm_conv_nc4",                  VK_SHADER_MASK_DERIVE, false },
    { "topk_argsort",                  VK_SHADER_MASK_DERIVE, false },
    { "count_equal",                   VK_SHADER_MASK_DERIVE, false },
    { "fa_mask_opt",                   VK_SHADER_MASK_DERIVE, false },
    { "hardsigmoid",                   VK_SHADER_MASK_DERIVE, false },
    { "concat_i16",                    VK_SHADER_MASK_DERIVE, true },
    { "fa_prefill",                    VK_SHADER_MASK_DERIVE, false },
    { "group_norm",                    VK_SHADER_MASK_DERIVE, false },
    { "leaky_relu",                    VK_SHADER_MASK_DERIVE, false },
    { "top_k_mask",                    VK_SHADER_MASK_DERIVE, false },
    { "col2im_1d",                     VK_SHADER_MASK_DERIVE, false },
    { "concat_i8",                     VK_SHADER_MASK_DERIVE, true },
    { "hardswish",                     VK_SHADER_MASK_DERIVE, false },
    { "rwkv_wkv6",                     VK_SHADER_MASK_DERIVE, false },
    { "rwkv_wkv7",                     VK_SHADER_MASK_DERIVE, false },
    { "solve_tri",                     VK_SHADER_MASK_DERIVE, false },
    { "add1_f16",                      VK_SHADER_MASK_DERIVE, false },
    { "add1_f32",                      VK_SHADER_MASK_DERIVE, false },
    { "get_rows",                      VK_SHADER_MASK_DERIVE, false },
    { "out_prod", 1u << 2, true },
    { "softplus",                      VK_SHADER_MASK_DERIVE, false },
    { "ssm_conv",                      VK_SHADER_MASK_DERIVE, false },
    { "ssm_scan",                      VK_SHADER_MASK_DERIVE, false },
    { "sum_rows",                      VK_SHADER_MASK_DERIVE, false },
    { "argsort",                       VK_SHADER_MASK_DERIVE, false },
    { "sigmoid",                       VK_SHADER_MASK_DERIVE, false },
    { "upscale",                       VK_SHADER_MASK_DERIVE, false },
    { "arange",                        VK_SHADER_MASK_DERIVE, false },
    { "argmax",                        VK_SHADER_MASK_DERIVE, false },
    { "concat",                        VK_SHADER_MASK_DERIVE, true },
    { "cumsum",                        VK_SHADER_MASK_DERIVE, false },
    { "pool1d",                        VK_SHADER_MASK_DERIVE, false },
    { "pool2d",                        VK_SHADER_MASK_DERIVE, false },
    { "repeat",                        (1u << 2), true },
    { "clamp",                         VK_SHADER_MASK_DERIVE, false },
    { "expm1",                         VK_SHADER_MASK_DERIVE, false },
    { "floor",                         VK_SHADER_MASK_DERIVE, false },
    { "round",                         VK_SHADER_MASK_DERIVE, false },
    { "scale",                         (1u << 2), true },
    { "snake", 1u << 3, true },
    { "trunc",                         VK_SHADER_MASK_DERIVE, false },
    { "xielu",                         VK_SHADER_MASK_DERIVE, false },
    { "ceil",                          VK_SHADER_MASK_DERIVE, false },
    { "diag",                          (1u << 2), true },
    { "fill",                          VK_SHADER_MASK_DERIVE, false },
    { "gelu",                          VK_SHADER_MASK_DERIVE, false },
    { "norm",                          VK_SHADER_MASK_DERIVE, false },
    { "relu",                          VK_SHADER_MASK_DERIVE, false },
    { "roll",                          (1u << 2), true },
    { "silu",                          VK_SHADER_MASK_DERIVE, false },
    { "sqrt",                          VK_SHADER_MASK_DERIVE, false },
    { "step",                          VK_SHADER_MASK_DERIVE, false },
    { "tanh",                          VK_SHADER_MASK_DERIVE, false },
    { "abs",                           VK_SHADER_MASK_DERIVE, false },
    { "acc",                           VK_SHADER_MASK_DERIVE, false },
    { "cos",                           VK_SHADER_MASK_DERIVE, false },
    { "div",                           VK_SHADER_MASK_DERIVE, false },
    { "elu",                           VK_SHADER_MASK_DERIVE, false },
    { "exp",                           VK_SHADER_MASK_DERIVE, false },
    { "log",                           (1u << 2), true },
    { "mul",                           VK_SHADER_MASK_DERIVE, false },
    { "neg",                           VK_SHADER_MASK_DERIVE, false },
    { "pad",                           VK_SHADER_MASK_DERIVE, true },
    { "sgn",                           VK_SHADER_MASK_DERIVE, false },
    { "sin",                           VK_SHADER_MASK_DERIVE, false },
    { "sqr",                           VK_SHADER_MASK_DERIVE, false },
    { "sub",                           VK_SHADER_MASK_DERIVE, false },
    { "tri",                           (1u << 2), true },
};
