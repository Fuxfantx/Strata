"""Make the trellis codebook live in constant memory (default), stop charging it to shared memory, and add a wide
column tile that only fits because of that.

Measured basis:
  * shmem_limit is 32768 on this device, so the tile budget is (BM + BN) * 17 * 8 bytes;
  * with 8 KiB charged for the codebook that is BM + BN <= 180, which leaves only 64x64 -> 7 column tiles;
  * reading the codebook from constant memory measures neutral (17.74 vs 17.87 tok/s on Gyro-M prefill), so it is free
    to make it the default and reclaim the 8 KiB.
  * 64x128 then fits at 26.1 KiB, keeps the row tiling unchanged, and drops the column tiles to 4 for a 430-token pass.
"""
import io

# ---------------------------------------------------------------- 1. generator default
P = r"third_party/llama.cpp\ggml\src\ggml-vulkan\vulkan-shaders\vulkan-shaders-gen.cpp"
s = io.open(P, encoding="utf-8-sig").read()
old = """    // Experiment knob (build time): trellis shaders read the codebook straight from the
    // constant array instead of copying it into shared memory per workgroup.
    std::map<std::string, std::string> defines = defines_in;
    if (getenv("GGML_VK_TQ_LUT_DIRECT") && (defines.count("DATA_A_TQ2_T") || defines.count("DATA_A_TQK6") || defines.count("DATA_A_TQK7"))) {
        defines["TQ_LUT_DIRECT"] = "1";
    }"""
new = """    // Trellis shaders read the codebook from the constant array instead of copying 8 KiB into shared memory per
    // workgroup. Measured neutral on Gyro-M prefill, and it returns the 8 KiB to the tile budget, which is what lets a
    // wider column tile fit at all. GGML_VK_TQ_LUT_SHARED=1 restores the shared copy for comparison.
    std::map<std::string, std::string> defines = defines_in;
    if (!getenv("GGML_VK_TQ_LUT_SHARED") && (defines.count("DATA_A_TQ2_T") || defines.count("DATA_A_TQK6") || defines.count("DATA_A_TQK7"))) {
        defines["TQ_LUT_DIRECT"] = "1";
    }"""
assert old in s, "generator knob not found"
s = s.replace(old, new, 1)
io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("generator: direct LUT is now the default")

# ---------------------------------------------------------------- 2. host-side accounting
P = r"third_party\llama.cpp\ggml\src\ggml-vulkan\ggml-vulkan.cpp"
s = io.open(P, encoding="utf-8-sig").read()
old = """    case GGML_TYPE_TQ2_T:
        lut_size = 4*2048; // packed-fp16 Hyb codebook (tq2_t_lut.glsl)
        break;"""
new = """    case GGML_TYPE_TQ2_T:
        // The codebook is a shader constant (TQ_LUT_DIRECT in tq2_t_lut.glsl, the default), so it does not occupy
        // shared memory and must not be charged to the tile budget. Charging it costs the wide tile.
        lut_size = 0;
        break;"""
assert old in s, "TQ2_T lut_size row not found"
s = s.replace(old, new, 1)

# ---------------------------------------------------------------- 3. the wide tile
anchor = "    const uint32_t mul_mat_mm_warp_16 = std::min(mul_mat_subgroup_size_16, 64u);"
assert anchor in s
tile = """

    // Trellis column tile: 64 rows by 128 columns, the largest that fits 32 KiB once the codebook is in constant
    // memory ((64 + 128) * 17 * 8 = 26.1 KiB). Row tiling is unchanged, so only the column passes drop, 7 -> 4 for a
    // 430-token prompt. Register blocking matches the standard 128x128 tile.
    const std::vector<uint32_t> l_warptile_mmq_trellis = { 128, 64, 128, 32, 64, 64, 2, 4, 2, 1, 64 };
    const std::array<uint32_t, 3> l_mmq_trellis_wg_denoms = { 64, 128, 1 };"""
s = s.replace(anchor, anchor + tile, 1)

# Enable it as the large slot for the trellis types only, when it fits.
marker = "            // The q8_1 mmq path has its own (larger) shmem layout, check it separately."
assert marker in s
insert = """            switch (t) {
                case GGML_TYPE_TQ2_T: case GGML_TYPE_TQ1_0: case GGML_TYPE_TQ2_0: case GGML_TYPE_PTQ1_0:
                    if (ggml_vk_matmul_shmem_support(device, l_warptile_mmq_trellis, false, t)) device->mul_mat_l[i] = true;
                    if (ggml_vk_matmul_shmem_support(device, l_warptile_mmq_trellis, true,  t)) device->mul_mat_id_l[i] = true;
                    break;
                default:
                    break;
            }

"""
s = s.replace(marker, insert + marker, 1)

# Config vectors next to the existing ones, and point the trellis macros at them.
cfg = "        std::vector<vk_tile_config> tc_mmqid = {{s_warptile_mmqid, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid, l_mmq_wg_denoms, l_align}};"
assert cfg in s
s = s.replace(cfg, cfg + """
        std::vector<vk_tile_config> tc_mmq_trellis = {{s_warptile_mmq, s_mmq_wg_denoms, s_align}, {m_warptile_mmq, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_trellis, l_mmq_trellis_wg_denoms, s_align}};
        std::vector<vk_tile_config> tc_mmqid_trellis = {{s_warptile_mmqid, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_trellis, l_mmq_trellis_wg_denoms, s_align}};""", 1)

# Split the trellis types out of the generic LUT macro loops.
old_sg = """    #define X_SG(TYPE, tstr) \\
            sg_create({TYPE, GGML_TYPE_F32, false, true},  tc_mmq, "matmul_" #tstr "_f32_f16acc", SPV_DOT2_F16ACC(matmul_##tstr##_f32), sizeof(vk_mat_mat_push_constants), 3); \\
            sg_create({TYPE, GGML_TYPE_F32, false, false}, tc_mmq, "matmul_" #tstr "_f32",        SPV_DOT2(matmul_##tstr##_f32),        sizeof(vk_mat_mat_push_constants), 3);
            FOR_EACH_LUT_TYPE(X_SG)
#undef X_SG"""
assert old_sg in s, "X_SG block shape changed"
new_sg = """    #define X_SG(TYPE, tstr) \\
            sg_create({TYPE, GGML_TYPE_F32, false, true},  tc_mmq, "matmul_" #tstr "_f32_f16acc", SPV_DOT2_F16ACC(matmul_##tstr##_f32), sizeof(vk_mat_mat_push_constants), 3); \\
            sg_create({TYPE, GGML_TYPE_F32, false, false}, tc_mmq, "matmul_" #tstr "_f32",        SPV_DOT2(matmul_##tstr##_f32),        sizeof(vk_mat_mat_push_constants), 3);
    #define X_SG_TRELLIS(TYPE, tstr) \\
            sg_create({TYPE, GGML_TYPE_F32, false, true},  tc_mmq_trellis, "matmul_" #tstr "_f32_f16acc", SPV_DOT2_F16ACC(matmul_##tstr##_f32), sizeof(vk_mat_mat_push_constants), 3); \\
            sg_create({TYPE, GGML_TYPE_F32, false, false}, tc_mmq_trellis, "matmul_" #tstr "_f32",        SPV_DOT2(matmul_##tstr##_f32),        sizeof(vk_mat_mat_push_constants), 3);
            FOR_EACH_LUT_TYPE_NONFP4(X_SG)
            FOR_EACH_TRELLIS_TYPE(X_SG_TRELLIS)
#undef X_SG
#undef X_SG_TRELLIS"""
s = s.replace(old_sg, new_sg, 1)

old_sgid = """        #define X_SG_ID(TYPE, tstr) \\
                sg_create({TYPE, GGML_TYPE_F32, true, true},  tc_mmqid, "matmul_id_" #tstr "_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_##tstr##_f32), sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count); \\
                sg_create({TYPE, GGML_TYPE_F32, true, false}, tc_mmqid, "matmul_id_" #tstr "_f32",        SPV_DOT2(matmul_id_##tstr##_f32),        sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
                FOR_EACH_LUT_TYPE(X_SG_ID)
#undef X_SG_ID"""
if old_sgid in s:
    new_sgid = """        #define X_SG_ID(TYPE, tstr) \\
                sg_create({TYPE, GGML_TYPE_F32, true, true},  tc_mmqid, "matmul_id_" #tstr "_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_##tstr##_f32), sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count); \\
                sg_create({TYPE, GGML_TYPE_F32, true, false}, tc_mmqid, "matmul_id_" #tstr "_f32",        SPV_DOT2(matmul_id_##tstr##_f32),        sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
        #define X_SG_ID_TRELLIS(TYPE, tstr) \\
                sg_create({TYPE, GGML_TYPE_F32, true, true},  tc_mmqid_trellis, "matmul_id_" #tstr "_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_##tstr##_f32), sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count); \\
                sg_create({TYPE, GGML_TYPE_F32, true, false}, tc_mmqid_trellis, "matmul_id_" #tstr "_f32",        SPV_DOT2(matmul_id_##tstr##_f32),        sizeof(vk_mat_mat_id_push_constants), mul_mat_id_param_count);
                FOR_EACH_LUT_TYPE_NONFP4(X_SG_ID)
                FOR_EACH_TRELLIS_TYPE(X_SG_ID_TRELLIS)
#undef X_SG_ID
#undef X_SG_ID_TRELLIS"""
    s = s.replace(old_sgid, new_sgid, 1)
    print("id macro split")
else:
    print("WARNING: id macro block not found, trellis id pipelines keep the stock tile")

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("done")
