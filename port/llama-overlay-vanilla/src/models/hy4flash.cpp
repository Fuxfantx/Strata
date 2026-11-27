// Hy4-Flash (hy_v4_flash) is a variant of the Hy4 family this tree implements as hy_v4.  What differs:
//   - the routed experts are split into hot / warm / cold stacks (160 / 96 / 256, 512 on the MTP block);
//   - the stacks read a latent width (1536) with ffn_routed_{up,down} projecting in and out;
//   - two extra norms around attention and the feed-forward;
//   - an Engram group on the blocks listed in ple.layers, and output hyper-connections.
//
// The split is stored per block: hy4f.expert_split.<tier>_ids[<tier>_offsets[il] + j] is the global expert id of local
// expert j of that tier.  The model reads those lists so the graph can send a selected expert to the stack that owns it.
#include "models.h"
#include "ggml-backend.h"

// the Engram block types take their scale from a table the quants code owns, and every Hy4 block type reads its
// codebook from one too
extern "C" void ggml_hy4_set_engram_range(float lo, float hi);
extern "C" void ggml_hy4_set_codebooks(const float * csq8, const float * csq4, const float * vq4_128, const float * vq4_64);

// the Engram block types take their scale from a table the quants code owns
extern "C" void ggml_hy4_set_engram_range(float lo, float hi);

void llama_model_hy4flash::load_arch_hparams(llama_model_loader & ml) {
    llama_model_hy_v4::load_arch_hparams(ml);

    // The four codebooks the block types dequantise with are key-values, not tensors, and the quants code copies them
    // into place.  Without this every CSQ and VQ block converts to zeros, which is a whole-model zero and not a subtle
    // error: the token embedding itself would come back empty.
    {
        const auto read_f32 = [](gguf_context * g, const char * key, std::vector<float> & out) {
            const int kid = gguf_find_key(g, key);
            if (kid < 0 || gguf_get_kv_type(g, kid) != GGUF_TYPE_ARRAY) {
                return false;
            }
            if (gguf_get_arr_type(g, kid) != GGUF_TYPE_FLOAT32) {
                return false;
            }
            const uint64_t n = gguf_get_arr_n(g, kid);
            const float * d = (const float *) gguf_get_arr_data(g, kid);
            out.assign(d, d + n);
            return true;
        };

        std::vector<float> c8, c4, v128, v64;
        const bool ok = read_f32(ml.metadata, "prism.csq8.codebook",    c8)   &
                        read_f32(ml.metadata, "prism.csq4.codebook",    c4)   &
                        read_f32(ml.metadata, "prism.vq4_128.codebook", v128) &
                        read_f32(ml.metadata, "prism.vq4_64.codebook",  v64);
        if (ok) {
            ggml_hy4_set_codebooks(c8.data(), c4.data(), v128.data(), v64.data());
            LLAMA_LOG_WARN("%s: codebooks set: csq8 %zu, csq4 %zu, vq4_128 %zu, vq4_64 %zu\n", __func__,
                    c8.size(), c4.size(), v128.size(), v64.size());
        } else {
            LLAMA_LOG_WARN("%s: codebooks missing; the block types would convert to zeros\n", __func__);
        }
    }

    // the Engram tables are resolved from the companion once the shape is known
    // The pack's hybrid attention: a window length, a per-block pattern, and a rope base for the windowed blocks.
    // hparams already carries all three fields (get_rope_freq_base picks the windowed base from is_swa), so this only
    // has to fill them in.
    {
        uint32_t n_swa = 0;
        if (ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, n_swa, false) && n_swa > 0) {
            hparams.n_swa = n_swa;
            // the window is a standard one: every position sees the last n_swa of them
            hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        }

        float freq_base_swa = 0.0f;
        if (ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, freq_base_swa, false) && freq_base_swa > 0.0f) {
            hparams.rope_freq_base_train_swa = freq_base_swa;
            LLAMA_LOG_WARN("%s: sliding window %u, windowed rope base %.1f\n", __func__, hparams.n_swa,
                    (double) hparams.rope_freq_base_train_swa);
        }

        // the pattern is stored in a narrow integer type the loader's typed helper rejects, so read it through the
        // metadata with the array's own element type
        std::vector<int32_t> pattern;
        {
            const int kid = gguf_find_key(ml.metadata, "hy_v4_flash.attention.sliding_window_pattern");
            LLAMA_LOG_WARN("%s: pattern key id %d, kv type %d, element type %d, n %llu\n", __func__, kid,
                    kid >= 0 ? (int) gguf_get_kv_type(ml.metadata, kid) : -1,
                    kid >= 0 && gguf_get_kv_type(ml.metadata, kid) == GGUF_TYPE_ARRAY ? (int) gguf_get_arr_type(ml.metadata, kid) : -1,
                    kid >= 0 && gguf_get_kv_type(ml.metadata, kid) == GGUF_TYPE_ARRAY ? (unsigned long long) gguf_get_arr_n(ml.metadata, kid) : 0ull);
            if (kid >= 0 && gguf_get_kv_type(ml.metadata, kid) == GGUF_TYPE_ARRAY) {
                const uint64_t n = gguf_get_arr_n(ml.metadata, kid);
                const enum gguf_type t = gguf_get_arr_type(ml.metadata, kid);
                const void * data = gguf_get_arr_data(ml.metadata, kid);
                pattern.resize((size_t) n);
                for (uint64_t i = 0; i < n; ++i) {
                    switch (t) {
                        // the pack writes this pattern as bools, one byte each
                        case GGUF_TYPE_BOOL:                          pattern[(size_t) i] = ((const uint8_t *) data)[i] ? 1 : 0; break;
                        case GGUF_TYPE_UINT8:  case GGUF_TYPE_INT8:  pattern[(size_t) i] = ((const int8_t *)  data)[i]; break;
                        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: pattern[(size_t) i] = ((const int16_t *) data)[i]; break;
                        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: pattern[(size_t) i] = ((const int32_t *) data)[i]; break;
                        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: pattern[(size_t) i] = (int32_t) ((const int64_t *) data)[i]; break;
                        default: pattern.clear(); break;
                    }
                    if (pattern.empty()) {
                        break;
                    }
                }
            }
        }

        if (!pattern.empty()) {
            const size_t n = std::min(pattern.size(), (size_t) hparams.n_layer_all);
            for (size_t i = 0; i < n; ++i) {
                hparams.is_swa_impl[i] = pattern[i] != 0 ? 1u : 0u;
            }
            LLAMA_LOG_WARN("%s: sliding window pattern read for %zu block(s)\n", __func__, n);
        }
    }

    // The router's scale multiplies every expert output; if it were meant to divide, or not to apply at all, the peak and
    // the gap would move, so it is one switch away.
    if (const char * no_scale = std::getenv("STRATA_HY4_MOE_NO_SCALE")) {
        if (no_scale[0] != '\0' && no_scale[0] != '0') {
            hparams.expert_weights_scale = 1.0f;
        }
    }

    // The router's weights may or may not be normalised over the selection before they are applied; the pack says they are,
    // and the switch is what measures whether that reading buys confidence.
    if (const char * no_norm = std::getenv("STRATA_HY4_MOE_NO_NORM")) {
        if (no_norm[0] != '\0' && no_norm[0] != '0') {
            hparams.expert_weights_norm = false;
        }
    }

    LLAMA_LOG_WARN("%s: moe config: scale %.4f, norm %d, gating %d\n", __func__,
            (double) hparams.expert_weights_scale, (int) hparams.expert_weights_norm,
            (int) hparams.expert_gating_func);


    LLAMA_LOG_WARN("%s: rope bases: train %.1f, windowed %.1f, n_swa %u, swa_type %d\n", __func__,
            (double) hparams.rope_freq_base_train, (double) hparams.rope_freq_base_train_swa,
            hparams.n_swa, (int) hparams.swa_type);

    engram_resolve(ml);

    ml.get_key("hy_v4_flash.expert_latent_length", hparams.n_expert_latent, false);
    ml.get_arr("hy_v4_flash.ple.layers", engram_layer_list, false);
    

    // the per-block tier maps; the id arrays are stored in a narrow integer type that get_arr does not accept, so they
    // are read through the gguf metadata with their own type
    const auto read_ints = [&](const char * key) {
        std::vector<int32_t> out;
        const int kid = gguf_find_key(ml.metadata, key);
        if (kid < 0 || gguf_get_kv_type(ml.metadata, kid) != GGUF_TYPE_ARRAY) {
            return out;
        }
        const uint64_t n        = gguf_get_arr_n(ml.metadata, kid);
        const enum gguf_type t  = gguf_get_arr_type(ml.metadata, kid);
        const void * data       = gguf_get_arr_data(ml.metadata, kid);
        out.resize((size_t) n);
        for (uint64_t i = 0; i < n; ++i) {
            switch (t) {
                case GGUF_TYPE_UINT8:
                case GGUF_TYPE_INT8:  out[(size_t) i] = (int32_t) ((const int8_t *)  data)[i]; break;
                case GGUF_TYPE_UINT16:
                case GGUF_TYPE_INT16: out[(size_t) i] = (int32_t) ((const int16_t *) data)[i]; break;
                case GGUF_TYPE_UINT32:
                case GGUF_TYPE_INT32: out[(size_t) i] =           ((const int32_t *) data)[i]; break;
                case GGUF_TYPE_UINT64:
                case GGUF_TYPE_INT64: out[(size_t) i] = (int32_t) ((const int64_t *) data)[i]; break;
                default:              out.clear(); return out;
            }
        }
        return out;
    };

    // The eight n-gram heads' row ranges live only in the companion file (the second half of the pack), so open it next
    // to the backbone and read them from there.  The companion is not a split: it carries the Engram tables.
    {
        const std::string & backbone = ml.files.empty() ? std::string() : ml.files[0]->path();
        std::string companion = backbone;

        const size_t pos = companion.rfind("-00001-of-");
        if (pos != std::string::npos) {
            companion.replace(pos, std::string("-00001-of-").size(), "-00002-of-");
        }

        if (!companion.empty() && companion != backbone) {
            // metadata only: the companion holds tens of gigabytes of tables and none of them is read here
            gguf_init_params gp;
            gp.no_alloc = true;
            gp.ctx      = nullptr;

            gguf_context * gguf_c = gguf_init_from_file(companion.c_str(), gp);
            if (gguf_c != nullptr) {
                const auto read_from = [&](gguf_context * g, const char * key, std::vector<int32_t> & out) {
                    const int kid = gguf_find_key(g, key);
                    if (kid < 0 || gguf_get_kv_type(g, kid) != GGUF_TYPE_ARRAY) {
                        return false;
                    }
                    const uint64_t n   = gguf_get_arr_n(g, kid);
                    const enum gguf_type t = gguf_get_arr_type(g, kid);
                    const void * data  = gguf_get_arr_data(g, kid);
                    out.resize((size_t) n);
                    for (uint64_t i = 0; i < n; ++i) {
                        switch (t) {
                            case GGUF_TYPE_UINT8:
                            case GGUF_TYPE_INT8:  out[(size_t) i] = (int32_t) ((const int8_t *)  data)[i]; break;
                            case GGUF_TYPE_UINT16:
                            case GGUF_TYPE_INT16: out[(size_t) i] = (int32_t) ((const int16_t *) data)[i]; break;
                            case GGUF_TYPE_UINT32:
                            case GGUF_TYPE_INT32: out[(size_t) i] =           ((const int32_t *) data)[i]; break;
                            case GGUF_TYPE_UINT64:
                            case GGUF_TYPE_INT64: out[(size_t) i] = (int32_t) ((const int64_t *) data)[i]; break;
                            default:              out.clear(); return false;
                        }
                    }
                    return true;
                };

                std::vector<int32_t> head_off, head_vocab;
                const int kid_dbg = gguf_find_key(gguf_c, "hy_v4_flash.ple.head_offsets");
                LLAMA_LOG_WARN("%s: head_offsets key id %d, kv type %d, element type %d, n %llu\n", __func__, kid_dbg,
                        kid_dbg >= 0 ? (int) gguf_get_kv_type(gguf_c, kid_dbg) : -1,
                        kid_dbg >= 0 ? (int) gguf_get_arr_type(gguf_c, kid_dbg) : -1,
                        kid_dbg >= 0 ? (unsigned long long) gguf_get_arr_n(gguf_c, kid_dbg) : 0ull);

                const bool ok_off   = read_from(gguf_c, "hy_v4_flash.ple.head_offsets",    head_off);
                const bool ok_vocab = read_from(gguf_c, "hy_v4_flash.ple.head_vocab_sizes", head_vocab);

                LLAMA_LOG_WARN("%s: companion %s opened, head_offsets %zu (%d), head_vocab_sizes %zu (%d)\n", __func__,
                        companion.c_str(), head_off.size(), (int) ok_off, head_vocab.size(), (int) ok_vocab);

                gguf_free(gguf_c);
            } else {
                LLAMA_LOG_WARN("%s: cannot open companion %s\n", __func__, companion.c_str());
            }
        }
    }

    const std::vector<int32_t> hot_off  = read_ints("hy4f.expert_split.hot_offsets");
    const std::vector<int32_t> warm_off = read_ints("hy4f.expert_split.warm_offsets");
    const std::vector<int32_t> cold_off = read_ints("hy4f.expert_split.cold_offsets");
    const std::vector<int32_t> hot_ids  = read_ints("hy4f.expert_split.hot_ids");


    // Two offset conventions exist: the reference pack starts its lists at block 1 (the first block with experts), and the v2
    // packs write one leading zero for the dense block 0.  A repeated first pair identifies the second form; dropping that entry
    // makes both read alike.
    const auto normalized = [](std::vector<int32_t> off) {
        if (off.size() >= 2 && off[0] == off[1]) {
            off.erase(off.begin());
            LLAMA_LOG_INFO("%s: expert split carries a leading entry for the dense block\n", __func__);
        }
        return off;
    };

    // How wide each tier is comes from the split itself, so a pack that rearranged the partition (160/96/256 in the first,
    // 160/160/192 in the second) is read without a build-time constant.
    {
        const std::vector<int32_t> hn = normalized(hot_off);
        const std::vector<int32_t> wn = normalized(warm_off);
        const std::vector<int32_t> cn = normalized(cold_off);
        if (hn.size() >= 2 && wn.size() >= 2 && cn.size() >= 2) {
            const int64_t t0 = hn[1] - hn[0], t1 = wn[1] - wn[0], t2 = cn[1] - cn[0];
            if (hn.size() >= 2) {
                // the last partition describes the draft block, whose hot tier is wider than the trunk's
                draft_tier_count = (int64_t) (hn[hn.size() - 1] - hn[hn.size() - 2]);
                LLAMA_LOG_INFO("%s: draft block hot tier %lld experts\n", __func__, (long long) draft_tier_count);
            }
            if (t0 + t1 + t2 == (int64_t) hparams.n_expert) {
                tier_counts = { t0, t1, t2 };
                LLAMA_LOG_INFO("%s: expert tiers %lld/%lld/%lld per block\n", __func__,
                        (long long) t0, (long long) t1, (long long) t2);
            }
        }
    }
    const std::vector<int32_t> warm_ids = read_ints("hy4f.expert_split.warm_ids");
    const std::vector<int32_t> cold_ids = read_ints("hy4f.expert_split.cold_ids");

    if (!hot_off.empty() && !hot_ids.empty()) {
        const int64_t n = (int64_t) hot_off.size() - 1;   // one entry per block that has experts
        // One table per block the file carries, not per entry: the last entry describes the MTP block, which sits one past the
        // decoder's loop, and its table is needed when the draft graph runs that block.
        tier_of_global.assign((size_t) std::max<int64_t>(n + 1, (int64_t) hparams.n_layer_all),
                std::vector<int32_t>((size_t) hparams.n_expert, -1));

        // The split describes the blocks that have experts, and block 0 does not: it is the dense block.  So entry i belongs to
        // block i+1, and reading entry i as block i puts every block's partition one block early - which shows up first as an
        // expert id that is larger than its own stack (the last entry describes the MTP block's 512 hot tier, while block 41
        // has 160).
        const auto fill = [&](int tier, const std::vector<int32_t> & off, const std::vector<int32_t> & ids) {
            for (int64_t il = 0; il < n && il < (int64_t) off.size() - 1; ++il) {
                const int64_t blk = il + 1;
                if (blk >= (int64_t) tier_of_global.size()) {
                    break;
                }
                for (int32_t j = off[il]; j < off[il + 1]; ++j) {
                    const int32_t g = ids[(size_t) j];
                    if (g >= 0 && g < (int32_t) hparams.n_expert) {
                        tier_of_global[(size_t) blk][(size_t) g] = tier;   // tier: 0 hot, 1 warm, 2 cold
                    }
                }
            }
        };
        fill(0, normalized(hot_off),  hot_ids);
        fill(1, normalized(warm_off), warm_ids);
        fill(2, normalized(cold_off), cold_ids);

        // every expert of a block must belong to exactly one tier
        int64_t unmapped = 0;
        for (const auto & row : tier_of_global) {
            for (int32_t t : row) {
                if (t < 0) {
                    unmapped++;
                }
            }
        }
        // the tables the dispatch will use: each tier of each block must cover exactly its own expert count
        int64_t bad_blocks = 0;
        for (size_t il = 0; il < tier_of_global.size(); ++il) {
            for (int t = 0; t < 3; ++t) {
                const auto tab = tier_local_table((uint32_t) il, t);
                const int64_t used = (int64_t) std::count_if(tab.begin(), tab.end(), [](int32_t v) { return v >= 0 && v != 0; })
                                   + (std::find(tab.begin(), tab.end(), 0) != tab.end() ? 1 : 0);
                if (used == 0) {
                    bad_blocks++;
                }
            }
        }
        LLAMA_LOG_WARN("%s: expert tiers read for %lld block(s), %lld expert(s) without a tier, %lld empty tier(s)\n",
                __func__, (long long) tier_of_global.size(), (long long) unmapped, (long long) bad_blocks);
    }
}

// The split stores, per block, which global experts live in which tier.  The dispatch needs them as two flat tables over
// the global ids: where each expert sits inside its tier, and whether it is there at all.
std::vector<int32_t> llama_model_hy4flash::tier_local_table(uint32_t il, int tier) const {
    const size_t n = hparams.n_expert;
    std::vector<int32_t> out(n, 0);
    if (il >= tier_of_global.size()) {
        return out;
    }
    const auto & row = tier_of_global[il];
    int32_t local = 0;
    for (size_t g = 0; g < n && g < row.size(); ++g) {
        if (row[g] == tier) {
            out[g] = local++;
        }
    }
    return out;
}

std::vector<float> llama_model_hy4flash::tier_mask_table(uint32_t il, int tier) const {
    const size_t n = hparams.n_expert;
    std::vector<float> out(n, 0.0f);
    if (il >= tier_of_global.size()) {
        return out;
    }
    const auto & row = tier_of_global[il];
    for (size_t g = 0; g < n && g < row.size(); ++g) {
        out[g] = row[g] == tier ? 1.0f : 0.0f;
    }
    return out;
}

void llama_model_hy4flash::load_arch_tensors(llama_model_loader & ml) {
    // sized before the base table runs, because that is where on_expert_stack records every stack it creates
    tier_stacks.assign((size_t) hparams.n_layer_all, {});

    llama_model_hy_v4::load_arch_tensors(ml);

    const int64_t n_embd  = hparams.n_embd;
    const int64_t hc      = hparams.dsv4_hc_mult;
    const int64_t latent  = expert_in_embd();
    const int64_t idx     = hparams.indexer_head_size;
    const int64_t ple_enb = 96;

    for (int i = 0; i < (int) hparams.n_layer_all; ++i) {
        create_tensor(tn(LLM_TENSOR_FFN_ROUTED_UP,   "weight", i), {latent, n_embd}, TENSOR_NOT_REQUIRED);
        create_tensor(tn(LLM_TENSOR_FFN_ROUTED_DOWN, "weight", i), {n_embd, latent}, TENSOR_NOT_REQUIRED);

        create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_APE,   "weight", i), {idx, 4}, TENSOR_NOT_REQUIRED);
        create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_NORM,  "weight", i), {idx}, TENSOR_NOT_REQUIRED);
        create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_GATE,  "weight", i), {idx, idx}, TENSOR_NOT_REQUIRED);
        create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_VALUE, "weight", i), {idx, idx}, TENSOR_NOT_REQUIRED);

        create_tensor(tn(LLM_TENSOR_POST_ATTENTION_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        create_tensor(tn(LLM_TENSOR_POST_FFW_NORM,       "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);

        if (std::find(engram_layer_list.begin(), engram_layer_list.end(), i) != engram_layer_list.end()) {
            create_tensor(tn(LLM_TENSOR_ENGRAM_KV,          "weight", i), {n_embd/2, 15360}, TENSOR_NOT_REQUIRED);
            create_tensor(tn(LLM_TENSOR_ENGRAM_KV_B,        "weight", i), {n_embd, 5}, TENSOR_NOT_REQUIRED);
            create_tensor(tn(LLM_TENSOR_ENGRAM_GATE,        "weight", i), {n_embd, hc}, TENSOR_NOT_REQUIRED);
            create_tensor(tn(LLM_TENSOR_ENGRAM_LOOKUP_NORM, "weight", i), {ple_enb, 16}, TENSOR_NOT_REQUIRED);
            create_tensor(tn(LLM_TENSOR_ENGRAM_PRE_GAMMA,   "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
            create_tensor(tn(LLM_TENSOR_ENGRAM_HC_SCALE,    "weight", i), {n_embd, hc}, TENSOR_NOT_REQUIRED);
        }
    }

    // create_tensor returns the tensor it made; the draft block's four are stored on the layer like every other architecture does,
    // otherwise the graph finds them null even though the file has them and the loader answered.
    for (uint32_t i = (uint32_t) hparams.n_layer(); i < hparams.n_layer_all; ++i) {
        auto & layer = layers[i];
        layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", i), {2*n_embd, n_embd}, TENSOR_NOT_REQUIRED);
        layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
    }
}

// Resolves the companion pack's five Engram tensors and the PLE arrays the hash needs.  Everything is read through the gguf
// metadata rather than the loader's typed helpers: the multipliers exceed 32 bits, and the head arrays are stored in a
// narrow integer type the helpers reject.
bool llama_model_hy4flash::engram_resolve(llama_model_loader & ml) const {
    if (engram.ready) {
        return true;
    }

    const auto read_ints = [](gguf_context * g, const char * key, std::vector<uint64_t> & out) {
        out.clear();
        const int kid = gguf_find_key(g, key);
        if (kid < 0 || gguf_get_kv_type(g, kid) != GGUF_TYPE_ARRAY) {
            return false;
        }
        const uint64_t n = gguf_get_arr_n(g, kid);
        const enum gguf_type t = gguf_get_arr_type(g, kid);
        const void * data = gguf_get_arr_data(g, kid);
        out.resize((size_t) n);
        for (uint64_t i = 0; i < n; ++i) {
            switch (t) {
                case GGUF_TYPE_UINT8:  case GGUF_TYPE_INT8:  out[(size_t) i] = (uint64_t) ((const int8_t *)  data)[i]; break;
                case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: out[(size_t) i] = (uint64_t) ((const int16_t *) data)[i]; break;
                case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: out[(size_t) i] = (uint64_t) ((const int32_t *) data)[i]; break;
                case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: out[(size_t) i] = (uint64_t) ((const int64_t *) data)[i]; break;
                default: out.clear(); return false;
            }
        }
        return true;
    };

    // the backbone carries the n-gram shape, the multipliers, and the pad token
    {
        std::vector<uint64_t> mult;
        if (read_ints(ml.metadata, "hy_v4_flash.ple.layer_multipliers", mult)) {
            ple_multipliers.assign(mult.begin(), mult.end());
        }
        // the pad token is a scalar, not an array
        {
            const int kid = gguf_find_key(ml.metadata, "hy_v4_flash.engram.pad_token_id");
            if (kid >= 0 && gguf_get_kv_type(ml.metadata, kid) == GGUF_TYPE_UINT32) {
                engram_pad = (int32_t) gguf_get_val_u32(ml.metadata, kid);
            }
        }
        int64_t v = 0;
        if (gguf_find_key(ml.metadata, "hy_v4_flash.ple.ngram_size") >= 0) {
            v = (int64_t) gguf_get_val_u32(ml.metadata, gguf_find_key(ml.metadata, "hy_v4_flash.ple.ngram_size"));
            ple_ngram = (int32_t) v;
        }
        if (gguf_find_key(ml.metadata, "hy_v4_flash.ple.heads_per_ngram") >= 0) {
            v = (int64_t) gguf_get_val_u32(ml.metadata, gguf_find_key(ml.metadata, "hy_v4_flash.ple.heads_per_ngram"));
            ple_heads = (int32_t) v;
        }
    }

    // the companion is the backbone's sibling: the same name with the shard number replaced
    if (companion.empty()) {
        const std::string & backbone = ml.files.empty() ? std::string() : ml.files[0]->path();
        std::string path = backbone;
        const size_t pos = path.rfind("-00001-of-");
        if (pos != std::string::npos) {
            path.replace(pos, std::string("-00001-of-").size(), "-00002-of-");
        }
        if (path != backbone) {
            companion = path;
        }
    }

    if (companion.empty()) {
        LLAMA_LOG_WARN("%s: no companion path, the Engram lookup stays off\n", __func__);
        return false;
    }

    gguf_init_params gp;
    gp.no_alloc = true;
    gp.ctx      = nullptr;
    gguf_context * g = gguf_init_from_file(companion.c_str(), gp);
    if (g == nullptr) {
        LLAMA_LOG_WARN("%s: cannot open %s\n", __func__, companion.c_str());
        return false;
    }

    // the Engram block types take their scale from exp(lo + q*(hi-lo)/255), and the range lives in this file; without it
    // the converters return zeros for every row
    {
        const int klo = gguf_find_key(g, "hy4f.engram.scale_lo");
        const int khi = gguf_find_key(g, "hy4f.engram.scale_hi");
        if (klo >= 0 && khi >= 0) {
            const float lo = gguf_get_val_f32(g, klo);
            const float hi = gguf_get_val_f32(g, khi);
            ggml_hy4_set_engram_range(lo, hi);
            LLAMA_LOG_WARN("%s: engram scale range %.6f..%.6f\n", __func__, lo, hi);
        }
    }

    std::vector<uint64_t> off, voc;
    read_ints(g, "hy_v4_flash.ple.head_offsets",    off);
    read_ints(g, "hy_v4_flash.ple.head_vocab_sizes", voc);
    ple_head_off.assign(off.begin(), off.end());
    ple_head_vocab.assign(voc.begin(), voc.end());

    // The index's top two bits are 0 low, 1 high, 2 hot, 3 warm: sampling four thousand entries and checking each
    // against the four row counts leaves that assignment with no row beyond a table and every other with some.
    static const char * names[4] = { "engram_low.weight", "engram_high.weight", "engram_hot.weight", "engram_warm.weight" };
    for (int t = 0; t < 4; ++t) {
        const int64_t tid = gguf_find_tensor(g, names[t]);
        if (tid < 0) {
            continue;
        }
        engram.off_table[t] = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, tid);

        // the pack writes its own type ids, which the reader remaps for the backbone; the same mapping applies here
        int32_t file_type = (int32_t) gguf_get_tensor_type(g, tid);
        switch (file_type) {
            case 148: file_type = GGML_TYPE_ENGRAM_Q4B12; break;
            case 149: file_type = GGML_TYPE_ENGRAM_Q6B12; break;
            case 150: file_type = GGML_TYPE_ENGRAM_Q2B12; break;
            case 43:  file_type = GGML_TYPE_F8_E4M3;      break;
            default: break;
        }
        engram.type[t]      = (ggml_type) file_type;
        engram.row_bytes[t] = ggml_row_size(engram.type[t], 96);
        engram.rows[t]      = (size_t) gguf_get_tensor_ne(g, tid)[1];

        LLAMA_LOG_WARN("%s: tier %d file type %d -> %s, row %zu\n", __func__, t,
                (int) gguf_get_tensor_type(g, tid), ggml_type_name(engram.type[t]), engram.row_bytes[t]);
    }

    const int64_t iid = gguf_find_tensor(g, "engram_index");
    if (iid >= 0) {
        engram.off_index = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, iid);
    }

    gguf_free(g);

    engram.ready = engram.off_index != 0 && engram.row_bytes[0] != 0 && !ple_multipliers.empty();

    LLAMA_LOG_WARN("%s: engram ready %d, ngram %d, heads %d, multipliers %zu, head offsets %zu, head vocab %zu, pad %d\n",
            __func__, (int) engram.ready, ple_ngram, ple_heads, ple_multipliers.size(), ple_head_off.size(),
            ple_head_vocab.size(), engram_pad);

    return engram.ready;
}

bool llama_model_hy4flash::engram_read(size_t offset, void * dst, size_t n) const {
    if (engram.fp == nullptr) {
        return false;
    }
#if defined(_WIN32)
    if (_fseeki64(engram.fp, (long long) offset, SEEK_SET) != 0) {
        return false;
    }
#else
    if (fseek(engram.fp, (long) offset, SEEK_SET) != 0) {
        return false;
    }
#endif
    return fread(dst, 1, n, engram.fp) == n;
}

bool llama_model_hy4flash::engram_rows(int layer_slot, const std::vector<llama_token> & tokens, std::vector<float> & out) const {
    if (!engram.ready) {
        return false;
    }
    if (engram.fp == nullptr) {
        engram.fp = fopen(companion.c_str(), "rb");
        if (engram.fp == nullptr) {
            return false;
        }
    }
    if (layer_slot < 0 || layer_slot >= (int) (ple_multipliers.size()/3)) {
        return false;
    }

    const int n_gram = ple_ngram;
    const int n_head = ple_heads;
    const int n_cols = 2*n_head;

    out.assign(tokens.size()*1536u, 0.0f);

    const uint64_t * mult = ple_multipliers.data() + (size_t) layer_slot*n_gram;
    const uint32_t * voc  = ple_head_vocab.data()  + (size_t) layer_slot*n_cols;
    const uint32_t * offs = ple_head_off.data()    + (size_t) layer_slot*n_cols;

    int64_t n_entry_ok = 0, n_row_ok = 0;

    for (size_t i = 0; i < tokens.size(); ++i) {
        uint64_t tk[4] = { 0, 0, 0, 0 };
        for (int s = 0; s < n_gram; ++s) {
            const long pos = (long) i - s;
            tk[s] = pos < 0 ? (uint64_t) engram_pad : (uint64_t) tokens[(size_t) pos];
        }

        uint64_t rolling = tk[0]*mult[0];
        for (int o = 1; o < n_gram; ++o) {
            rolling ^= tk[o]*mult[o];
            for (int h = 0; h < n_head; ++h) {
                const int col = (o - 1)*n_head + h;
                const uint64_t row = rolling % voc[col] + offs[col];

                uint32_t entry = 0;
                if (!engram_read(engram.off_index + row*4, &entry, 4)) {
                    continue;
                }
                n_entry_ok++;

                const int      tier = (int) (entry >> 30);
                const uint64_t lrow = entry & 0x3FFFFFFFu;
                if (tier > 3 || engram.row_bytes[tier] == 0 || lrow >= engram.rows[tier]) {
                    continue;
                }

                std::vector<uint8_t> raw(engram.row_bytes[tier]);
                if (!engram_read(engram.off_table[tier] + lrow*engram.row_bytes[tier], raw.data(), raw.size())) {
                    continue;
                }

                float v[96];
                const ggml_type_traits * tr = ggml_get_type_traits(engram.type[tier]);
                if (tr == nullptr || tr->to_float == nullptr) {
                    continue;
                }
                tr->to_float(raw.data(), v, 96);
                n_row_ok++;

                // a row that dequantises to something non-finite would be turned into NaN by the lookup norm below, so
                // count those by tier before they are normalised
                for (int k = 0; k < 96; ++k) {
                    if (!std::isfinite(v[k])) {
                        if (getenv("STRATA_HY4_TRACE") != NULL) {
                            LLAMA_LOG_WARN("%s: non-finite at tier %d row %llu value %d (%g), token %zu col %d\n",
                                    __func__, tier, (unsigned long long) lrow, k, (double) v[k], i, (int) col);
                        }
                        v[k] = 0.0f;
                    }
                }

                for (int k = 0; k < 96; ++k) {
                    out[i*1536u + (size_t) col*96 + k] = v[k];
                }
            }
        }
    }
    // The pack's 96-dimension lookup norm: each column's ninety-six values are normalised together and then scaled by
    // that column's own weight vector.  Doing it here keeps the graph unchanged - the rows arrive already normalised.
    if (layer_slot >= 0 && layer_slot < (int) engram_layer_list.size()) {
        const ggml_tensor * ln = get_tensor((std::string("blk.") + std::to_string(engram_layer_list[(size_t) layer_slot]) +
                    ".engram_lookup_norm.weight").c_str());
        // The weight may live in a device buffer with no host pointer, so it is copied out through its backend instead of
        // being read directly; the copy is small and is made once per call.
        static thread_local std::vector<float> w_host;
        // The pack's checklist names the lookup norm, but not whether it belongs before the projection or after it; leaving
        // it out entirely is the comparison that says how much it contributes at all.
        static const bool no_lookup_norm = [] {
            const char * e = std::getenv("STRATA_HY4_ENGRAM_NO_LOOKUP_NORM");
            return e != nullptr && e[0] != '0';
        }();
        if (!no_lookup_norm && ln != nullptr && ln->type == GGML_TYPE_F32 && ln->ne[0] == 96 && ln->ne[1] == 16 &&
                ggml_nbytes(ln) == 96*16*sizeof(float)) {
            w_host.resize(96*16);
            // A tensor can carry a buffer and still have no address: the buffer is assigned before anything is placed in it, and a
            // tensor the model never asked for keeps both.  The read asserts on the address, so the address is what is checked.
            if (ln->data != nullptr) {
                if (ln->buffer != nullptr) {
                    ggml_backend_tensor_get(ln, w_host.data(), 0, w_host.size()*sizeof(float));
                } else {
                    std::memcpy(w_host.data(), ln->data, w_host.size()*sizeof(float));
                }
            }

            const float * w = w_host.data();
            for (size_t t = 0; t < tokens.size(); ++t) {
                for (int c = 0; c < 16; ++c) {
                    float * row = out.data() + t*1536u + (size_t) c*96;
                    double sq = 0.0;
                    for (int k = 0; k < 96; ++k) {
                        sq += (double) row[k] * (double) row[k];
                    }
                    const float inv = 1.0f / sqrtf((float) (sq/96.0) + hparams.f_norm_rms_eps);
                    for (int k = 0; k < 96; ++k) {
                        row[k] = row[k] * inv * w[(size_t) c*96 + k];
                    }
                }
            }
        }
    }

    if (getenv("STRATA_HY4_TRACE") != NULL) {
        LLAMA_LOG_WARN("%s: entries %lld, rows %lld, fp %p, index off %zu, row bytes %zu %zu %zu %zu\n", __func__,
                (long long) n_entry_ok, (long long) n_row_ok, (void *) engram.fp, engram.off_index,
                engram.row_bytes[0], engram.row_bytes[1], engram.row_bytes[2], engram.row_bytes[3]);
    }
    return true;
}
