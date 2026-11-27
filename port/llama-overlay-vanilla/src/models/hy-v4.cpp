#include "models.h"

#include <cstdlib>
#include <set>
#include <string>

#include <cstdio>
#include "llama-kv-cache-dsa-iswa.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-dsa.h"

#include <cmath>

// iHC (independent Hyper-Connections) helpers. Same layout as the DeepSeek-V4 HC, but without
// the comb/sinkhorn term: hc_fn makes only 2*hc coefficients (pre + post). The streams mix
// through the pre-reduce / post-distribute round trip instead.

static size_t hy_v4_elem_offset(const ggml_tensor * t, int64_t i) {
    return ggml_row_size(t->type, i);
}

static ggml_tensor * hy_v4_view_1d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t i0) {
    return ggml_view_1d(ctx, t, ne0, hy_v4_elem_offset(t, i0));
}

static ggml_tensor * hy_v4_view_2d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t ne1, int64_t i0) {
    return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], hy_v4_elem_offset(t, i0));
}

void llama_model_hy_v4::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,      hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,    hparams.n_embd_head_k_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,  hparams.n_embd_head_v_mla_impl);
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm,  false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func,   false);

    // routed-expert SwiGLU logits clamp (shared/dense experts are NOT clamped, so
    // swiglu_clamp_shexp is intentionally left at its 0 default)
    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP, hparams.swiglu_clamp_exp, hparams.n_layer_all, false);

    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,     hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,   hparams.dsv4_hc_eps);
    ml.get_key(LLM_KV_HYPER_CONNECTION_MAGNITUDE, hparams.hc_magnitude);

    // DSA is absent on the all-full_attention checkpoints, so indexer_top_k stays 0 there
    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head,    false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size, false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k,     false);

    if (hparams.indexer_top_k > 0) {
        // the reference plumbs rms_norm_eps into the indexer k_norm LayerNorm, and build_norm
        // reads f_norm_eps for LLM_NORM
        hparams.f_norm_eps = hparams.f_norm_rms_eps;

        if (hparams.indexer_n_head == 0 || hparams.indexer_head_size <= hparams.n_rot()) {
            throw std::runtime_error("hy_v4: bad indexer head count / key length");
        }

        ml.get_key_or_arr(LLM_KV_ATTENTION_INDEXER_TYPES, hparams.is_indexer_full_impl, hparams.n_layer(), false);
        if (!hparams.is_indexer_full(0)) {
            throw std::runtime_error("hy_v4: layer 0 must own an indexer, nothing precedes it to share");
        }
    }

    GGML_ASSERT(hparams.is_mla());

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_hy_v4::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_head_k_mla   = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla   = hparams.n_embd_head_v_mla();
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;
    GGML_ASSERT(n_embd_head_qk_nope >= 1);

    const int64_t q_lora_rank     = hparams.n_lora_q;
    const int64_t kv_lora_rank    = hparams.n_lora_kv;
    const int64_t n_ff_exp        = hparams.n_ff_exp();
    const int64_t n_expert_shared = hparams.n_expert_shared;
    const int64_t hc              = hparams.dsv4_hc_mult;

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    // global iHC head (collapses hc streams before the final norm)
    hc_head_fn    = create_tensor(tn(LLM_TENSOR_HC_HEAD_FN,    "weight"), {hc * n_embd, hc}, 0);
    hc_head_base  = create_tensor(tn(LLM_TENSOR_HC_HEAD_BASE,  "weight"), {hc}, 0);
    hc_head_scale = create_tensor(tn(LLM_TENSOR_HC_HEAD_SCALE, "weight"), {1}, 0);

    for (int i = 0; i < (int) tensor_layers(); ++i) {
        auto & layer = layers[i];

        // the head count is per block in this family: a pack may mix 32-head and 64-head blocks
        const int64_t n_head_i = hparams.n_head(i);

        layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM,     "weight", i), {n_embd}, 0);
        layer.attn_sinks    = create_tensor(tn(LLM_TENSOR_ATTN_SINKS,    "weight", i), {n_head_i}, 0);

        layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", i), {n_embd, q_lora_rank}, 0);
        layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, 0);
        layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", i), {q_lora_rank, n_head_i * n_embd_head_k_mla}, 0);
        layer.wkv_a_mqa     = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), {n_embd, kv_lora_rank + n_embd_head_qk_rope}, 0);
        layer.attn_kv_a_norm= create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM,"weight", i), {kv_lora_rank}, 0);
        layer.wk_b          = create_tensor(tn(LLM_TENSOR_ATTN_K_B,      "weight", i), {n_embd_head_qk_nope, kv_lora_rank, n_head_i}, 0);
        layer.wv_b          = create_tensor(tn(LLM_TENSOR_ATTN_V_B,      "weight", i), {kv_lora_rank, n_embd_head_v_mla, n_head_i}, 0);
        layer.wo            = create_tensor(tn(LLM_TENSOR_ATTN_OUT,      "weight", i), {n_head_i * n_embd_head_v_mla, n_embd}, 0);
        layer.wqkv_gate     = create_tensor(tn(LLM_TENSOR_ATTN_GATE,     "weight", i), {n_embd, n_head_i * n_embd_head_v_mla}, 0);

        // only "full" indexer layers ship weights; "shared" layers reuse their top-k
        if (hparams.indexer_top_k > 0 && hparams.is_indexer_full(i)) {
            const int64_t n_indexer_head = hparams.indexer_n_head;
            const int64_t n_embd_indexer = hparams.indexer_head_size;

            layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, n_indexer_head * n_embd_indexer}, 0);
            layer.indexer_attn_k   = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K,   "weight", i), {n_embd, n_embd_indexer}, 0);
            layer.indexer_k_norm   = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "weight", i), {n_embd_indexer}, 0);
            layer.indexer_k_norm_b = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "bias",   i), {n_embd_indexer}, TENSOR_NOT_REQUIRED);
            layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, n_indexer_head}, 0);
        }

        if (block_has_hc(i)) {
            layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc * n_embd, 2 * hc}, 0);
            layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {2 * hc}, 0);
            layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {2}, 0);
            layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc * n_embd, 2 * hc}, 0);
            layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {2 * hc}, 0);
            layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {2}, 0);
        }

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        if (i < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
        } else {
            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, TENSOR_NOT_REQUIRED);

            if (n_expert == 0) {
                throw std::runtime_error("n_expert must be > 0");
            }
            if (n_expert_used == 0) {
                throw std::runtime_error("n_expert_used must be > 0");
            }

            // one routed stack per tier: a single stack for the plain hy_v4 packs, hot/warm/cold for a split variant
            const int64_t in_embd = expert_in_embd();
            for (int t = 0; t < expert_tiers(); ++t) {
                llm_tensor e_gate = LLM_TENSOR_FFN_GATE_EXPS;
                llm_tensor e_up   = LLM_TENSOR_FFN_UP_EXPS;
                llm_tensor e_down = LLM_TENSOR_FFN_DOWN_EXPS;
                expert_tier_enums(t, e_gate, e_up, e_down);

                const int64_t n_exp = expert_tier_count(t, i);
                if (n_exp <= 0) {
                    continue;   // this variant has no such tier on this block
                }
                ggml_tensor * gate = create_tensor(tn(e_gate, "weight", i), {in_embd, n_ff_exp, n_exp}, 0);
                ggml_tensor * up   = create_tensor(tn(e_up,   "weight", i), {in_embd, n_ff_exp, n_exp}, 0);
                ggml_tensor * down = create_tensor(tn(e_down, "weight", i), {n_ff_exp, in_embd, n_exp}, 0);

                if (t == 0) {
                    layer.ffn_gate_exps = gate;
                    layer.ffn_up_exps   = up;
                    layer.ffn_down_exps = down;
                }
                on_expert_stack((uint32_t) i, t, 0, gate);
                on_expert_stack((uint32_t) i, t, 1, up);
                on_expert_stack((uint32_t) i, t, 2, down);
            }

            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd}, 0);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
        }
    }
}

llama_rope_params llama_model_hy_v4::get_rope_params_indexer(const llama_cparams & cparams, int il) const {
    auto res = llama_model::get_rope_params_indexer(cparams, il);

    // [nope | rope], the same as the MLA keys
    res.n_offs = hparams.indexer_head_size - res.n_rot;
    res.type   = hparams.rope_type;

    return res;
}

std::unique_ptr<llm_graph_context> llama_model_hy_v4::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(params);
    }
    return std::make_unique<graph>(params);
}

// The draft graph: the trunk's last hidden state and the next token go through the two norms, the projection and block 42, then
// the shared head.  It is being built against the reference in qwen4exp, which is the same family and already has one.
llama_model_hy_v4::graph_mtp::graph_mtp(const llm_graph_params & params) : graph(params, no_build{}) {
    LLAMA_LOG_WARN("%s: draft graph begins\n", __func__);
    // Block 42 is the draft block: it is not part of the decoder's loop, and it never was - the loop runs n_layer() blocks and
    // this one sits at exactly that index.
    const int il_mtp = (int) hparams.n_layer();
    const auto & layer = model.layers[il_mtp];
    const int64_t hc = hparams.dsv4_hc_mult;
    const float kq_scale = 1.0f / sqrtf(float(hparams.n_embd_head_k_mla()));

    // Inputs: the trunk's last hidden state (n_embd by n_tokens) and the token whose successor the draft is for.
    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd_inp(), hparams.n_embd_out());
    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);
    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);
    LLAMA_LOG_WARN("%s: draft graph: n_embd_inp %lld n_embd_out %lld n_layer %u nextn %u\n", __func__,
            (long long) hparams.n_embd_inp(), (long long) hparams.n_embd_out(),
            hparams.n_layer(), hparams.n_layer_nextn);
    inp->h = build_inp_embd_state(inp.get(), hparams.n_embd_out());
    LLAMA_LOG_WARN("%s: draft graph: state input %p\n", __func__, (void *) inp->h);
    // Both inputs are captured before the object is handed to the result: after the move the pointer is empty, and reading either
    // tensor through it is a null dereference.
    ggml_tensor * h_in   = inp->h;
    ggml_tensor * tok_in = inp->tokens;
    res->add_input(std::move(inp));

    ggml_tensor * inp_pos = build_inp_pos();

    LLAMA_LOG_WARN("%s: draft graph: attention inputs next\n", __func__);
    llm_graph_input_attn_k     * inp_attn     = nullptr;
    llm_graph_input_attn_k_dsa * inp_attn_dsa = nullptr;
    llm_graph_input_attn_k     * inp_attn_swa = nullptr;
    const bool is_dsa = hparams.indexer_top_k > 0;
    const bool memory_is_hybrid = dynamic_cast<const llama_kv_cache_dsa_iswa_context *>(params.mctx) != nullptr;
    LLAMA_LOG_WARN("%s: draft graph: mctx %p is_dsa %d hybrid %d\n", __func__, (void *) params.mctx, (int) is_dsa, (int) memory_is_hybrid);
    if (!is_dsa) {
        inp_attn = build_attn_inp_k();
    // The paired input carries a windowed half as well; a block that is not windowed never uses it, and the tensors it builds are then
    // not part of the graph at all - so it must not be the form chosen for such a block.
    // The paired builder is the one that knows this context's memory: it produces an input for each half with the right context
    // behind it. Asking for the DSA-only input against a paired memory hands the graph a context of the wrong kind.
    } else if (memory_is_hybrid && std::getenv("STRATA_HY4_MTP_PLAIN_ATTN") == nullptr) {
        auto * bi = build_attn_inp_k_dsa_iswa();
        LLAMA_LOG_WARN("%s: draft graph: paired input %p\n", __func__, (void *) bi);
        inp_attn_dsa = bi->get_dsa();
        inp_attn_swa = bi->get_swa();
        LLAMA_LOG_WARN("%s: draft graph: attn %p swa %p dsa %p\n", __func__, (void *) inp_attn, (void *) inp_attn_swa, (void *) inp_attn_dsa);
    } else {
        inp_attn_dsa = build_attn_inp_k_dsa();
    }

    ggml_tensor * last_top_k = nullptr;
    const std::map<size_t, ggml_tensor *> no_engram;

    // Everything this graph touches, printed once: a null in this list is what a crash inside the constructor would be about, and
    // naming it here costs one line of log instead of another round of stack reading.
    LLAMA_LOG_WARN("%s: draft graph: h %p tokens %p eh_proj %p enorm %p hnorm %p head_norm %p tok_embd %p output %p\n", __func__,
            (void *) h_in, (void *) tok_in,
            (void *) layer.nextn.eh_proj, (void *) layer.nextn.enorm, (void *) layer.nextn.hnorm,
            (void *) layer.nextn.shared_head_norm, (void *) model.tok_embd, (void *) model.output);

    // Projection: the hidden state and the token embedding, normed, concatenated embedding first, projected back to the width the
    // block expects.
    // the state arrives as n_embd by tokens already, so it is normed as it is: reshaping it to three dimensions here puts the token
    // axis where the concatenation does not expect it
    ggml_tensor * h_n = build_norm(h_in, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il_mtp);
    cb(h_n, "mtp_hnorm", il_mtp);
    ggml_tensor * e_r = ggml_get_rows(ctx0, model.tok_embd, tok_in);
    // the embedding table stores rotated rows, so the draft's lookup needs the same inverse transform the decoder applies before
    // anything norms it: without it the table is read in a basis its rows were not written in
    static const bool no_invrot = [] {
        const char * e = std::getenv("STRATA_HY4_MTP_NO_INVROT");
        return e != nullptr && e[0] != '0';
    }();
    if (hdmd && !no_invrot) {
        if (const auto it = hdmd->inv.find(model.tok_embd); it != hdmd->inv.end()) {
            e_r = llama_mul_mat_hadamard(ctx0, e_r, it->second.rot);
            if (it->second.signs) {
                e_r = ggml_mul(ctx0, e_r, it->second.signs);
            }
        }
    }
    ggml_tensor * e_n = build_norm(e_r, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il_mtp);
    cb(e_n, "mtp_enorm", il_mtp);
    LLAMA_LOG_WARN("%s: draft head: e_r {%lld,%lld,%lld,%lld} e_n {%lld,%lld,%lld,%lld} h_n {%lld,%lld,%lld,%lld}\n", __func__,
            (long long) e_r->ne[0], (long long) e_r->ne[1], (long long) e_r->ne[2], (long long) e_r->ne[3],
            (long long) e_n->ne[0], (long long) e_n->ne[1], (long long) e_n->ne[2], (long long) e_n->ne[3],
            (long long) h_n->ne[0], (long long) h_n->ne[1], (long long) h_n->ne[2], (long long) h_n->ne[3]);
    ggml_tensor * res_hc = build_lora_mm(layer.nextn.eh_proj, ggml_concat(ctx0, e_n, h_n, 0));
    cb(res_hc, "mtp_eh_proj", il_mtp);

    // the draft block carries no hyper-connections, so it takes the single stream rather than the decoder's hc of them
    res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, 1, n_tokens);
    cb(res_hc, "mtp_inp", il_mtp);
    ggml_tensor * cur = build_block(il_mtp, res_hc, is_dsa, inp_pos, kq_scale, inp_attn, inp_attn_swa, inp_attn_dsa,
            last_top_k, no_engram);
    cb(cur, "mtp_block", il_mtp);

    // The head: the drafts' own norm, then the shared output weight.
    // the block's output is n_embd wide per stream, and a draft block carries one stream rather than hc of them, so the flattened
    // width is taken from the tensor rather than assumed
    ggml_tensor * flat = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], n_tokens);
    res->t_h_nextn = flat;
    cb(res->t_h_nextn, "h_nextn", il_mtp);

    ggml_tensor * normed = build_norm(flat, layer.nextn.shared_head_norm, nullptr, LLM_NORM_RMS, il_mtp);
    cb(normed, "mtp_head_norm", il_mtp);
    // the draft's own embedding output is only needed when a caller asks for it; making it unconditional puts a second output on the
    // graph every run, and a node nothing consumes is a node the scheduler may leave without a buffer
    static const bool no_embd_out = [] {
        const char * e = std::getenv("STRATA_HY4_MTP_NO_EMBD");
        return e != nullptr && e[0] != '0';
    }();
    if (!no_embd_out) {
        res->t_embd = normed;
    }

    ggml_tensor * logits = build_lora_mm(model.output, normed);
    cb(logits, "result_output", -1);
    res->t_logits = logits;
}

// reduce hc streams x[:,i,:] weighted by w[i,:] -> [n_embd, n_tokens]
// reference runs this in fp32 (inside the float() / autocast(fp32) context)
static ggml_tensor * hy_v4_hc_reduce(ggml_context * ctx0, ggml_tensor * x, ggml_tensor * w, int64_t hc, int64_t n_embd, int64_t nt, ggml_type out_type) {
    ggml_tensor * x_f32 = ggml_cast(ctx0, x, GGML_TYPE_F32);
    static const bool hc_normalised = [] {
        const char * e = std::getenv("STRATA_HY4_HC_NORM");
        return e != nullptr && e[0] != '0';
    }();

    ggml_tensor * result = nullptr;
    for (int64_t ih = 0; ih < hc; ++ih) {
        ggml_tensor * xh = ggml_view_2d(ctx0, x_f32, n_embd, nt, x_f32->nb[2], ih * x_f32->nb[1]);
        ggml_tensor * wh = ggml_view_2d(ctx0, w, 1, nt, w->nb[1], ih * w->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, xh, wh);
        result = result ? ggml_add(ctx0, result, cur) : cur;
    }
    if (hc_normalised) {
        // divide by the total gate so the streams combine as a weighted average
        ggml_tensor * w_f32 = ggml_cast(ctx0, w, GGML_TYPE_F32);
        ggml_tensor * sum   = ggml_scale_bias(ctx0, ggml_sum_rows(ctx0, w_f32), 1.0f, 1e-6f);
        ggml_tensor * denom = ggml_reshape_2d(ctx0, sum, 1, nt);
        result = ggml_div(ctx0, result, denom);
    }

    return ggml_cast(ctx0, result, out_type);
}

llama_model_hy_v4::graph::hc_pre
llama_model_hy_v4::graph::build_hc_pre(
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base,
        int il) const {
    const int64_t hc  = hparams.dsv4_hc_mult;
    const int64_t nt  = x->ne[2];
    GGML_ASSERT(x->ne[0] == n_embd && x->ne[1] == (il < (int) hparams.n_layer() ? hc : 1));

    if (il >= (int) hparams.n_layer()) {
        // plain pre-norm: the sublayer reads the single stream, and the post step adds it back unchanged
        LLAMA_LOG_WARN("%s: plain pre il %d: x {%lld,%lld,%lld,%lld} -> {%lld,%lld}\n", __func__, il,
                (long long) x->ne[0], (long long) x->ne[1], (long long) x->ne[2], (long long) x->ne[3],
                (long long) n_embd, (long long) nt);
        hc_pre plain;
        plain.cur  = ggml_reshape_2d(ctx0, x, n_embd, nt);
        plain.post = nullptr;
        return plain;
    }

    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, hc * n_embd, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, hparams.f_norm_rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx0, hc_fn, flat_norm); // [2*hc, nt]
    cb(mixes, "hc_mixes", il);

    ggml_tensor * scale_pre  = hy_v4_view_1d(ctx0, hc_scale, 1, 0);
    ggml_tensor * scale_post = hy_v4_view_1d(ctx0, hc_scale, 1, 1);
    ggml_tensor * base_pre   = hy_v4_view_1d(ctx0, hc_base, hc, 0);
    ggml_tensor * base_post  = hy_v4_view_1d(ctx0, hc_base, hc, hc);

    // Which half of the mixes belongs to the pre gate and which to the post gate is the one piece the pack carries no
    // reference for, so the other reading can be tried for a run without a rebuild.
    // The reading that produced coherent text on the test turn is the default; STRATA_HY4_HC_SWAP=0 restores the other
    // one, which is kept because a single sample is weak evidence and the two are one switch apart.
    static const bool hc_swap = [] {
        const char * e = std::getenv("STRATA_HY4_HC_SWAP");
        return e == nullptr || e[0] != '0';
    }();
    const int64_t half_pre  = hc_swap ? hc : 0;
    const int64_t half_post = hc_swap ? 0  : hc;

    // pre = sigmoid(mixes[:hc]*scale_pre + base_pre) + eps
    ggml_tensor * pre = hy_v4_view_2d(ctx0, mixes, hc, nt, half_pre);
    pre = ggml_mul(ctx0, pre, scale_pre);
    pre = ggml_add(ctx0, pre, base_pre);
    pre = ggml_sigmoid(ctx0, pre);
    pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.dsv4_hc_eps);
    cb(pre, "hc_pre", il);

    // post = magnitude*sigmoid(mixes[hc:2hc]*scale_post + base_post) + eps
    ggml_tensor * post = hy_v4_view_2d(ctx0, mixes, hc, nt, half_post);
    post = ggml_mul(ctx0, post, scale_post);
    post = ggml_add(ctx0, post, base_post);
    post = ggml_sigmoid(ctx0, post);
    post = ggml_scale(ctx0, post, hparams.hc_magnitude);
    post = ggml_scale_bias(ctx0, post, 1.0f, hparams.dsv4_hc_eps);
    cb(post, "hc_post_gate", il);

    // Whether a sublayer reads the gated sum of the streams or one stream directly is another reading the pack does not
    // settle; the first stream is the alternative and it is one switch away.
    static const bool hc_pick0 = [] {
        const char * e = std::getenv("STRATA_HY4_HC_PICK0");
        return e != nullptr && e[0] != '0';
    }();
    if (hc_pick0) {
        ggml_tensor * first = ggml_cont(ctx0, ggml_view_2d(ctx0, x, n_embd, nt, x->nb[2], 0));
        return { first, post };
    }

    return { hy_v4_hc_reduce(ctx0, x, pre, hc, n_embd, nt, x->type), post };
}

ggml_tensor * llama_model_hy_v4::graph::build_hc_post(
        ggml_tensor * x,
        ggml_tensor * residual,
        ggml_tensor * post,
        int il) const {
    const int64_t hc = hparams.dsv4_hc_mult;
    ggml_tensor * cur = nullptr;
    const int64_t nt = x->ne[1];
    GGML_ASSERT(x->ne[0] == n_embd);

    if (il >= (int) hparams.n_layer()) {
        // no streams on this layer: the sublayer's output is added straight back, which is what a post-mix of one stream does
        LLAMA_LOG_WARN("%s: plain post il %d: x {%lld,%lld,%lld,%lld} residual {%lld,%lld,%lld,%lld}\n", __func__, il,
                (long long) x->ne[0], (long long) x->ne[1], (long long) x->ne[2], (long long) x->ne[3],
                (long long) residual->ne[0], (long long) residual->ne[1], (long long) residual->ne[2], (long long) residual->ne[3]);
        GGML_UNUSED(post);
        // both sides are given the same four-dimensional shape so a sublayer output that arrives flat and a residual that arrives
        // with a stream axis still add: at reservation time the token count is the batch's, not one
        const int64_t nt_x = x->ne[1];
        ggml_tensor * a = ggml_reshape_4d(ctx0, x, n_embd, 1, nt_x, 1);
        ggml_tensor * b = ggml_reshape_4d(ctx0, residual, n_embd, 1, nt_x, 1);
        return ggml_add(ctx0, b, a);
    }
    GGML_ASSERT(residual->ne[1] == hc);

    // reference HC post runs entirely in fp32 to avoid bf16 rounding accumulation
    // across 78 layers: post.float() * x.float() + residual.float() -> .to(dtype)
    ggml_tensor * x_f32    = ggml_cast(ctx0, x, GGML_TYPE_F32);
    ggml_tensor * post_f32 = ggml_cast(ctx0, post, GGML_TYPE_F32);
    ggml_tensor * res_f32  = ggml_cast(ctx0, residual, GGML_TYPE_F32);

    ggml_tensor * out = nullptr;
    for (int64_t i = 0; i < hc; ++i) {
        ggml_tensor * res_i  = ggml_view_2d(ctx0, res_f32, n_embd, nt, res_f32->nb[2], i * res_f32->nb[1]);
        ggml_tensor * post_i = ggml_view_2d(ctx0, post_f32, 1, nt, post_f32->nb[1], i * post_f32->nb[0]);
        ggml_tensor * cur = ggml_add(ctx0, res_i, ggml_mul(ctx0, x_f32, post_i));
        cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, nt);
        out = out ? ggml_concat(ctx0, out, cur, 1) : cur;
    }

    // cast back to the original type (bf16)
    out = ggml_cast(ctx0, out, residual->type);
    return out; // [n_embd, hc, nt]
}

ggml_tensor * llama_model_hy_v4::graph::build_hc_head(
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base) const {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];

    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, hc * n_embd, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, hparams.f_norm_rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx0, hc_fn, flat_norm); // [hc, nt]
    cb(mixes, "hc_head_mixes", -1);

    ggml_tensor * pre = ggml_mul(ctx0, mixes, hc_scale);
    pre = ggml_add(ctx0, pre, hc_base);
    pre = ggml_sigmoid(ctx0, pre);
    pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.dsv4_hc_eps);

    // Whether the head weights the streams by that gate at all is another reading the pack does not settle: the plain
    // mean of the streams is the alternative, and it is one switch away.
    static const bool head_mean = [] {
        const char * e = std::getenv("STRATA_HY4_HEAD_MEAN");
        return e != nullptr && e[0] != '0';
    }();
    if (head_mean) {
        pre = ggml_scale_bias(ctx0, pre, 0.0f, 1.0f/float(hc));
    }
    cb(pre, "hc_head_pre", -1);

    return hy_v4_hc_reduce(ctx0, x, pre, hc, n_embd, nt, x->type);
}

ggml_tensor * llama_model_hy_v4::graph::build_attention(
        llm_graph_input_attn_k * inp_attn,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        float kq_scale,
        int il) const {
    const auto & layer = model.layers[il];

    const int64_t n_embd_head_k       = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;
    const uint32_t kv_lora_rank       = hparams.n_lora_kv;
    // the head count is per block in this family
    const int64_t n_head_il = hparams.n_head(il);
    if (getenv("STRATA_GGML_SHAPE_TRACE") != NULL && il < 4) {
        LLAMA_LOG_WARN("%s: il %d n_head %lld (default %u, arr0 %u arr2 %u)\n", __func__, il,
                (long long) n_head_il, hparams.n_head(), hparams.n_head_arr[0], hparams.n_head_arr[2]);
    }

    ggml_tensor * q = build_lora_mm(layer.wq_a, cur);
    q = build_norm(q, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    q = build_lora_mm(layer.wq_b, q);

    ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head_il, n_tokens,
            ggml_row_size(q->type, n_embd_head_k), ggml_row_size(q->type, n_embd_head_k) * n_head_il, 0);
    ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head_il, n_tokens,
            ggml_row_size(q->type, n_embd_head_k), ggml_row_size(q->type, n_embd_head_k) * n_head_il,
            ggml_row_size(q->type, n_embd_head_qk_nope));

    ggml_tensor * kv_cmpr_pe = build_lora_mm(layer.wkv_a_mqa, cur);
    ggml_tensor * kv_cmpr = ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
    ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));

    q_pe = build_rope(q_pe, inp_pos, il);
    cb(q_pe, "q_pe", il);
    k_pe = build_rope(k_pe, inp_pos, il);
    cb(k_pe, "k_pe", il);

    kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(kv_cmpr, "kv_cmpr", il);

    // MLA absorption: q_nope @ wk_b -> compressed space
    q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
    ggml_tensor * q_nope_absorbed = build_lora_mm(layer.wk_b, q_nope);
    q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);

    // note: rope must go first for in-place context shifting in build_rope_shift()
    ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);

    kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
    ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
    ggml_tensor * Vcur = kv_cmpr;

    // MLA-as-MQA; wo applied manually below so the gated-MLA gate can sit before o_proj
    // The sinks are extra logits in the softmax denominator; whether this pack means them to apply is one switch away.
    static const bool no_sinks = [] {
        const char * e = std::getenv("STRATA_HY4_NO_SINKS");
        return e != nullptr && e[0] != '0';
    }();
    ggml_tensor * attn = build_attn(inp_attn,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, no_sinks ? nullptr : layer.attn_sinks, layer.wv_b, kq_scale, il);
    cb(attn, "attn_kqv", il); // [n_head_il * n_embd_head_v, n_tokens]

    // gated MLA: elementwise gate on the decompressed attention output.  Which nonlinearity the pack means is one switch
    // away - the sigmoid saturates at one, a silu does not.
    static const bool gate_silu = [] {
        const char * e = std::getenv("STRATA_HY4_ATTN_GATE_SILU");
        return e != nullptr && e[0] != '0';
    }();
    static const bool no_gate = [] {
        const char * e = std::getenv("STRATA_HY4_NO_ATTN_GATE");
        return e != nullptr && e[0] != '0';
    }();
    if (!no_gate) {
        ggml_tensor * gate = build_lora_mm(layer.wqkv_gate, cur);
        gate = gate_silu ? ggml_silu(ctx0, gate) : ggml_sigmoid(ctx0, gate);
        attn = ggml_mul(ctx0, attn, gate);
    }
    cb(attn, "attn_gated", il);

    ggml_tensor * out = build_lora_mm(layer.wo, attn);
    cb(out, "attn_out", il);

    return out;
}

ggml_tensor * llama_model_hy_v4::graph::build_indexer_top_k(
        llm_graph_input_attn_k_dsa * inp_attn_dsa,
        ggml_tensor * cur,
        ggml_tensor * qr,
        ggml_tensor * inp_pos,
        int il) const {
    const auto & layer = model.layers[il];

    const int64_t n_indexer_head       = hparams.indexer_n_head;
    const int64_t n_embd_indexer       = hparams.indexer_head_size;

    // nope rows come first, so rope only the last n_embd_indexer_rope rows, same as the MLA path
    ggml_tensor * iq = ggml_mul_mat(ctx0, layer.indexer_attn_q_b, qr);

    iq = ggml_reshape_3d(ctx0, iq, n_embd_indexer, n_indexer_head, n_tokens);

    iq = build_rope_indexer(iq, inp_pos, il);
    cb(iq, "indexer_q", il);

    ggml_tensor * ik = ggml_mul_mat(ctx0, layer.indexer_attn_k, cur);

    ik = build_norm(ik, layer.indexer_k_norm, layer.indexer_k_norm_b, LLM_NORM, il);

    ik = ggml_reshape_3d(ctx0, ik, n_embd_indexer, 1, n_tokens);

    ik = build_rope_indexer(ik, inp_pos, il);
    cb(ik, "indexer_k", il);

    // the reference applies a Hadamard rotation here, but it only helps its FP8 kernels.
    // it is orthogonal, so it does not change q.k and we can skip it.

    const auto * mctx_lid   = inp_attn_dsa->mctx->get_lid();
    const auto & k_idxs_lid = inp_attn_dsa->get_k_idxs_lid();
    ggml_build_forward_expand(gf, mctx_lid->cpy_k(ctx0, ik, k_idxs_lid, il));

    ggml_tensor * iw = ggml_mul_mat(ctx0, layer.indexer_proj, cur);

    ik = mctx_lid->get_k(ctx0, il);

    const auto n_stream = ik->ne[3];
    iq = ggml_view_4d(ctx0, iq, iq->ne[0], iq->ne[1], iq->ne[2]/n_stream, n_stream,
            iq->nb[1], iq->nb[2], iq->nb[3]/n_stream, 0);
    iw = ggml_view_4d(ctx0, iw, iw->ne[0], iw->ne[1]/n_stream, iw->ne[2], n_stream,
            iw->nb[1], iw->nb[2]/n_stream, iw->nb[3]/n_stream, 0);

    // fold both reference scale factors into the weights before the big score tensor
    iw = ggml_scale(ctx0, iw, 1.0f / sqrtf(float(n_embd_indexer * n_indexer_head)));

    ggml_tensor * score = ggml_lightning_indexer(ctx0, iq, ik, iw, inp_attn_dsa->self_attn_mask_lid.raw);
    cb(score, "indexer_score", il);

    const uint32_t n_top_k = score->ne[0] < (int64_t) hparams.indexer_top_k ? score->ne[0] : hparams.indexer_top_k;

    return ggml_cont(ctx0, ggml_top_k(ctx0, score, n_top_k));
}

llama_model_hy_v4::graph::attn_dsa
llama_model_hy_v4::graph::build_attention_dsa(
        llm_graph_input_attn_k_dsa * inp_attn_dsa,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        ggml_tensor * last_top_k,
        float kq_scale,
        int il) const {
    const auto & layer = model.layers[il];

    const int64_t n_embd_head_k       = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;
    const uint32_t kv_lora_rank       = hparams.n_lora_kv;
    const int64_t n_head_il = hparams.n_head(il);

    ggml_tensor * qr = build_lora_mm(layer.wq_a, cur);
    qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);

    ggml_tensor * top_k = last_top_k;
    if (hparams.is_indexer_full(il)) {
        top_k = build_indexer_top_k(inp_attn_dsa, cur, qr, inp_pos, il);
        cb(top_k, "top_k", il);
    }
    GGML_ASSERT(top_k != nullptr);

    ggml_tensor * q = build_lora_mm(layer.wq_b, qr);

    ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head_il, n_tokens,
            ggml_row_size(q->type, n_embd_head_k), ggml_row_size(q->type, n_embd_head_k) * n_head_il, 0);
    ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head_il, n_tokens,
            ggml_row_size(q->type, n_embd_head_k), ggml_row_size(q->type, n_embd_head_k) * n_head_il,
            ggml_row_size(q->type, n_embd_head_qk_nope));

    ggml_tensor * kv_cmpr_pe = build_lora_mm(layer.wkv_a_mqa, cur);
    ggml_tensor * kv_cmpr = ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
    ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
            ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));

    q_pe = build_rope(q_pe, inp_pos, il);
    cb(q_pe, "q_pe", il);
    k_pe = build_rope(k_pe, inp_pos, il);
    cb(k_pe, "k_pe", il);

    kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(kv_cmpr, "kv_cmpr", il);

    q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
    ggml_tensor * q_nope_absorbed = build_lora_mm(layer.wk_b, q_nope);
    q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);

    ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);

    kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
    ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
    ggml_tensor * Vcur = kv_cmpr;

    ggml_tensor * attn = build_attn(inp_attn_dsa,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, layer.attn_sinks, layer.wv_b, top_k, kq_scale, il);
    cb(attn, "attn_kqv", il);

    ggml_tensor * gate = build_lora_mm(layer.wqkv_gate, cur);
    gate = ggml_sigmoid(ctx0, gate);
    attn = ggml_mul(ctx0, attn, gate);
    cb(attn, "attn_gated", il);

    ggml_tensor * out = build_lora_mm(layer.wo, attn);
    cb(out, "attn_out", il);

    return { out, top_k };
}

// Unbuffered progress marks for the graph builder: a device backend that dies mid-build leaves nothing in the buffered log
// to say which stage it reached, and these cost nothing unless the variable is set.
#define HY4_MARK(tag) do { if (getenv("STRATA_HY4_BUILD_TRACE") != NULL) { std::fprintf(stderr, "[hy4] %s\n", tag); std::fflush(stderr); } } while (0)

// One block, for any caller that needs exactly one: the decoder's loop becomes a call to it, and the draft graph runs block 42
// through the same code rather than a copy of it.
ggml_tensor * llama_model_hy_v4::graph::build_block(
        int il,
        ggml_tensor * inpL,
        bool is_dsa,
        ggml_tensor * inp_pos,
        float kq_scale,
        llm_graph_input_attn_k * inp_attn,
        llm_graph_input_attn_k * inp_attn_swa,
        llm_graph_input_attn_k_dsa * inp_attn_dsa,
        ggml_tensor * & last_top_k,
        const std::map<size_t, ggml_tensor *> & engram_rows_tensor) {
    ggml_tensor * cur = nullptr;
    const int64_t hc = hparams.dsv4_hc_mult;
        if (getenv("STRATA_HY4_BUILD_TRACE") != NULL && il % 10 == 0) { std::fprintf(stderr, "[hy4] block %d\n", il); std::fflush(stderr); }
        // The Engram write sits in a closure so that its place in the block can be chosen without duplicating it: the
        // streams are captured by reference, so the write reads and updates inpL either way.
        auto apply_engram = [&]() {
        // the same shape the DeepSeek-V4.1 builder uses: the looked-up rows are projected to one key slab per residual
        // stream plus one value slab, and the gate comes from their similarity
        if (const auto it_e = engram_rows_tensor.find((size_t) il); it_e != engram_rows_tensor.end()) {
            const ggml_tensor * kv_w   = model.get_tensor((std::string("blk.") + std::to_string(il) + ".engram_kv.weight").c_str());
            const ggml_tensor * gate_w = model.get_tensor((std::string("blk.") + std::to_string(il) + ".engram_gate.weight").c_str());
            if (kv_w != nullptr) {
                ggml_tensor * rows = it_e->second;
                ggml_tensor * kv = build_lora_mm(const_cast<ggml_tensor *>(kv_w), rows);
                cb(kv, "engram_kv", il);

                // Which slab of the projection feeds the write is another guess the pack does not settle: the last one, or
                // the first, with the keys shifted along accordingly.
                // Measured: taking the value from the first slab and the keys from the rest gives a peak of 10.490 with a gap
                // of 1.520, against 8.465 and 0.606 the other way round.  The other reading stays behind the variable.
                // Measured twice: on the rough prompt the value from the first slab looked better, and on a properly formatted
                // turn the last slab does - peak 10.712 with a gap of 2.675 against 9.837 and 1.314.  The formatted
                // measurement is the one that counts, so the last slab is the default and the first stays one variable away.
                static const int value_slab = [] {
                    const char * e = std::getenv("STRATA_HY4_ENGRAM_VALUE_FIRST");
                    return e != nullptr && e[0] != '0' ? 1 : 0;
                }();
                const int64_t key_off   = value_slab == 0 ? 0 : n_embd;
                const int64_t value_off = value_slab == 0 ? hc*n_embd : 0;
                ggml_tensor * key   = ggml_cont(ctx0, ggml_view_3d(ctx0, kv, n_embd, hc, n_tokens,
                        ggml_row_size(kv->type, n_embd), kv->nb[1], ggml_row_size(kv->type, key_off)));
                ggml_tensor * value = ggml_cont(ctx0, ggml_view_3d(ctx0, kv, n_embd, 1, n_tokens,
                        ggml_row_size(kv->type, n_embd), kv->nb[1], ggml_row_size(kv->type, value_off)));

                // a quantised tensor has no binary-op kernel, so the gate has to be materialised in f32 first
                // the pack's Engram block also carries a five-slot bias for the projection, a pre-scale for the block
                // input and a per-stream scale for the value, none of which the plain builder has
                const ggml_tensor * kv_b = model.get_tensor((std::string("blk.") + std::to_string(il) +
                            ".engram_kv_b.weight").c_str());
                const ggml_tensor * pre_gamma = model.get_tensor((std::string("blk.") + std::to_string(il) +
                            ".engram_pre_gamma.weight").c_str());
                const ggml_tensor * hc_scale = model.get_tensor((std::string("blk.") + std::to_string(il) +
                            ".engram_hc_scale.weight").c_str());

                // Whether the projection carries the bias at all is a guess; it can be left out for a comparison.
                static const bool no_kv_bias = [] {
                    const char * e = std::getenv("STRATA_HY4_ENGRAM_NO_BIAS");
                    return e != nullptr && e[0] != '0';
                }();

                if (!no_kv_bias && kv_b != nullptr && kv_b->ne[0] * kv_b->ne[1] == kv->ne[0]) {
                    // the bias has one value per projection row, laid out as five slabs of n_embd
                    ggml_tensor * b = ggml_is_quantized(kv_b->type) ? ggml_cast(ctx0, const_cast<ggml_tensor *>(kv_b), GGML_TYPE_F32)
                                                                   : const_cast<ggml_tensor *>(kv_b);
                    ggml_tensor * kv3 = ggml_reshape_3d(ctx0, kv, n_embd, hc + 1, n_tokens);
                    ggml_tensor * b3  = ggml_reshape_3d(ctx0, b,  n_embd, hc + 1, 1);
                    kv = ggml_reshape_2d(ctx0, ggml_add(ctx0, kv3, b3), kv->ne[0], kv->ne[1]);
                    cb(kv, "engram_kv_biased", il);
                }

                ggml_tensor * egate = nullptr;
                if (gate_w != nullptr) {
                    ggml_tensor * g = const_cast<ggml_tensor *>(gate_w);
                    egate = ggml_is_quantized(g->type) ? ggml_cast(ctx0, g, GGML_TYPE_F32) : g;
                }
                // The pre-scale is another piece the pack ships without saying how it is used; leaving it out is the reading
                // that measures how much it contributes.
                static const bool no_pre_gamma = [] {
                    const char * e = std::getenv("STRATA_HY4_ENGRAM_NO_PRE_GAMMA");
                    return e != nullptr && e[0] != '0';
                }();

                ggml_tensor * x_pre = inpL;
                if (!no_pre_gamma && pre_gamma != nullptr) {
                    ggml_tensor * g = ggml_is_quantized(pre_gamma->type) ? ggml_cast(ctx0, const_cast<ggml_tensor *>(pre_gamma), GGML_TYPE_F32)
                                                                        : const_cast<ggml_tensor *>(pre_gamma);
                    x_pre = ggml_mul(ctx0, inpL, g);
                    cb(x_pre, "engram_pre_gamma", il);
                }

                ggml_tensor * dot = ggml_mul(ctx0, ggml_mul(ctx0,
                        ggml_rms_norm(ctx0, x_pre, hparams.f_norm_rms_eps),
                        ggml_rms_norm(ctx0, key,  hparams.f_norm_rms_eps)), egate);
                dot = ggml_scale(ctx0, ggml_sum_rows(ctx0, dot), 1.0f/sqrtf((float) n_embd));
                cb(dot, "engram_dot", il);

                // Two readings of the gate: the signed root of the similarity, and the similarity itself.  The pack gives no
                // formula for it, so the second is selectable while the difference is measured.
                // The similarity itself measured better than its signed root - a higher peak and an eight times wider gap -
                // so that is the default and the root stays one variable away.
                static const bool gate_root = [] {
                    const char * e = std::getenv("STRATA_HY4_ENGRAM_GATE_ROOT");
                    return e != nullptr && e[0] != '0';
                }();
                const bool gate_plain = !gate_root;
                ggml_tensor * gate = nullptr;
                if (gate_plain) {
                    gate = ggml_sigmoid(ctx0, dot);
                } else {
                    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, dot), 1e-6f, INFINITY));
                    ggml_tensor * sign = ggml_scale_bias(ctx0, ggml_step(ctx0, ggml_neg(ctx0, dot)), -2.0f, 1.0f);
                    gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, mag, sign));
                }
                cb(gate, "engram_gate", il);

                ggml_tensor * v_rep = ggml_repeat_4d(ctx0, value, n_embd, hc, n_tokens, 1);
                if (hc_scale != nullptr) {
                    ggml_tensor * s = ggml_is_quantized(hc_scale->type) ? ggml_cast(ctx0, const_cast<ggml_tensor *>(hc_scale), GGML_TYPE_F32)
                                                                       : const_cast<ggml_tensor *>(hc_scale);
                    v_rep = ggml_mul(ctx0, v_rep, s);
                    cb(v_rep, "engram_hc_scaled", il);
                }

                inpL = ggml_add(ctx0, inpL, ggml_mul(ctx0, v_rep, gate));
                cb(inpL, "engram_out", il);
            }
        }
        };

        // Where the write belongs in the block is the one part of it the pack does not settle; both readings are one
        // variable apart and the closure makes them one call each.
        static const bool engram_late = [] {
            const char * e = std::getenv("STRATA_HY4_ENGRAM_LATE");
            return e != nullptr && e[0] != '0';
        }();
        if (!engram_late) {
            apply_engram();
        }

        ggml_tensor * residual = inpL;
        hc_pre attn_hc = build_hc_pre(inpL, model.layers[il].hc_attn_fn, model.layers[il].hc_attn_scale,
                model.layers[il].hc_attn_base, il);
        cur = attn_hc.cur;
        cur = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        if (is_dsa) {
            attn_dsa attn = (hparams.is_swa(il) && inp_attn_swa != nullptr)
                    ? attn_dsa{ build_attention(inp_attn_swa, cur, inp_pos, kq_scale, il), nullptr }
                    : build_attention_dsa(inp_attn_dsa, cur, inp_pos, last_top_k, kq_scale, il);
            cur = attn.cur;
            last_top_k = attn.top_k;
        } else {
            cur = build_attention(inp_attn, cur, inp_pos, kq_scale, il);
        }

        // the pack carries a second norm after attention and after the feed-forward; a plain hy_v4 pack has neither, and
        // the lookups return null there so nothing changes for it.  Where those norms belong in the block is a reading the
        // pack does not settle, so they can be left out entirely for a measurement.
        static const bool no_post_norms = [] {
            const char * e = std::getenv("STRATA_HY4_NO_POST_NORMS");
            return e != nullptr && e[0] != '0';
        }();
        if (const ggml_tensor * post_attn = no_post_norms ? nullptr : model.get_tensor((std::string("blk.") + std::to_string(il) +
                    ".post_attention_norm.weight").c_str())) {
            cur = build_norm(cur, const_cast<ggml_tensor *>(post_attn), nullptr, LLM_NORM_RMS, il);
            cb(cur, "post_attention_norm", il);
        }

        inpL = build_hc_post(cur, residual, attn_hc.post, il);
        cb(inpL, "hc_attn_out", il);

        residual = inpL;
        hc_pre ffn_hc = build_hc_pre(inpL, model.layers[il].hc_ffn_fn, model.layers[il].hc_ffn_scale,
                model.layers[il].hc_ffn_base, il);
        cur = ffn_hc.cur;
        cur = build_norm(cur, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        const auto & layer = model.layers[il];
        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                    layer.ffn_up,   NULL, NULL,
                    layer.ffn_gate, NULL, NULL,
                    layer.ffn_down, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // A variant whose expert stacks read a latent width projects into that width first and back out after the
            // MoE; a pack without those tensors (every plain hy_v4 one) runs the builder on `cur` unchanged.
            // the names describe the residual direction, not the projection: down maps n_embd into the latent and up maps
            // the latent back out
            const ggml_tensor * routed_in   = model.get_tensor((std::string("blk.") + std::to_string(il) + ".ffn_routed_down.weight").c_str());
            const ggml_tensor * routed_out  = model.get_tensor((std::string("blk.") + std::to_string(il) + ".ffn_routed_up.weight").c_str());

            if (il == 1) {
                LLAMA_LOG_WARN("%s: il %d routed_in %p routed_out %p\n", __func__, il,
                        (const void *) routed_in, (const void *) routed_out);
            }
            ggml_tensor * moe_in = routed_in ? build_lora_mm(const_cast<ggml_tensor *>(routed_in), cur) : cur;
            if (routed_in) {
                cb(moe_in, "ffn_routed_in", il);
            }

            // A variant that splits its routed experts into tiers hands the builder the three stacks of each tier plus the
            // two lookup tables, so a selected expert is only ever looked up in the tier that owns it.
            const auto & model_hy4 = static_cast<const llama_model_hy_v4 &>(model);
            llm_moe_tiers tiers;
            if (model_hy4.expert_tiers() > 1) {
                auto inp = std::make_unique<llm_graph_input_expert_tiers>();
                for (int t = 0; t < model_hy4.expert_tiers(); ++t) {
                    tiers.gate.push_back(model_hy4.tier_stack(il, t, 0));
                    tiers.up.push_back  (model_hy4.tier_stack(il, t, 1));
                    tiers.down.push_back(model_hy4.tier_stack(il, t, 2));

                    // one value per row, so a row lookup returns exactly the ids' shape; a one-dimensional table would
                    // come back as one row of n_expert values per id
                    ggml_tensor * lt = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, hparams.n_expert);
                    ggml_set_name(lt, "expert_tier_local");
                    ggml_tensor * mt = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, hparams.n_expert);
                    ggml_set_name(mt, "expert_tier_mask");

                    tiers.local.push_back(lt);
                    tiers.mask.push_back(mt);

                    inp->local.push_back(lt);
                    inp->mask.push_back(mt);
                    inp->local_host.push_back(model_hy4.tier_local_table(il, t));
                    inp->mask_host.push_back(model_hy4.tier_mask_table(il, t));
                }
                res->add_input(std::move(inp));
            }

            // The pack writes expert_gating_func as a number, and what a number means depends on the enum it was written against:
            // in the upstream convention 1 is softmax and 2 is sigmoid, while this tree's enum has 0 softmax, 1 sigmoid and 2 a
            // raw-weight mode.  The value is forwarded as-is by default and can be overridden here for a run.
            static const char * gating_env = std::getenv("STRATA_HY4_GATING");
            const int gating_op = gating_env != nullptr ? std::atoi(gating_env) : (int) hparams.expert_gating_func;
            // the router reads cur (n_embd wide); the expert stacks read the projected tensor when the pack has
            // ffn_routed_up, which is what the last argument below selects
            ggml_tensor * moe_out = build_moe_ffn(cur,
                    layer.ffn_gate_inp,
                    layer.ffn_up_exps,
                    layer.ffn_gate_exps,
                    layer.ffn_down_exps,
                    layer.ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) gating_op,
                    il,
                    nullptr,   // probs_in
                    nullptr,   // gate_up_exps
                    nullptr,   // selected_experts_in
                    nullptr,   // post_exp_norm
                    1e-5f,     // post_exp_norm_eps
                    routed_in ? moe_in : nullptr,    // cur_experts_in
                    tiers.empty() ? nullptr : &tiers);

            if (routed_out) {
                moe_out = build_lora_mm(const_cast<ggml_tensor *>(routed_out), moe_out);
                cb(moe_out, "ffn_routed_out", il);
            }

            // The pack also carries one shared expert per MoE block: the same gate/up/down path evaluated on the block's
            // input and added to the routed output.  Without it every MoE block is missing an addend.
            if (layer.ffn_up_shexp != nullptr && layer.ffn_gate_shexp != nullptr && layer.ffn_down_shexp != nullptr) {
                // the shared expert reads the block input at n_embd and works at the expert intermediate width
                ggml_tensor * shexp = build_ffn(cur,
                        layer.ffn_up_shexp,   nullptr, nullptr,
                        layer.ffn_gate_shexp, nullptr, nullptr,
                        layer.ffn_down_shexp, nullptr, nullptr,
                        nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(shexp, "ffn_shexp_out", il);
                moe_out = ggml_add(ctx0, moe_out, shexp);
                cb(moe_out, "ffn_moe_plus_shexp", il);
            }
            cb(moe_out, "ffn_moe_out", il);

            // The shared expert was already added above, where the MoE output is built; adding it here as well took it into
            // every block twice, which is the kind of error that leaves every token slightly wrong rather than one part of the
            // graph obviously broken.
            cur = moe_out;
            cb(cur, "ffn_out", il);
        }

            // the pack carries a second norm after the feed-forward as well; a plain hy_v4 block has no such tensor
        if (const ggml_tensor * post_ffn = no_post_norms ? nullptr : model.get_tensor((std::string("blk.") + std::to_string(il) +
                    ".post_ffw_norm.weight").c_str())) {
            cur = build_norm(cur, const_cast<ggml_tensor *>(post_ffn), nullptr, LLM_NORM_RMS, il);
            cb(cur, "post_ffw_norm", il);
        }

    inpL = build_hc_post(cur, residual, ffn_hc.post, il);
        if (engram_late) {
            apply_engram();
        }
        cb(inpL, "l_out", il);
    return inpL;
}

llama_model_hy_v4::graph::graph(const llm_graph_params & params) :
    llm_graph_context(params) {
    build(params);
}

void llama_model_hy_v4::graph::build(const llm_graph_params & params) {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t n_embd_head_k = hparams.n_embd_head_k_mla();
    // The scale for the compressed queries and keys follows the width the score is taken over, which for an absorbed MLA
    // is the compressed width and not the original head width; a factor over it makes that reading checkable.
    const float kq_scale_factor = [] {
        const char * e = std::getenv("STRATA_HY4_KQ_SCALE_FACTOR");
        return e != nullptr ? (float) std::atof(e) : 1.0f;
    }();
    const float kq_scale = kq_scale_factor / sqrtf(float(n_embd_head_k));

    ggml_tensor * cur;

    // The sparse path can be turned off for a run; the tensors stay in the table either way, only the builder changes.
    static const bool dsa_off = [] {
        const char * e = std::getenv("STRATA_HY4_NO_DSA");
        return e != nullptr && e[0] != '\0' && e[0] != '0';
    }();
    const bool is_dsa = hparams.indexer_top_k > 0 && !dsa_off;

    HY4_MARK("begin");
    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();

    // The four Engram blocks read a table lookup: the rows are gathered on the host from the companion pack and uploaded,
    // then projected by that block's engram_kv.  The projection is built here so its shape can be checked against the
    // block's remaining Engram tensors before anything is added to the residual stream.
    std::map<size_t, ggml_tensor *> engram_rows_tensor;

    // The rows tensor is built here rather than at the write, because a graph input whose consumer is skipped would be left
    // unreferenced; an ablation therefore has to switch the whole path off at its source.
    static const bool no_engram = [] {
        const char * e = std::getenv("STRATA_HY4_NO_ENGRAM");
        return e != nullptr && e[0] != '0';
    }();

    if (!no_engram && params.ubatch.token != nullptr && params.ubatch.n_tokens > 0) {
        const auto & hy4e = static_cast<const llama_model_hy_v4 &>(model);
        for (int slot = 0; slot < 4; ++slot) {
            const auto ple_layers = hy4e.engram_layers();
            if (slot >= (int) ple_layers.size()) {
                break;
            }
            const int il_e = ple_layers[(size_t) slot];
            const ggml_tensor * kv = model.get_tensor((std::string("blk.") + std::to_string(il_e) + ".engram_kv.weight").c_str());
            if (kv == nullptr) {
                continue;
            }

            std::vector<llama_token> tk(params.ubatch.token, params.ubatch.token + params.ubatch.n_tokens);
            std::vector<float> rows;
            if (!hy4e.engram_rows(slot, tk, rows)) {
                continue;
            }

            auto inp_e = std::make_unique<llm_graph_input_engram>();
            inp_e->rows = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1536, n_tokens);
            ggml_set_name(inp_e->rows, "engram_rows");
            inp_e->rows_host = std::move(rows);

            engram_rows_tensor[(size_t) il_e] = inp_e->rows;
            res->add_input(std::move(inp_e));
        }
    }

    if (getenv("STRATA_HY4_TRACE") != NULL && params.ubatch.token != nullptr && params.ubatch.n_tokens >= 3) {
        const auto & hy4 = static_cast<const llama_model_hy_v4 &>(model);
        std::vector<llama_token> tk(params.ubatch.token + params.ubatch.n_tokens - 3,
                                    params.ubatch.token + params.ubatch.n_tokens);
        std::vector<float> rows;
        if (hy4.engram_rows(0, tk, rows)) {
            double sq = 0.0;
            for (float v : rows) {
                sq += (double) v * (double) v;
            }
            LLAMA_LOG_WARN("%s: engram layer 0 gathered %zu values for %zu tokens, norm %.6f\n", __func__,
                    rows.size(), tk.size(), std::sqrt(sq));
        } else {
            LLAMA_LOG_WARN("%s: engram gather returned false\n", __func__);
        }
    }
    // A pack that windows some blocks as well as indexing others needs the paired input: the full-attention and indexer
    // half, and the windowed half whose cache is limited to n_swa positions.
    const bool memory_is_hybrid = dynamic_cast<const llama_kv_cache_dsa_iswa_context *>(params.mctx) != nullptr;

    llm_graph_input_attn_k           * inp_attn     = is_dsa ? nullptr : build_attn_inp_k();
    llm_graph_input_attn_k_dsa       * inp_attn_dsa = nullptr;
    llm_graph_input_attn_k           * inp_attn_swa = nullptr;

    if (is_dsa) {
        if (memory_is_hybrid) {
            auto * inp = build_attn_inp_k_dsa_iswa();
            inp_attn_dsa = inp->get_dsa();
            inp_attn_swa = inp->get_swa();
        } else {
            inp_attn_dsa = build_attn_inp_k_dsa();
        }
    }
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // top-k of the last "full" indexer layer, reused by the following "shared" layers
    ggml_tensor * last_top_k = nullptr;

    // expand the single embedding into hc parallel residual streams
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    HY4_MARK("loop");
    for (int il = 0; il < n_layer; ++il) {
        inpL = build_block(il, inpL, is_dsa, inp_pos, kq_scale, inp_attn, inp_attn_swa, inp_attn_dsa,
                last_top_k, engram_rows_tensor);
    }

    // prune to the requested output rows once, after all HC streams are done
    if (inp_out_ids) {
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd * hc, n_tokens);
        flat = ggml_get_rows(ctx0, flat, inp_out_ids);
        inpL = ggml_reshape_3d(ctx0, flat, n_embd, hc, n_outputs);
    }

    HY4_MARK("head");
    // The final norm can sit before the stream collapse or after it; the pack's tensor names do not settle that, and the
    // two places are one switch apart.
    static const bool norm_before_head = [] {
        const char * e = std::getenv("STRATA_HY4_NORM_BEFORE_HEAD");
        return e != nullptr && e[0] != '0';
    }();
    if (norm_before_head) {
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd * hc, n_outputs);
        flat = build_norm(flat, model.output_norm, nullptr, LLM_NORM_RMS, -1);
        inpL = ggml_reshape_3d(ctx0, flat, n_embd, hc, n_outputs);
        cb(inpL, "result_norm_pre", -1);
    }

    cur = build_hc_head(inpL, model.hc_head_fn, model.hc_head_scale, model.hc_head_base);
    cb(cur, "hc_head", -1);

    // The draft step's first half, for measurement only: whether the three nextn tensors can be read at all. It is kept off the
    // graph's outputs, because replacing one changes what the allocator plans and that question is being asked separately.
    static const bool mtp_probe = [] {
        const char * e = std::getenv("STRATA_HY4_MTP");
        return e != nullptr && e[0] != '0';
    }();
    if (mtp_probe) {
        const ggml_tensor * eh    = model.get_tensor("blk.42.nextn.eh_proj.weight");
        const ggml_tensor * enorm = model.get_tensor("blk.42.nextn.enorm.weight");
        const ggml_tensor * hnorm = model.get_tensor("blk.42.nextn.hnorm.weight");
        if (eh != nullptr && enorm != nullptr && hnorm != nullptr) {
            ggml_tensor * h_n = build_norm(cur, const_cast<ggml_tensor *>(hnorm), nullptr, LLM_NORM_RMS, -1);
            ggml_tensor * both = ggml_concat(ctx0, h_n, h_n, 0);
            ggml_tensor * draft = build_lora_mm(const_cast<ggml_tensor *>(eh), both);
            cb(draft, "mtp_draft", -1);
            res->t_embd = draft;   // an output, so the scheduler evaluates it
        }
    }

    if (!norm_before_head) {
        cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    }
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    // What the forward pass actually reads, when asked: every weight that appears as a node source, printed once per build.
    // Comparing this with the file's tensor list is how a tensor that loads and is never used shows up - a failure mode no
    // shape check and no switch can see.
    if (std::getenv("STRATA_HY4_COVERAGE") != nullptr) {
        std::set<std::string> used;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx0); t != nullptr; t = ggml_get_next_tensor(ctx0, t)) {
            for (int i = 0; i < GGML_MAX_SRC; ++i) {
                const ggml_tensor * src = t->src[i];
                if (src != nullptr && src->name[0] != '\0') {
                    used.insert(src->name);
                }
            }
        }
        for (const auto & name : used) {
            std::fprintf(stderr, "[hy4-use] %s\n", name.c_str());
        }
    }
}
