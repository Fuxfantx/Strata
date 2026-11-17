// vk-mtp.cpp - the MTP draft head.  Mirrors common_speculative_impl_draft_mtp for the case this product ships: one
// trained nextn layer in its own GGUF, one sequence, one head (no chained heads, no shared KV).
#include "vk-mtp.h"

#include "llama-ext.h"

#include "vk-common.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace strata {
namespace {

void batch_set(llama_batch & b, int i, llama_token tok, int pos) {
    b.token[i]     = tok;
    b.pos[i]       = pos;
    b.n_seq_id[i]  = 1;
    b.seq_id[i][0] = 0;
    b.logits[i]    = 0;
}

} // namespace

bool Mtp::load(const std::string & path, llama_context * target_ctx, int n_ctx, int threads,
               ggml_backend_dev_t * devices) {
    if (!mtp_supported_by_build()) {
        status_ = "drafting measures slower than plain decoding on this GPU (5.8 against 7.0 tok/s on the 27B, 11.9 "
                  "against 36.2 on the 35B-A3B, because a drafted token pays a full-vocabulary LM matvec); set "
                  "STRATA_VK_MTP=1 to use it anyway";
        return false;
    }
    target_ = target_ctx;

    llama_model_params mp = llama_model_default_params();
    mp.load_mtp      = true;                  // the nextn tensors are the point of this file
    mp.n_gpu_layers  = 999;
    mp.devices       = devices;               // one device, named by the caller
    mp.load_mode     = LLAMA_LOAD_MODE_MMAP;
    mp.check_tensors = false;

    model_ = llama_model_load_from_file(path.c_str(), mp);
    if (model_ == nullptr) {
        status_ = "cannot load the MTP head";
        return false;
    }
    const int n_mtp = llama_model_n_layer_nextn(model_);
    if (n_mtp <= 0) {
        status_ = "the file carries no nextn layer";
        logf("[strata-vk] MTP: %s\n", status_.c_str());
        return false;
    }
    n_embd_ = llama_model_n_embd_out(model_);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = (uint32_t) n_ctx;
    cp.n_batch         = 8;                   // one token plus the drafts, one at a time
    cp.n_ubatch        = 8;
    cp.n_seq_max       = 1;
    cp.n_threads       = threads > 0 ? threads : (int) std::thread::hardware_concurrency();
    cp.n_threads_batch = cp.n_threads;
    cp.ctx_type        = LLAMA_CONTEXT_TYPE_MTP;
    // `embeddings` must stay false: it switches on the pooled-embedding output (llm_graph_context::build_pooling), and the
    // DECODER_MTP graph never fills `res->t_embd`, so the first decode asserts on "missing result_norm/result_embd". The
    // hidden state the head needs comes from `llama_set_embeddings_nextn` below, which is a different switch.
    cp.embeddings      = false;
    // The head reads its own hidden state back with llama_get_embeddings_nextn_ith, so the buffer has to be reserved when
    // the context is created - llama_set_embeddings_nextn after the fact flips the flag but the buffer is already sized.
    // Unmasked: `observe` needs every row's hidden state and asks for no logits at all, and a batch with no logits row
    // skips the LM head matmul entirely - which is 1.2 GB of weights on this model, i.e. ~12 ms per priming pass.
    cp.embeddings_nextn        = true;
    cp.embeddings_nextn_masked = false;
    cp.no_perf         = true;

    ctx_ = llama_init_from_model(model_, cp);
    if (ctx_ == nullptr) {
        status_ = "cannot create the MTP context";
        return false;
    }
    n_ctx_ = (int) llama_n_ctx(ctx_);

    // The target must emit its hidden state, the head must consume it.  `masked` on the head matches the upstream
    // driver: the head only reads the rows it is given, never behind them.
    llama_set_embeddings_nextn(target_, true, false);
    llama_set_embeddings_nextn(ctx_, true, false);

    llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    sp.no_perf = true;
    smpl_ = llama_sampler_chain_init(sp);
    // Greedy takes the global maximum, which is inside any top-k by construction: the top_k pass over the whole vocabulary
    // could not change a token, so it is not built.
    llama_sampler_chain_add(smpl_, llama_sampler_init_greedy());

    // llama_batch_init allocates the embeddings but not the tokens; MTP batches carry both.
    batch_ = llama_batch_init(8, n_embd_, 1);
    batch_.token = (llama_token *) std::malloc(sizeof(llama_token) * 8);
    if (batch_.token == nullptr) {
        status_ = "out of memory for the MTP batch";
        return false;
    }

    pending_h_.assign((size_t) n_embd_, 0.f);
    status_ = "ready";
    logf("[strata-vk] MTP: %d nextn layer(s), hidden %d, ctx %d, drafts chain through the head\n", n_mtp, n_embd_, n_ctx_);
    return true;
}

void Mtp::unload() {
    if (smpl_) { llama_sampler_free(smpl_); smpl_ = nullptr; }
    // The batch carries both tokens and embeddings here, so the token array is this class's own malloc; free the batch
    // whenever any part of it was allocated, not only when the tokens were.
    if (batch_.token || batch_.embd) { llama_batch_free(batch_); batch_ = {}; }
    if (ctx_) { llama_free(ctx_); ctx_ = nullptr; }
    if (model_) { llama_model_free(model_); model_ = nullptr; }
    pending_h_.clear();
    n_embd_ = 0;
    status_ = "off";
}

void Mtp::clear() {
    if (!ctx_) return;
    llama_memory_clear(llama_get_memory(ctx_), true);
    std::fill(pending_h_.begin(), pending_h_.end(), 0.f);
}

bool Mtp::observe(const llama_token * toks, int n, int pos0) {
    if (!ctx_ || n <= 0) return false;
    // A shorter accepted prefix than the last draft leaves stale cells: drop everything from this position on.
    llama_memory_seq_rm(llama_get_memory(ctx_), 0, pos0, -1);

    // The head at position p consumes the token at p and the *target's* hidden state at p-1, so the whole prompt can be
    // replayed with the target's own output rows - they are all there, unmasked, indexed by position.  The head's batch
    // is eight tokens wide, so a 512-token prompt is primed in 64 chunks.
    const int width = 8;
    int last = 0;
    for (int base = 0; base < n; base += width) {
        const int take = std::min(width, n - base);
        for (int i = 0; i < take; ++i) {
            const int gi = base + i;
            const float * src = llama_get_embeddings_nextn_ith(target_, gi == 0 ? 0 : gi - 1);
            if (src == nullptr) return false;
            batch_set(batch_, i, toks[gi], pos0 + gi);
            std::memcpy(batch_.embd + (size_t) i * n_embd_, src, sizeof(float) * (size_t) n_embd_);
        }
        // No logits anywhere: the head's LM matmul is skipped and only the hidden state is produced, which is all this
        // pass is for.
        // batch_set() already cleared every row's logits flag; the second pass here was redundant.
        batch_.n_tokens = take;
        last = take - 1;
        if (llama_decode(ctx_, batch_) != 0) {
            logf("[strata-vk] MTP: the draft could not follow the target at position %d\n", pos0 + base);
            return false;
        }
    }

    const float * tail = llama_get_embeddings_nextn_ith(ctx_, last);
    if (tail == nullptr) return false;
    pending_h_.assign(tail, tail + n_embd_);
    primed_ = pos0 + n;   // the head's cache now covers [0, pos0+n)
    return true;
}

std::vector<llama_token> Mtp::draft(llama_token last, int pos0, int n_max) {
    std::vector<llama_token> out;
    if (!ctx_ || n_max <= 0 || !primed_ || pos0 != primed_) return out;

    llama_memory_seq_rm(llama_get_memory(ctx_), 0, pos0, -1);   // drop the KV of a rejected draft

    pending_h_.resize((size_t) n_embd_);
    std::vector<float> h = pending_h_;
    llama_token cur = last;
    for (int i = 0; i < n_max; ++i) {
        batch_set(batch_, 0, cur, pos0 + i);
        batch_.logits[0] = 1;
        std::memcpy(batch_.embd, h.data(), sizeof(float) * (size_t) n_embd_);
        batch_.n_tokens = 1;
        if (llama_decode(ctx_, batch_) != 0) break;

        const llama_token id = llama_sampler_sample(smpl_, ctx_, 0);
        const float * hr = llama_get_embeddings_nextn_ith(ctx_, 0);
        if (hr == nullptr) break;
        out.push_back(id);
        h.assign(hr, hr + n_embd_);
        cur = id;
    }
    return out;
}

void Mtp::accept(int n_accepted) {
    // The next draft starts at the token that follows the accepted prefix, and the head needs the target's hidden state
    // at the position just before it.  That row is still in the target's output buffer from the pass that produced it, so
    // there is nothing to keep from the draft's own rows.
    const float * h = llama_get_embeddings_nextn_ith(target_, std::max(n_accepted, 0));
    if (h == nullptr) return;
    pending_h_.assign(h, h + n_embd_);
}

} // namespace strata
