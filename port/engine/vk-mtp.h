// vk-mtp.h - Strata's MTP: a second GGUF holding one trained nextn head, drafting the next token from the target's
// hidden state.
//
// The head is not a standalone model: it consumes the target's hidden state at position p and the token at p+1, so its
// context is fed batches that carry *both* a token and an embedding row.  That is the `LLAMA_CONTEXT_TYPE_MTP` context
// type and the `llama_set_embeddings_nextn` / `llama_get_embeddings_nextn_ith` pair from llama-ext.h.
//
// One generation step is:
//   draft()   propose up to n_max tokens, chaining the single head (each step's own hidden state feeds the next);
//   target    decode [token, draft...] in one pass, logits on every row;
//   observe() rebuild the draft's KV for those rows and keep the target's hidden rows;
//   accept()  pick the hidden row that pairs with the next token, so the next draft starts from the right state.
// A draft token is only ever *proposed*: the target samples every row itself, so the accepted sequence is exactly the
// sequence the target would have produced without MTP.
#pragma once

#include "ggml-backend.h"
#include "llama.h"

#include <cstdlib>
#include <string>
#include <vector>

namespace strata {

// The head drafts correctly (docs/NPU-FEASIBILITY.md, "The MTP works now"), but a drafted token costs a full-vocabulary
// LM matvec - the head is 1.7-2.2 GB - so MTP speculation still measures slower than plain decoding on both packs here
// (27B 5.8 against 7.0 tok/s, 35B-A3B 11.9 against 36.2, measured with a 67% and a 28% acceptance after the matvec
// pass got cheaper).  It is therefore opt-in rather than on by default, and the opt-in is an environment variable so
// that a stale `--mtp` in a saved run configuration cannot quietly make the app slower.
inline bool mtp_supported_by_build() {
    static const bool v = std::getenv("STRATA_VK_MTP") != nullptr;
    return v;
}

class Mtp {
public:
    ~Mtp() { unload(); }

    // `target_ctx` is the decoding context; it is switched to emit nextn hidden states here.  `devices` names the device
    // the head runs on, null-terminated: naming it is not optional, because llama.cpp's default device list is every
    // graphics device in the build, and this build registers OpenCL and the HTP beside Vulkan - the head would be split
    // across all three, and the HTP slot fails before the first token.
    bool load(const std::string & path, llama_context * target_ctx, int n_ctx, int threads,
              ggml_backend_dev_t * devices);
    void unload();

    bool ready() const { return ctx_ != nullptr; }
    const char * status() const { return status_.c_str(); }

    // Forget the draft KV and the carried hidden state (a new conversation).
    void clear();

    // After every target decode: rebuild the draft's KV over [pos0, pos0+n) and keep the target's hidden rows.
    bool observe(const llama_token * toks, int n, int pos0);

    // Propose up to `n_max` tokens following `last`, which sits at `pos0`.
    std::vector<llama_token> draft(llama_token last, int pos0, int n_max);

    // `n_accepted` draft tokens were accepted; carry the matching hidden row into the next draft.
    void accept(int n_accepted);

private:
    llama_model * model_ = nullptr;
    llama_context * ctx_ = nullptr;
    llama_context * target_ = nullptr;
    llama_sampler * smpl_ = nullptr;
    llama_batch batch_ = {};
    int n_embd_ = 0;
    int n_ctx_ = 0;
    std::vector<float> pending_h_;   // the target's hidden state at the position before the next draft
    int primed_ = 0;                 // the head's cache covers [0, primed_)
    std::string status_ = "off";
};

} // namespace strata
