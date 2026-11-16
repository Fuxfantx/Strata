// vk-engine.cpp - reuse the conversation prefix, generate, and stop cleanly.
//
// The load path (devices, model, context) is vk-engine-load.cpp; this file is everything a turn does.
#include "vk-engine.h"

#include <chrono>
#include <cstdlib>
#include <cstring>

namespace strata {

void Engine::unload() {
    mtp_.unload();
    if (batch_.token != nullptr) { llama_batch_free(batch_); batch_ = {}; }
    if (ctx_)   { llama_free(ctx_); ctx_ = nullptr; }
    if (model_) { llama_model_free(model_); model_ = nullptr; }
    devices_.clear();
}

void Engine::reset() {
    llama_memory_clear(llama_get_memory(ctx_), true);
    kv_tokens_.clear();
    ngram_.clear();
    ngram_indexed_ = 0;
    mtp_.clear();
    mtp_primed_ = false;
}

std::string Engine::describe() const {
    char buf[320];
    std::snprintf(buf, sizeof(buf), "engine=%s backend=%s model=%s n_ctx=%d npu=%s mtp=%s",
                  kEngineVersion, gpu_backend_name(backend_), arch_.c_str(), n_ctx_,
                  npu_.active() ? ggml_backend_dev_name(npu_.dev) : npu_.found() ? "idle" : "none",
                  mtp_.ready() ? "on" : "off");
    return buf;
}

// The head needs its KV to mirror the target's up to the last decoded position; one catch-up decode does that.
void Engine::mtp_prime(const llama_token * toks, int n, int pos0) {
    mtp_primed_ = n > 0 && mtp_.observe(toks, n, pos0);
    if (!mtp_primed_) logf("[strata-vk] MTP: cannot follow the target's hidden state; running without drafts\n");
}


void Engine::trim_to_common_prefix(llama_memory_t mem, const std::vector<llama_token> & ids) {
    int common = (int) std::min(kv_tokens_.size(), ids.size());
    int n = 0;
    while (n < common && kv_tokens_[n] == ids[n]) ++n;
    common = n;

    if (common < (int) kv_tokens_.size()) {
        // The prompt diverges (an edited message, a new conversation): drop what follows the shared prefix.  A
        // recurrent layer (GDN) cannot be cut part way, and then the whole state goes and the prompt is read from
        // token 0 - still correct, just slower.
        if (!(common > 0 && llama_memory_seq_rm(mem, 0, common, -1))) {
            llama_memory_clear(mem, true);
            common = 0;
        }
        kv_tokens_.resize(common);
    }

    // A prompt that is exactly the cached prefix leaves nothing to decode, and the logits still in the context are
    // the previous turn's.  Re-read its last token so sampling starts from this prompt's own last position.
    if (common == (int) ids.size() && common > 0) {
        if (llama_memory_seq_rm(mem, 0, common - 1, -1)) {
            kv_tokens_.pop_back();
        } else {
            llama_memory_clear(mem, true);
            kv_tokens_.clear();
        }
    }
}

bool Engine::decode(const llama_token * toks, int n, int pos0, bool want_logits) {
    for (int i = 0; i < n; ++i) {
        batch_.token[i]     = toks[i];
        batch_.pos[i]       = pos0 + i;
        batch_.n_seq_id[i]  = 1;
        batch_.seq_id[i][0] = 0;
        batch_.logits[i]    = (want_logits && i == n - 1) ? 1 : 0;
    }
    batch_.n_tokens = n;
    const int rc = llama_decode(ctx_, batch_);
    if (rc != 0) {
        logf("[strata-vk] llama_decode failed: %d (n=%d pos=%d)\n", rc, n, pos0);
        return false;
    }
    return true;
}

bool Engine::decode_all(const llama_token * toks, int n, int pos0) {
    for (int i = 0; i < n; ++i) {
        batch_.token[i]     = toks[i];
        batch_.pos[i]       = pos0 + i;
        batch_.n_seq_id[i]  = 1;
        batch_.seq_id[i][0] = 0;
        batch_.logits[i]    = 1;      // the verifier samples every position
    }
    batch_.n_tokens = n;
    const int rc = llama_decode(ctx_, batch_);
    if (rc != 0) {
        logf("[strata-vk] llama_decode failed: %d (n=%d pos=%d, speculative)\n", rc, n, pos0);
        return false;
    }
    return true;
}

// The last n-1 tokens plus the token just sampled form the pattern; an earlier occurrence of it gives the continuation
// to draft.  The occurrence is found through an index built as the history grows, so the cost is one hash and one map
// lookup per token instead of a backward scan over the whole conversation.
namespace {
uint64_t ngram_hash(const llama_token * t, size_t n) {
    uint64_t h = 1469598103934665603ull;   // FNV-1a
    for (size_t i = 0; i < n; ++i) {
        h ^= (uint64_t) (uint32_t) t[i];
        h *= 1099511628211ull;
    }
    return h;
}
} // namespace

void Engine::index_ngrams(const std::vector<llama_token> & hist) {
    const size_t n = (size_t) lookup_n_;
    if (n == 0) return;
    if (hist.size() < ngram_indexed_) {          // the history was trimmed: the positions moved, so rebuild
        ngram_.clear();
        ngram_indexed_ = 0;
    }
    // A window starting at i needs n tokens, and it must not be the newest one: a draft starting at i+n needs room past
    // it, which the newest window never has.
    for (size_t i = ngram_indexed_; i + n <= hist.size(); ++i) {
        NgramSlots & slots = ngram_[ngram_hash(hist.data() + i, n)];
        std::memmove(slots.pos + 1, slots.pos, sizeof(slots.pos) - sizeof(slots.pos[0]));
        slots.pos[0] = (int32_t) i;              // newest first
        if (slots.n < (uint8_t) (sizeof(slots.pos) / sizeof(slots.pos[0]))) ++slots.n;
    }
    ngram_indexed_ = hist.size();
}

std::vector<llama_token> Engine::lookup_draft(const std::vector<llama_token> & hist, llama_token sampled,
                                              int budget) {
    std::vector<llama_token> out;
    const size_t n = (size_t) lookup_n_;
    const size_t m = std::min((size_t) std::max(budget, 0), (size_t) lookup_m_);
    if (n == 0 || m == 0) return out;
    index_ngrams(hist);
    if (hist.size() < n) return out;

    // pattern = hist[len-n+1 .. len-1] + sampled
    llama_token pattern[8] = {};
    if (n > sizeof(pattern) / sizeof(pattern[0])) return out;
    for (size_t j = 0; j + 1 < n; ++j) pattern[j] = hist[hist.size() - n + 1 + j];
    pattern[n - 1] = sampled;

    static const bool lk_debug = std::getenv("STRATA_VK_LOOKUP_DEBUG") != nullptr;
    auto it = ngram_.find(ngram_hash(pattern, n));
    if (it == ngram_.end()) {
        if (lk_debug) {
            logf("[lookup] miss hist=%zu idx=%zu keys=%zu pattern=%d,%d,%d\n", hist.size(), ngram_indexed_,
                 ngram_.size(), (int) pattern[0], (int) pattern[1], (int) pattern[2]);
        }
        return out;
    }

    // Prefer a match that leaves a full draft behind it: the newest occurrence in repetitive text is the tail of the
    // same run and has nothing after it, and taking it was why an earlier version drafted nothing at all.  Any draft is
    // only a proposal, so a match from this bounded list costs nothing but a shorter draft when it is imperfect.
    size_t match = 0, room_best = 0;
    for (uint8_t s = 0; s < it->second.n; ++s) {
        const size_t pos = (size_t) it->second.pos[s];
        if (pos + n > hist.size()) continue;
        const size_t room = hist.size() - (pos + n);
        if (room >= m) { match = pos; room_best = room; break; }
        if (room > room_best) { match = pos; room_best = room; }
    }
    if (match == 0 || room_best < 1) {
        if (lk_debug) {
            logf("[lookup] hit but no room: slots=%u match=%zu room=%zu hist=%zu\n",
                 (unsigned) it->second.n, match, room_best, hist.size());
        }
        return out;
    }
    if (lk_debug) {
        logf("[lookup] draft from %zu of %zu (room %zu, m %zu)\n", match, hist.size(), room_best, m);
    }

    const size_t copy = std::min(m, hist.size() - (match + n));
    if (copy < 1) return out;
    out.assign(hist.begin() + (ptrdiff_t) (match + n), hist.begin() + (ptrdiff_t) (match + n + copy));
    return out;
}

// Decode [cur, draft...] in one target pass and keep the prefix the target agrees with.  Every row's logits decide one
// acceptance, so the accepted sequence is exactly what the target would have produced one token at a time.
bool Engine::verify_draft(llama_sampler * smpl, llama_token cur, const std::vector<llama_token> & draft,
                          std::vector<llama_token> & agreed, llama_token & next, bool use_mtp) {
    const int base = (int) kv_tokens_.size();
    std::vector<llama_token> batch;
    batch.reserve(draft.size() + 1);
    batch.push_back(cur);
    batch.insert(batch.end(), draft.begin(), draft.end());
    const Clock::time_point t_pass = Clock::now();
    if (!decode_all(batch.data(), (int) batch.size(), base)) return false;
    const double pass_ms = ms_since(t_pass);
    if (use_mtp) mtp_.observe(batch.data(), (int) batch.size(), base);

    int accepted = 0;
    next = llama_sampler_sample(smpl, ctx_, accepted);
    while (accepted < (int) draft.size() && next == draft[(size_t) accepted]) {
        ++accepted;
        next = llama_sampler_sample(smpl, ctx_, accepted);
    }

    kv_tokens_.push_back(cur);
    kv_tokens_.insert(kv_tokens_.end(), draft.begin(), draft.begin() + accepted);
    agreed.assign(draft.begin(), draft.begin() + accepted);

    // Drop everything the draft added past the accepted prefix.  The cache is the only state that has to be undone:
    // the sampler is deterministic by construction here (speculation requires greedy), so nothing else diverged.
    if (!llama_memory_seq_rm(llama_get_memory(ctx_), 0, base + 1 + accepted, -1)) {
        if (spec_rm_failed_ == 0) {
            logf("[strata-vk] the cache refused to drop %d drafted position(s) at %d; speculation stops here\n",
                 (int) draft.size() - accepted, base + 1 + accepted);
        }
        ++spec_rm_failed_;
        // A cache that cannot drop a suffix cannot speculate: keep the accepted tokens and stop drafting.
        spec_ok_ = false;
        mtp_primed_ = false;
    }
    if (use_mtp) mtp_.accept(accepted);
    // STRATA_VK_PROFILE=1: what a draft attempt costs, split into the verification pass and the rollback that follows it.
    // Drafting measured slower than plain decoding on both model families, and this is the instrument that says where.
    static const bool profile = std::getenv("STRATA_VK_PROFILE") != nullptr;
    if (profile) {
        logf("[strata-vk] draft: pass %d tok = %.2f ms, rollback %.2f ms, kept %d/%d\n",
             (int) batch.size(), pass_ms, ms_since(t_pass) - pass_ms, accepted, (int) draft.size());
    }
    return true;
}

// Strata's per-request sampling keys (serve/server.py's `sampling_keys`) -> a llama.cpp sampler chain.
llama_sampler * Engine::make_sampler(const std::string & keys, bool * greedy) const {
    double temperature = -1.0, top_p = -1.0, min_p = -1.0;
    int    top_k = -1;
    double p_repeat = 1.0, p_freq = 0.0, p_present = 0.0;
    int    p_last_n = 64;
    uint32_t seed = LLAMA_DEFAULT_SEED;

    size_t i = 0;
    while (i < keys.size()) {
        while (i < keys.size() && keys[i] == ' ') ++i;
        size_t j = i;
        while (j < keys.size() && keys[j] != ' ') ++j;
        const std::string kv = keys.substr(i, j - i);
        i = j;
        const size_t eq = kv.find('=');
        if (eq == std::string::npos) continue;
        const std::string k = kv.substr(0, eq);
        const std::string v = kv.substr(eq + 1);
        const double d = std::atof(v.c_str());
        if      (k == "temperature")     temperature = d;
        else if (k == "top_p")           top_p = d;
        else if (k == "top_k")           top_k = (int) d;
        else if (k == "min_p")           min_p = d;
        else if (k == "penalty_repeat")  p_repeat = d;
        else if (k == "penalty_freq")    p_freq = d;
        else if (k == "penalty_present") p_present = d;
        else if (k == "penalty_last_n")  p_last_n = (int) d;
        else if (k == "seed")            seed = (uint32_t) std::strtoul(v.c_str(), nullptr, 10);
        // `pen_from` (the penalty window's start) has no llama.cpp equivalent: the last-n window stands in for it.
    }

    llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    sp.no_perf = false;
    llama_sampler * chain = llama_sampler_chain_init(sp);
    if (p_repeat != 1.0 || p_freq != 0.0 || p_present != 0.0) {
        llama_sampler_chain_add(chain, llama_sampler_init_penalties(
            (int32_t) llama_vocab_n_tokens(vocab_), p_last_n,
            (float) p_repeat, (float) p_freq, (float) p_present));
    }
    // Only a greedy chain is a pure function of the logits, which is what lets a draft be verified against the
    // model's own choice.  Penalties keep a window of history that a rejected draft would corrupt, so they also
    // disqualify speculation.
    *greedy = temperature <= 0.0 && p_repeat == 1.0 && p_freq == 0.0 && p_present == 0.0;
    if (temperature <= 0.0) {
        // Strata's default when no temperature is asked for is greedy
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
        return chain;
    }
    if (top_k > 0)                    llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k));
    if (top_p > 0.0 && top_p < 1.0)   llama_sampler_chain_add(chain, llama_sampler_init_top_p((float) top_p, 1));
    if (min_p > 0.0 && min_p <= 1.0)  llama_sampler_chain_add(chain, llama_sampler_init_min_p((float) min_p, 1));
    llama_sampler_chain_add(chain, llama_sampler_init_temp((float) temperature));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));
    return chain;
}

} // namespace strata









