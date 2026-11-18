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
    // A chat turn appends, so the first mismatch sits at the end of the shorter array: walk backwards and stop in a few
    // tokens instead of touching the whole cached prefix on every turn.
    const int common_max = (int) std::min(kv_tokens_.size(), ids.size());
    int n = common_max;
    while (n > 0 && kv_tokens_[n - 1] == ids[n - 1]) --n;
    int common = n;

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
    if (n > batch_cap_) { logf("[strata-vk] batch of %d exceeds the %d rows allocated; refusing\n", n, batch_cap_); return false; }
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

// The id of the token the front end writes once per picture width: the model's own <|image_pad|>.  A model without one
// (no projector, or another family) answers -1 and the request is refused rather than guessed at.
int Engine::image_pad_token() const {
    if (vocab_ == nullptr) return -1;
    static const char * kPad = "<|image_pad|>";
    llama_token tok = 0;
    const int n = llama_tokenize(vocab_, kPad, (int32_t) std::strlen(kPad), &tok, 1, false, true);
    if (n != 1) return -1;
    return (int) tok;
}

bool Engine::decode_embd(const float * embd, int n, int pos0, bool want_logits) {
    if (n_embd_ <= 0) { logf("[strata-vk] picture rows without an embedding width\n"); return false; }
    if (batch_embd_cap_ < n) {
        // rows are read in n_batch-sized groups, so this happens at most once per context
        if (batch_embd_.token != nullptr || batch_embd_.embd != nullptr) llama_batch_free(batch_embd_);
        batch_embd_ = llama_batch_init(n, n_embd_, 1);
        batch_embd_cap_ = n;
    }
    std::memcpy(batch_embd_.embd, embd, sizeof(float) * (size_t) n * (size_t) n_embd_);
    for (int i = 0; i < n; ++i) {
        batch_embd_.pos[i]       = pos0 + i;
        batch_embd_.n_seq_id[i]  = 1;
        batch_embd_.seq_id[i][0] = 0;
        batch_embd_.logits[i]    = (want_logits && i == n - 1) ? 1 : 0;
    }
    batch_embd_.n_tokens = n;
    const int rc = llama_decode(ctx_, batch_embd_);
    if (rc != 0) {
        logf("[strata-vk] llama_decode (picture) failed: %d (n=%d pos=%d)\n", rc, n, pos0);
        return false;
    }
    return true;
}

bool Engine::decode_all(const llama_token * toks, int n, int pos0) {
    if (n > batch_cap_) { logf("[strata-vk] batch of %d exceeds the %d rows allocated; refusing\n", n, batch_cap_); return false; }
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

namespace {
int argmax_logits(const float * lg, int n) {
    int best = 0;
    float bv = lg[0];
    for (int i = 1; i < n; ++i) {
        if (lg[i] > bv) { bv = lg[i]; best = i; }
    }
    return best;
}
} // namespace

void Engine::debug_logits_row(const char * tag, int32_t idx) const {
    const int nv = llama_vocab_n_tokens(vocab_);
    const float * lg = llama_get_logits_ith(ctx_, idx);
    if (lg == nullptr) {
        logf("[strata-vk] %s: index %d has no logits\n", tag, (int) idx);
        return;
    }
    const int b0 = argmax_logits(lg, nv);
    float second = -1e30f;
    int b1 = -1;
    for (int i = 0; i < nv; ++i) {
        if (i != b0 && lg[i] > second) { second = lg[i]; b1 = i; }
    }
    logf("[strata-vk] %s: index %d -> top %d (%.3f), then %d (%.3f)\n", tag, (int) idx, b0, (double) lg[b0],
         b1, (double) second);
}

// Decode N sequences in one batch.  The point is the arithmetic of a bandwidth-bound step: the dense weights are read once
// for the whole batch, so N tokens can cost little more than one.  Only the experts scale with the batch, and they are the
// smaller half of a token's traffic.
Engine::SlotStats Engine::bench_slots(const std::vector<llama_token> & prompt, int steps, int slots) {
    SlotStats out;
    out.slots = slots;
    const int P = (int) prompt.size();
    if (!ready() || slots < 1 || steps < 1 || P < 1) return out;
    if ((int) llama_n_seq_max(ctx_) < slots) {
        logf("[strata-vk] slots: %d asked for, the context holds %d sequences\n", slots, (int) llama_n_seq_max(ctx_));
        return out;
    }
    // Each sequence keeps its own copy of the prompt and its own recurrent state, so the cache needs slots * (P + steps).
    const long long need = (long long) (P + steps + 1) * slots;
    if (need > (long long) n_ctx_) {
        logf("[strata-vk] slots: %d sequences of %d tokens need %lld cache cells, the context has %d\n",
             slots, P + steps + 1, need, n_ctx_);
        return out;
    }

    llama_memory_clear(llama_get_memory(ctx_), true);
    kv_tokens_.clear();

    // Prefill one sequence at a time.  It sits outside the timed window, and it leaves each sequence's last-token logits at
    // a batch index that is known here.
    const int nv = llama_vocab_n_tokens(vocab_);
    std::vector<llama_token> cur((size_t) slots, 0);
    for (int s = 0; s < slots; ++s) {
        int last_rows = 0;
        for (int off = 0; off < P; off += batch_cap_) {
            const int n = std::min(batch_cap_, P - off);
            for (int i = 0; i < n; ++i) {
                batch_.token[i]     = prompt[(size_t) off + i];
                batch_.pos[i]       = off + i;
                batch_.n_seq_id[i]  = 1;
                batch_.seq_id[i][0] = s;
                batch_.logits[i]    = (off + i == P - 1) ? 1 : 0;
            }
            batch_.n_tokens = n;
            if (llama_decode(ctx_, batch_) != 0) {
                logf("[strata-vk] slots: prefill of sequence %d failed at %d/%d\n", s, off + n, P);
                return out;
            }
            last_rows = n;
        }
        const float * lg = llama_get_logits_ith(ctx_, last_rows - 1);
        if (lg == nullptr) { logf("[strata-vk] slots: no logits for sequence %d\n", s); return out; }
        cur[(size_t) s] = (llama_token) argmax_logits(lg, nv);
    }

    // The measured loop: one row per sequence, all rows in one batch.
    const Clock::time_point t0 = Clock::now();
    out.t0_epoch = epoch_seconds();
    for (int t = 0; t < steps; ++t) {
        for (int s = 0; s < slots; ++s) {
            batch_.token[s]     = cur[(size_t) s];
            batch_.pos[s]       = P + t;
            batch_.n_seq_id[s]  = 1;
            batch_.seq_id[s][0] = s;
            batch_.logits[s]    = 1;
        }
        batch_.n_tokens = slots;
        if (llama_decode(ctx_, batch_) != 0) {
            logf("[strata-vk] slots: decode failed at step %d of %d\n", t, steps);
            return out;
        }
        for (int s = 0; s < slots; ++s) {
            const float * lg = llama_get_logits_ith(ctx_, s);
            if (lg == nullptr) { logf("[strata-vk] slots: no logits at step %d\n", t); return out; }
            cur[(size_t) s] = (llama_token) argmax_logits(lg, nv);
        }
        out.steps   = t + 1;
        out.tokens += slots;
    }
    out.ms = ms_since(t0);
    out.t1_epoch = epoch_seconds();
    return out;
}

int Engine::slot_active() const {
    int n = 0;
    for (const SlotState & s : slot_st_) {
        if (s.active) ++n;
    }
    return n;
}

// Admit one request into `slot`: check it fits, drop whatever the slot held, and keep its prompt for the chunk reader.
// Nothing is decoded here, so the caller can run a step for the slots already in flight between two chunks.
Engine::SlotAdmit Engine::slot_begin(int slot, const std::vector<llama_token> & ids, int max_new,
                                     const std::string & keys) {
    SlotAdmit out;
    if (slot < 0 || slot >= (int) slot_st_.size()) { out.error = "no such free slot"; return out; }
    SlotState & s = slot_st_[(size_t) slot];
    if (s.active || s.prefilling || s.ready) { out.error = "slot busy"; return out; }
    if (ids.empty()) { out.error = "empty prompt"; return out; }
    if (max_new < 1) { out.error = "a bad max_new"; return out; }
    // The slot owns n_ctx/n_seq_max cells, so a request that would not fit is refused before anything is written: a
    // prompt that silently lost its tail would answer from the wrong text.
    if ((long long) ids.size() + max_new + 1 > (long long) slot_ctx_) {
        out.error = "prompt + max_new does not fit the slot (" + std::to_string(ids.size()) + " + " +
                    std::to_string(max_new) + " > " + std::to_string(slot_ctx_) + ")";
        return out;
    }

    llama_memory_seq_rm(llama_get_memory(ctx_), slot, -1, -1);
    s.prompt      = ids;
    s.prefill_off = 0;
    s.prefilling  = true;
    s.ready       = false;
    s.cancel      = false;
    s.max_new     = max_new;
    s.pp_t0       = Clock::now();
    s.admit       = SlotAdmit{};
    s.admit.error.clear();

    bool greedy = false;
    if (s.smpl != nullptr) llama_sampler_free(s.smpl);
    s.smpl = make_sampler(keys, &greedy);
    out.ok = true;
    return out;
}

// Read the next `chunk` prompt tokens into the slot's sequence.  The last row of the last chunk is where the admission's
// first token comes from, so that is the only row asked for logits.
int Engine::slot_prefill_chunk(int slot, int chunk) {
    if (slot < 0 || slot >= (int) slot_st_.size()) return 0;
    SlotState & s = slot_st_[(size_t) slot];
    if (!s.prefilling) return 0;
    const int total = (int) s.prompt.size();
    const int n = std::min({chunk, batch_cap_, total - s.prefill_off});
    if (n <= 0) {
        s.prefilling = false;
        s.ready = true;
        return 0;
    }
    for (int i = 0; i < n; ++i) {
        const int at = s.prefill_off + i;
        batch_.token[i]     = s.prompt[(size_t) at];
        batch_.pos[i]       = at;
        batch_.n_seq_id[i]  = 1;
        batch_.seq_id[i][0] = slot;
        batch_.logits[i]    = (at == total - 1) ? 1 : 0;
    }
    batch_.n_tokens = n;
    if (llama_decode(ctx_, batch_) != 0) {
        s.admit.error = "the prompt did not decode";
        s.admit.ok = false;
        llama_memory_seq_rm(llama_get_memory(ctx_), slot, -1, -1);
        s.prefilling = false;
        s.prompt.clear();
        if (s.smpl != nullptr) llama_sampler_free(s.smpl);
        s.smpl = nullptr;
        return 0;
    }
    s.prefill_off += n;
    if (s.prefill_off < total) return n;

    // The prompt is in: sample the token the admission answers with, which counts towards `max_new`.  `-1` is the last
    // row that asked for logits, which is the one that completes the prompt: after a chunk that asked for none, the
    // output rows of this decode hold only this one.  The prompt is not fed to the sampler, exactly as the solo path
    // does not feed it: `llama_sampler_sample` accepts what it returns, so that is the whole window either path builds.
    s.admit.pp_ms = ms_since(s.pp_t0);
    static const bool admit_dbg = std::getenv("STRATA_VK_ADMIT_DEBUG") != nullptr;
    if (admit_dbg) {
        char tag[64];
        std::snprintf(tag, sizeof(tag), "slot %d row n-1", slot);
        debug_logits_row(tag, n - 1);
        std::snprintf(tag, sizeof(tag), "slot %d row -1", slot);
        debug_logits_row(tag, -1);
    }
    s.admit.first = llama_sampler_sample(s.smpl, ctx_, n - 1);
    if (s.admit.first == 0) {
        // A row that never got its logits reads as all zeros, and a greedy sample of zeros is token 0.  Say so rather
        // than answer from an empty row.
        logf("[strata-vk] slot %d: the admission row read as token 0 (logits %s)\n", slot,
             llama_get_logits_ith(ctx_, n - 1) != nullptr ? "present" : "missing");
    }
    s.admit.eog = !ignore_eog_ && llama_vocab_is_eog(vocab_, s.admit.first);
    s.admit.continues = !s.admit.eog && s.max_new > 1;
    s.admit.ok = true;
    s.produced = 1;
    s.pos      = total;
    s.next     = s.admit.first;
    s.prefilling = false;
    s.ready      = true;
    s.prompt.clear();
    return n;
}

// Hand the caller the `T`/`DONE`/`BADM` fields and put the slot into (or out of) the batch.
bool Engine::slot_admit_report(int slot, SlotAdmit & out) {
    if (slot < 0 || slot >= (int) slot_st_.size()) return false;
    SlotState & s = slot_st_[(size_t) slot];
    if (!s.ready) {
        out = s.admit;                    // a failed chunk leaves its `error` here
        s.ready = false;
        return false;
    }
    out = s.admit;
    // `next` is the token the slot will decode in its first step, so it is by construction the token this admission
    // answers with: reading it from there keeps the `T` line and the batch in step whatever happened above.
    out.first = s.next;
    s.ready = false;
    s.active = out.continues;
    if (!out.continues && s.smpl != nullptr) {
        llama_sampler_free(s.smpl);
        s.smpl = nullptr;
    }
    return out.ok;
}

void Engine::slot_stop(int slot) {
    if (slot >= 0 && slot < (int) slot_st_.size()) slot_st_[(size_t) slot].cancel = true;
}

// One token for every active slot, in a single batch: the dense weights are read once for the whole batch, which is the
// point of running them together at all (PERFORMANCE.md section 20.45).
void Engine::slot_step(std::vector<SlotEvent> & out) {
    out.clear();
    std::vector<int> live;
    for (int i = 0; i < (int) slot_st_.size(); ++i) {
        if (slot_st_[(size_t) i].active) live.push_back(i);
    }
    if (live.empty()) return;
    for (size_t k = 0; k < live.size(); ++k) {
        const SlotState & s = slot_st_[(size_t) live[k]];
        batch_.token[k]     = s.next;
        batch_.pos[k]       = s.pos;
        batch_.n_seq_id[k]  = 1;
        batch_.seq_id[k][0] = live[k];
        batch_.logits[k]    = 1;
    }
    batch_.n_tokens = (int) live.size();
    const bool ok = llama_decode(ctx_, batch_) == 0;
    for (size_t k = 0; k < live.size(); ++k) {
        const int slot = live[k];
        SlotState & s = slot_st_[(size_t) slot];
        SlotEvent e;
        e.slot = slot;
        // A failed step ends every slot in it: the alternative is to keep stepping on a cache that may not have moved,
        // which would put the same token in twice.
        const bool stop_now = !ok || s.cancel || s.produced >= s.max_new;
        if (stop_now) {
            e.done = true;
            e.produced = s.produced;
            e.eog = !ignore_eog_ && llama_vocab_is_eog(vocab_, s.next);
            s.active = false;
            s.cancel = false;
            if (s.smpl != nullptr) llama_sampler_free(s.smpl);
            s.smpl = nullptr;
            out.push_back(e);
            continue;
        }
        // `llama_sampler_sample` already accepted what it returned: accepting again would double the token in the
        // penalties window and make a slot answer differently from a solo run on the same request.
        s.next = llama_sampler_sample(s.smpl, ctx_, (int32_t) k);
        ++s.pos;
        ++s.produced;
        e.tok = s.next;
        e.fresh = true;
        e.produced = s.produced;
        e.eog = !ignore_eog_ && llama_vocab_is_eog(vocab_, s.next);
        // An end-of-generation token is emitted, then the slot ends on the *next* step, exactly as the solo path prints
        // the stop token and stops.  `produced >= max_new` is checked here so the last token is not followed by a decode.
        if (e.eog || s.produced >= s.max_new) {
            e.done = true;
            s.active = false;
            s.cancel = false;
            if (s.smpl != nullptr) llama_sampler_free(s.smpl);
            s.smpl = nullptr;
        }
        out.push_back(e);
    }
}

// The last n-1 tokens plus the token just sampled form the pattern; an earlier occurrence of it gives the continuation// to draft.  The occurrence is found through an index built as the history grows, so the cost is one hash and one map
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
//
// One chain per call, owned by the caller.  It used to be cached by key, which cannot work: the solo path frees what it
// is given, so the cache held a freed pointer for the next request, and the slots all shared one chain and freed it once
// each.  A penalties chain also carries the request's window, which no two requests may share.  What the cache saved was
// one vocabulary-sized allocation per request.
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
    sp.no_perf = true;   // nothing reads llama_perf_sampler here; the clock reads are pure cost
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









