// vk-engine.h - the model, the context, and one generation.
//
// Two self-speculative paths share one verifier, because a draft is only ever a *proposal*: the target samples every
// position itself, so an accepted draft yields exactly the tokens the target would have produced alone.
//
//   * the MTP head (--mtp FILE) drafts from the model's own hidden state: general purpose, needs a second GGUF;
//   * prompt lookup drafts the continuation of an n-gram seen earlier in the same conversation: free, but it only
//     pays where the text repeats (code, quoting, lists).
#pragma once

#include "llama.h"

#include "vk-common.h"
#include "vk-devices.h"
#include "vk-mtp.h"
#include "vk-options.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata {

/** Token ids back to text, for a vocabulary that may not have a loaded model behind it (--detok). */
std::string detokenize_ids(const llama_vocab * vocab, const std::vector<llama_token> & ids);

struct Stats {
    int         prompt_tokens = 0;
    int         reused        = 0;
    double      prompt_ms     = 0;
    double      decode_ms     = 0;
    double      first_step_ms = 0;   // the first decode step on its own: it pays page faults and pipeline use
    double      decode_cpu_ms = 0;   // CPU time the process burned inside the same window
    int         generated     = 0;
    int         drafted       = 0;   // tokens offered by a drafter
    int         accepted      = 0;   // of those, the ones the target agreed with
    std::string finish        = "stop";   // stop | length | error
};

// set by the dry run once the resolved policy has been printed (see --dry-run)
extern bool dry_run_finished;

// RenderDoc in-application capture (vk-rdc.cpp): desired for one decode step, chosen by STRATA_VK_RDC=<step>.
bool rdc_capture_wanted(int step);
void rdc_capture_begin();
void rdc_capture_end();

class Engine {
public:
    ~Engine() { unload(); }

    bool load(const Options & opt, const NpuPlan & npu);
    void unload();

    int  n_ctx() const { return n_ctx_; }
    bool ready() const { return ctx_ != nullptr; }
    std::string describe() const;

    /** Tokenise text the way the app's prompt would be tokenised (BOS added when the model wants one). */
    std::vector<llama_token> tokenize(const std::string & text) const;

    /** The inverse, for reading a token stream back as text (--detok, and the comparison scripts). */
    std::string detokenize(const std::vector<llama_token> & ids) const;

    /** In serve mode the protocol lines (T, PP) go to stdout; a one-shot or benchmark run prints its own summary. */
    void set_wire(bool wire) { wire_ = wire; }

    /** Benchmark mode: keep decoding past an end-of-generation token, which a dummy prompt hits immediately. */
    void set_ignore_eog(bool v) { ignore_eog_ = v; }

    // One generation.  `ids` is the whole prompt as the server builds it every turn; only the suffix past the tokens
    // already in the KV cache is decoded.  `on_token` returns false to stop early.
    size_t moe_cache_mib_ = 0;   // decided at load time by the expert placement policy

    template <typename OnToken>
    Stats generate(const std::vector<llama_token> & ids, int max_new, const std::string & keys,
                   OnToken on_token, bool print_pp) {
        Stats st;
        st.prompt_tokens = (int) ids.size();
        if (!ready()) {
            st.finish = "error";
            return st;
        }

        const llama_memory_t mem = llama_get_memory(ctx_);
        trim_to_common_prefix(mem, ids);
        st.reused = (int) kv_tokens_.size();

        Clock::time_point tp = Clock::now();
        const int total = (int) ids.size() - st.reused;
        const llama_token * last_chunk = nullptr;
        int last_chunk_n = 0, last_chunk_pos = 0;
        for (int done = 0; done < total;) {
            if (g_stop) {
                // a partial prompt leaves a prefix that is not a prompt: drop it so the next turn reads cleanly
                st.finish = "stop";
                reset();
                return st;
            }
            const int n = std::min(opt_.n_batch, total - done);
            last_chunk_pos = (int) kv_tokens_.size();
            if (!decode(ids.data() + st.reused + done, n, last_chunk_pos, done + n == total)) {
                st.finish = "error";
                return st;
            }
            last_chunk = ids.data() + st.reused + done;
            last_chunk_n = n;
            kv_tokens_.insert(kv_tokens_.end(), ids.begin() + st.reused + done, ids.begin() + st.reused + done + n);
            done += n;
            if (print_pp && wire_) {
                const double ms = ms_since(tp);
                char buf[160];
                std::snprintf(buf, sizeof(buf), "PP %d %d %.3f %.3f", done, total, ms,
                              ms > 0 ? done * 1000.0 / ms : 0.0);
                out_line(buf);
            }
        }
        // Perf cap: hand the rest of the wall time to the machine (browsers and Office need the GPU too) and report the
        // capped rate, so the printed tok/s is what the machine delivers at this share.
        {
            const double busy = ms_since(tp);
            if (opt_.perf_cap < 100) {
                sleep_ms(busy * (100.0 - opt_.perf_cap) / opt_.perf_cap);
                st.prompt_ms = busy + busy * (100.0 - opt_.perf_cap) / opt_.perf_cap;   // wall time, idle included
            } else {
                st.prompt_ms = busy;
            }
        }
        if (max_new <= 0) return st;          // a prompt-only run (the prefill benchmark)

        bool greedy = false;
        llama_sampler * smpl = make_sampler(keys, &greedy);

        // A non-greedy chain or a penalty window has state that a rejected draft would corrupt, so those requests never
        // speculate; the answer must stay exactly what a plain run would produce.
        const bool can_mtp    = greedy && mtp_.ready() && last_chunk_n > 0;
        const bool can_lookup = greedy && spec_ok_ && opt_.lookup != Tri::Off;
        if (can_mtp) mtp_prime(last_chunk, last_chunk_n, last_chunk_pos);

        if (!opt_.dump_logits.empty() && !dump_done_) {
            dump_done_ = true;
            const int nv = llama_vocab_n_tokens(vocab_);
            const float * logits = llama_get_logits_ith(ctx_, -1);
            if (logits != nullptr) {
                std::FILE * f = std::fopen(opt_.dump_logits.c_str(), "wb");
                if (f != nullptr) {
                    std::fwrite(logits, sizeof(float), (size_t) nv, f);
                    std::fclose(f);
                    logf("[strata-vk] wrote %d logits to %s\n", nv, opt_.dump_logits.c_str());
                } else {
                    logf("[strata-vk] cannot write %s\n", opt_.dump_logits.c_str());
                }
            }
        }

        llama_token tok = llama_sampler_sample(smpl, ctx_, -1);
        Clock::time_point tg = Clock::now();
        const double decode_cpu0 = process_cpu_ms();

        bool first_decode_pending = true;
        while (true) {
            if (g_stop) { st.finish = "stop"; break; }
            if (!ignore_eog_ && llama_vocab_is_eog(vocab_, tok)) { st.finish = "stop"; break; }

            if (wire_) out_line("T " + std::to_string((int) tok));
            if (!on_token(tok)) { st.finish = "stop"; break; }
            ++st.generated;

            // The last token is printed but not decoded: the cache then holds exactly the text the server gets back
            // next turn (prompt + accepted tokens), and the next turn reuses it whole.
            if (st.generated >= max_new) { st.finish = "length"; break; }

            const int pos  = (int) kv_tokens_.size();
            const int room = std::min(max_new - st.generated, n_ctx_ - pos - 1);

            std::vector<llama_token> proposals;
            bool from_mtp = false;
            if (room > 0) {
                if (can_mtp && mtp_primed_) {
                    proposals = mtp_.draft(tok, pos, std::min(room, mtp_draft_n()));
                    from_mtp  = !proposals.empty();
                } else if (can_lookup) {
                    proposals = lookup_draft(kv_tokens_, tok, room);
                }
                if ((int) proposals.size() > room) proposals.resize((size_t) room);
            }

            if (proposals.empty()) {
                const Clock::time_point ts = Clock::now();
                // STRATA_VK_RDC=<step>: wrap exactly one decode step in a RenderDoc frame capture (vk-rdc.cpp).  The
                // capture therefore holds one graph, with every pipeline and descriptor of a single token in it.
                const bool rdc = rdc_capture_wanted(st.generated);
                if (rdc) rdc_capture_begin();
                if (!decode(&tok, 1, pos, true)) { st.finish = "error"; break; }
                if (rdc) rdc_capture_end();
                if (first_decode_pending) {
                    // st.generated already counts the token being decoded, so the first step is tracked by hand
                    first_decode_pending = false;
                    st.first_step_ms = ms_since(ts);
                }
                pace_step(opt_.perf_cap);
                kv_tokens_.push_back(tok);
                if (can_mtp && mtp_primed_) mtp_.observe(&tok, 1, pos);
                tok = llama_sampler_sample(smpl, ctx_, -1);
                continue;
            }

            st.drafted += (int) proposals.size();
            std::vector<llama_token> agreed;
            llama_token next = tok;
            if (!verify_draft(smpl, tok, proposals, agreed, next, from_mtp)) { st.finish = "error"; break; }
            st.accepted += (int) agreed.size();

            // The agreed tokens are already in the cache; emit them, then continue from the token after the last one.
            bool stop_after = false;
            for (const llama_token t : agreed) {
                if (g_stop || (!ignore_eog_ && llama_vocab_is_eog(vocab_, t))) { st.finish = "stop"; stop_after = true; break; }
                if (wire_) out_line("T " + std::to_string((int) t));
                if (!on_token(t)) { st.finish = "stop"; stop_after = true; break; }
                ++st.generated;
                if (st.generated >= max_new) { st.finish = "length"; stop_after = true; break; }
            }
            if (stop_after) break;
            tok = next;
        }

        st.decode_ms = ms_since(tg);
        st.decode_cpu_ms = process_cpu_ms() - decode_cpu0;
        // decode_ms is already wall time (the pacing slept inside the loop), so it needs no rescaling.
        llama_sampler_free(smpl);
        if (st.finish == "error") reset();
        return st;
    }

    /** Forget the KV cache (a new conversation, or after a failed decode). */
    void reset();

    static void request_stop() { g_stop = true; }
        static void clear_stop() { g_stop = false; }

private:
    // Devices, model and context.  Split out of load() because a placement the NPU cannot serve is retried without it.
    std::vector<ggml_backend_dev_t> plan_devices(const Options & opt, const NpuPlan & npu);
    llama_model * load_model(const Options & opt, const NpuPlan & npu);

    void trim_to_common_prefix(llama_memory_t mem, const std::vector<llama_token> & ids);
    bool decode(const llama_token * toks, int n, int pos0, bool want_logits);
    bool decode_all(const llama_token * toks, int n, int pos0);
    std::vector<llama_token> lookup_draft(const std::vector<llama_token> & hist, llama_token sampled, int budget);
    // Bring the n-gram index up to the history it is asked about; rebuilds it when the history was trimmed.
    void index_ngrams(const std::vector<llama_token> & hist);
    // Decode the draft in one target pass, keep the prefix the target agrees with, and report the token that follows.
    bool verify_draft(llama_sampler * smpl, llama_token cur, const std::vector<llama_token> & draft,
                      std::vector<llama_token> & agreed, llama_token & next, bool use_mtp);
    void mtp_prime(const llama_token * toks, int n, int pos0);
    llama_sampler * make_sampler(const std::string & keys, bool * greedy) const;

    Options opt_;
    NpuPlan npu_;
    GpuBackend backend_ = GpuBackend::Auto;
    Mtp mtp_;
    bool mtp_primed_ = false;
    llama_model * model_ = nullptr;
    llama_context * ctx_ = nullptr;
    const llama_vocab * vocab_ = nullptr;
    llama_batch batch_ = {};
    int n_ctx_ = 0;
    Clock::time_point load_t0_{};
    bool wire_ = false;                        // serve mode: the protocol lines own stdout
    bool ignore_eog_ = false;                  // benchmark mode: an end-of-generation token does not stop the loop
    std::string arch_ = "unknown";
    std::vector<llama_token> kv_tokens_;
    bool dump_done_ = false;   // --dump-logits writes once

    // Prompt-lookup drafting needs "where did this n-gram appear before".  A backward scan over the history answers it
    // in O(history) per token; an index built as the history grows answers it in O(1) with two hash multiplies, which is
    // what makes drafting cheap enough to leave on by default.  One entry per history position, bounded per key.
    struct NgramSlots { int32_t pos[4]; uint8_t n = 0; };
    std::unordered_map<uint64_t, NgramSlots> ngram_;
    size_t ngram_indexed_ = 0;                 // positions whose window is in the index

    std::vector<ggml_backend_dev_t> devices_;   // `llama_model_params::devices`, null-terminated
    std::vector<float> split_;                  // `llama_model_params::tensor_split`: 0 for an NPU, layers elsewhere
    bool spec_ok_  = true;                      // false once the cache cannot drop a suffix
    // how often the cache refused to drop a rejected draft's suffix, which turns drafting off for the rest of the run;
    // the first refusal is logged, the count only keeps the log to one line
    long spec_rm_failed_ = 0;

    // Draft shape.  A draft is verified by one target pass over the drafted tokens, and that pass runs at *prefill*
    // speed: on Gyro-M prefill is 7.4 tok/s against 8.7 decode, so a pass costs more per token than the decode it
    // replaces, and only a short draft amortises the pass.  Measured on Gyro-M, repetitive prompt, 32 tokens:
    // m=1 9.1 tok/s, m=2 8.6, m=4 8.9, m=8 3.6, no drafting 7.6.  A model whose prefill is much faster than its
    // decode (the normal case) wants a longer draft - STRATA_VK_LOOKUP_NM=n,m overrides.
    int  lookup_n_ = 3;
    int  lookup_m_ = 2;

    // How many tokens the MTP head is asked for per round.  Each one is a full draft pass whose LM head is 1.7-2.2 GB of
    // weights on the packs here, so the useful draft length is a trade against that cost, not as long as possible.
    // Measured on the 27B, repetitive prompt, 24 tokens: n=1 7.4 tok/s (10/12 kept), n=2 6.9 (11/16), n=4 6.4 (16/23);
    // on the 35B-A3B every length loses to plain decoding (n=1 17.1, n=2 12.6 against 36.2), so a caller that wants
    // drafts on a dense model should ask for one at a time.
    int  mtp_draft_n() const {
        static const int n = [] { const char * e = std::getenv("STRATA_VK_MTP_N"); return e ? std::max(1, std::atoi(e)) : 1; }();
        return n;
    }

    static inline bool g_stop = false;
};

} // namespace strata





