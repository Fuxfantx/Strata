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

struct Options;
void log_pack_plan(const Options & opt);

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
    // The picture path: `--mmproj` says the front end has a reader for this model, and `image_pad_token` is the
    // id a picture's rows replace (serve/server.py writes one per image token).
    const std::string & mmproj() const { return opt_.mmproj; }
    bool vision_enabled() const { return opt_.vision || !opt_.mmproj.empty(); }
    int  n_embd() const { return n_embd_; }
    int  image_pad_token() const;
    bool ready() const { return ctx_ != nullptr; }
    // The loaded model and this context, for the concurrent-context benchmark: a second llama_context on the same model
    // shares its weight buffers, so two decodes can run at once on one copy of the weights.
    llama_model   * model()   const { return model_; }
    llama_context * context() const { return ctx_; }
    std::string describe() const;

    /** Tokenise text the way the app's prompt would be tokenised (BOS added when the model wants one). */
    std::vector<llama_token> tokenize(const std::string & text) const;

    /** The inverse, for reading a token stream back as text (--detok, and the comparison scripts). */
    std::string detokenize(const std::vector<llama_token> & ids) const;

    /** In serve mode the protocol lines (T, PP) go to stdout; a one-shot or benchmark run prints its own summary. */
    void set_wire(bool wire) { wire_ = wire; }

    /** Benchmark mode: keep decoding past an end-of-generation token, which a dummy prompt hits immediately. */
    void set_ignore_eog(bool v) { ignore_eog_ = v; }

    // Concurrent decode: `slots` independent sequences, each prefilled with `prompt`, then one token per sequence per
    // llama_decode.  Sampling is argmax, so no sampler state is shared between sequences.  `slots` = 1 and `slots` > 1 run
    // through this same function, which is what makes their numbers comparable.
    // `t0_epoch`/`t1_epoch` bracket the measured loop only, not the per-sequence prefills, so an outside power sampler
    // can price the decode steps without the prompt work of `slots` sequences in the same window.
    struct SlotStats { int slots = 0, steps = 0, tokens = 0; double ms = 0.0, t0_epoch = 0.0, t1_epoch = 0.0; };
    SlotStats bench_slots(const std::vector<llama_token> & prompt, int steps, int slots);

    // --parallel N: N request slots, each one KV sequence, all stepping together in a single batch.  This is the serve
    // path's concurrency: one process, one copy of the weights, N requests in flight (PERFORMANCE.md section 20.44).
    // Only the protocol layer above decides what a slot means; the engine owns the sequences and the step.
    struct SlotAdmit { bool ok = false; llama_token first = 0; double pp_ms = 0.0; bool continues = false; bool eog = false;
                       std::string error; };
    // `fresh` says whether `tok` is a token this step produced; a slot that is only being closed reports no token.
    struct SlotEvent { int slot = 0; llama_token tok = 0; bool fresh = false; bool done = false; bool eog = false;
                       int produced = 0; };
    int  parallel_slots() const { return (int) slot_st_.size(); }
    int  slot_ctx() const { return slot_ctx_; }
    int  slot_active() const;
    // An admission reads its prompt in chunks so that the slots already running keep decoding between them: a prompt
    // read to the end in one call turns the wait into wall time that no other request can use (section 20.46).
    SlotAdmit slot_begin(int slot, const std::vector<llama_token> & ids, int max_new, const std::string & keys);
    int  slot_prefill_chunk(int slot, int chunk);          // prompt tokens read, 0 when the prompt is complete
    bool slot_admit_report(int slot, SlotAdmit & out);      // fills the `T`/`DONE`/`BADM` fields once it is complete
    void slot_stop(int slot);
    // One token for every active slot, in one batch.  A slot that ends is reported with `done`: it has no further token,
    // and its sequence is dropped so the next request on that slot starts from an empty cache.
    void slot_step(std::vector<SlotEvent> & out);

    // One picture's embeddings: the rows that stand in for `ids[pos .. pos+n_tokens)`.  The front end writes one
    // <|image_pad|> per image token there (serve/server.py, the `ids = out` loop), so a picture is a run of that token
    // and this span says which run it is.  The engine never runs the projector: serve/server.py encodes the picture and
    // hands the file over on the GENI line (port/docs/features/VISION.md).
    struct ImageSpan {
        int pos = 0;                    // first prompt position the picture replaces
        int n_tokens = 0;               // how many positions it holds
        const float * embd = nullptr;   // n_tokens * n_embd floats, row-major
    };

    // One generation.  `ids` is the whole prompt as the server builds it every turn; only the suffix past the tokens
    // already in the KV cache is decoded.  `on_token` returns false to stop early.  `images` is empty for a text
    // request, which is then byte for byte the run it always was.
    size_t moe_cache_mib_ = 0;   // decided at load time by the expert placement policy

    template <typename OnToken>
    Stats generate(const std::vector<llama_token> & ids, int max_new, const std::string & keys,
                   OnToken on_token, bool print_pp,
                   const std::vector<ImageSpan> & images = std::vector<ImageSpan>()) {
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
        // A picture's rows must go through the embedding path, so the prefill stops at each span: text before it, the
        // span itself as embeddings, text after it.  Only the segments change; every call continues the same positions.
        std::vector<ImageSpan> spans;
        for (const ImageSpan & im : images) {
            if (im.n_tokens > 0 && im.embd != nullptr && im.pos >= st.reused) spans.push_back(im);
        }
        if (!spans.empty()) {
            // never start the read inside a picture: a run of <|image_pad|> that begins before the reused prefix has no
            // embeddings for its head, so its prefix is dropped and read again
            for (const ImageSpan & im : spans) {
                if (im.pos < st.reused) st.reused = im.pos;
            }
        }
        // STRATA_VK_SEG_LOG=1: one line per prefill segment (kind, rows, ms).  A picture costs a fixed ~1 s whatever
        // its row count (port/docs/features/VISION.md 3.2), so this is what separates "the rows are slow" from
        // "something around them is".
        const bool seg_log = std::getenv("STRATA_VK_SEG_LOG") != nullptr;
        size_t next_span = 0;
        for (int done = 0; done < total;) {
            const Clock::time_point seg_t0 = Clock::now();
            if (g_stop) {
                // a partial prompt leaves a prefix that is not a prompt: drop it so the next turn reads cleanly
                st.finish = "stop";
                reset();
                return st;
            }
            const int at = st.reused + done;
            if (next_span < spans.size() && spans[next_span].pos == at) {
                const ImageSpan & im = spans[next_span];
                for (int off = 0; off < im.n_tokens;) {
                    const int n = std::min(opt_.n_batch, im.n_tokens - off);
                    last_chunk_pos = (int) kv_tokens_.size();
                    if (!decode_embd(im.embd + (size_t) off * (size_t) n_embd_, n, last_chunk_pos,
                                     done + off + n == total)) {
                        st.finish = "error";
                        return st;
                    }
                    kv_tokens_.insert(kv_tokens_.end(), ids.begin() + st.reused + done + off,
                                      ids.begin() + st.reused + done + off + n);
                    off += n;
                    done += n;
                }
                if (seg_log) {
                    logf("[strata-vk] seg picture n=%d pos=%d %.1f ms\n", im.n_tokens, im.pos, ms_since(seg_t0));
                }
                last_chunk = nullptr;      // a picture is not a text row: nothing to prime a draft on
                last_chunk_n = 0;
                ++next_span;
                continue;
            }
            // text up to the next picture (or to the end of the prompt)
            int limit = total - done;
            if (next_span < spans.size() && spans[next_span].pos > at) limit = spans[next_span].pos - at;
            const int n = std::min(opt_.n_batch, limit);
            last_chunk_pos = (int) kv_tokens_.size();
            if (!decode(ids.data() + st.reused + done, n, last_chunk_pos, done + n == total)) {
                st.finish = "error";
                return st;
            }
            last_chunk = ids.data() + st.reused + done;
            last_chunk_n = n;
            kv_tokens_.insert(kv_tokens_.end(), ids.begin() + st.reused + done, ids.begin() + st.reused + done + n);
            done += n;
            if (seg_log) {
                logf("[strata-vk] seg text n=%d pos=%d %.1f ms\n", n, at, ms_since(seg_t0));
            }
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
        // The wait first: with async submission the last decode returns as soon as it is submitted, so without this the
        // prompt clock below would report the submit instead of the work (port/docs/ADRENO.md, "Async submission").
        // It is one fence per prompt, and it is what makes an async run measurable at all.
        llama_synchronize(ctx_);
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
        // Tri::Auto resolves to off here: the option's own note records that a verify pass on this backend costs more than
        // the tokens it can save, so only an explicit --lookup on takes the risk.
        const bool can_lookup = greedy && spec_ok_ && opt_.lookup == Tri::On;
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

        static const bool admit_dbg = std::getenv("STRATA_VK_ADMIT_DEBUG") != nullptr;
        if (admit_dbg) debug_logits_row("solo row -1", -1);
        llama_token tok = llama_sampler_sample(smpl, ctx_, -1);
        Clock::time_point tg = Clock::now();
        const double decode_cpu0 = process_cpu_ms();

        // STRATA_VK_PROFILE=1: the per-token loop is decode plus sampling; timing the two separately is what says
        // whether the wall time outside the graph belongs to the backend or to the sampler.
        static const bool profile_tok = std::getenv("STRATA_VK_PROFILE") != nullptr;
        double acc_decode = 0.0, acc_sample = 0.0;
        int    n_steps    = 0;

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
                static const bool rdc_on = std::getenv("STRATA_VK_RDC") != nullptr;
                const bool rdc = rdc_on && rdc_capture_wanted(st.generated);
                if (rdc) rdc_capture_begin();
                if (!decode(&tok, 1, pos, true)) { st.finish = "error"; break; }
                if (profile_tok) {
                    acc_decode += ms_since(ts);
                    ++n_steps;
                }
                if (rdc) rdc_capture_end();
                if (first_decode_pending) {
                    // st.generated already counts the token being decoded, so the first step is tracked by hand
                    first_decode_pending = false;
                    st.first_step_ms = ms_since(ts);
                }
                pace_step(opt_.perf_cap);
                kv_tokens_.push_back(tok);
                if (can_mtp && mtp_primed_) mtp_.observe(&tok, 1, pos);
                {
                    const Clock::time_point tsamp = Clock::now();
                    tok = llama_sampler_sample(smpl, ctx_, -1);
                    if (profile_tok) {
                        acc_sample += ms_since(tsamp);
                    }
                }
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

        // The sampler reads the logits every step, which already waits for that step's graph; this makes the decode
        // clock airtight for a run whose last step did not sample (async, ADRENO.md).
        llama_synchronize(ctx_);
        st.decode_ms = ms_since(tg);
        st.decode_cpu_ms = process_cpu_ms() - decode_cpu0;
        if (profile_tok && n_steps > 0) {
            const double wall = st.decode_ms;
            logf("[strata-vk] token loop: decode %.2f ms, sample %.2f ms, rest %.2f ms per token (%d steps)\n",
                 acc_decode / n_steps, acc_sample / n_steps,
                 (wall - acc_decode - acc_sample) / n_steps, n_steps);
        }
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
    bool decode_embd(const float * embd, int n, int pos0, bool want_logits);   // a picture's rows (ImageSpan)
    bool decode_all(const llama_token * toks, int n, int pos0);
    std::vector<llama_token> lookup_draft(const std::vector<llama_token> & hist, llama_token sampled, int budget);
    // Bring the n-gram index up to the history it is asked about; rebuilds it when the history was trimmed.
    void index_ngrams(const std::vector<llama_token> & hist);

    // The sampler chain is a pure function of the request's sampling keys but it also carries that request's penalty
    // window, so every caller gets its own and frees it (see make_sampler).
    // Decode the draft in one target pass, keep the prefix the target agrees with, and report the token that follows.
    bool verify_draft(llama_sampler * smpl, llama_token cur, const std::vector<llama_token> & draft,
                      std::vector<llama_token> & agreed, llama_token & next, bool use_mtp);
    void mtp_prime(const llama_token * toks, int n, int pos0);
    llama_sampler * make_sampler(const std::string & keys, bool * greedy) const;
    // STRATA_VK_ADMIT_DEBUG=1: summarise a logits row by index, so a first token that differs between a solo run and a
    // slot can be blamed on the row (the model state) or on the sampler (the chain) rather than guessed at.
    void debug_logits_row(const char * tag, int32_t idx) const;

    Options opt_;
    NpuPlan npu_;
    GpuBackend backend_ = GpuBackend::Auto;
    Mtp mtp_;
    bool mtp_primed_ = false;
    llama_model * model_ = nullptr;
    llama_context * ctx_ = nullptr;
    const llama_vocab * vocab_ = nullptr;
    int         n_embd_ = 0;      // the model's embedding width: the row size of a picture's embeddings
    llama_batch batch_ = {};
    llama_batch batch_embd_ = {};   // the picture rows        // allocated on the first picture: llama_batch_init gives a batch either tokens or
                                    // embeddings, never both, which is why a picture is its own decode call
    int batch_embd_cap_ = 0;
    int         batch_cap_ = 0;   // rows batch_ can hold; the drafters must never exceed it
    int n_ctx_ = 0;
    // --parallel N: one entry per slot, empty unless `--serve` was asked for more than one.  `slot_ctx_` is the cache a
    // slot owns, `n_ctx / n_seq_max`, which is what bounds a slot's prompt plus its reply.
    struct SlotState {
        bool active = false, cancel = false;
        int  max_new = 0, produced = 0, pos = 0;
        llama_token next = 0;
        llama_sampler * smpl = nullptr;
        // A slot that is still reading its prompt: `prompt`/`prefill_off` walk it a chunk at a time, and `ready` says the
        // first token has been sampled and the admission's `T`/`DONE`/`BADM` are owed.
        bool prefilling = false, ready = false;
        std::vector<llama_token> prompt;
        int  prefill_off = 0;
        Clock::time_point pp_t0{};
        SlotAdmit admit;
    };
    std::vector<SlotState> slot_st_;
    int slot_ctx_ = 0;
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





