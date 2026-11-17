// vk-serve.cpp - the protocol serve/server.py speaks: one command per line, one answer line per command.
//
// READY / INFO / T / PP / DONE are what the server parses; the field layout of DONE is the released engine's, because
// serve/server.py indexes into it positionally.
#include "vk-serve.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace strata {
namespace {

std::atomic<bool> g_quit{false};

std::vector<std::string> split_words(const std::string & line, size_t from) {
    std::vector<std::string> parts;
    size_t i = from;
    while (i <= line.size()) {
        const size_t j = line.find(' ', i);
        if (j == std::string::npos) {
            parts.push_back(line.substr(i));
            break;
        }
        parts.push_back(line.substr(i, j - i));
        i = j + 1;
        if (i > line.size()) break;
    }
    return parts;
}

std::string first_word(const std::string & line) {
    const size_t j = line.find(' ');
    return j == std::string::npos ? line : line.substr(0, j);
}

// Prompt tokens per read between two batch steps.  The released engine reads 32 at a time so that the slots already in
// flight keep their cadence while a request is admitted.  On this machine that is a loss: a prefill pass streams the
// whole dense set (3.06 GiB of weights) whatever its size, so a prompt read in two chunks reads those weights twice.
// Measured at 48 tokens on 4 slots: whole prompt 9.17 s of wall time, 32-token chunks 11.23 s, for the same 64 tokens
// (section 20.47).  The default is therefore one read; STRATA_VK_PREFILL_CHUNK=N turns the pipelined reader on.
int prefill_chunk() {
    static const int n = [] {
        const char * e = std::getenv("STRATA_VK_PREFILL_CHUNK");
        const int v = e != nullptr ? std::atoi(e) : 0;
        return v > 0 ? v : (1 << 20);
    }();
    return n;
}

void emit_slot_events(const std::vector<Engine::SlotEvent> & events, bool stream);

void handle_gen(Engine & engine, const std::string & line) {
    // GEN <max_new>[ key=value]* <ids>: the keys are space-separated `name=value` words and the ids are
    // COMMA-separated, so the first word without '=' starts the ids and every following word belongs to them.
    const std::vector<std::string> parts = split_words(line, 4);
    if (parts.empty()) {
        out_line("ERR malformed GEN");
        return;
    }
    int max_new = std::atoi(parts[0].c_str());
    if (max_new < 0) max_new = 0;

    std::string keys, id_text;
    size_t p = 1;
    for (; p < parts.size() && parts[p].find('=') != std::string::npos; ++p) keys += " " + parts[p];
    for (; p < parts.size(); ++p) {
        if (!id_text.empty()) id_text += ' ';
        id_text += parts[p];
    }

    const std::vector<llama_token> ids = parse_ids(id_text);
    if (ids.empty()) {
        out_line("ERR empty prompt");
        return;
    }
    if (max_new == 0) {
        out_line("DONE 0 " + std::to_string(ids.size()) + " 0.000 0.000 stop 0 0 0 0 0 0 0 0.0 0 0");
        return;
    }

    logf("[strata-vk] GEN %d tokens, prompt %zu%s\n", max_new, ids.size(), keys.c_str());
    Engine::clear_stop();
    // One `GEN <id>` per token, because the app paints the answer as it arrives: a summary-only reply would leave the
    // panel empty until the whole answer was finished.  `stream=0` turns it off for a caller that wants only the summary.
    const bool stream = keys.find("stream=0") == std::string::npos;
    const Stats st = engine.generate(ids, max_new, keys,
        [stream](llama_token id) {
            if (stream) out_line("GEN " + std::to_string(id));
            return true;
        }, true);

    char buf[512];
    std::snprintf(buf, sizeof(buf), "DONE %d %d %.3f %.3f %s 0 0 %d 0 0 0 0 0.0 %d 0",
                  st.generated, st.prompt_tokens, st.prompt_ms, st.decode_ms, st.finish.c_str(),
                  st.reused, st.prompt_tokens - st.reused);
    out_line(buf);
    logf("[strata-vk] prompt %d tokens = %d reused + %d read in %.0f ms (%.1f tok/s), %d generated in %.0f ms "
         "(%.1f tok/s)%s\n",
         st.prompt_tokens, st.reused, st.prompt_tokens - st.reused, st.prompt_ms,
         st.prompt_ms > 0 ? (st.prompt_tokens - st.reused) * 1000.0 / st.prompt_ms : 0.0,
         st.generated, st.decode_ms,
         st.decode_ms > 0 ? st.generated * 1000.0 / st.decode_ms : 0.0,
         st.drafted > 0 ? (" [" + std::to_string(st.accepted) + "/" + std::to_string(st.drafted) + " drafts kept]").c_str() : "");
    Engine::clear_stop();
}

// BGEN <slot> <max_new>[ key=value]* <ids>: the same admission the released engine gives a batch slot.  The prompt is
// read into that slot's own sequence, the first token is answered as `T`, the admission ends with `DONE` and `BADM`
// says whether the slot goes on; every token after the first arrives as `BT <slot> <id>` from the shared step.
void handle_bgen(Engine & engine, const std::string & line, bool stream) {
    const std::vector<std::string> parts = split_words(line, 5);
    if (parts.size() < 2) {
        out_line("ERR malformed BGEN");
        return;
    }
    const int slot = std::atoi(parts[0].c_str());
    int max_new = std::atoi(parts[1].c_str());
    if (max_new < 0) max_new = 0;

    std::string keys, id_text;
    size_t p = 2;
    for (; p < parts.size() && parts[p].find('=') != std::string::npos; ++p) keys += " " + parts[p];
    for (; p < parts.size(); ++p) {
        if (!id_text.empty()) id_text += ' ';
        id_text += parts[p];
    }
    const std::vector<llama_token> ids = parse_ids(id_text);
    if (ids.empty()) {
        out_line("ERR empty prompt");
        return;
    }
    if (max_new == 0) {
        out_line("DONE 0 " + std::to_string(ids.size()) + " 0.000 0.000 stop 0 0 0 0 0 0 0 0.0 0 0");
        out_line("BADM " + std::to_string(slot) + " 0");
        return;
    }

    Engine::SlotAdmit ad = engine.slot_begin(slot, ids, max_new, keys);
    if (!ad.ok) {
        // The server reads this as "try another slot" and falls back to a solo GEN after BGEN_REFUSALS_MAX refusals.
        out_line("ERR BGEN: " + ad.error);
        return;
    }
    logf("[strata-vk] BGEN slot %d: %d tokens, prompt %zu%s\n", slot, max_new, ids.size(), keys.c_str());
    // The prompt is read a chunk at a time, and a batch step runs between chunks: otherwise the read is wall time no
    // other request can use.  `PP` is the progress line the server already parses for a solo prompt read.
    int read = 0;
    int n = 0;
    const int chunk = prefill_chunk();
    // A prompt that fits one read is admitted with no step in between: alternating a prefill-shaped batch with a
    // step-shaped one is what made the step itself slower here (781 to 1193 ms, section 20.47).
    const bool pipelined = chunk < (int) ids.size();
    while ((n = engine.slot_prefill_chunk(slot, chunk)) > 0) {
        read += n;
        char pp[96];
        std::snprintf(pp, sizeof(pp), "PP %d %d 1 1", read, (int) ids.size());
        out_line(pp);
        if (pipelined && engine.slot_active() > 0) {
            std::vector<Engine::SlotEvent> events;
            engine.slot_step(events);
            emit_slot_events(events, stream);
        }
    }
    if (!engine.slot_admit_report(slot, ad) || !ad.ok) {
        out_line("ERR BGEN: " + ad.error);
        return;
    }
    out_line("T " + std::to_string((int) ad.first));
    char buf[512];
    // The field layout of the solo `DONE`: generated, prompt tokens, prompt ms, decode ms, finish, traffic, reused.
    std::snprintf(buf, sizeof(buf), "DONE 1 %d %.3f 0.000 %s 0 0 0 0 0 0 0 0.0 %d 0", (int) ids.size(), ad.pp_ms,
                  ad.eog ? "stop" : "length", (int) ids.size());
    out_line(buf);
    out_line("BADM " + std::to_string(slot) + " " + (ad.continues ? "1" : "0"));
    if (!ad.continues) {
        out_line("BDONE " + std::to_string(slot) + " 1 " + (ad.eog ? "stop" : "length") + " 0.0");
    }
}

// One `BT` (and, when it ends, a `BDONE`) per slot that moved.  The produced count includes the admission's token, as
// the released engine's does, so a slot with max_new 1 reports 1.
void emit_slot_events(const std::vector<Engine::SlotEvent> & events, bool stream) {
    for (const Engine::SlotEvent & e : events) {
        if (stream && e.fresh) out_line("BT " + std::to_string(e.slot) + " " + std::to_string((int) e.tok));
        if (e.done) {
            out_line("BDONE " + std::to_string(e.slot) + " " + std::to_string(e.produced) + " " +
                     (e.eog ? "stop" : "length") + " 0.0");
        }
    }
}

} // namespace

void CommandQueue::push(std::string line) {
    {
        std::lock_guard<std::mutex> lock(mu_);
        q_.push_back(std::move(line));
    }
    cv_.notify_one();
}

bool CommandQueue::pop(std::string & out, int timeout_ms) {
    std::unique_lock<std::mutex> lock(mu_);
    if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] { return !q_.empty() || closed_; })) {
        return false;
    }
    if (q_.empty()) return false;
    out = std::move(q_.front());
    q_.pop_front();
    return true;
}

void CommandQueue::close() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        closed_ = true;
    }
    cv_.notify_all();
}

bool CommandQueue::closed() const {
    std::lock_guard<std::mutex> lock(mu_);
    return closed_;
}

bool quit_requested() { return g_quit.load(); }

void start_stdin_reader(CommandQueue & q) {
    std::thread([&q] {
        // A 32K-token prompt is a few hundred KB of ids on one line, so the line is accumulated rather than read
        // into a fixed buffer (MSVC has no getline).
        std::string line;
        char buf[65536];
        while (std::fgets(buf, sizeof(buf), stdin) != nullptr) {
            line += buf;
            if (line.empty() || line.back() != '\n') continue;
            while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
            if (line == "STOP") {
                Engine::request_stop();
            } else if (line == "QUIT" || line == "EXIT") {
                g_quit.store(true);
                q.push(line);
                break;
            } else {
                q.push(line);
            }
            line.clear();
        }
        // End of input is not a request to stop: the lines already read are still commands to answer.  Closing the
        // queue lets serve() drain it and then end, which is what the app sees when it closes the engine's stdin.
        q.close();
    }).detach();
}

int serve(Engine & engine, CommandQueue & q) {
    // READY comes last: the app waits for it with a timeout, and the INFO lines fill the Monitor.
    const int slots = engine.parallel_slots();
    out_line("INFO " + engine.describe());
    // `batch_slots` is what the server reads to decide whether it may send BGEN at all: 0 (or absent) means it keeps
    // one request per engine, which is what every build before --parallel did.
    out_line("INFO batch_slots=" + std::to_string(slots >= 2 ? slots : 0));
    out_line("READY " + std::to_string(engine.n_ctx()) + " stop");

    logf("[strata-vk] serve: %d request slot%s, %d tokens of cache each\n", slots, slots == 1 ? "" : "s",
         slots >= 2 ? engine.slot_ctx() : engine.n_ctx());

    std::string line;
    std::vector<Engine::SlotEvent> events;
    bool stream_slots = true;   // a `stream=0` BGEN turns the per-token lines off for the request it admits
    while (true) {
        // While a slot is live the queue is polled, not waited on: a BGEN that arrived during the last step is admitted
        // now, and then this step carries the new slot too.  A command always wins over a step, so a STOP or a new
        // request never waits for the batch.
        const bool busy = engine.slot_active() > 0;
        if (q.pop(line, busy ? 0 : 200)) {
            const std::string cmd = first_word(line);
            if (cmd == "QUIT" || cmd == "EXIT") {
                break;
            } else if (cmd == "GEN") {
                // A solo request while slots are live: drain them first, so the answer cannot interleave with their
                // tokens on the same stdout.
                while (engine.slot_active() > 0) {
                    engine.slot_step(events);
                    emit_slot_events(events, stream_slots);
                }
                handle_gen(engine, line);
            } else if (cmd == "BGEN" && slots >= 2) {
                stream_slots = line.find("stream=0") == std::string::npos;
                handle_bgen(engine, line, stream_slots);
            } else if (cmd == "BGEN") {
                out_line("ERR this engine runs one request at a time (no batch slots)");
            } else if (cmd == "BSTOP") {
                const std::vector<std::string> f = split_words(line, 6);
                if (!f.empty()) engine.slot_stop(std::atoi(f[0].c_str()));
            } else if (cmd == "GENI" || cmd == "BGENI") {
                out_line("ERR this build reads text only: pictures need Strata's image encoder, which this engine does not run");
            } else if (cmd == "VRAM") {
                out_line("VRAM expert_slots=0 expert_cache_mib=0 vram_free_mib=0");
            } else if (cmd == "SAVE" || cmd == "RESTORE") {
                out_line("ERR disk sessions are not supported by the Vulkan engine");
            } else if (cmd == "INFO") {
                out_line("INFO " + engine.describe());
            } else if (cmd == "STOP") {
                Engine::request_stop();
            } else if (!line.empty()) {
                out_line("ERR unknown command " + line.substr(0, 32));
            }
            continue;
        }
        if (engine.slot_active() > 0) {
            engine.slot_step(events);
            emit_slot_events(events, stream_slots);
            continue;
        }
        if (q.closed()) break;     // stdin ended and every command it carried has been answered
    }
    return 0;
}

} // namespace strata

