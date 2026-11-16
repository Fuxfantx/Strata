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
    out_line("INFO " + engine.describe());
    out_line("INFO batch_slots=0");
    out_line("READY " + std::to_string(engine.n_ctx()) + " stop");

    std::string line;
    while (true) {
        if (!q.pop(line, 200)) {
            if (q.closed()) break;     // stdin ended and every command it carried has been answered
            continue;
        }
        const std::string cmd = first_word(line);
        if (cmd == "QUIT" || cmd == "EXIT") {
            break;
        } else if (cmd == "GEN") {
            handle_gen(engine, line);
        } else if (cmd == "GENI" || cmd == "BGENI") {
            out_line("ERR this build reads text only: pictures need Strata's image encoder, which this engine does not run");
        } else if (cmd == "BGEN") {
            out_line("ERR this engine runs one request at a time (no batch slots)");
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
    }
    return 0;
}

} // namespace strata

