// vk-common.cpp - the two output channels: the protocol on stdout, the log on stderr.
#include "vk-common.h"

#include "ggml.h"
#include "llama.h"

#include <cstdlib>
#include <mutex>

namespace strata {
namespace {
std::mutex g_log_mu;

// The backend's log is emitted for every graph the context builds, and ggml's default callback prints every level: a
// chat turn used to write a few hundred lines per token into the app's engine log.  Warnings and errors are what the
// log is for; everything else comes back with STRATA_VK_VERBOSE=1.
enum ggml_log_level g_last_kept = GGML_LOG_LEVEL_NONE;
bool g_verbose = false;

void backend_log(enum ggml_log_level level, const char * text, void * user_data) {
    GGML_UNUSED(user_data);
    if (level == GGML_LOG_LEVEL_CONT) {
        if (g_last_kept < GGML_LOG_LEVEL_WARN) return;
    } else if (!g_verbose && level < GGML_LOG_LEVEL_WARN) {
        g_last_kept = level;
        return;
    }
    g_last_kept = level;
    std::lock_guard<std::mutex> lock(g_log_mu);
    std::fputs(text, stderr);
    std::fflush(stderr);
}
} // namespace

void quiet_backend_logs() {
    g_verbose = std::getenv("STRATA_VK_VERBOSE") != nullptr;
    llama_log_set(backend_log, nullptr);
}

void logf(const char * fmt, ...) {
    std::lock_guard<std::mutex> lock(g_log_mu);
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fflush(stderr);
}

void out_line(const std::string & line) {
    std::fwrite(line.data(), 1, line.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

} // namespace strata
