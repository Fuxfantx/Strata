// vk-common.h - what every part of the Vulkan engine shares.
//
// The engine is the Strata `--serve` component contract on top of ggml's Vulkan backend.  stdout carries the protocol
// and nothing else; stderr carries the log that serve/server.py narrates.  The Adreno decisions (which backend
// knobs, whether the NPU takes the dense tensors) live in vk-devices.cpp, not spread through the engine.
#pragma once

#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace strata {

inline constexpr const char * kEngineVersion = "0.2.13-arm64-vulkan";

// The context the app asks for by default, and the prompt chunk.  A 2048-token batch makes the Vulkan backend
// allocate a 3.2 GiB compute buffer on Adreno, so the chunk stays at half of that: 1024 measured 25% faster prefill than
// 512 (pp512 96.6 against 77.4 tok/s, pp256 83.8 against 66.5, pp64 45.8 against 41.7, tokens identical), and 2048 would
// quadruple the buffer for a chunk no prompt on this machine reaches.
inline constexpr int kDefaultCtx   = 32768;
inline constexpr int kDefaultBatch = 1024;

using Clock = std::chrono::steady_clock;

inline double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// Wall-clock seconds since the Unix epoch: the units an outside sampler (port/tools/energy-sample.ps1 writes the same)
// timestamps with, so a power window can be cut to the engine's own measurement.
inline double epoch_seconds() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

// CPU time this process has burned so far, in ms. Comparing it with the wall time of a decode step says whether the
// step is bound by the host (graph build, recording, sampling) or by the GPU, which the Vulkan perf logger cannot tell
// apart: its timestamps are written with eAllCommands, so a host stall lands inside the next op's window.
inline double process_cpu_ms() {
#ifdef _WIN32
    FILETIME create{}, exit{}, kernel{}, user{};
    if (!GetProcessTimes(GetCurrentProcess(), &create, &exit, &kernel, &user)) {
        return 0.0;
    }
    const auto to_ms = [](const FILETIME & t) {
        return (double) (((uint64_t) t.dwHighDateTime << 32) | t.dwLowDateTime) / 10000.0;   // 100 ns units
    };
    return to_ms(kernel) + to_ms(user);
#else
    return 0.0;
#endif
}

// Windows' default timer granularity is ~15.6 ms, so a plain sleep_for overshoots short waits by an order of magnitude; the
// last couple of milliseconds are spun instead, which costs far less than the overshoot would.
inline void sleep_ms(double ms) {
    if (ms <= 0.0) return;
    const Clock::time_point t0 = Clock::now();
    for (;;) {
        const double left = ms - ms_since(t0);
        if (left <= 0.0) return;
        if (left > 2.0) {
            // Sleep in short steps and keep only the last millisecond spinning: a yield loop over twenty milliseconds burned a
            // core for the whole idle share of a paced run, which is the opposite of what pacing is for.
            std::this_thread::sleep_for(std::chrono::milliseconds((long long) std::min(left - 1.0, 4.0)));
        } else if (left > 1.0) {
            std::this_thread::yield();
        }
    }
}

// Sleeps the idle share owed for the step that just ran, keeping (100 - perf_cap)% of the wall time free for the rest of the
// machine (browsers, Office). A no-op at perf_cap 100.
inline void pace_step(int perf_cap) {
    static const bool pace_dbg = std::getenv("STRATA_VK_PACEDBG") != nullptr;
    if (perf_cap >= 100) return;
    static thread_local Clock::time_point last = Clock::now();
    const double busy = ms_since(last);
    const double idle = busy * (100.0 - perf_cap) / perf_cap;
    if (pace_dbg) {
        std::fprintf(stderr, "[pace] cap=%d busy=%.2f sleep=%.2f\n", perf_cap, busy, idle);
    }
    sleep_ms(idle);
    last = Clock::now();   // after the sleep, so the next step measures work only
}



void logf(const char * fmt, ...);
void out_line(const std::string & line);

/** Keep warnings and errors from llama.cpp/ggml, drop the per-graph debug chatter unless STRATA_VK_VERBOSE is set. */
void quiet_backend_logs();

} // namespace strata
