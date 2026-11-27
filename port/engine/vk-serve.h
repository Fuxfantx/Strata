// vk-serve.h - the `strata --serve` line protocol on stdin/stdout.
#pragma once

#include "vk-engine.h"

#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>

namespace strata {

// stdin is read on its own thread so STOP and QUIT are seen while a long prompt is being decoded.
class CommandQueue {
public:
    void push(std::string line);
    bool pop(std::string & out, int timeout_ms);
    void close();
    bool closed() const;

private:
    mutable std::mutex      mu_;
    std::condition_variable cv_;
    std::deque<std::string> q_;
    bool                    closed_ = false;
};

void start_stdin_reader(CommandQueue & q);
bool quit_requested();

int serve(Engine & engine, CommandQueue & q);

} // namespace strata
