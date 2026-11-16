// membw - practical DRAM bandwidth: stream a large buffer through the caches and reduce it with integer adds, which is
// cheap enough that the loop is bound by memory rather than by the reduce.  The 27B decode already moves ~116 GB/s of
// weights, so this is the number that 116 has to be judged against (the part is specified at 228 GB/s over three
// channels).
//
//   clang-cl /O2 /std:c++17 /EHsc membw.cpp /Fe:membw.exe
//   membw.exe 8 18
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

int main(int argc, char ** argv) {
    const size_t gb = argc > 1 ? (size_t) atoi(argv[1]) : 8;
    const unsigned threads = argc > 2 ? (unsigned) atoi(argv[2]) : 18;
    const size_t bytes = gb * 1024ull * 1024ull * 1024ull;
    const size_t n64 = bytes / sizeof(uint64_t);

    std::vector<uint64_t> buf(n64);
    for (size_t i = 0; i < n64; i += 512) {
        buf[i] = i;   // one write per page
    }

    std::atomic<unsigned> ready{0};
    std::atomic<uint64_t> sink{0};
    auto body = [&](unsigned t) {
        const size_t lo = n64 * t / threads;
        const size_t hi = n64 * (t + 1) / threads;
        // four independent accumulators so the loop keeps several loads in flight
        uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        ready.fetch_add(1);
        while (ready.load() < threads) {
        }
        const uint64_t * p = buf.data() + lo;
        size_t i = 0;
        for (; i + 4 <= hi - lo; i += 4) {
            a0 += p[i + 0];
            a1 += p[i + 1];
            a2 += p[i + 2];
            a3 += p[i + 3];
        }
        for (; i < hi - lo; ++i) {
            a0 += p[i];
        }
        sink.fetch_add(a0 + a1 + a2 + a3);
    };

    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < threads; ++t) {
        pool.emplace_back(body, t);
    }
    for (auto & th : pool) {
        th.join();
    }
    auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    std::printf("%zu GiB read with %u threads in %.3f s = %.1f GB/s (sink %llu)\n", gb, threads, secs,
                (double) bytes / secs / 1e9, (unsigned long long) sink.load());
    return 0;
}
