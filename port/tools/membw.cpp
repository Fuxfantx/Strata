// membw - practical DRAM bandwidth, and the reason it is not the pin rate.
//
// The pin rate of this part is 228 GB/s (three LPDDR channels).  What a stream can actually draw is bounded by how
// many cache lines are in flight:  BW = in-flight bytes / latency, and a core can only track as many outstanding
// misses as it has L1 fill buffers.  So the interesting question is not "what is the peak" but "at what unroll depth
// and thread count does the curve flatten, and where is the plateau".  This tool sweeps both.
//
//   clang-cl /O2 /std:c++17 /EHsc membw.cpp /Fe:membw.exe
//   membw.exe <GiB> <threads> <depth>
//
// <depth> is the number of independent accumulators per thread, i.e. how many loads the loop can keep in flight.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

template <int DEPTH>
static void stream(const uint64_t * p, size_t n, uint64_t * out) {
    uint64_t acc[DEPTH] = {};
    size_t i = 0;
    for (; i + DEPTH <= n; i += DEPTH) {
        for (int d = 0; d < DEPTH; ++d) acc[d] += p[i + d];
    }
    for (; i < n; ++i) acc[0] += p[i];
    uint64_t s = 0;
    for (int d = 0; d < DEPTH; ++d) s += acc[d];
    *out = s;
}

int main(int argc, char ** argv) {
    const size_t  gb      = argc > 1 ? (size_t) atoi(argv[1]) : 8;
    const unsigned threads = argc > 2 ? (unsigned) atoi(argv[2]) : 18;
    const int      depth   = argc > 3 ? atoi(argv[3]) : 4;

    const size_t bytes = gb * 1024ull * 1024ull * 1024ull;
    const size_t n64   = bytes / sizeof(uint64_t);

    std::vector<uint64_t> buf(n64);
    for (size_t i = 0; i < n64; i += 512) {
        buf[i] = i;   // one write per page, so the pages are resident before timing
    }

    std::atomic<unsigned> ready{0};
    std::atomic<uint64_t> sink{0};
    auto body = [&](unsigned t, uint64_t * out) {
        const size_t lo = n64 * t / threads;
        const size_t hi = n64 * (t + 1) / threads;
        const uint64_t * p = buf.data() + lo;
        const size_t n = hi - lo;
        ready.fetch_add(1);
        while (ready.load() < threads) {
        }
        switch (depth) {
            case  1: stream< 1>(p, n, out); break;
            case  2: stream< 2>(p, n, out); break;
            case  4: stream< 4>(p, n, out); break;
            case  8: stream< 8>(p, n, out); break;
            case 16: stream<16>(p, n, out); break;
            default: std::fprintf(stderr, "depth must be 1,2,4,8,16\n"); std::exit(2);
        }
    };

    std::vector<uint64_t> outs(threads, 0);
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < threads; ++t) {
        pool.emplace_back(body, t, &outs[t]);
    }
    for (auto & th : pool) {
        th.join();
    }
    auto t1 = std::chrono::steady_clock::now();
    for (uint64_t v : outs) sink.fetch_add(v);

    const double secs = std::chrono::duration<double>(t1 - t0).count();
    const double gbs  = (double) bytes / secs / 1e9;
    std::printf("%2zu GiB  threads %2u  depth %2d  %6.3f s  %6.1f GB/s  %5.2f GB/s/thread  (sink %llu)\n",
                gb, threads, depth, secs, gbs, gbs / threads, (unsigned long long) sink.load());
    return 0;
}
