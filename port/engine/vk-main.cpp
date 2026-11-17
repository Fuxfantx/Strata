// vk-main.cpp - the entry point: the device report, the one-shot run, the benchmarks, and the serve loop.
#include "vk-common.h"
#include "vk-devices.h"
#include "vk-engine.h"

#ifdef _WIN32
#include <windows.h>
#include <crtdbg.h>
#include <excpt.h>
#include <process.h>
#endif
#include <exception>
#include "vk-options.h"
#include "vk-serve.h"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>

using namespace strata;

namespace {

#ifdef _WIN32
// STRATA_VK_BACKTRACE=1: install the three hooks a fail-fast can go through - std::terminate, the invalid-parameter
// handler and the unhandled-exception filter - and print a module+offset backtrace before dying.  The x64 build under
// Prism dies with 0xC0000409 inside _invoke_watson without a word, and this is what says who called it.
static void bt_print() {
    void * frames[48] = { 0 };
    const USHORT n = RtlCaptureStackBackTrace(1, 48, frames, nullptr);
    for (USHORT i = 0; i < n; ++i) {
        PVOID base = nullptr;
        if (RtlPcToFileHeader(frames[i], &base) && base != nullptr) {
            char path[MAX_PATH] = { 0 };
            GetModuleFileNameA((HMODULE) base, path, MAX_PATH);
            const char * name = strrchr(path, '\\');
            std::fprintf(stderr, "[backtrace] #%02u %s + 0x%llx\n", i, name ? name + 1 : path,
                         (unsigned long long) ((uintptr_t) frames[i] - (uintptr_t) base));
        } else {
            std::fprintf(stderr, "[backtrace] #%02u 0x%llx\n", i, (unsigned long long) (uintptr_t) frames[i]);
        }
    }
    std::fflush(stderr);
}

static void bt_terminate() {
    std::fprintf(stderr, "[backtrace] std::terminate\n");
    bt_print();
    _Exit(70);
}

static void bt_invalid_parameter(const wchar_t * expr, const wchar_t * func, const wchar_t * file,
                                 unsigned int line, uintptr_t) {
    std::fprintf(stderr, "[backtrace] invalid parameter in %ls (%.80ls:%u), expr %.80ls\n",
                 func ? func : L"?", file ? file : L"?", line, expr ? expr : L"?");
    bt_print();
    _Exit(71);
}

static void install_backtrace_hooks() {
    if (std::getenv("STRATA_VK_BACKTRACE") == nullptr) {
        return;
    }
    std::set_terminate(bt_terminate);
    _set_invalid_parameter_handler(bt_invalid_parameter);
    SetUnhandledExceptionFilter([](EXCEPTION_POINTERS * ep) -> LONG {
        std::fprintf(stderr, "[backtrace] unhandled exception 0x%08lx at %p\n",
                     ep->ExceptionRecord->ExceptionCode, ep->ExceptionRecord->ExceptionAddress);
        bt_print();
        return EXCEPTION_EXECUTE_HANDLER;
    });
    std::fprintf(stderr, "[backtrace] hooks installed\n");
}
#else
static void install_backtrace_hooks() {}
#endif

int run_geometry(const Options & opt, const ModelFacts & facts) {
    llama_backend_init();
    const NpuPlan npu = plan_npu(opt.npu, facts, opt.npu_layers);
    std::printf("strata %s\n", kEngineVersion);
    print_devices(npu, true);
    llama_backend_free();
    return 0;
}

int run_one_shot(Engine & engine, const Options & opt) {
    std::vector<llama_token> ids = opt.tokens;
    if (ids.empty() && !opt.prompt.empty()) ids = engine.tokenize(opt.prompt);
    if (ids.empty()) ids.push_back(1);
    // STRATA_VK_ADMIT_DEBUG=1: the ids this prompt became, so the serve protocol can be driven with exactly what a real
    // request carries instead of a synthetic id list (which this model answers with degenerate zeros; see 20.49).
    if (std::getenv("STRATA_VK_ADMIT_DEBUG") != nullptr) {
        std::string text;
        for (llama_token t : ids) {
            text += std::to_string((int) t);
            text += ',';
        }
        logf("[strata-vk] prompt ids: %s\n", text.c_str());
    }
    const int n = opt.max_new > 0 ? opt.max_new : 32;
    const Stats st = engine.generate(ids, n, "", [](llama_token t) {
        std::printf("%d,", (int) t);
        return true;
    }, false);
    {
        // A steady rate that leaves the first step out, and the first step itself: averaging a cold first token into
        // the rate is what made this project read 6.5 tok/s where the steady state was 11.
        const double steady_ms = st.decode_ms - st.first_step_ms;
        const int    steady_n  = st.generated - (st.first_step_ms > 0 ? 1 : 0);
        std::printf("\n%.1f tok/s decode (steady %.1f over %d), first step %.0f ms, prompt %.0f ms, cpu %.0f ms of %.0f ms wall",
                    st.decode_ms > 0 ? st.generated * 1000.0 / st.decode_ms : 0.0,
                    steady_ms > 0 ? steady_n * 1000.0 / steady_ms : 0.0, steady_n,
                    st.first_step_ms, st.prompt_ms, st.decode_cpu_ms, st.decode_ms);
    }
    if (st.drafted > 0) std::printf(", %d/%d drafts kept", st.accepted, st.drafted);
    std::printf("\n");
    return 0;
}

int run_bench(Engine & engine, const Options & opt) {
    // A real prompt when one is given (its tokenisation is the app's), otherwise token 1 repeated: no tokenizer is
    // needed then and the cost is the same as any other prompt of that length.
    std::vector<llama_token> ids = opt.tokens;
    if (ids.empty() && !opt.prompt.empty()) ids = engine.tokenize(opt.prompt);
    if (ids.empty() && opt.bench_pp) ids.assign((size_t) opt.bench_n, (llama_token) 1);
    if (ids.empty()) ids.assign(64, (llama_token) 1);
    // --bench-pp N with a prompt longer than N measures the first N tokens of it.
    if (opt.bench_pp && (int) ids.size() > opt.bench_n) ids.resize((size_t) opt.bench_n);

    // The first llama_decode of a context allocates the backend's compute buffer for that graph's shape (~700 ms for a
    // 2048-token batch on this Adreno), so a warmup pass of the same shape - with a prompt that shares no prefix, or the
    // measured pass would reuse the cache - keeps that fixed cost out of the number below.
    std::vector<llama_token> warm(ids.size(), (llama_token) 2);
    engine.generate(warm, 0, "", [](llama_token) { return true; }, false);

    if (opt.bench_pp) {
        const Stats st = engine.generate(ids, 0, "", [](llama_token) { return true; }, false);
        std::printf("pp %d tokens in %.1f ms = %.2f tok/s\n", (int) ids.size(), st.prompt_ms,
                    st.prompt_ms > 0 ? ids.size() * 1000.0 / st.prompt_ms : 0.0);
        return 0;
    }
    // A dummy prompt of repeated tokens reaches an end-of-generation token almost at once, which used to end the loop
    // after two tokens and report 0.6 tok/s: the benchmark wants `--bench-tg N` decodes whatever the sampler picks.
    engine.set_ignore_eog(true);
    if (opt.slots > 0) {
        // --batch-slots N: N sequences, one token each per step.  N = 1 is the same code as N > 1, so the two numbers are
        // a paired comparison and not a comparison of two harnesses.
        const Engine::SlotStats ss = engine.bench_slots(ids, opt.bench_n, opt.slots);
        engine.set_ignore_eog(false);
        if (ss.steps <= 0 || ss.ms <= 0) {
            std::printf("slots %d: no step ran\n", opt.slots);
            return 1;
        }
        const double agg = ss.tokens * 1000.0 / ss.ms;
        std::printf("slots %d: %d tokens in %.1f ms = %.2f tok/s aggregate (%.2f per sequence, %.1f ms/step)\n", ss.slots,
                    ss.tokens, ss.ms, agg, agg / ss.slots, ss.ms / ss.steps);
        // The two epochs bracket the measured steps only, so port/tools/energy-ab.py can price a power sample window
        // against this run without the prefills of `slots` sequences landing inside it.
        logf("[strata-vk] bench window %.3f %.3f\n", ss.t0_epoch, ss.t1_epoch);
        return 0;
    }
    const Stats st = engine.generate(ids, opt.bench_n, "", [](llama_token) { return true; }, false);
    engine.set_ignore_eog(false);
    std::printf("tg %d tokens in %.1f ms = %.2f tok/s (prompt %.0f ms, cpu %.0f ms)\n", st.generated, st.decode_ms,
                st.decode_ms > 0 ? st.generated * 1000.0 / st.decode_ms : 0.0, st.prompt_ms, st.decode_cpu_ms);
    return 0;
}
// Several llama_contexts on ONE model: the weights are the model's buffers, so this is the only concurrency that does not
// duplicate a single byte of them.  Each context decodes one token per step on its own thread; the aggregate is the sum of
// the rates, the same convention as two-serve.py.  --dual-contexts 1 runs the same code with one context, so the two
// numbers come from one harness.
int run_dual_bench(Engine & engine, const Options & opt) {
    std::vector<llama_token> ids = opt.tokens;
    if (ids.empty() && !opt.prompt.empty()) ids = engine.tokenize(opt.prompt);
    if (ids.empty()) ids.assign(64, (llama_token) 1);
    const int P     = (int) ids.size();
    const int steps = opt.bench_dual;
    const int nctx  = std::max(1, opt.dual_contexts);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = (uint32_t) std::max(engine.n_ctx(), P + steps + 8);
    cp.n_batch   = 1024;
    cp.n_ubatch  = 1024;
    cp.n_seq_max = 1;
    std::vector<llama_context *> ctxs((size_t) nctx, nullptr);
    ctxs[0] = engine.context();
    for (int i = 1; i < nctx; ++i) {
        ctxs[(size_t) i] = llama_init_from_model(engine.model(), cp);
        if (ctxs[(size_t) i] == nullptr) {
            std::printf("dual: context %d could not be created\n", i);
            return 1;
        }
    }

    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(engine.model()));
    std::vector<double> ms((size_t) nctx, 0.0);
    std::vector<int>    got((size_t) nctx, 0);
    std::atomic<int>    ready{0};
    std::atomic<bool>   go{false};

    auto worker = [&](int idx) {
        llama_context * c = ctxs[(size_t) idx];
        llama_batch b = llama_batch_init(1024, 0, 1);
        auto fill = [&](int row, llama_token t, int pos, int lg) {
            b.token[row] = t; b.pos[row] = pos; b.n_seq_id[row] = 1; b.seq_id[row][0] = 0; b.logits[row] = lg;
        };
        int rows = 0;
        for (int off = 0; off < P; off += 1024) {
            const int n = std::min(1024, P - off);
            for (int i = 0; i < n; ++i) fill(i, ids[(size_t) (off + i)], off + i, (off + i == P - 1) ? 1 : 0);
            b.n_tokens = n;
            rows = n;
            if (llama_decode(c, b) != 0) { llama_batch_free(b); ready.fetch_add(1); return; }
        }
        auto argmax = [&](const float * l) {
            int best = 0; float bv = l[0];
            for (int v = 1; v < nv; ++v) if (l[v] > bv) { bv = l[v]; best = v; }
            return (llama_token) best;
        };
        const float * l0 = llama_get_logits_ith(c, rows - 1);
        if (l0 == nullptr) { llama_batch_free(b); ready.fetch_add(1); return; }
        llama_token cur = argmax(l0);
        ready.fetch_add(1);
        while (!go.load()) std::this_thread::yield();
        const auto t0 = std::chrono::steady_clock::now();
        for (int t = 0; t < steps; ++t) {
            fill(0, cur, P + t, 1);
            b.n_tokens = 1;
            if (llama_decode(c, b) != 0) break;
            const float * l = llama_get_logits_ith(c, 0);
            if (l == nullptr) break;
            cur = argmax(l);
            got[(size_t) idx] = t + 1;
        }
        ms[(size_t) idx] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        llama_batch_free(b);
    };

    std::vector<std::thread> th;
    th.reserve((size_t) nctx);
    for (int i = 0; i < nctx; ++i) th.emplace_back(worker, i);
    while (ready.load() < nctx) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    go.store(true);
    for (auto & t : th) t.join();

    double agg = 0.0;
    for (int i = 0; i < nctx; ++i) {
        const double rate = ms[(size_t) i] > 0 ? got[(size_t) i] * 1000.0 / ms[(size_t) i] : 0.0;
        agg += rate;
        std::printf("ctx %d: %d tokens in %.1f ms = %.2f tok/s\n", i, got[(size_t) i], ms[(size_t) i], rate);
    }
    std::printf("dual %d contexts: %.2f tok/s aggregate (one model, one copy of the weights)\n", nctx, agg);

    for (int i = 1; i < nctx; ++i) llama_free(ctxs[(size_t) i]);
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    install_backtrace_hooks();
    quiet_backend_logs();
    const Options opt = parse_args(argc, argv);
    // Both must be in the environment before any backend is loaded: the Adreno knobs before the first Vulkan device is
    // created, the HTP mapping budget before the Hexagon backend reads it.  An unresolved Auto still applies the Vulkan
    // knobs, because Auto resolves to Vulkan whenever that backend exists.
    const ModelFacts facts = opt.version || opt.model.empty() ? ModelFacts{} : read_model_facts(opt.model);
    apply_adreno_tuning(opt.gpu, facts.dense_matmul, facts.q20_only, facts.lowbit_pack, opt.shader_family);
    Options opt_load = opt;
    opt_load.expert_bytes = facts.expert_bytes;
    opt_load.disk_bytes   = facts.disk_bytes;
#ifdef _WIN32
    // Leave the reserved cores genuinely free for the rest of the machine: the engine's threads (and ggml's) are pinned to
    // the cores plan_affinity leaves us, which is 20% of the machine with a one-big-two-small floor.
    //
    // With --gpu cpu the CPU backend runs every node itself, so the mask is the big-core class instead: a barrier opens and closes
    // each node of each decode step, a heterogeneous pool pays its slowest core on every one of them, and the six 3072 MHz cores
    // measured more expensive at the barrier than they return in work (PERFORMANCE.md, the thread sweep).
    {
        const AffinityPlan ap = opt.gpu == GpuBackend::CPU ? plan_cpu_affinity(opt.threads)
                              : opt.perf_cap < 100        ? plan_affinity(opt.perf_cap)
                                                          : AffinityPlan{};
        const int total = (int) std::thread::hardware_concurrency();
        if (!ap.cpus.empty() && (int) ap.cpus.size() < total && total <= 64) {
            DWORD_PTR mask = 0;
            for (int c : ap.cpus) mask |= ((DWORD_PTR) 1 << c);
            if (!SetProcessAffinityMask(GetCurrentProcess(), mask)) {
                logf("[strata-vk] could not pin the engine to its %d cores; the OS decides\n", (int) ap.cpus.size());
            }
        }
        if (opt.gpu == GpuBackend::CPU && !ap.cpus.empty()) {
            logf("[strata-vk] CPU backend: %d threads on the big-core class (%d of %d cores)\n",
                 ap.n_threads, (int) ap.cpus.size(), total);
        }
    }
#endif
    configure_npu_budget(opt.npu);

    if (opt.version) {
        std::printf("strata %s\n", kEngineVersion);
        return 0;
    }
    if (opt.geometry) return run_geometry(opt, facts);

    llama_backend_init();
    if (opt.npu_probe) {
        const NpuPlan npu = plan_npu(opt.npu == NpuMode::Off ? NpuMode::Auto : opt.npu, facts, opt.npu_layers);
        print_devices(npu, true);
        llama_backend_free();
        return npu.active() ? 0 : 1;
    }

    if (opt.model.empty()) {
        std::fprintf(stderr, "strata-vk: no model given (--native MODEL.gguf)\n");
        llama_backend_free();
        return 2;
    }

    // --detok reads a token stream back as text. Nothing but the vocabulary is needed, so this skips the device plan, the
    // expert placement and the whole 40 s load: it is the other half of `llama-tokenize --ids`, for checking a generated
    // token stream against a reference implementation's text.
    if (opt.detok) {
        llama_model_params mp = llama_model_default_params();
        mp.vocab_only = true;
        llama_model * model = llama_model_load_from_file(opt.model.c_str(), mp);
        if (model == nullptr) {
            std::fprintf(stderr, "strata-vk: cannot read the tokenizer of %s\n", opt.model.c_str());
            llama_backend_free();
            return 1;
        }
        const std::string text = detokenize_ids(llama_model_get_vocab(model), opt.tokens);
        std::printf("%s\n", text.c_str());
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }

    const NpuPlan npu = plan_npu(opt.npu, facts, opt.npu_layers);
    Engine engine;
    if (!engine.load(opt_load, npu)) {
        return strata::dry_run_finished ? 0 : 3;
    }

    int rc = 0;
    if (opt.serve) {
        engine.set_wire(true);
        CommandQueue q;
        start_stdin_reader(q);
        rc = serve(engine, q);
    } else if (opt.bench_dual > 0) {
        rc = run_dual_bench(engine, opt);
    } else if (opt.bench_pp || opt.bench_tg) {
        rc = run_bench(engine, opt);
    } else {
        rc = run_one_shot(engine, opt);
    }

    engine.unload();
    llama_backend_free();
    return rc;
}


