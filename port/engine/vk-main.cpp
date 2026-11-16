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
    const Stats st = engine.generate(ids, opt.bench_n, "", [](llama_token) { return true; }, false);
    engine.set_ignore_eog(false);
    std::printf("tg %d tokens in %.1f ms = %.2f tok/s (prompt %.0f ms, cpu %.0f ms)\n", st.generated, st.decode_ms,
                st.decode_ms > 0 ? st.generated * 1000.0 / st.decode_ms : 0.0, st.prompt_ms, st.decode_cpu_ms);
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
    apply_adreno_tuning(opt.gpu, facts.dense_matmul, facts.q20_only, facts.lowbit_pack);
    Options opt_load = opt;
    opt_load.expert_bytes = facts.expert_bytes;
    opt_load.disk_bytes   = facts.disk_bytes;
#ifdef _WIN32
    // Leave the reserved cores genuinely free for the rest of the machine: the engine's threads (and ggml's) are pinned to
    // the cores plan_affinity leaves us, which is 20% of the machine with a one-big-two-small floor.
    {
        const AffinityPlan ap = opt.perf_cap < 100 ? plan_affinity(opt.perf_cap) : AffinityPlan{};
        const int total = (int) std::thread::hardware_concurrency();
        if (!ap.cpus.empty() && (int) ap.cpus.size() < total && total <= 64) {
            DWORD_PTR mask = 0;
            for (int c : ap.cpus) mask |= ((DWORD_PTR) 1 << c);
            if (!SetProcessAffinityMask(GetCurrentProcess(), mask)) {
                logf("[strata-vk] could not pin the engine to its %d cores; the OS decides\n", (int) ap.cpus.size());
            }
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
    } else if (opt.bench_pp || opt.bench_tg) {
        rc = run_bench(engine, opt);
    } else {
        rc = run_one_shot(engine, opt);
    }

    engine.unload();
    llama_backend_free();
    return rc;
}


