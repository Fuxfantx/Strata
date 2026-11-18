// strata-vision.cpp - the picture reader the front end spawns, on llama.cpp's mtmd.
//
// serve/server.py (class Vision) starts this program with "--mmproj <file> --model <file> [--gpu] [--threads N]
// [--max-tokens N] [--min-tokens N]", in a working directory of its own, and then speaks two lines with it:
//
//     stdout:  READY <n_embd> mtmd          once, when the encoder is up
//     stdin:   ENC <image> <output>         once per picture, both names relative to that working directory
//     stdout:  OK <n_tokens>                after <output> is written (or ERR <why>)
//
// It is a drop-in for Strata's own reader: the front end caches the result by the picture's hash and never parses
// the file, it only concatenates one request's pictures into one file and hands that path to the engine on its
// GENI/BGENI line.  So <output> holds one self-describing record per picture, which is what makes concatenation
// work (port/docs/features/VISION.md):
//
//     u32 magic 'SVM1' (0x314D5653), u32 n_tokens, u32 n_embd, u32 flags, f32 embd[n_tokens * n_embd]
//
// flags bit 0 is set: the embeddings are row-major [n_tokens][n_embd].  The engine supplies the placeholder token
// itself, because that token belongs to the text model's vocabulary, which this program does not have to ship.
//
// The encoder is opt-in: the engine only reads pictures when it was given --mmproj, so an engine run without one
// keeps exactly the bytes and timing it had.

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include "ggml-backend.h"
#include "gguf.h"

#include <cerrno>
#include <cmath>
#include <cstring>
#include <direct.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kSveMagic = 0x314D5653u;   // 'S','V','M','1' little endian

struct Options {
    std::string mmproj;
    std::string model;
    bool gpu = false;
    bool help = false;
    int  threads = 0;
    int  max_tokens = 0;
    int  min_tokens = 0;
};

void usage(FILE * f) {
    std::fprintf(f,
                 "strata-vision %s - pictures for the Strata engine, on llama.cpp's mtmd\n"
                 "usage: strata-vision --mmproj FILE --model FILE [--gpu] [--threads N]\n"
                 "                     [--max-tokens N] [--min-tokens N]\n"
                 "speaks: ENC <image> <output> on stdin -> OK <n_tokens> on stdout\n",
                 "0.1");
}

bool parse_int(const char * s, int & out) {
    if (s == nullptr || *s == '\0') return false;
    char * end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (end == s || (end != nullptr && *end != '\0')) return false;
    out = (int) v;
    return true;
}

// Returns 0 when the arguments are usable, 1 for "--help", 2 for a usage error (printed here).
int parse_args(int argc, char ** argv, Options & o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto take = [&](std::string & dst) {
            if (i + 1 >= argc) return false;
            dst = argv[++i];
            return true;
        };
        if (a == "-h" || a == "--help") {
            o.help = true;
            return 1;
        } else if (a == "--mmproj") {
            if (!take(o.mmproj)) return 2;
        } else if (a == "--model") {
            if (!take(o.model)) return 2;
        } else if (a == "--gpu") {
            o.gpu = true;
        } else if (a == "--threads") {
            std::string v;
            if (!take(v) || !parse_int(v.c_str(), o.threads)) return 2;
        } else if (a == "--max-tokens") {
            std::string v;
            if (!take(v) || !parse_int(v.c_str(), o.max_tokens)) return 2;
        } else if (a == "--min-tokens") {
            std::string v;
            if (!take(v) || !parse_int(v.c_str(), o.min_tokens)) return 2;
        } else {
            // The front end passes the released reader's whole option set; a flag this build does not need is
            // accepted so a newer server never fails to start an older encoder (the engine's own drop-in rule).
            std::fprintf(stderr, "[strata-vision] ignoring '%s'\n", a.c_str());
        }
    }
    if (o.mmproj.empty() || o.model.empty()) return 2;
    return 0;
}

bool write_sve(const std::string & path, const float * embd, int n_tokens, int n_embd, std::string & err) {
    if (embd == nullptr) {
        err = "the encoder produced no embeddings";
        return false;
    }
    std::FILE * f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        // The reader runs in a directory the server made (serve/server.py, Vision.work_dir): say where it is and what
        // the OS said, because "cannot write x.sve" alone cannot tell a missing directory from a permission.
        char cwd[1024] = {};
        const char * here = _getcwd(cwd, (int) sizeof(cwd)) != nullptr ? cwd : "?";
        std::fprintf(stderr, "[strata-vision] cannot open '%s' for writing in '%s': %s\n", path.c_str(), here,
                     std::strerror(errno));
        err = "cannot write " + path;
        return false;
    }
    const uint32_t header[4] = {kSveMagic, (uint32_t) n_tokens, (uint32_t) n_embd, 1u};
    bool ok = std::fwrite(header, sizeof(header), 1, f) == 1;
    const size_t n_floats = (size_t) n_tokens * (size_t) n_embd;
    ok = ok && std::fwrite(embd, sizeof(float), n_floats, f) == n_floats;
    ok = (std::fclose(f) == 0) && ok;
    if (!ok) err = "cannot write " + path;
    return ok;
}

// One u32 out of a GGUF header, without loading any tensor: the key is tried with the file's own architecture as the
// prefix, then bare.  Returns 0 when neither key is there.
uint32_t gguf_u32(const char * path, const char * suffix, bool with_arch = true) {
    struct gguf_init_params gp = {true /*no_alloc*/, nullptr};
    ggml_context * meta = nullptr;
    gp.ctx = &meta;
    gguf_context * g = gguf_init_from_file(path, gp);
    if (g == nullptr) return 0;
    char key[192];
    uint32_t out = 0;
    std::string arch;
    if (with_arch) {
        const int64_t a = gguf_find_key(g, "general.architecture");
        if (a >= 0 && gguf_get_val_str(g, a) != nullptr) arch = gguf_get_val_str(g, a);
    }
    const std::string keys[2] = {arch.empty() ? std::string() : arch + "." + suffix, std::string(suffix)};
    for (const std::string & k : keys) {
        if (k.empty()) continue;
        std::snprintf(key, sizeof(key), "%s", k.c_str());
        const int64_t id = gguf_find_key(g, key);
        if (id < 0) continue;
        const enum gguf_type t = gguf_get_kv_type(g, id);
        if (t == GGUF_TYPE_UINT32) out = gguf_get_val_u32(g, id);
        else if (t == GGUF_TYPE_INT32 && gguf_get_val_i32(g, id) > 0) out = (uint32_t) gguf_get_val_i32(g, id);
        else if (t == GGUF_TYPE_UINT64) out = (uint32_t) gguf_get_val_u64(g, id);
        if (out > 0) break;
    }
    gguf_free(g);
    if (meta != nullptr) ggml_free(meta);
    return out;
}

// The width of the rows mtmd hands back: its own contract says llama_model_n_embd_inp (mtmd.h, above
// mtmd_get_output_embd).  A vocab-only load reports every accessor as 0 in this tree - the vocabulary is loaded but
// the hyperparameters are not - so the model's own metadata is read out of the GGUF header instead.  Both must agree
// with the projector's own output width, which is printed at start so a mismatch is visible in the log.
int embd_width(const llama_model * model, const std::string & model_path) {
    for (int n : {llama_model_n_embd_inp(model), llama_model_n_embd(model), llama_model_n_embd_out(model)}) {
        if (n > 0) return n;
    }
    return (int) gguf_u32(model_path.c_str(), "embedding_length");
}

// One picture: load, tokenize at the media marker, run the projector, write the record.  Returns the number of
// image tokens, or -1 with `err` set.
int encode_one(mtmd_context * mctx, const llama_model * model, int n_embd, const char * image, const char * output,
               std::string & err) {
    struct mtmd_helper_init_opt opt = mtmd_helper_init_opt_default();
    struct mtmd_helper_bitmap_wrapper bmp = mtmd_helper_bitmap_init_from_file(mctx, image, false, opt);
    if (bmp.bitmap == nullptr) {
        err = std::string("cannot read the picture ") + image;
        return -1;
    }

    int n_tokens = -1;
    struct mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const struct mtmd_bitmap * bitmaps[1] = {bmp.bitmap};
    const char * marker = mtmd_get_marker(mctx);
    if (marker == nullptr) marker = mtmd_default_marker();
    struct mtmd_input_text text = {marker, std::strlen(marker), false, true};
    const int32_t rc = mtmd_tokenize(mctx, chunks, &text, bitmaps, 1);
    if (rc != 0) {
        err = "the picture does not fit this model (mtmd_tokenize " + std::to_string(rc) + ")";
    } else {
        const size_t n_chunks = mtmd_input_chunks_size(chunks);
        for (size_t i = 0; i < n_chunks; ++i) {
            const struct mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
            if (chunk == nullptr || mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_IMAGE) continue;
            const struct mtmd_image_tokens * toks = mtmd_input_chunk_get_tokens_image(chunk);
            if (toks == nullptr) continue;
            if (mtmd_encode_chunk(mctx, chunk) != 0) {
                err = "the encoder failed on this picture";
                break;
            }
            float * embd = mtmd_get_output_embd(mctx);
            const int n = (int) mtmd_image_tokens_get_n_tokens(toks);
            if (n <= 0 || n_embd <= 0) {
                err = "the encoder reported no tokens";
                break;
            }
            if (!write_sve(output, embd, n, n_embd, err)) break;
            n_tokens = n;
            break;
        }
        if (n_tokens < 0 && err.empty()) err = "the picture produced no image chunk";
    }

    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bmp.bitmap);
    if (bmp.video_ctx != nullptr) mtmd_helper_video_free(bmp.video_ctx);
    return n_tokens;
}

}  // namespace

int main(int argc, char ** argv) {
    Options opt;
    const int pr = parse_args(argc, argv, opt);
    if (pr == 1) {
        usage(stdout);
        return 0;
    }
    if (pr != 0) {
        usage(stderr);
        return 2;
    }

    llama_backend_init();

    // Only the metadata and the vocabulary are wanted: the picture's embeddings are the projector's output, and the
    // text model's tensors are not read.  A vocab-only load keeps this program's footprint at the mmproj file.
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(opt.model.c_str(), mp);
    if (model == nullptr) {
        mp.vocab_only = false;                       // a tree that needs the real weights still works
        mp.n_gpu_layers = opt.gpu ? 999 : 0;
        model = llama_model_load_from_file(opt.model.c_str(), mp);
    }
    if (model == nullptr) {
        std::fprintf(stderr, "[strata-vision] cannot load the model %s\n", opt.model.c_str());
        std::printf("ERR cannot load the model\n");
        std::fflush(stdout);
        return 2;
    }

    mtmd_context_params cp = mtmd_context_params_default();
    // --gpu names the device rather than letting mtmd pick "the GPU": on this machine the Hexagon NPU is registered as
    // an accelerator as well, and the default pick landed on it, where the session cannot be opened from this process
    // (0x200, the same unsigned-skel path the engine hits).  Only a Vulkan device is used, and no Vulkan device means
    // the processor, said out loud.
    if (opt.gpu) {
        // By name, not by type: in this fork the Vulkan device is registered as an accelerator (type 2) and the
        // Hexagon NPU as a GPU (type 1), so a type filter picks the NPU - which is exactly what mtmd's own default
        // did before this, and the session cannot be opened from this process.
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            const char * name = d != nullptr ? ggml_backend_dev_name(d) : nullptr;
            if (name != nullptr && std::strncmp(name, "Vulkan", 6) == 0) {
                cp.device = d;
                break;
            }
        }
        if (cp.device == nullptr) {
            std::fprintf(stderr, "[strata-vision] --gpu was asked for but this process sees no Vulkan device; "
                                 "reading on the processor\n");
        }
    }
    cp.use_gpu        = cp.device != nullptr;
    cp.print_timings  = false;
    cp.n_threads      = opt.threads > 0 ? opt.threads : 4;
    cp.media_marker   = mtmd_default_marker();
    cp.warmup         = false;
    cp.image_min_tokens = opt.min_tokens > 0 ? opt.min_tokens : 0;
    cp.image_max_tokens = opt.max_tokens > 0 ? opt.max_tokens : 0;

    mtmd_context * mctx = mtmd_init_from_file(opt.mmproj.c_str(), model, cp);
    if (mctx == nullptr) {
        std::fprintf(stderr, "[strata-vision] cannot load the projector %s\n", opt.mmproj.c_str());
        std::printf("ERR cannot load the projector\n");
        std::fflush(stdout);
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }

    const int n_embd = embd_width(model, opt.model);
    if (n_embd <= 0) {
        std::fprintf(stderr, "[strata-vision] the model reports no embedding width (n_embd %d, n_embd_inp %d, n_embd_out %d)\n",
                     llama_model_n_embd(model), llama_model_n_embd_inp(model), llama_model_n_embd_out(model));
        std::printf("ERR the model reports no embedding width\n");
        std::fflush(stdout);
        mtmd_free(mctx);
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }
    std::printf("READY %d mtmd\n", n_embd);
    std::fflush(stdout);
    const uint32_t proj = gguf_u32(opt.mmproj.c_str(), "vision.projection_dim");
    char where[96];
    if (cp.device != nullptr) {
        std::snprintf(where, sizeof(where), "%s", ggml_backend_dev_name(cp.device));
    } else {
        std::snprintf(where, sizeof(where), "the processor");
    }
    std::fprintf(stderr, "[strata-vision] ready on %s, n_embd %d, projector %u, %d threads\n",
                 where, n_embd, proj, cp.n_threads);

    std::string line;
    char buf[4096];
    while (std::fgets(buf, sizeof(buf), stdin) != nullptr) {
        line = buf;
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        if (line.empty()) continue;
        if (line == "QUIT" || line == "EXIT") break;

        if (line.rfind("ENC ", 0) != 0) {
            std::printf("ERR unknown command\n");
            std::fflush(stdout);
            continue;
        }
        // "ENC <image> <output>", both without spaces (the front end guarantees that; #480)
        const size_t sp1 = line.find(' ', 4);
        if (sp1 == std::string::npos) {
            std::printf("ERR malformed ENC\n");
            std::fflush(stdout);
            continue;
        }
        const std::string image  = line.substr(4, sp1 - 4);
        const std::string output = line.substr(sp1 + 1);
        if (image.empty() || output.empty()) {
            std::printf("ERR malformed ENC\n");
            std::fflush(stdout);
            continue;
        }

        std::string err;
        const int n = encode_one(mctx, model, n_embd, image.c_str(), output.c_str(), err);
        if (n > 0) {
            std::printf("OK %d\n", n);
        } else {
            std::printf("ERR %s\n", err.empty() ? "the picture could not be read" : err.c_str());
            std::fprintf(stderr, "[strata-vision] %s: %s\n", image.c_str(), err.c_str());
        }
        std::fflush(stdout);
    }

    mtmd_free(mctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
