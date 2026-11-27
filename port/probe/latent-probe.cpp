// latent-probe.cpp - run one GGML_OP_LATENT_MOE on a repacked companion and print what it produced.
//
// The latent block's arithmetic is checked twice: `port/tools/latent-ref.py --forward` computes it in numpy from the
// pack's own bytes, and this tool runs the same block through ggml's op on a chosen backend.  Both read their input
// vectors from the same directory, so the comparison is value for value and not just shape for shape.
//
//   python port/tools/latent-ref.py --pack DIR --companion C.gguf --dump-inputs IN
//   latent-probe --companion C.gguf --inputs IN --backend CPU
//
// The companion is the repacked GGUF (`port/tools/edge0-repack.py`), which holds latent_d, latent_u, latent_labels,
// ffn_gate_exps, ffn_up_exps and ffn_down_exps with the shapes the op expects.
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct gguf_tensor {
    std::string name;
    std::vector<int64_t> dims;
    uint32_t type = 0;
    uint64_t offset = 0;
};

// minimal GGUF reader: the metadata block is skipped by value size, then the tensor table is returned
bool read_gguf(const std::string & path, std::vector<gguf_tensor> & out, uint64_t & data_start, size_t & file_size) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) { return false; }
    char magic[4];
    if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "GGUF", 4) != 0) { std::fclose(f); return false; }
    uint32_t ver = 0; uint64_t n_tensors = 0, n_kv = 0;
    if (std::fread(&ver, 4, 1, f) != 1 || std::fread(&n_tensors, 8, 1, f) != 1 || std::fread(&n_kv, 8, 1, f) != 1) {
        std::fclose(f); return false;
    }
    auto rstr = [&](std::string & s) {
        uint64_t n = 0;
        if (std::fread(&n, 8, 1, f) != 1) { return false; }
        s.resize(n);
        return n == 0 || std::fread(&s[0], 1, n, f) == n;
    };
    auto skip = [&](uint32_t t) -> bool {
        size_t n = 0;
        switch (t) {
            case 0: case 1: case 7: n = 1; break;
            case 2: case 3: n = 2; break;
            case 4: case 5: case 6: n = 4; break;
            case 10: case 11: case 12: n = 8; break;
            case 8: { uint64_t len = 0; if (std::fread(&len, 8, 1, f) != 1) { return false; } n = (size_t) len; break; }
            default: return false;
        }
        return n == 0 || std::fseek(f, (long) n, SEEK_CUR) == 0;
    };
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key;
        if (!rstr(key)) { std::fclose(f); return false; }
        uint32_t t = 0;
        if (std::fread(&t, 4, 1, f) != 1) { std::fclose(f); return false; }
        if (t == 9) {
            uint32_t at = 0; uint64_t n = 0;
            if (std::fread(&at, 4, 1, f) != 1 || std::fread(&n, 8, 1, f) != 1) { std::fclose(f); return false; }
            for (uint64_t k = 0; k < n; ++k) {
                if (!skip(at)) { std::fclose(f); return false; }
            }
        } else if (!skip(t)) { std::fclose(f); return false; }
    }
    for (uint64_t i = 0; i < n_tensors; ++i) {
        gguf_tensor t;
        if (!rstr(t.name)) { std::fclose(f); return false; }
        uint32_t nd = 0;
        if (std::fread(&nd, 4, 1, f) != 1) { std::fclose(f); return false; }
        t.dims.resize(nd);
        for (uint32_t k = 0; k < nd; ++k) {
            uint64_t d = 0;
            if (std::fread(&d, 8, 1, f) != 1) { std::fclose(f); return false; }
            t.dims[k] = (int64_t) d;
        }
        if (std::fread(&t.type, 4, 1, f) != 1 || std::fread(&t.offset, 8, 1, f) != 1) { std::fclose(f); return false; }
        out.push_back(t);
    }
    data_start = ((uint64_t) std::ftell(f) + 31) / 32 * 32;
    std::fseek(f, 0, SEEK_END);
    file_size = (size_t) std::ftell(f);
    std::fclose(f);
    return true;
}

bool read_file(const std::string & path, void * dst, size_t n) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) { return false; }
    const bool ok = std::fread(dst, 1, n, f) == n;
    std::fclose(f);
    return ok;
}

double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char ** argv) {
    std::string companion, inputs, backend_name = "CPU";
    bool verify_f32 = false;
    std::string blob_file, blob_offset_text = "0";
    bool vk_copy = false;
    std::string dump_out;
    std::string dump_row_file;
    long long dump_row_off = 0;
    int dump_row_type = 11;
    int dump_row_n = 1792;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](std::string & out) { if (i + 1 < argc) { out = argv[++i]; } };
        if (a == "--companion") { next(companion); }
        else if (a == "--inputs") { next(inputs); }
        else if (a == "--backend") { next(backend_name); }
        else if (a == "--verify-f32") { verify_f32 = true; }
        else if (a == "--blob-file") { next(blob_file); }
        else if (a == "--blob-offset") { next(blob_offset_text); }
        else if (a == "--vk-copy") { vk_copy = true; }
        else if (a == "--dump-out") { next(dump_out); }
        else if (a == "--dump-row") {
            dump_row_file = argv[++i];
            if (i + 1 < argc) { dump_row_off = std::strtoll(argv[++i], nullptr, 10); }
            if (i + 1 < argc) { dump_row_type = std::atoi(argv[++i]); }
            if (i + 1 < argc) { dump_row_n = std::atoi(argv[++i]); }
        }
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (companion.empty() || inputs.empty()) {
        std::fprintf(stderr, "usage: latent-probe --companion C.gguf --inputs DIR [--backend CPU|Vulkan0]\n");
        return 2;
    }

    if (!dump_row_file.empty()) {
        const ggml_type type = (ggml_type) dump_row_type;
        const size_t row_bytes = ggml_row_size(type, dump_row_n);
        std::vector<char> raw(row_bytes);
        FILE * f = std::fopen(dump_row_file.c_str(), "rb");
        if (!f) { std::fprintf(stderr, "cannot read %s\n", dump_row_file.c_str()); return 2; }
        std::fseek(f, (long) dump_row_off, SEEK_SET);
        if (std::fread(raw.data(), 1, row_bytes, f) != row_bytes) {
            std::fprintf(stderr, "short read at %lld\n", dump_row_off); return 2;
        }
        std::fclose(f);
        std::vector<float> out((size_t) dump_row_n);
        const ggml_to_float_t to_f = ggml_get_type_traits(type)->to_float;
        if (to_f == nullptr) { std::fprintf(stderr, "no converter for type %d\n", dump_row_type); return 2; }
        to_f(raw.data(), out.data(), dump_row_n);
        printf("row type %d off %lld: %zu bytes\n", dump_row_type, dump_row_off, row_bytes);
        for (int i = 0; i < dump_row_n; ++i) {
            printf("%.7g%s", out[i], (i % 8 == 7) ? "\n" : " ");
        }
        printf("\n");
        return 0;
    }

    std::vector<gguf_tensor> table;
    uint64_t data_start = 0;
    size_t file_size = 0;
    if (!read_gguf(companion, table, data_start, file_size)) {
        std::fprintf(stderr, "cannot read %s\n", companion.c_str());
        return 2;
    }
    printf("companion: %s, %zu tensors, %.2f MiB\n", companion.c_str(), table.size(), file_size / 1048576.0);

    // the tensors this probe needs
    auto find = [&](const char * suffix) -> const gguf_tensor * {
        for (const auto & t : table) { if (t.name.size() >= std::strlen(suffix) &&
                                          t.name.compare(t.name.size() - std::strlen(suffix), std::strlen(suffix), suffix) == 0) {
            return &t; } }
        return nullptr;
    };
    const gguf_tensor * td  = find(".latent_d.weight");
    const gguf_tensor * tu  = find(".latent_u.weight");
    const gguf_tensor * tg  = find(".ffn_gate_exps.weight");
    const gguf_tensor * tp  = find(".ffn_up_exps.weight");
    const gguf_tensor * tdn = find(".ffn_down_exps.weight");
    const gguf_tensor * tl  = find(".latent_labels");
    if (!td || !tu || !tg || !tp || !tdn || !tl) {
        std::fprintf(stderr, "the companion is missing one of the latent tensors\n");
        return 2;
    }

    // the inputs the Python side wrote
    std::vector<float> x, w, want;
    std::vector<int32_t> ids;
    const int64_t n_embd   = td->dims[0];
    const int64_t latent   = tu->dims[0];
    const int64_t groups   = tu->dims[2];
    const int64_t n_ff     = tg->dims[1];
    const int64_t n_expert = tg->dims[2];
    if (!read_file(inputs + "/x.f32", x.data(), 0)) { /* sized below */ }
    x.resize((size_t) n_embd);
    if (!read_file(inputs + "/x.f32", x.data(), x.size() * sizeof(float))) {
        std::fprintf(stderr, "cannot read %s/x.f32\n", inputs.c_str()); return 2;
    }
    int64_t n_tokens = 1, n_used = 1;
    {
        FILE * f = std::fopen((inputs + "/shapes.txt").c_str(), "rb");
        if (!f) { std::fprintf(stderr, "cannot read %s/shapes.txt\n", inputs.c_str()); return 2; }
        if (std::fscanf(f, "%lld %lld", (long long *) &n_tokens, (long long *) &n_used) != 2) {
            std::fclose(f); std::fprintf(stderr, "shapes.txt must hold two integers\n"); return 2;
        }
        std::fclose(f);
    }
    ids.resize((size_t) (n_used * n_tokens));
    w.resize((size_t) (n_used * n_tokens));
    if (!read_file(inputs + "/ids.i32", ids.data(), ids.size() * sizeof(int32_t)) ||
        !read_file(inputs + "/weights.f32", w.data(), w.size() * sizeof(float))) {
        std::fprintf(stderr, "cannot read the ids or weights input\n"); return 2;
    }
    bool have_want = false;
    want.resize((size_t) n_embd);
    have_want = read_file(inputs + "/out.f32", want.data(), want.size() * sizeof(float));

    // build the graph
    ggml_backend_ptr backend;
    {
        // the backends are separate DLLs here, so they have to be registered before they can be enumerated
        ggml_backend_load_all();
        ggml_backend_dev_t dev = nullptr;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (backend_name == ggml_backend_dev_name(d)) { dev = d; break; }
        }
        if (!dev) {
            std::fprintf(stderr, "no backend named %s; this build has:", backend_name.c_str());
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                std::fprintf(stderr, " %s", ggml_backend_dev_name(ggml_backend_dev_get(i)));
            }
            std::fprintf(stderr, "\n");
            return 2;
        }
        backend.reset(ggml_backend_dev_init(dev, nullptr));
        if (!backend) { std::fprintf(stderr, "cannot init %s\n", backend_name.c_str()); return 2; }
        printf("backend: %s\n", ggml_backend_dev_name(dev));
    }
    ggml_init_params ip = { 512u * 1024u * 1024u, nullptr, true };
    ggml_context_ptr ctx(ggml_init(ip));

    auto make_c = [&](const gguf_tensor * t) {
        ggml_tensor * r = ggml_new_tensor(ctx.get(), (ggml_type) t->type, (int) t->dims.size(), t->dims.data());
        std::string full = t->name;
        ggml_set_name(r, full.c_str());
        return r;
    };
    ggml_tensor * td_t  = make_c(td);
    ggml_tensor * tu_t  = make_c(tu);
    ggml_tensor * tg_t  = make_c(tg);
    ggml_tensor * tp_t  = make_c(tp);
    ggml_tensor * tdn_t = make_c(tdn);
    ggml_tensor * tl_t  = make_c(tl);

    ggml_tensor * cur = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_embd, n_tokens);
    // the Vulkan plumbing probe is only enabled for this name, so a real model can never reach it
    ggml_set_name(cur, vk_copy ? "latent_probe" : "cur");
    ggml_tensor * ids_t = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_used, n_tokens);
    ggml_set_name(ids_t, "ids");
    ggml_tensor * w_t = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 1, n_used, n_tokens);
    ggml_set_name(w_t, "weights");

    ggml_tensor * out = ggml_latent_moe(ctx.get(), cur, td_t, tu_t, tg_t, tp_t, tdn_t, tl_t, ids_t, w_t);

    // the Vulkan block writes its intermediate D projection here; the CPU reference does not need it
    ggml_tensor * scr_t = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, latent * groups * n_tokens);
    ggml_set_name(scr_t, "latent_scratch");
    out->src[9] = scr_t;

    ggml_set_name(out, "out");

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, out);

    // one buffer for every tensor in the context, the way mm-probe does it: no graph allocator, so no reuse questions
    ggml_backend_buffer_ptr tensor_buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!tensor_buf) { std::fprintf(stderr, "allocation failed\n"); return 2; }

    // upload the tensors straight from the companion's byte ranges
    FILE * cf = std::fopen(companion.c_str(), "rb");
    if (!cf) { std::fprintf(stderr, "cannot reopen the companion\n"); return 2; }
    auto upload = [&](ggml_tensor * dst, const gguf_tensor * src) {
        const size_t n = ggml_nbytes(dst);
        std::vector<char> buf(n);
        std::fseek(cf, (long) (data_start + src->offset), SEEK_SET);
        if (std::fread(buf.data(), 1, n, cf) != n) { std::fprintf(stderr, "short read for %s\n", src->name.c_str()); std::exit(2); }
        ggml_backend_tensor_set(dst, buf.data(), 0, n);
    };
    upload(td_t, td);
    upload(tu_t, tu);
    upload(tg_t, tg);
    upload(tp_t, tp);
    upload(tdn_t, tdn);
    upload(tl_t, tl);
    std::fclose(cf);
    ggml_backend_tensor_set(cur, x.data(), 0, x.size() * sizeof(float));
    ggml_backend_tensor_set(ids_t, ids.data(), 0, ids.size() * sizeof(int32_t));
    ggml_backend_tensor_set(w_t, w.data(), 0, w.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend.get(), gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "compute failed\n"); return 2;
    }
    std::vector<float> got((size_t) n_embd);
    // requested: write the output tensor so another implementation can be compared value for value
    ggml_backend_tensor_get(out, got.data(), 0, got.size() * sizeof(float));

    if (!dump_out.empty()) {
        FILE * df = std::fopen(dump_out.c_str(), "wb");
        if (!df) { std::fprintf(stderr, "cannot write %s\n", dump_out.c_str()); return 2; }
        std::fwrite(got.data(), sizeof(float), got.size(), df);
        std::fclose(df);
        printf("output written to %s (%zu floats)\n", dump_out.c_str(), got.size());
    }

    double sum = 0.0, max_abs = 0.0, diff = 0.0;
    for (int64_t i = 0; i < n_embd; ++i) {
        sum += got[i];
        max_abs = std::max(max_abs, (double) std::fabs(got[i]));
        // The blob form: one byte view of a layer's experts, which is how a latent pack stores them.  The typed form above
    // came from a repacked companion; if both are given, their results must be identical, because the bytes and the
    // arithmetic are the same and only the addressing differs.
    if (!blob_file.empty()) {
        const size_t gu_row = ggml_row_size((ggml_type) tg->type, latent);
        const size_t dn_row = ggml_row_size((ggml_type) tdn->type, n_ff);
        const int64_t blob_len = (int64_t) (2 * n_ff * gu_row + latent * dn_row);
        const long long blob_offset = std::atoll(blob_offset_text.c_str());

        ggml_init_params ip3 = { 512u * 1024u * 1024u, nullptr, true };
        ggml_context_ptr ctx3(ggml_init(ip3));
        ggml_tensor * d3  = ggml_new_tensor(ctx3.get(), (ggml_type) td->type, (int) td->dims.size(), td->dims.data());
        ggml_tensor * u3  = ggml_new_tensor(ctx3.get(), (ggml_type) tu->type, (int) tu->dims.size(), tu->dims.data());
        ggml_tensor * ex3 = ggml_new_tensor_1d(ctx3.get(), GGML_TYPE_I8, blob_len * n_expert);
        ggml_tensor * lb3 = ggml_new_tensor_1d(ctx3.get(), GGML_TYPE_I32, n_expert);
        ggml_tensor * cur3 = ggml_new_tensor_2d(ctx3.get(), GGML_TYPE_F32, n_embd, n_tokens);
        ggml_tensor * ids3 = ggml_new_tensor_2d(ctx3.get(), GGML_TYPE_I32, n_used, n_tokens);
        ggml_tensor * w3   = ggml_new_tensor_3d(ctx3.get(), GGML_TYPE_F32, 1, n_used, n_tokens);
        ggml_tensor * out3 = ggml_latent_moe_blob(ctx3.get(), cur3, d3, u3, ex3, lb3, ids3, w3,
                                                  (int32_t) tg->type, (int32_t) tdn->type,
                                                  latent, n_ff, n_expert, blob_len);
    {
        // the same scratch the typed form uses: groups x latent floats, one token
        ggml_tensor * scr3 = ggml_new_tensor_1d(ctx3.get(), GGML_TYPE_F32, latent * groups * n_tokens);
        ggml_set_name(scr3, "latent_scratch");
        out3->src[9] = scr3;
    }
        ggml_cgraph * gf3 = ggml_new_graph(ctx3.get());
        ggml_build_forward_expand(gf3, out3);
        const double t_ctx = now_ms();
        ggml_backend_buffer_ptr buf3(ggml_backend_alloc_ctx_tensors(ctx3.get(), backend.get()));
        if (!buf3) { std::fprintf(stderr, "blob allocation failed\n"); return 2; }
        printf("  blob: context+allocation %.1f ms\n", now_ms() - t_ctx);

        FILE * cf3 = std::fopen(companion.c_str(), "rb");
        auto upload_q3 = [&](ggml_tensor * dst, const gguf_tensor * src) {
            std::vector<char> buf(ggml_nbytes(dst));
            std::fseek(cf3, (long) (data_start + src->offset), SEEK_SET);
            if (std::fread(buf.data(), 1, buf.size(), cf3) != buf.size()) { std::exit(2); }
            ggml_backend_tensor_set(dst, buf.data(), 0, buf.size());
        };
        upload_q3(d3, td); upload_q3(u3, tu); upload_q3(lb3, tl);
        std::fclose(cf3);

        const double t_read = now_ms();
        FILE * bf = std::fopen(blob_file.c_str(), "rb");
        if (!bf) { std::fprintf(stderr, "cannot read %s\n", blob_file.c_str()); return 2; }
        std::vector<char> raw((size_t) (blob_len * n_expert));
        std::fseek(bf, (long) blob_offset, SEEK_SET);
        if (std::fread(raw.data(), 1, raw.size(), bf) != raw.size()) {
            std::fprintf(stderr, "short read from %s at %lld\n", blob_file.c_str(), blob_offset); return 2;
        }
        std::fclose(bf);
        printf("  blob: read %.0f MiB in %.1f ms\n", raw.size() / 1048576.0, now_ms() - t_read);
        const double t_set = now_ms();
        ggml_backend_tensor_set(ex3, raw.data(), 0, raw.size());
        printf("  blob: upload %.1f ms\n", now_ms() - t_set);
        {
            // read the blob back: if the bytes on the device match the file, a wrong result is the kernel's arithmetic
            // rather than a missing upload
            std::vector<char> back(raw.size());
            ggml_backend_tensor_get(ex3, back.data(), 0, back.size());
            size_t first = back.size();
            for (size_t i = 0; i < back.size(); ++i) {
                if (back[i] != raw[i]) { first = i; break; }
            }
            if (first == back.size()) {
                printf("  blob: read-back matches the file (%zu bytes)\n", back.size());
            } else {
                printf("  blob: read-back DIFFERS at byte %zu (device %d, file %d)\n",
                       first, (int) (unsigned char) back[first], (int) (unsigned char) raw[first]);
            }
        }
        ggml_backend_tensor_set(cur3, x.data(), 0, x.size() * sizeof(float));
        ggml_backend_tensor_set(ids3, ids.data(), 0, ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(w3, w.data(), 0, w.size() * sizeof(float));
        const double t_compute = now_ms();
        if (ggml_backend_graph_compute(backend.get(), gf3) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "blob compute failed\n"); return 2;
        }
        printf("  blob: compute %.1f ms\n", now_ms() - t_compute);
        std::vector<float> got3((size_t) n_embd);
        ggml_backend_tensor_get(out3, got3.data(), 0, got3.size() * sizeof(float));
        double dmax = 0.0;
        for (int64_t i = 0; i < n_embd; ++i) { dmax = std::max(dmax, (double) std::fabs(got[i] - got3[i])); }
        const double tol3 = 1e-5 * std::max(1.0, max_abs);
        printf("blob form (one byte view, %lld bytes/ expert): max abs difference vs typed form %.6g -> %s\n",
               (long long) blob_len, dmax, dmax <= tol3 ? "ok" : "MISMATCH");

        // the comparison that decides: the blob form against the reference's own output, the same yardstick the typed
        // form is held to.  Comparing two device runs only says whether they agree, not whether either is right.
        std::vector<float> ref3((size_t) n_embd);
        {
            const std::string rf = inputs + "/out.f32";
            FILE * f = std::fopen(rf.c_str(), "rb");
            if (!f) { std::fprintf(stderr, "cannot read %s\n", rf.c_str()); return 2; }
            if (std::fread(ref3.data(), sizeof(float), ref3.size(), f) != ref3.size()) {
                std::fprintf(stderr, "short read from %s\n", rf.c_str()); return 2;
            }
            std::fclose(f);
        }
        double dref3 = 0.0;
        for (int64_t i = 0; i < n_embd; ++i) {
            dref3 = std::max(dref3, (double) std::fabs(ref3[(size_t) i] - got3[i]));
        }
        const double tolref3 = 1e-3 * std::max(1.0, max_abs);
        printf("blob form against %s: max abs difference %.6g -> %s\n",
               inputs.c_str(), dref3, dref3 <= tolref3 ? "ok" : "MISMATCH");
        if (dref3 > tolref3) { return 1; }
    }

    if (vk_copy) {
        double dmax = 0.0;
        for (int64_t i = 0; i < n_embd; ++i) {
            dmax = std::max(dmax, (double) std::fabs(got[i] - x[i]));
        }
        printf("vulkan copy probe: max abs difference against the activation %.6g -> %s\n", dmax,
               dmax == 0.0 ? "ok" : "MISMATCH");
        return dmax == 0.0 ? 0 : 1;
    }

    if (have_want) { diff = std::max(diff, (double) std::fabs(got[i] - want[i])); }
    }
    printf("result: sum %.6f  max abs %.6f  first four %.6f %.6f %.6f %.6f\n", sum, max_abs,
           got[0], got[1], got[2], got[3]);

    // The same block over the same weights dequantised to f32 by ggml: the quantised path and the f32 path must agree,
    // and a difference here means the op reads the wrong bytes rather than that the dequantiser is imprecise.
    if (verify_f32) {
        ggml_init_params ip2 = { 1024u * 1024u * 1024u, nullptr, true };
        ggml_context_ptr ctx2(ggml_init(ip2));
        auto as_f32 = [&](const gguf_tensor * t) {
            ggml_tensor * src = ggml_new_tensor(ctx2.get(), (ggml_type) t->type, (int) t->dims.size(), t->dims.data());
            ggml_set_name(src, (t->name + ".q").c_str());
            ggml_tensor * dst = ggml_new_tensor(ctx2.get(), GGML_TYPE_F32, (int) t->dims.size(), t->dims.data());
            ggml_set_name(dst, t->name.c_str());
            return std::pair<ggml_tensor *, ggml_tensor *>(src, dst);
        };
        auto [d_q, d_f]   = as_f32(td);
        auto [u_q, u_f]   = as_f32(tu);
        auto [g_q, g_f]   = as_f32(tg);
        auto [p_q, p_f]   = as_f32(tp);
        auto [dn_q, dn_f] = as_f32(tdn);
        ggml_tensor * cur2  = ggml_new_tensor_2d(ctx2.get(), GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_name(cur2, "cur");
        ggml_tensor * ids2  = ggml_new_tensor_2d(ctx2.get(), GGML_TYPE_I32, n_used, n_tokens);
        ggml_tensor * w2    = ggml_new_tensor_3d(ctx2.get(), GGML_TYPE_F32, 1, n_used, n_tokens);
        ggml_tensor * out2  = ggml_latent_moe(ctx2.get(), cur2, d_f, u_f, g_f, p_f, dn_f, tl_t, ids2, w2);
        ggml_set_name(out2, "out_f32");
        ggml_cgraph * gf2 = ggml_new_graph(ctx2.get());
        ggml_build_forward_expand(gf2, out2);
        ggml_backend_buffer_ptr buf2(ggml_backend_alloc_ctx_tensors(ctx2.get(), backend.get()));
        if (!buf2) { std::fprintf(stderr, "second allocation failed\n"); return 2; }

        FILE * cf2 = std::fopen(companion.c_str(), "rb");
        auto upload_q = [&](ggml_tensor * dst, const gguf_tensor * src) {
            std::vector<char> buf(ggml_nbytes(dst));
            std::fseek(cf2, (long) (data_start + src->offset), SEEK_SET);
            if (std::fread(buf.data(), 1, buf.size(), cf2) != buf.size()) { std::exit(2); }
            ggml_backend_tensor_set(dst, buf.data(), 0, buf.size());
        };
        upload_q(d_q, td); upload_q(u_q, tu); upload_q(g_q, tg); upload_q(p_q, tp); upload_q(dn_q, tdn);
        // dequantise on the host with ggml's own converter, then upload as f32
        auto dq = [&](ggml_tensor * q, ggml_tensor * f) {
            std::vector<char> raw(ggml_nbytes(q));
            ggml_backend_tensor_get(q, raw.data(), 0, raw.size());
            const ggml_to_float_t to_f = ggml_get_type_traits(q->type)->to_float;
            const int64_t n = ggml_nelements(q);
            std::vector<float> f32((size_t) n);
            if (to_f) {
                // a converter takes a row at a time: its block arithmetic is per row, so passing the whole tensor would
                // make it read a row as long as every row together
                const int64_t ne0 = q->ne[0];
                const size_t  row_bytes = ggml_row_size(q->type, ne0);
                for (int64_t r = 0; r < n / ne0; ++r) {
                    to_f(raw.data() + (size_t) r * row_bytes, f32.data() + (size_t) r * ne0, ne0);
                }
            } else if (q->type == GGML_TYPE_F32) {
                std::memcpy(f32.data(), raw.data(), (size_t) n * sizeof(float));
            } else {
                std::fprintf(stderr, "no converter for type %d\n", (int) q->type); std::exit(2);
            }
            ggml_backend_tensor_set(f, f32.data(), 0, f32.size() * sizeof(float));
        };
        dq(d_q, d_f); dq(u_q, u_f); dq(g_q, g_f); dq(p_q, p_f); dq(dn_q, dn_f);
        std::fclose(cf2);
        ggml_backend_tensor_set(cur2, x.data(), 0, x.size() * sizeof(float));
        ggml_backend_tensor_set(ids2, ids.data(), 0, ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(w2, w.data(), 0, w.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend.get(), gf2) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "f32 compute failed\n"); return 2;
        }
        std::vector<float> got2((size_t) n_embd);
        ggml_backend_tensor_get(out2, got2.data(), 0, got2.size() * sizeof(float));
        double dmax = 0.0;
        for (int64_t i = 0; i < n_embd; ++i) { dmax = std::max(dmax, (double) std::fabs(got[i] - got2[i])); }
        const double tol2 = 1e-4 * std::max(1.0, max_abs);
        printf("quantised vs f32 weights: max abs difference %.6g (tolerance %.6g) -> %s\n", dmax, tol2,
               dmax <= tol2 ? "ok" : "MISMATCH");
        if (dmax > tol2) { return 1; }
    }
    if (have_want) {
        // a relative tolerance: the reference accumulates in numpy float32 and this side in double, so the two agree to
        // about float32 precision on values of any magnitude, and an absolute threshold would only fit one scale
        const double tol = 1e-5 * std::max(1.0, max_abs);
        printf("against %s/out.f32: max abs difference %.6g, relative tolerance %.6g -> %s\n",
               inputs.c_str(), diff, tol, diff <= tol ? "ok" : "MISMATCH");
        return diff <= tol ? 0 : 1;
    }
    return 0;
}
