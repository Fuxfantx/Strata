# `llama-ple-disk` — the N-Gram / PLE table kept in the file (staged, not wired yet)

Transplanted verbatim from `agentionai/llama.cpp`, branch `perf/strix-vulkan` at `3c75be0` (MIT licence):

* `src/llama-ple-disk.h`
* `src/llama-ple-disk.cpp` (411 lines)

## Why it is the answer to both open questions

The 3.8 Flash Next packs carry `per_layer_token_embd` — 15.27 GiB of the Gyro-S pack (dims `(160, 320001536)`) — which this
tree loads into RAM like any other `GET_ROWS` tensor. The resident set then becomes 24.02 GiB of experts + ~4 GiB dense +
15.27 GiB of table = **54 GiB on a 40 GiB machine**, so the expert buffers get paged out and every decode step re-reads the
~0.5 GB of experts it uses: 4.2 tok/s, against 13.2 tok/s on the Q2_0 pack. The kernels are not the limit (a q4_0 matvec of
the same shape measures 54.6 GB/s on this machine).

The wheel's own choice, for exactly the "which mode does not page" question:

```cpp
// O_DIRECT / FILE_FLAG_NO_BUFFERING, so the rows never enter the page cache either
bool direct_io = true;
size_t cache_bytes = 256u << 20;  // direct-mapped cache of raw rows
int32_t n_threads = 64;           // random reads on NVMe need queue depth
```

So the table is neither mapped (which would hold page-cache pages) nor pread (which pollutes it): unbuffered positional reads,
a bounded direct-mapped row cache, and a reader pool over `FILE_FLAG_OVERLAPPED` (the comment in the source explains that a
synchronous handle would serialize the pool, and that `FILE_FLAG_NO_BUFFERING` requires aligned buffers).

## What wiring it needs (next round)

1. `llama-model-loader`: when `params.ple_io == disk`, skip allocating/reading `per_layer_token_embd` and record its file offset
   and type instead (`gguf` gives both).
2. `qwen4exp.cpp`: replace the `GET_ROWS` on that tensor with a `gather` at batch build time (the wheel takes `int32_t * idx`,
   `n`, and a float `dst`), which is where the 16 hashed rows per token are known.
3. `common/arg.cpp` + `llama_model_params`: the `--ple-io ram|disk` switch.
4. `--ple-cache-mib` / `--ple-threads` knobs if the defaults need tuning on this machine.

Then the resident set drops to ~39 GiB (fits), and the measurement to beat is the Q2_0 pack's pp64 31.9 / tg32 13.20 tok/s.

## Status

Staged here rather than in `port/llama-overlay-vanilla/` because it is not wired: adding it to the overlay would put a file in the
build that nothing calls yet, and the overlay/third_party parity check expects one-to-one mirrors.

## 施工记录（按顺序，每步都可独立构建）

1. `include/llama.h:332` 的 `struct llama_model_params`（参照同段的 `moe_cache_size`/`moe_cache_cpu_share`）加 `int32_t ple_io_threads`、`int32_t ple_cache_mb`；`src/llama-model.cpp` 的 `llama_model_default_params()` 给 64 / 256。
2. `common/arg.cpp` 加 `--ngram-on-disk`、`--ngram-io-threads`、`--ngram-cache-mb`，经 `common/common.h` 的 params 传到 `mparams`（片段见 `wiring/common__arg.cpp.txt`、`wiring/common__common.h.txt`）。
3. 引擎侧 `port/engine/vk-options.h`/`vk-engine-load.cpp` 暴露同名开关并透传（默认 off）。
4. `src/models/qwen4exp.cpp`：19 处上下文见 `wiring/src__models__qwen4exp.cpp.txt`；把 `per_layer_token_embd` 的 `GET_ROWS` 换成按批 `gather`，并在 loader 跳过该张量的分配，记录 `{file, offs, type, ne0, nrows}`。
5. 验证：`--tokens 9707 --n 8` 的 token 必须仍是 `16451,18,13,20,12,21135,487,328`；随后看 decode 是否 4.2 -> 10+ tok/s。
