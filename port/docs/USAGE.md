# 使用 / Usage

## 1. 构建 / Build

```powershell
# 取一份 llama.cpp（pin 见 port/llama-overlay-vanilla/UPSTREAM.md），叠加 overlay，构建引擎与工具
powershell -NoProfile -ExecutionPolicy Bypass -File port/tools/fetch-llama.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File port/tools/build-engine.ps1
# 可选：后端自检工具
powershell -NoProfile -ExecutionPolicy Bypass -File port/tools/build-engine.ps1 -Target test-backend-ops
```

产物在 `build\bin\`：`strata-vk.exe`、`llama.dll`、`ggml-vulkan.dll`、`ggml-cpu.dll`。
The build reads `third_party/llama.cpp`; the overlay in `port/llama-overlay-vanilla` is composed into it (see `MAINTAINER.md`).

## 2. 跑起来 / Run

```powershell
# 默认：整模型在 GPU（Vulkan）
build\bin\strata-vk.exe --native D:\models\pack.gguf --tokens 9707 --n 32

# 纯 CPU 后端（需要支持的自定义算子，见 BACKEND-OPTIONS.md）
build\bin\strata-vk.exe --gpu cpu --native D:\models\pack.gguf --tokens 9707 --n 16
```

`--tokens` 是**确定的**输入路径（不走模板），用于复现与门禁；`--prompt` 的裸文本对这个模型是 off-distribution 的，
见 `BACKEND-OPTIONS.md` §13（不要用它做门禁）。

## 3. 常用开关 / Flags worth knowing

| 开关 | 作用 |
| --- | --- |
| `--native FILE` | 模型包 |
| `--gpu vulkan\|cpu` | 后端 |
| `--tokens a,b,c` / `--prompt TEXT` | 输入（token 路径 / 文本路径） |
| `-c N` / `--ctx-size` | 上下文（默认 32768） |
| `--kv f16\|q8_0\|q4_0` | KV 类型（实测对 decode 无影响） |
| `--shader-family auto\|f16\|fp32` | **诊断开关，默认路径已经不需要它。** 2026-11-18 之后，int-dot 的 **mat-mat** 固定取自 fp32 模块，mat-vec 与 elementwise 留在 f16 模块，于是默认同时拿到 prefill 49.80 与 decode 10.12（成对 3 轮中位数）。`fp32` 现在只把 elementwise 也换成 fp32：decode 5.5、prefill 46.63，两头都更差。**别再用它换 prefill**（`PERFORMANCE.md` §1） |
| `--prefill N` | prefill 分块（默认 1024；此项带来 pp512 +25%） |
| `--npu off\|dense\|split` | NPU 布置（本机被守卫硬关闭，见 BACKEND-OPTIONS.md） |
| `--perf-cap 1..100` | 限速（默认 100）；`--gpu cpu` 时不生效，见下面的 `-t` |
| `-t N` / `--threads N` | 后端线程数。`--gpu cpu` 的默认是**大核类的核数减一**（本机 11，逻辑 CPU 6-17 里留一个给宿主）：小核在每个算子的屏障上都在收钱，而占满整类会让主线程/驱动/OS 抢核（`PERFORMANCE.md` §3）。`-t` 超过大核数会**主动把掩码放宽到全部核**，所以 `--gpu cpu -t 18 --bench-pp` 能拿回 CPU prefill 的 19.4 tok/s |
| `--lookup on\|off` | 投机解码（实测 0% 接受率，见 PERFORMANCE.md） |

## 4. 测量 / Measuring

| 工具 | 用途 |
| --- | --- |
| `STRATA_VK_TIME=1` | GPU 侧时间桶（record/submit/wait/dispatch/barrier/fusion/end） |
| `STRATA_VK_NODE_LOG=1` / `STRATA_VK_PIPE_LOG=1` | 逐节点 / 逐 pipeline 的 dispatch 记录 |
| `STRATA_CPU_TIME=1` | CPU 后端逐算子计时（退出时按算子名打印） |
| `STRATA_VK_OP_TIME=1` | 逐 dispatch 的 GPU 时间戳（**行级金额不可信**，见 PERFORMANCE.md） |
| `GGML_VK_PERF_LOGGER=1` | 每图 GPU `Total time`（**这个可信**） |
| `python port/tools/ab_pp.py` | ABBA 成对 A/B（含 `--control` 零对照） |
| `python port/tools/op-times.py` | 解析 perf-logger 输出 |
| `build\bin\mm-probe.exe -b Vulkan0 -t tqk6 -m 2560 -n 1 -k 2560 --n-as 512 --n-used 10` | **single-kernel isolation**: one shape, one dispatch, wall clock. `--n-as/--n-used` runs `MUL_MAT_ID` (the MoE expert matvec, half of a single-stream decode step); the trellis types work too (2026-11-19: `type_from_name` dereferenced a null name for the holes in the type table). `--verify` is `ops=1` only |
| `python port/tools/node-count.py` | 逐算子 dispatch 计数 |
| `python port/tools/logit-diff.py` | 两次 `--dump-logits` 的质量对比（top-1/top-k/KL） |
| `--batch-slots N` + `--bench-tg T` | **批量解码基准**：N 条序列各走 T 步，每步一个 batch。`N = 1` 与 `N > 1` 是同一段代码，所以两个数是一对成对比较。测得的曲线（cooperative 路径）：8 槽 8.89、16 槽 18.45、24 槽 25.05、32 槽 30.51、48 槽 32.24、**64 槽 33.62** tok/s 聚合；96 槽起 `allocateMemory` 失败。`slots >= 9` 才会进 matmul 路径（8 及以下走慢的 matvec，8 是个坑）。要求 `n_seq_max >= N`（同一个 `--batch-slots` 会同时设 `cp.n_seq_max`）以及 `n_ctx >= N × (prompt + T + 1)`；`--serve` 用同一个旗标就能让 `serve/server.py` 走多槽 |
| `python port/tools/energy-ab.py EXE MODEL -- <引擎参数> + <引擎参数>` | **成对能耗**：一边跑引擎一边用 `energy-sample.ps1` 采样 Energy Meter，把样本裁到引擎自己打印的 `bench window`（只含被计时的解码步），报每 1000 token 的能耗。计数器是未标定标度，**只有同一次探针内的比值有效**；两次运行要生成相同的 token 数 |
| `--parallel N` / `--batch N` | **`--serve` 的请求槽数**（`--batch` 是 `serve/replicas.py` 的写法）。N ≥ 2 时 `INFO` 报 `batch_slots=N`，引擎接受 `BGEN <slot> …` 并把 N 个请求放进同一批、每步各出一个 token，**一份权重**。每槽的 cache 是 `n_ctx / N`。**正确但当前没有收益**：32 槽、每请求 64 token 实测槽 0 与 solo **64/64 相同**，但每步 3178 ms ≈ 32 × 单条 99 ms，聚合 **1.00×**（`PERFORMANCE.md` §4；早先的 4.27× 是坏内核的读数，已撤回） |
| `python port/tools/batch-serve.py EXE MODEL --slots N --tokens T --engine …` | **协议级批量**：同一进程先发一个 `GEN` 再发 N 个 `BGEN`，按同槽相邻 token 的时间差报每步时间（不含 prefill），并核对槽 0 的 token 流与单独运行是否逐 token 相同 |
| `python port/tools/gguf-budget.py MODEL.gguf` | 一个包的字节预算：按角色、按类型、按张量，带宽论证都用它（只读元数据，60 GB 的包也是秒级） |
| `python port/tools/gguf-tokenizer.py MODEL.gguf --out pack/full/tokenizer` | 从包里造出 `serve/server.py` 要的 tokenizer 目录（`vocab.json`/`merges.txt`/`token_type.json`/`tokenizer.json`），并用 `tools/strata_tokenizer.py` 本体做往返验证。**tokenizer 必须来自被服务的那个模型**；`strata_tokenizer` 只认 `qwen35` / `deepseek*` 两种 pre-tokenizer，`qwen2`（Tiny-Qwen）会被明确拒绝 |
| `python port/tools/e2e-replicas.py --exe build/bin/strata-vk.exe --model M.gguf --out-json tmp.json` | 并发那条线的端到端：起一个 server、两个副本（Vulkan + CPU）、并发两个请求，断言双方完成且日志里有 `cards 0 \| cpu` |
| `python port/tools/two-serve.py` | 两个引擎（默认 + `--gpu cpu`）各发一个 GEN，断言各自的 READY/后端/`DONE` token 数（不经过 Python server） |
| `STRATA_CPU_REPACK=0\|1` | 压过引擎按内存预算给 repack 的判断；`0` 复现"repack 从未生效"的旧行为，用来做 A/B |
| `STRATA_CPU_MASK=<hex>` / `STRATA_CPU_THREADS=N` | 压过 CPU 后端的大核类亲和计划（`plan_cpu_affinity`），用来扫掩码/线程数 |
| `STRATA_VK_SKIP=<子串>` | **测量用消融**：pipeline 名含该子串的 dispatch 照常记账但不发射，输出必然是错的，墙钟就是这一族的价格。一次构建问"这一族要多少钱"，比抓帧便宜（`PERFORMANCE.md` §2 与 `ADRENO.md` 是它的第一份结果）。**结果比基线更慢 = 垃圾数据，不是这一族便宜** |
| `STRATA_VK_ADMIT_DEBUG=1` | 在准入处与 solo 首 token 处打印采样行的 top-1/top-2 和两个索引（`n-1` 与 `-1`）各自解析到的行；用它区分"行是空的 / 行号映射错 / 模型真的选了别的"。“第二个请求不一样”那条就是靠它定位的（`PERFORMANCE.md` §4） |
| `STRATA_VK_KV_UNIFIED=1` | 把 `kv_unified` 打开（`llama_context_default_params` 默认是 false，而 llama.cpp 自己的工具都走 `common_params` 的 true）。实测：单序列 logits 回到 n_seq_max=1 的值，但第二个序列直接 NaN，**是负结果**，留在代码里是为了下次不用重写（`PERFORMANCE.md` §4） |
| **只换 `build\bin\ggml-cpu.dll` 做 A/B** | 内核级 A/B 最干净的办法：把两个变体各存一份 DLL，每跑一次换一次文件。它把"重新构建"从测量里去掉，成对交替之后噪声底比换源码重编小得多（本轮的 trellis 融合就是用这个办法判成中性的） |
| **跑 app / JS 侧的测试** | 这台机器的 PATH 上没有 `node`，但 DSH 自己跑在 Node 上：`$env:ELECTRON_RUN_AS_NODE='1'; & "$env:LOCALAPPDATA\Programs\DeepSeek Harness\DeepSeek Harness.exe" --version` 报 **Node v24**，`-e "..."` 可直接执行脚本。`apps/host` 的 `.mjs` 与 `serve/` 的 Python 测试因此都能在本机跑，不必等打包 |
| `serve/server.py --replicas-cpu N` | 在卡片副本旁边再加 N 个 **CPU 后端**副本（一张卡的机器唯一能有的第二个 worker）。实测 +39% 聚合（`PERFORMANCE.md` §4），代价是超大包上第二个请求慢。端到端验证脚本：`python port/tools/e2e-replicas.py --exe build\bin\strata-vk.exe --model M.gguf --tokenizer T --out-json X --cpu-via-cli` |
| `python port/tools/gguf-tokenizer.py M.gguf --out DIR` | 从包元数据造出 `serve/server.py` 要的 tokenizer 目录，并用 `strata_tokenizer` 做编码往返验证。**vocab 必须来自被服务的那个模型**；`pre` 只支持 `qwen35` / `deepseek*` |
| `python port/tools/two-serve.py EXE MODEL` | 按 `serve/replicas.py` 的写法同时起两个引擎（一个默认后端、一个 `--gpu cpu`），各发一个 GEN，断言两侧都 READY、都 DONE、token 数对得上。这是「CPU 副本」那条路的引擎侧契约检查（`PERFORMANCE.md` §4） |
| `STRATA_VK_F16ACC=1` | 选 f16-accumulate 的 cooperative matrix。**默认已关闭**：它与 CPU logits 的相关性只有 0.02（27B/35B/Gyro-S 三个包都测了），输出是错的；打开只用于"只看形状"的基准（prefill 221 tok/s 那种读数就是它，`PERFORMANCE.md` §4.1）。**2026-11-18 更正**：那个 0.02 来自一个零次循环的 tile（`PERFORMANCE.md` §4.4/§4.5），tile 修好之后这条开关**只建分组的（`MUL_MAT_ID`）管线**，该内核已被 oracle 认证正确（`-o MUL_MAT_ID -p type_a=q5_K`/`tqk6` 各 3/3），Gyro-S pp512 达 **145 tok/s**；但整模型仍非有限，原因在 dense 一侧的回落路径（§4.5 末），所以**还不能当默认** |
| `STRATA_VK_F16ACC_DENSE=1` | 连同 dense 的 cooperative 管线一起创建。默认不建，因为 dense 那份 shader 会把 `-inf` 写进每个输出的第一个 BM tile（`PERFORMANCE.md` §4.5）；留着这个开关是给修 staging 的人做 A/B，不是给用户用的 |
| `STRATA_VK_F16=off` | 强制整机 fp32 shaders（等价于 `--shader-family fp32`）。**它已经不是 prefill 的捷径**：默认路径把 fp32 只用在 int-dot 的 mat-mat 上，pp512 反而比它快（49.80 对 46.63）且 decode 不受影响（10.12 对 5.48）。留作诊断 |
| `STRATA_VK_INT8_COOPMAT=1` | 让 Qualcomm 设备也创建 upstream 只给 AMD RDNA3/4 建的 **int8 cooperative MMQ** 管线（本机硬件有 64×64×32 的 Sint8→Sint32 矩阵核心，**int32 累加、精确**）。实测：oracle 在 q5_K/q6_K/q8_0 上 **0 失败**，但 Gyro-S pp512 16.18 → 15.62、tg16 10.37 → 10.31 —— **中性**，默认关（`PERFORMANCE.md` §4.2 与 §5-P3） |
| `--dump-logits FILE` | 把第一次生成的首行 logits 写盘（fp32 词表长度）。与 `--gpu cpu` 配对就是后端等价性门禁（`USAGE.md` 第 5 节第 0 条，`PERFORMANCE.md` §4.1） |

## 5. 门禁 / Gates

交付前跑一次，全部通过才提交：

```powershell
# 0) 后端等价性：同一模型、同一模板化 prompt，CPU 与 Vulkan 各存一份 logits，比相关性与 argmax
#    （token 流比较会漏掉整类错误：f16-accumulate 的 coopmat 内核就是在 32-token 比较下活了很久，见 §4.1）
build\bin\strata-vk.exe --native M.gguf --tokens 248045,846,198,9419,248046,198,248045,74455,198 --n 1 --gpu cpu --dump-logits tmp-cpu.bin
build\bin\strata-vk.exe --native M.gguf --tokens 248045,846,198,9419,248046,198,248045,74455,198 --n 1 --dump-logits tmp-gpu.bin
#    期望：corr >= 0.999 且 argmax 相同；实测 27B 0.99967、35B 0.99906、Gyro-S 0.9982（cooperative 默认开）/ 0.9983（关掉），argmax 均相同
#    （模板化 prompt 的 id 用 tools/strata_tokenizer.py 的 Tokenizer.from_gguf(...).encode(tmpl, parse_special=True) 取）

# 1) 参考 token 流（GPU 路径必须逐 token 相同）
build\bin\strata-vk.exe --native D:\models\pack.gguf --tokens 9707 --n 32
#    期望：16451,18,13,20,12,21135,487,328,31788,24894,8015,19,9546,487,328,74,24050,12283,17,13,20,487,328,40819,12,20,1293,198,262,28539,283,498

# 2) 后端 oracle
build\bin\test-backend-ops.exe -o MUL_MAT -b Vulkan0     # 期望 0 failures
#    注意：它只覆盖默认管线；STRATA_VK_F16ACC=1 的分组 cooperative 管线要单独跑 `-o MUL_MAT_ID`（实测 1168/1168 通过）

# 2b) cooperative 内核的最小自检（不需要模型，秒级）。tile 退化时这一步是**全零**：
#     `dev/cpu` 比值应落在 0.98-1.00；全零或 nmse 1.0 就是 MMA 循环没跑（PERFORMANCE.md §4.4/§4.5）
$env:STRATA_VK_F16ACC='1'; build\bin\mm-probe.exe -b Vulkan0 -t q4_0 -m 512 -n 64 -k 512 --verify
Remove-Item env:STRATA_VK_F16ACC
#    已知：2026-11-19 起整模型 `--dump-logits` 是有限的（trellis 类型的 cooperative shader 选错已修，PERFORMANCE.md 4.2）

# 3) 文档与编码
python port/tools/check-encoding.py        # 期望 0 damaged
python port/tools/audit-docs.py            # 期望 0 problems

# 4) overlay 与 third_party 一致
python - <<'PY'
import hashlib, pathlib
r = pathlib.Path('port/llama-overlay-vanilla'); t = pathlib.Path('third_party/llama.cpp')
files = [p for p in r.rglob('*') if p.is_file()]
mis = [str(p.relative_to(r)) for p in files if not (t/p.relative_to(r)).exists()
       or hashlib.sha256(p.read_bytes()).hexdigest() != hashlib.sha256((t/p.relative_to(r)).read_bytes()).hexdigest()]
print('overlay', len(files), 'files', len(mis), 'mismatches')
PY
```
