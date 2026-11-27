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
| `--shader-family auto\|f16\|fp32` | **诊断开关，默认路径已经不需要它。** 2026-11-18 之后，int-dot 的 **mat-mat** 固定取自 fp32 模块，mat-vec 与 elementwise 留在 f16 模块，于是默认同时拿到 prefill 49.80 与 decode 10.12（成对 3 轮中位数）。`fp32` 现在只把 elementwise 也换成 fp32：decode 5.5、prefill 46.63，两头都更差。**别再用它换 prefill**（`PERFORMANCE-LOG.md` §1） |
| `--prefill N` | prefill 分块（默认 1024；此项带来 pp512 +25%） |
| `--npu off\|dense\|split` | NPU 布置（本机被守卫硬关闭，见 BACKEND-OPTIONS.md） |
| `--perf-cap 1..100` | 限速（默认 100）；`--gpu cpu` 时不生效，见下面的 `-t` |
| `-t N` / `--threads N` | 后端线程数。`--gpu cpu` 的默认是**大核类的核数减一**（本机 11，逻辑 CPU 6-17 里留一个给宿主）：小核在每个算子的屏障上都在收钱，而占满整类会让主线程/驱动/OS 抢核（`PERFORMANCE-LOG.md` §3）。`-t` 超过大核数会**主动把掩码放宽到全部核**，所以 `--gpu cpu -t 18 --bench-pp` 能拿回 CPU prefill 的 19.4 tok/s |
| `--lookup on\|off` | 投机解码（实测 0% 接受率，见 PERFORMANCE.md） |

#### 并行相关 / Parallelism

| 开关 | 说明 |
| --- | --- |
| `--split-mode layer\|row\|none` | `layer` 是分层流水（默认）；`row` 是**真张量并行**，需要后端暴露 `ggml_backend_split_buffer_type` **且**两侧都有 split 张量算子；本机只有 1 个 Vulkan 设备、CPU 侧没有 split 算子，所以引擎会在加载前说明缺哪一侧并回落到分层（`PERFORMANCE-LOG.md` §5.5.16/§5.5.17） |
| `--tensor-split a,b,...` | 多设备份额。列出两个设备时第二个是 CPU；据此得到的是**分层**而非行切分 |

## 4. 测量 / Measuring

测量这一节的前提：**这台机器上的读数会漂**。预填在同一窗口内可以差 12–16%，解码只差约 0.6%。所以
"哪个工具可信"比"有多少工具"重要。

### 4.1 可以信的读数

| 手段 | 为什么可信 |
| --- | --- |
| `GGML_VK_PERF_LOGGER=1` 的每图 `Total time` | 直接来自时间戳差，只报总量不报归属；**不要用它的 per-name 表排名**（错位一个节点，`PERFORMANCE-LOG.md` 有证据） |
| `python port/tools/ab_pp.py` | ABBA 成对交替 + 中位数，带 `--control` 零对照；预填改动只能用这种方式判定 |
| `build\bin\mm-probe.exe -b Vulkan0 -t tqk6 -m 2560 -n 1 -k 2560 --n-as 512 --n-used 10` | 单形状、单派发的隔离墙钟；`MUL_MAT_ID` 也能测（`--verify` 仅 `ops=1`） |
| `python port/tools/node-count.py` | 逐算子派发计数（派发条数是图的性质，不受窗口影响） |
| `python port/tools/logit-diff.py` | 两次 `--dump-logits` 的 top-1/top-k/KL 比较，用于后端等价性 |
| `python port/tools/gguf-budget.py MODEL.gguf` | 包的字节预算（按角色/类型），带宽论证的依据；只读元数据 |
| `python port/tools/gguf-tokenizer.py MODEL.gguf --out DIR` | 造出 `serve/server.py` 要的 tokenizer 目录并做编码往返验证；**vocab 必须来自被服务的模型**，`pre` 只支持 `qwen35` / `deepseek*` |
| **只换 `build\bin\ggml-cpu.dll` 做 A/B** | 把"重新构建"从测量里去掉，成对交替后噪声底比改源码重编小得多 |

### 4.2 会改变行为的开关（不是仪器）

| 开关 | 作用 |
| --- | --- |
| `STRATA_VK_F16ACC=1` | 建 f16-accumulate 的**分组**（`MUL_MAT_ID`）协同矩阵管线；该内核已被 oracle 认证，但它不是默认路径，整模型在 dense 一侧仍有回落问题（`PERFORMANCE-LOG.md` §4.5） |
| `STRATA_VK_F16ACC_DENSE=1` | 连 dense 的协同矩阵管线一起建；dense 那份 shader 会把 `-inf` 写进每个输出的第一个 BM tile，**给做 staging 的人做 A/B，不是给用户用的** |
| `STRATA_VK_F16ACC_NOID=1` | 只关分组那一半，用来隔离两臂 |
| `STRATA_VK_F16=off` | 强制全机 fp32 shader（等价 `--shader-family fp32`）；**它不是预填捷径**，留作诊断 |
| `STRATA_VK_MM_TILE` / `MMQ_TILE` | 浮点 / 量化矩阵乘的每线程输出瓦片，默认 `8x4` / `8x8`（这两个默认值是实测最优，见 `PERFORMANCE.md`） |
| `STRATA_VK_MAX_MATVEC_COLS=N` | 多少列以上让位给 mat-mat；默认 8。`ne1 == 1` 有硬编码捷径仍走 matvec |
| `STRATA_VK_STD_RM_MULT=N` | 标准量化 matvec 的每工作组行数倍数；默认值已是最优点，**加大到 12 会明显变慢** |
| `STRATA_VK_TQ_RM_MULT` / `_ID` | trellis matvec 的行数倍数；默认 20 之后基本饱和 |
| `STRATA_VK_REPACK_Q4=<MB>` | q2_0 → q4_0 重打包缓存的内存预算，默认 **512 MB**；设 `0` 关闭整条重打包。q4_0 的整数点积内核在这台机器上更快，所以预填宽形状时回本、解码步没有张量够格时零成本 |
| `STRATA_VK_REPACK_MIN_N=<N>` | 参与重打包的最小列数，默认 1024（低于它的形状重打包不划算）|
| `STRATA_VK_REPACK_RESIDENT_PCT=<pct>` | 允许被重打包副本占用的显存百分比，默认 90，夹在 10-95 之间 |
| `STRATA_VK_KV_UNIFIED=1` | 打开 `kv_unified`；实测第二个序列直接 NaN，**负结果**，保留只为下次不用重写 |
| `STRATA_VK_PREFILL_CHUNK` / `STRATA_VK_NO_WARMUP` / `STRATA_VK_NO_COMPANION` / `STRATA_VK_COMPANION_DIR` | 引擎侧：预填分块、跳过预热、关闭/移动 f16 companion 缓存 |
| 屏障回退 | 默认是"按缓冲收窄 + 按缓冲去重 + 空脏表跳过 + split-K 事件对"。`STRATA_VK_NO_BARRIER=1` 直接跳过屏障（**会读到陈旧数据，只用于给依赖本身定价**）；`STRATA_VK_NO_BARRIER_DEDUPE=1` 关掉去重；`STRATA_VK_BARRIER_GLOBAL=1` 退回全局内存屏障；`STRATA_VK_BARRIER_EMPTY_OFF=1` 关掉空脏表跳过；`STRATA_VK_BARRIER_EXEC_ONLY=1` 只保留执行依赖；`STRATA_VK_BARRIER_WIDE=1` / `_WIDE_STAGE=1` 退回宽屏障；`STRATA_VK_SPLIT_EVENT_BARRIER=1` 让 split-K 退回单屏障；`STRATA_VK_PREALLOC_COARSE=1` 让预分配标志退回粗粒度清除。**这些只用于 A/B 与二分，默认值才是实测更快的那个** |
| `STRATA_VK_INTDOT_OFF=1` | 关整数点积（`GGML_VK_DISABLE_INTEGER_DOT_PRODUCT`）做二分；**它会让 q8_0 的 mat-mat 回落出错**，见 `KNOWN-ISSUES.md` |

### 4.3 只打印、不改变结果的诊断

| 开关 | 打印什么 |
| --- | --- |
| `STRATA_VK_TIME=1` | GPU 侧时间桶（record / submit / wait / dispatch / barrier / fusion / end） |
| `STRATA_VK_PIPE_LOG=1` | 逐节点标记 + 逐 dispatch 一行（pipeline 名 + 工作组数）；`node-count.py`、`attribute_pipes.py`、`zip-node-pipe.py` 读它 |
| `STRATA_VK_FUSEDBG=1` | 每次选中的融合名，以及每个被拒的 `MUL_MAT+ADD` 及原因 |
| `STRATA_VK_COOPDBG=1` / `COOPMAT_LOG=1` / `COOPMAT_DUMP=1` | 协同矩阵的门禁与形状判定 |
| `STRATA_VK_EVAL_GRAPH=1` | 只走图不读回，打印每个节点及其输入（静态检查图结构） |
| `STRATA_VK_ADMIT_DEBUG=1` | 准入处与首 token 处的 top-1/top-2 采样行 |
| `STRATA_VK_SEG_LOG=1` / `VERBOSE=1` / `BACKTRACE=1` | 分段日志、详细日志、崩溃回溯 |
| `STRATA_VK_REPACK_SELFTEST=1` / `ROUNDTRIP_SELFTEST=1` | 不需要模型的自检，秒级：前者比对 q2_0→q4_0 重打包与 CPU 的块结果，后者量一次主机往返（下/上行各 40 KB）的提交与同步耗时 |
| `STRATA_VK_RDC*` | RenderDoc 挂载；本驱动上未验证过抓帧 |

### 4.4 引擎侧开关（与后端无关）

| 开关 | 作用 |
| --- | --- |
| `STRATA_VK_GPU` | 压过设备选择 |
| `STRATA_VK_MTP` / `_N` | 打开 MTP 起草并设定草稿长度；默认关闭，实测本包上更慢 |
| `STRATA_VK_LOOKUP` / `_NM` / `_DEBUG` | 前缀复用（lookup）开关与调试输出 |
| `STRATA_VK_EXPERT_SPLIT` / `_ROW_SPLIT` / `_ROW_SPLIT_TENSORS` | 专家/行切分放置的实验开关；默认关闭 |
| `STRATA_VK_NPU_FORCE` / `_MBUF` / `_VMEM` | Hexagon NPU 放置；本驱动上开不了会话（`KNOWN-ISSUES.md`）|
| `STRATA_VK_PERF_CAP` / `STRATA_VK_PROFILE` | 采样与分段计时 |
| `STRATA_CPU_MASK` / `_THREADS` | 压过 CPU 后端的大核亲和与线程数 |
| `STRATA_VK_EVAL` / `_EVAL_GRAPH` | 只求值 / 只走图不读回，静态检查用 |

其余（`PACEDBG`、`ADRENO_TUNING`、`RDC_MB`、`RDC_PATH` 等）是开发过程中的临时调试项，保留在代码里但不作为文档承诺。

### 4.5 批量的当前结论

`--parallel N` / `--batch-slots N` 让 N 个请求共用一份权重，**正确性已验证**（32 槽时槽 0 的 token 流与单独运行
逐 token 相同）。但**不要引用任何历史加速比**：早先的 4.27× 是坏内核的读数，已撤回；随后一次 32 槽测量是
**约 1.00×**（每步 ≈ 32 × 单条时间）。当前能说的只有"协议与正确性成立，吞吐收益未获证实"。
历史曲线见 `PERFORMANCE-LOG.md`，不要复制到别处。

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
#     `dev/cpu` 比值应落在 0.98-1.00；全零或 nmse 1.0 就是 MMA 循环没跑（PERFORMANCE-LOG.md §4.4/§4.5）
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
