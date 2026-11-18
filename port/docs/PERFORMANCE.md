# Performance — 本机实测、已进入代码的改动、以及不要再试的方向

**本机量到的、当前有效的性能结论。** 没有本机测量的数字不写进来；是假设的地方会标出来。
Every number here comes from a reproducible run on this machine; hypotheses are labelled.

- 机器：Snapdragon X2 Elite Extreme（18 个 Oryon 核，47.6 GB 统一内存，三通道 LPDDR5X 标称 **228 GB/s**，
  Adreno X2-90，Vulkan 1.4，驱动 `30.0.228.10000`），Windows 11 ARM64。
- 目标包：`Gyro-S-TQ1_0.gguf`（54.47 GiB，`qwen4exp`，48 层，512 专家 top-10，d=2560，trellis 专家 24.02 GiB）。
- 方法：同一二进制、成对交替、取中位数；**绝对读数不要跨时段比**——同一二进制同一天的 decode 在
  10.1-11.3、pp512 在 43-50 tok/s 之间漂移（连续 GPU 工作之后偏低）。引用本页数字按 ±10% 读。
- 工具与开关：[`USAGE.md`](USAGE.md)。设备事实与逐项 Vulkan 测量：[`ADRENO.md`](ADRENO.md)。

---

## 1. 当前数字 / Current numbers

| 项目 | 实测 | 量法 |
| --- | --- | --- |
| **Vulkan 预填 pp512** | **85.55 tok/s** | 默认设置（2026-11-19 之后；同一二进制 `STRATA_VK_F16ACC=0` 是 42.51，配对同日） |
| Vulkan 预填，同一构建改之前 | 16.01 | mat-mat 从 fp16 模块取 shader 时 |
| Vulkan 预填，`--shader-family fp32` | 46.6 | 全部 shader 走 fp32，decode 掉到 5.5 |
| Vulkan 预填，分组 cooperative 关掉 | 42.51 | `STRATA_VK_F16ACC=0`，同一二进制 |
| Vulkan 预填，dense cooperative 也开 | 80.98 | `STRATA_VK_F16ACC_DENSE=1`，比默认慢 5% |
| **Vulkan 解码，单流** | 10.25 tok/s | `--bench-tg 16`；37.9 GB/s 权重流量 |
| **Vulkan 解码，32 槽** | **31.09 tok/s 聚合**（2.4× 单流） | `--batch-slots 32 --bench-tg 16`；同一二进制 `F16ACC=0` 是 25.47（+22%） |
| **Vulkan 解码，64 槽** | **33.62 / 36.65 / 36.73 tok/s 聚合**（3.3-3.6× 单流，三次运行） | `--batch-slots 64 --bench-tg 16`；96 槽装不下（§5.3） |
| **Vulkan 32 槽 + CPU 进程** | 29.1 tok/s 聚合（旧记录，未在新内核上复测） | 两个进程同时跑 |
| CPU 解码，单流 | 10.5-11.3 | `--gpu cpu`，大核类 11 线程 |
| CPU 预填 pp512 | 21.06 | 同上 |
| CPU 解码，`-t 12` / `-t 13` | 10.4 / 9.7 | 默认 11 线程已是最优 |

**每 token 的字节**（`gguf-roles.py` 从 GGUF 头算）：**3.598 GiB = 3.863 GB**。稠密 3.129 GiB（87%），专家
10/512 = 0.469 GiB。稠密里最大的是 `attn_qkv` 619 MiB、`output` 497、`attn_gate` 371、`ssm_out` 371、
`attn_q` 248、`hc_*up` 319（Q8_0）、`hc_*down` 246（Q6_K）、`ffn_gate_inp` 120（BF16）。
`per_layer_token_embd`（26.83 GiB IQ4_NL）与 `token_embd` 是 GET_ROWS，每 token 约 1.4 KiB，**与字节无关**。

## 2. 一个 decode token 的时间去向 / Where a token's time goes

### 2.1 两个后端的瓶颈完全不同

同一包、同一份 `--tokens 9707 --n 16`：

| 后端 | 最大的一族 | 占比 | 量法 |
| --- | --- | --- | --- |
| CPU | `MUL_MAT_ID`（trellis 专家 matvec） | **81.3%**（1470.7/1809.9 ms） | `STRATA_CPU_TIME=1` |
| CPU | `MUL_MAT`（稠密 matvec） | 9.7% | 同上 |
| GPU | `tqk` 专家 matvec | **52%**（813/1557 ms） | `STRATA_VK_SKIP=tqk` 消融 |
| GPU | `q5_K` / `q6_K` 稠密 matvec | 20% / 9% | 同上 |

GPU 侧另三族（`multi_add`、`glu`、`concat`、`dsv4_hc`）各自 **≤6 ms**，是噪声 ⇒ **不要按"减少 glue dispatch"排期**。

- **CPU 的 trellis 是 1.78 指令/权重**（`ggml_vec_dot_tq_q8_0_impl`，NEON 分支已存在）：2.36 G 活跃权重
  × 1.78 = 4.2 G 指令 ÷ 11 线程 ÷ 3.07 GHz × 1.4 IPC ≈ 1.42 s ≈ 实测 1.47 s。⇒ CPU 是**指令**受限，
  0.469 GiB / 87 ms = 5.8 GB/s。
- **GPU 的 trellis 是延迟/指令受限**：`mul_mat_vec_id_tqk6_f32` 的网格是 `32x10x1` = 320 个 workgroup，
  每 lane 沿一行串行推 6-bit 状态机、每步一次共享 LUT 读。`rm_tq`（每 workgroup 行数）已扫到 20 最优，
  所以换粒度没用。

### 2.2 带宽：单引擎到不了 228，双引擎也到不了

`membw.cpp` 现在扫线程数与每线程 unroll 深度，用来区分"请求并发不足"和"内存系统效率不足"：

| 线程 | 2 | 3 | 4 | 6 | 8 | 10 | 12 | **14** | 18 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GB/s | 101 | 126 | 147 | 149 | 140 | 147 | 147 | **154.6** | 149 |

深度 4/8/16 在 6/12/18 线程下是 141/134/137、129/111/111、133/119/119 —— **加深 unroll 无益**。
⇒ **4 线程就饱和（147）**，峰值 154.6（14 线程，标称值的 **68%**），18 线程倒赔 4%。
GPU 流式约 108 GB/s，两引擎同时 **177 GB/s（78%）**。
剩下的 22-34% 是这块 UMA 器件的内存读取效率，**任何内核都拿不到**。

**decode 在墙上的位置**：单流 37.7 GB/s（GPU 上限的 35%）、16 槽 58.4、**32 槽 103.1（95%）**。
⇒ 批量再摊销也换不到吞吐；更高的数字只能从 CPU 一侧来（§5）。

### 2.3 常驻集与缺页：只值 1-2%

`Win32_PerfRawData_PerfOS_Memory` 在 `--bench-tg 96` 里按 2 秒差值采样：`PagesInputPerSec` 在 t≈30 s 是
137 624/s（**538 MB/s**，首次填充 27 GiB 常驻集），t≈60 s 降到 12 378/s（48 MB/s），而 `PagesOutputPerSec`
全程 0-142/s。这是映射文件页的签名（页被从 standby 链直接丢弃、不写盘）。稳态 48 MB/s = 4.6 MB/token
≈ 1 182 次软缺页 ≈ **1-2 ms of ~100 ms**。⇒ 缺页解释不了"模型内比隔离慢"，那只发生在预热阶段；
**KV 未被换出**（`pages-out≈0`），不需要处理。

## 3. 已经进入代码、并且量到收益的改动 / Shipped, measured

| 改动 | 收益 | 在哪 |
| --- | --- | --- |
| **int-dot 的 mat-mat 从 fp32 模块取 shader**（mat-vec 与 elementwise 仍留 fp16 模块） | pp512 **16.01 → 43-50（+211%）**，decode 与 bench prompt −48% 不变 | `ggml-vulkan.cpp` `ggml_vk_load_shaders` 的三处 `sg_create_mmq` |
| CPU 线程放进**大核类** | decode 7.8-8.3 → **11.8-12.1 tok/s（+34%）** | `vk-devices.cpp` `plan_cpu_affinity` |
| CPU 的 repack 门禁按**实际拷贝的字节**算 | CPU prefill **+13%** | `vk-engine-load.cpp` |
| CPU prefill 不用全部 18 核 | prefill +5.5% | `plan_cpu_affinity` |
| Vulkan **int-dot（dp4a）** mat-mat 路径 | dense 包 prefill 数倍 | `decide_policy` |
| flash-attn：`n_ctx ≥ 4096` 且 ≥32 层默认开 | 2048 上下文 4.5× prefill | `vk-engine-load.cpp` |
| Vulkan prefill ubatch = 1 024 | 1 024 比 512 快 25% | `kDefaultBatch` |
| KV `q4_0` + `n_ctx = 262 144` | KV 显存 8× 小 | `--kv q4_0 --max-context 262144` |
| PLE 表 26.8 GiB **本来就是 host-only** | 释放约 27 GiB 常驻假设 | `llama-ple-disk.cpp` |
| 专家驻留：host 专家 + 设备侧 cache | MoE 在 47.6 GiB 上可跑 | `--moe-cache-size` |
| 采样器链每请求一条 | 正确性 | `vk-engine.cpp` `make_sampler` |
| `--parallel N` / `--batch-slots N` 请求槽 | **正确且有收益：32 槽 31.09 / 64 槽 33.62 tok/s 聚合**；两个旗标都能建槽（原先只有 `--parallel` 能，serve 因此广告 0 槽） | `vk-serve.cpp`、`vk-engine-load.cpp` |
| **companion 包合成**（`--compat FILE` + `strata.companion`） | 逐位不变的往返已验证 | `llama-model-loader.cpp` 三处、`vk-options.*`、`port/tools/gguf-companion.py` |
| **`requant-tensor`** 工具 | 秒级产出重打包 companion | `port/probe/requant-tensor.cpp` |

## 4. 负结果、以及被撤回的结论 / Ruled out, and what was withdrawn

### 4.1 不要再试的方向

| 方向 | 判定 | 数字 / 证据 |
| --- | --- | --- |
| **NPU 放置**（Hexagon HTP） | 关闭 | 每种放置都比 GPU 慢（27B decode 4.1-4.7 对 7.3；35B 10-14 对 39） |
| **raw FastRPC 开 cdsp 用户 PD** | 关闭 | `remote_handle64_open` 返回 `AEE_ERPC 0x200`，三种形状 |
| **f16-accumulate coopmat** | **默认开**（trellis 包） | 2026-11-19 之前关闭，因为分组那份写 `-inf`；根因是 trellis 类型没有 `MmTypeA` 分支、`buf_a` 从不写入（§4.2）。修好后 pp512 +101%、32 槽 +22%，质量门禁全过 |
| **`STRATA_VK_INT8_COOPMAT=1`** | 空操作 | 见 §4.2：管线从未被创建 |
| **host 指针 wrap / `--load-mode import` / 两个 context 共享权重** | 关闭 | 导入预算约 3 GiB；`direct` 峰值 59.25 GiB；同后端可共享但零增益 |
| **静态 CPU/GPU 层切分**、**token 级 CPU+GPU 并行** | 净损 | 层间串行；两进程 +33% 吞吐但 2.44× J/token；`--cpu-moe` −83% |
| **host 专家 + `--moe-cache-cpu-share`** | **5× 亏损** | `--n-cpu-moe 48 --moe-cache-size 4096`：decode 2.0 tok/s（基线 10.2），share 0.5 仍 2.1 |
| **MTP / NextN**、**n-gram lookup** | 负 / 中性 | 这个包 −31%；lookup 每次 tg16 在噪声内 |
| **准入的 prefill 切片** | 净亏 | 一个 prefill pass 就要把 3.06 GiB 稠密权重流一遍 |
| **把稠密张量降位宽**（companion 重打包） | **破输出** | attention 组 Q5_K→Q4_K：corr **0.978**、第 3 个 token 就变；"存得过厚"那组（`hc_*_up` Q8_0、`hc_*_down`/`output` Q6_K）降到 Q5_0/Q5_K：corr **0.991**、第 7 个 token 变 |
| **`--moe-cache-size` 调大**、**`-t 18`** | 中性/负 | 见 §1 与 §3 |
| **进程内存优先级** | 不存在 | `MEMORY_PRIORITY_*` 那组常量在 SDK 10.0.26100 里没有；文档标度只到 `NORMAL = 5`，而新进程默认就是 5 |
| **用 `mm-probe` 测 trellis** | 不可用 | `mm-probe -t tqk6` 在三种形状下都段错误（0xC0000005） |

### 4.2 cooperative matrix：根因是 trellis 类型没有 `MmTypeA` 分支（2026-11-19 修好）

**三个阶段，最后一个才是对的。**

**第一阶段（到 2026-11-18 之前）**：f16-accumulate 报 pp512 **221 tok/s**、logits 与 CPU 相关 **0.02**。
当时的判定是"f16 部分和 + 饱和，设计内的近似"。**这个机制是错的**：那颗 tile 的 `WN = 32` 而 `TN = 64`，
`cms_per_col = WN/TN` 整除成 0（`mul_mmq_cm1.comp:221-222`），于是 MMA 循环、epilogue 与 store **全部编译成空**
——它只做 staging。`mm-probe --verify` 的全零、0.02 的相关性、"又熟又省电"的读数都是同一件事。

**第二阶段（tile 修好之后）**：按本机矩阵形状重建 tile（`{BLOCK_SIZE=64, BM=64, BN=64, BK=32, WM=64, WN=64,
WMITER=1, TM=c_m, TN=c_n, TK=c_k, WARP=64}`，共享 26 624 B）后 MMA 真的跑：`mm-probe --verify` 每元素
dev/cpu 比值 **0.983-1.003**，pp512 **143.8 tok/s**。但整模型仍非有限。用 `STRATA_VK_F16ACC_NOID=1`
把两半分开：

| 组合 | 有限 | corr vs CPU | argmax |
| --- | --- | --- | --- |
| 完全不用 coop | True | 0.998352 | 77/77 ✓ |
| **dense 用 coop、分组不用** | **True** | **0.998420** | **77/77 ✓** |
| 两者都用 | False | nan | — |

⇒ 分组（`MUL_MAT_ID`）那份是 `-inf` 的来源。`GGML_VK_NAN_SCAN=*` 定位：坏值不是 nan 而是 `-inf`，
恰好是每张量**开头 64 个元素**（一个 `BM` tile = 一个 workgroup 的输出），第一个坏节点是 `blk.3` 的输入残差。

**第三阶段（2026-11-19，根因）**：不是 tile、不是 staging 的地址算术，**是 shader 选错了**。量化 matmul 有两种
SPIR-V：legacy/K-quant 共用一个按 `MmTypeA` 特化常量分派的 shader，而 **trellis 与 LUT 类型在那个 shader 里
没有分支**（`mul_mm_funcs.glsl` 的 `MmTypeA` 链只有 Q1_0/Q2_0/Q2_K/Q3_K/Q4_0/Q4_1/Q4_K/Q5_0/Q5_1/Q5_K/Q6_K/Q8_0/
TQ1_0/TQ2_0），它们只以**按类型编译的 SPIR-V** 存在（`matmul_tqk6_f16_f16acc_cm1` 等）。而创建代码把
`matmul_quant_f16_f16acc_cm1`（通用 shader）挂给了包括 TQK6/TQK7/TQ2_T/PTQ1_0 在内的每一个类型
（`ggml-vulkan.cpp` 的 `cmh_types`）。对 trellis 类型，那个 shader 的 `MmTypeA` 链**一个分支都不命中**，
`buf_a` 从不写入 ⇒ MMA 读未初始化的共享内存 ⇒ `-inf`。

为什么 oracle 一直是绿的：`test-backend-ops` 的 tqk6 用例是 `k=128`，低于 cooperative 的 `k >= 256` 形状门，
从未碰到这些管线；而 q4_0 的 `k=2880` 用例走的是通用 shader，那条链有 q4_0 分支。

修法：把 `cmh_types` 拆成"通用 shader 覆盖的类型"与"自带 SPIR-V 的类型"两组，后者用
`matmul_<type>_f16_f16acc_cm1` / `matmul_id_subgroup_<type>_f16_f16acc_cm1`，并改用不带 `MmTypeA` 的
`ggml_vk_mul_mm_spec`。同时把 TQK6/TQK7 加进 `read_model_facts` 的 trellis 计数（原来只数 TQ2_T/PTQ1_0/
TQ1_0/TQ2_0，Gyro-S 因此报"0 trellis"），策略按 `trellis > 0` 打开 cooperative，并删掉 host-expert
路径上"强制 STRATA_VK_F16ACC=0"的 workaround（它的 NaN 与这里是同一个根因）。

**修好后的实测（同一二进制，同日配对）：**

| 项目 | cooperative 开 | 关 | 差 |
| --- | --- | --- | --- |
| pp512 | **85.55 tok/s** | 42.51 | **+101%** |
| tg16 单流 | 10.25 | 10.19 | 中性 |
| 32 槽聚合 | **31.09** | 25.47 | **+22%** |
| 9-token prompt | 1192 ms | 1743 ms | −32% |

质量门禁：`--tokens 248045,... --n 1` 的 CPU vs Vulkan logits **corr 0.998239、argmax 77/77 相同**；
`--tokens 9707 --n 32` 的 32 个 token **与参考流逐 token 相同**；`test-backend-ops -o MUL_MAT` 2648/2648，
`-o MUL_MAT_ID`（`STRATA_VK_F16ACC=1`）1168/1168。dense cooperative 仍是净损失（80.98 对 85.55），保持关闭。

**未解决**：prefill 的 tile 浪费。512 token 分到 512 个专家 ⇒ 每个专家约 10 行，而 cooperative tile 的 N 是
64 行，**约 84% 的 MMA 行是空的**。设备上只有 64x64 与 32x32 两种形状（`ADRENO.md`），32x32 慢 57-77%，
所以这不是换 tile 能解决的；要做得多专家打包进一个 tile，需要另一种权重布局。这是 prefill 下一个结构性目标。

## 5. 还剩什么 / What is left

**2026-11-19 更新**：下面这张并发表是 dp4a 内核上测的。打开 cooperative 分组路径之后 32 槽从 25.47 到
**31.09**，所以"批量已经到墙上"这句话要按内核算——它当时成立是因为 dp4a 的专家 matmul 把 GPU 时间花在
指令上。表里的 CPU 并发数（+9%）没有在新内核上复测，仍是待办。

### 5.1 prefill 的钱花在哪（`STRATA_VK_SKIP` 消融，pp512，同一二进制）

| 跳过 | pp512 | 该类占用 |
| --- | --- | --- |
| 不跳（基线） | 85.55（5.98 s） | - |
| `matmul_id_coop`（专家 matmul） | 180.10（2.84 s） | **3.14 s = 52%** |
| `matmul`（全部 matmul） | 658.11（0.78 s） | 5.21 s = 87% ⇒ 稠密 matmul **2.07 s = 35%** |
| `lightning`（indexer） | 84.68（6.05 s） | ~0 |
| `flash_attn` | 41.15（12.44 s） | **不可用**：跳过它反而慢一倍，就是文档警告的"垃圾缓冲 → denormal → 整图变慢" |

所以 prefill = **52% 专家 matmul + 35% 稠密 matmul + 13% 其它**。专家 matmul 那 3.14 s 做了 655 GFLOP：
**209 GFLOP/s = cooperative 峰值（2700）的 8%**。

为什么只有 8%：tile 是 64 列 token，而 512 token 分到 512 个专家后**每个专家只有约 10 行**，于是 84% 的
MMA 列是空的。这是**阶跃**浪费——只要 rows/expert ≤ 64 就一直接近 84%，超过 64 才变好，所以
pp1024/pp2048 都是中性（89.20 / 83.36 对 85.55）：20 行还是 1 个 tile，浪费不变。要跨过 64 行需要
约 3.3k token 的 ubatch，而 ubatch 1024 已被证明是最优（更大反而慢）。

prefill 曲线（同一二进制，cooperative 默认开）：pp256 **72.84**、pp512 **85.55**、pp1024 **89.20**、
pp2048 **83.36**。

### 5.2 下一个结构性目标：分组 matmul 的窄 tile —— **两条路都测过，关了**

原计划是"用 dp4a 的自由 BN 换掉 cooperative 的 64 列 tile"。2026-11-19 把两条路都试到结论：

1. **trellis 类型没有 int8 dp4a matmul shader。** 生成器只为 `is_legacy_quant || is_k_quant || mxfp4 ||
   iq3_s || iq4_xs` 编译 `mul_mmq.comp`（`vulkan-shaders-gen.cpp:635`），而 trellis 的生成符号表里只有
   `matmul[_id][_subgroup]_tqk6_{f16,f32}[_f16acc][_cm1]`，**没有 `_q8_1`**。窄 tile 只能挂在 dp4a 上，
   所以这条路要先给 trellis 补一颗 int8 matmul 内核（A staging 走 `dequant_funcs.glsl` 的 TQK 解码），
   不是改 tile 常量。收益按 §5.1 的算式也只有 +10-14%（in-graph dp4a 实测 444 GFLOP/s issued，
   不是隔离峰值的 1182），性价比不成立。
2. **32x32 cooperative 矩阵这颗驱动不接受。** 加一颗
   `{BLOCK_SIZE=64, BM=64, BN=32, BK=32, WM=32, WN=32, WMITER=1, TM=32, TN=32, TK=c_k, WARP=64}`
   （`TM == 2*CM_ELEMS` 与 `WARP == 2*TN` 都成立，store 路径不变），按 `rows_per_expert < 32` 选它：
   `createComputePipeline` 直接 `ErrorUnknown`，编译日志是
   `Invalid mask for Layout[] ... @llvm.qgpu.coopmat.layout.annotate.v64f16(...) layout value: i32 17`。
   这台设备的 KHR_coopmat 只有 **64x64** 一种可用形状；ADRENO.md 里"32x32 慢 57-77%"那句说的是别的配置，
   不是一颗真的 32x32 矩阵。已回滚，回滚后 oracle 1168/1168。

**把两个阶段都校准到隔离速率（2026-11-19，`mm-probe`）：**

| 阶段 | ubatch 内的活 | 耗时 | 有效率 | 隔离峰值 |
| --- | --- | --- | --- | --- |
| 稠密 matmul（35%） | 512 token × 3.5 G 权重 = 3.6 TFLOP | 2.07 s | **1740 GFLOP/s = 隔离的 83%** | 2103（q5_K m=10240 n=512 k=2560） |
| 专家 matmul（52%） | 512 × 640 M = 0.66 TFLOP | 3.14 s | 210 GFLOP/s = 隔离的 8% | 2700 |

⇒ 稠密那一半基本没有余量（83% 已经很好）。专家那一半的 8% 是 **MMA 峰值**的 8%，但那个峰值不是它的地板：
专家 kernel 先把权重解码成 f16 再进 MMA，而解码速率已经被单流 decode 那条线标定过——
**~47.6 G 权重/s**（0.503 GB / 49.5 ms，1.708 bit/weight）。一个 512-token ubatch 会碰到全部 512 个专家，
即 120.8 G 权重，**解码地板 = 2.54 s**，实测 3.14 s ⇒ 专家阶段已经在解码地板的 **81%** 上。

> **更正（2026-11-19，本文上一版写错过）**：上一版说"tile 填满会让专家阶段 3.14 → 0.49 s、pp512 到
> ~154 tok/s"。那是把 MMA 峰值当成了地板，忽略了 staging 的解码。按解码地板算，完美的 tile 最多把
> 专家阶段压到 2.54 s，pp512 **85 → ~95 tok/s（+11%）**。所以 prefill 的余量不是 1.8 倍而是 1.12 倍，
> 而且它和解码共享同一堵墙：trellis 解码 47.6 G 权重/s。

⇒ **prefill 的 84% 空列在这台设备上无解**：矩阵核不能变窄，dp4a 没有 trellis 内核，ubatch 又不能跨过
64 行（那要 ~3.3k token）。但它也不再是主要余量——按解码地板算只值 +11%。85-95 tok/s 就是这套内核的上限，
而真正的天花板是**解码速率本身**：prefill 的专家阶段（81%）、64 槽的聚合吞吐（~10 GB/s 专家字节）、
单流 decode（49.5 ms / 0.503 GB）都在同一个数上收敛。要再上去只有两条路，都不是"调内核"：
换一种让解码更便宜的权重格式，或者换 decode 策略（并发已经用尽）。

### 5.3 多槽并发的曲线（cooperative 路径，2026-11-19 重测）

`--batch-slots N --bench-tg 16`，同一二进制：

| 槽 | 聚合 tok/s | 每步 | 备注 |
| --- | --- | --- | --- |
| 8 | 8.89 | 900 ms | 8 是坑（matvec 上限 `mul_mat_vec_max_cols`） |
| 16 | 18.45 | **867 ms** | 每步最快的一档 |
| 24 | 25.05 | 958 ms | |
| 32 | 30.51 | 1049 ms | 旧记录的"最优"（dp4a 时代 26.7） |
| 48 | 32.24 | 1489 ms | dp4a 时代这里是下陷 |
| **64** | **33.62 / 36.65** | 1903 / 1746 ms | 新的最优点（两次运行），比 32 槽 **+10-20%**、比单流 **3.3-3.6x** |
| 96 / 128 | — | — | `vk::Device::allocateMemory: ErrorInitializationFailed`（47.6 GiB 上装不下） |

⇒ 曲线在 cooperative 路径上**单调到 64 槽**，不再有 dp4a 时代的 48 槽下陷；96 槽起是内存装不下，不是
性能问题。每步时间从 16 槽的 867 ms 涨到 64 槽的 1903 ms（2.2x），聚合只涨 1.8x，所以 64 槽之后的
边际收益已经很薄。

**serve 路径的修复（2026-11-19）**：`--parallel N` 与 `--batch-slots N` 本来是同一套槽机制，但只有
`--parallel` 会创建槽（`vk-engine-load.cpp`），于是 `--batch-slots 64 --serve` 向 `serve/server.py`
广告 `batch_slots=0`，多槽服务根本走不到；那段注释还写着"槽不买吞吐"（cooperative 之前的老结论）。
现在两个旗标取 `max(parallel, slots)` 都能建槽，`batch-serve.py --slots 64 --engine --batch-slots 64`
实测 **slot 0 的 8 个 token 与单流逐 token 相同**，引擎广告 `batch_slots=64`。

注意 `batch-serve.py` 报的是**含准入**的突发速率（12.17 tok/s，"1-7 per slot"），和 `--bench-tg` 的
稳态每步速率（33.62）不是同一个量：前者受逐个槽的 prompt 读入与准入顺序支配。要比吞吐就看
`--bench-tg`；要比正确性就看 batch-serve 的 `token identical` 那一行。

### 5.4 decode：50% / 38% / 12%（`STRATA_VK_SKIP` 消融，tg16，同一二进制）

| 跳过 | decode | 每 token | 该类占用 |
| --- | --- | --- | --- |
| 不跳（基线） | 10.13 | 98.7 ms | - |
| `mul_mat_vec_id_tqk`（trellis 专家 matvec） | **20.31** | 49.2 ms | **49.5 ms = 50%** |
| `mul_mat_vec`（全部 matvec） | 81.99 | 12.2 ms | 86.5 ms = 88% ⇒ 稠密 matvec **38 ms = 38%** |
| `mul_mat_vec_q5_k`（单个最大的稠密张量） | 12.67 | 78.9 ms | 19.8 ms = 20% |

两个结论：

* **稠密那一半已经接近墙**：38 ms 搬 3.4 GB = **89 GB/s**，是 GPU 流式上限（108）的 82%。
* **trellis 那一半是全部的问题**：49.5 ms 搬 0.503 GB = **10.2 GB/s，上限的 9.4%**。按带宽它只该值
  6-9 ms（含 26 B 块的 sector 放大 1.2-2x），所以它不是带宽受限，是**延迟 / 寄存器受限**：
  每行要 hold `x0..x3` 四个 vec4 加四个累加器，20 行展开需要约 400 个寄存器，编译器必然缩展开深度；
  同时 8 KiB 的共享 LUT 把驻留压在每 CU 4 个工作组（ADRENO：32 KiB/CU ÷ 8 KiB）。

**共享 LUT 的驻留假设也测掉了（2026-11-19）。** `tq2_t_lut.glsl` 里本来就有个只用于测量的
`TQ_LUT_PAD4K`（再加 4 KiB 共享，把每 CU 的驻留从 4 个工作组压到 2 个）。把它接到 trellis 类型的
matvec 变体上重新编译后测 tg16：**10.13 对 10.13，完全中性**。⇒ 驻留不是瓶颈，原计划的"把 LUT 搬到
全局只读 buffer 换 occupancy"因此**撤回**，不做。

留下来的数字是：每个 warp 大约 13 个周期才完成一个 step（4 个权重），而一份 step 只有 2 次共享取数
+ 1 次乘法 + 8 次 FMA；同一线程手里本该有 20 行 × 4 step = 80 条独立链。所以问题不在 occupancy、
不在共享带宽、也不在 rows/WG，而在**编译器没有把 `accumulate_tqk` 展开到能重叠这些链**（每行要
4 个 u16 load + 4 个 vec4 + 4 个累加器，20 行就是 600 多条指令，`[[unroll]]` 很可能被忽略）。
下一步本来打算重写这个循环（显式 2-4 行一组的手工展开 + 缩短每行的活跃寄存器）。**2026-11-19 做了，
是负结果，见下。**

**2026-11-19 补测：把最后三个假设也测掉了。**

* **不是取数受限。** 把 4 个权重字和 `d` 换成由 `ib` 算出来的值（同样的依赖形状、同样的 ALU，但没有内存读）
  重编后测：**1.698 ms 对 1.70 ms**，取数完全不影响时间。
* **4 行手工展开没有收益（已回滚）。** `num_rows` 是特化常量的表达式、`NUM_ROWS` 是特化常量，两者都不是
  编译期常量，所以 `[[unroll]]` 从来没生效过；改成"外层 4 行一组、内层字面量 4 次展开"之后成对换
  `ggml-vulkan.dll`、5 组交替、每组 50 次迭代：m=2560 **1.602 对 1.606（中性）**、m=640
  **0.527 对 0.462（慢 14%）**。分组把那两行之间的依赖拆散了，但活跃寄存器翻倍，m=640 直接吃亏。
* **matvec 阈值（`ids->ne[1] <= 8`）实测确认是对的。** 用 `STRATA_VK_MATMUL_ID_COLS_MAX=0` 把
  `MUL_MAT_ID` 全部赶到矩阵核路径（n=1 也走 64x64 tile）测隔离形状：**13.204 ms 对 1.546 ms，慢 8.5 倍**。
  1 列数据配 64 列的 MMA 再加 row-map 那两个额外 kernel，代价远超它省下的东西；阈值不用动。
* **select-free 的 state 提取是中性的（已回滚）。** 把那套"三个 32 位窗口 + 3 路 select"换成"每一步直接读它
  自己的两个 word"（多 3 次取数、少 4 条 ALU）——按取数免费的前提应该更快：成对换 DLL 后
  m=2560 **1.572 对 1.607（−2%）**、m=640 **0.507 对 0.498（+2%）**，中性。
  这个改动第一版写错了地址（`rel>>4` 少了 `wb`，只有 lane 0 碰巧对），单测看起来 **0.864 对 1.60 ms
  = 2.4x**，而 `test-backend-ops -o MUL_MAT` 立刻以 tqk6 `ERR 1.1-3.0` 判死。**"快 2.4 倍"的来源是错的地址
  让所有 lane/行重复读同几个 word（聚簇/广播），不是内核变好。** 修正地址后 oracle 2648/2648 全过、性能归零。
  这条比任何性能数字都值得记：先过 oracle，再谈快慢。
* **tile 选择器已经是最优**（pp512，`STRATA_VK_FORCE_TILE`）：s 73.67 / **m 85.13（默认）** / l 83.73 tok/s。
  稠密 matmul 在 n=512 时选 idx=1（BM=64 BN=64），换哪一档都更慢。
* **方法学教训**：上面两个改动在**单次**读数里都像"+8%"，配对之后一个是 0、一个是 −14%。低于 10% 的差异
  必须换 DLL 成对测，单跑不可信（§6 的纪律，这次抓到实例）。

⇒ 把 decode 的账合成一个数：单流（matvec，n=1）和 64 槽（cooperative mat-mat，n=64）都落在
**专家字节 ~10 GB/s**，也就是 trellis 解码 **~49 G 权重/s**——两个不同的内核、两种 tile、两种并行结构，
同一个数。取数、码本、驻留、行数、工作组、展开、tile 全部测平，所以这就是这颗 GPU 解这个格式的速度；
剩下的只能是换格式（1.63 bpw 已经是下限，§5.4 前面算过 scalar 重打包要 4-5 倍字节）或换 decode 策略。

`STRATA_VK_DMMV_WG=large`（128 线程/工作组，4 倍 occupancy）实测 **6.99 对 9.65 tok/s**，更差：把工作组
放大 4 倍让每线程只剩 1.25 个 item，ILP 掉光，hybrid 归约还多一层共享。`rm_tq_id` 的扫描也是尖峰：
8 → 8.74、**20 → 10.09**、40 → 8.37、80 → 4.73 tok/s（40 以上开始溢出寄存器）。

**隔离测量（2026-11-19，`mm-probe --n-as/--n-used`，单次 dispatch，秒级）：**

| 形状 | 时间 | 实际速率 |
| --- | --- | --- |
| tqk6 ID, m=2560 n=1 k=2560, 10/64 专家 | 1.55-1.82 ms | **7.4-8.7 GB/s** |
| **q4_0 ID, 同一形状** | **0.549 ms** | 67.5 GB/s |
| **q4_K ID, 同一形状** | **0.516 ms** | 71.9 GB/s |
| q8_0 / q6_K / q5_0 | 0.924 / 0.705 / 0.720 ms | |
| tqk7 ID, m=2560 | 1.63-2.17 ms | 7.2-9.6 GB/s |
| tqk6 ID, m=512（gate/up） | 0.504 ms | 5.5 GB/s |
| 专家栈 16 个 vs 512 个（16 MB vs 671 MB） | 1.72 vs 1.65 ms | **无差别 ⇒ 不是内存/缓存** |
| n=4 / n=8 token | 6.37 / 11.07 ms | ≈ 4x / 7x 线性 ⇒ matvec 按 token 重读权重 |

**三个把范围收窄到"格式本身"的结果（同一轮）：**

1. **LUT 只值 6%。** 把 `tq2_t_step` 的索引换成常量（编译器于是把两次共享取数整个提到循环外）重建后测：
   **1.643 ms 对 1.75 ms**。共享码本、bank 冲突、hash 乘法、`unpackHalf2x16` 全部加起来只有 6%——所以
   "优化 LUT" 这一族（全局 LUT、更小码本、预取）都不值得做。
2. **激活值本来就被提出了行循环。** 显式把那 4 个 `data_b_v4` 载入提到 row 循环外重建，时间
   **1.792 ms 对 1.792 ms**（已回滚该改动）：`data_b` 是 `readonly`，地址与 row 无关，编译器早就 CSE 了。
   所以剩下的 5 次取数全是权重取数。
3. **权重是 1.63 bit/weight，scalar 重打包这条路是死的。** 专家集 24.023 GiB 覆盖
   48 层 × 512 专家 × 3 个矩阵 × 2560×640 = **120.8 G 个权重** ⇒ **1.63 bpw**（TQK6 是"每 4 个权重
   K=6 个新 bit"，即 1.5 bpw）。任何 scalar 格式都是它的 3-5 倍字节：Q6_K 约 100 GiB、Q8_0 约 128 GiB。
   40 GiB 的 RAM 连 24 GiB 的 trellis 专家集都只是刚好放得下（这也是这个格式存在的理由），换成
   100-128 GiB 的 companion 会让每个 token 的 0.8 GiB 全部从磁盘缺页进来 —— 单流 decode 会从 10 掉到
   个位数。**"重打包成更宽/更规整的格式"到此关闭。**

**剩下的解释**：每 16 个权重 5 次取数、2.35 G 权重/token ⇒ 7.35 亿次取数 / 49.5 ms ≈ **14.8 G 次/s**，
已经压在这颗 GPU 的取数发射上限上；16 个权重 4 次取数是这个布局的底线（4 个字里的两个要跨块回绕，
块只有 2 字节对齐，64 位窗口读不了），要把 4 次再降下去得让每个 lane 的 4 个字 8 字节对齐 ——
那会把 24 B 的块变成 64 B，2.7 倍字节，和上一条同样的理由关掉。

所以单流 decode 的 **10.25 tok/s 是"这个格式 + 这颗 GPU"的上限**：字节已经压到 1.63 bpw，kernel 的
取数发射也满了。要再上去只能换 decode 策略（并发：32 槽 31.09），或者重写一个把 state 提取从每
4 个权重一次降到每 16 个权重一次的 cooperatively-load 版本（每个 lane 读连续字节、再用 shuffle 分给
同行 lane）——后者是唯一还没试过的结构性想法，代价是一颗新内核。

`mm-probe` 现在支持 `MUL_MAT_ID`，trellis 类型也不再段错误（根因是 `type_from_name` 拿
`ggml_type_name` 的空指针去比字符串，类型表有空洞）。

单流 decode 仍是 **10.25**（本次 tg16 读数 10.13，机器漂移在 §1 的范围内）。剩下的按收益排：

1. **重写 `accumulate_tqk` 的累加循环**（上一段）：现在所有结构性旋钮都测平了，唯一的解释是展开深度
   不足。这一步要先有 harness，否则迭代成本太高。
2. **CPU 侧的 trellis 点积**（`ggml_vec_dot_tq_q8_0_impl`）。指令账：1.65-1.78 条/权重，其中 LUT 取数与
   操作数拼装 38%、state 提取 27%、MAC 只有 15%。已设计并微基准验证的 int8×int8 + `sdot` 版本是
   **1.33-1.37×**（corr 0.999964，非精确），但它有个前提警告：真实 decode 里每线程只有 2.33 G 权重/s，
   同一内核在微基准里是 10.6 G/s —— **4.6× 的差**说明 CPU decode 并非纯指令受限，端到端大概率低于 1.25×。
   它只影响 CPU 后端与 CPU+GPU 并发，不影响 Vulkan 单流。

**已关闭**：字节（§4.1 两组都破输出）、NPU（打不开）、`--shader-family fp32` 对 decode（−45%）、
dense cooperative（80.98 对 85.55）、更大/更小的 matvec 工作组与更多 rows/WG（§5.3）、LUT 驻留
（PAD4K 中性，§5.3）、pp512 的 84% 空列随 ubatch 变小（阶跃函数，§5.1）、窄 dp4a tile 与 32x32
cooperative tile（§5.2）。

## 6. 测量纪律 / Measurement discipline

- **成对、交替、同二进制**：A/B 之间只换一个变量；内核级 A/B 直接换 `build\bin\ggml-cpu.dll` 或
  `ggml-vulkan.dll`，把重新构建从测量里去掉。
- **不看单次运行**：中位数；每次跑前杀掉残留的 `strata-vk.exe`。
- **绝对读数不跨时段比**：§1 首段已给漂移范围。
- **正确性先于性能**：任何性能数都要先过 argmax + 相关性；`STRATA_VK_PERF_LOGGER` 的**逐行金额不可信**
  （op 名与时长错位配对：同一张表里有 18 MB 读 3.25 µs 的行），只有 `Total time` 与 dispatch/barrier 计数可信；
  单类价格用 `STRATA_VK_SKIP` 消融，且它的读数是**上界**不是净价。
- **负结果和正结果一样进文档**：推翻一条结论时在原处加撤回注记并指向新证据（§4.2 就是范例）。
