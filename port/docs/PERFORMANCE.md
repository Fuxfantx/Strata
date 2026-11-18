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
| **Vulkan 预填 pp512** | **43-50 tok/s** | 默认设置（2026-11-18 之后） |
| Vulkan 预填，同一构建改之前 | 16.01 | mat-mat 从 fp16 模块取 shader 时 |
| Vulkan 预填，`--shader-family fp32` | 46.6 | 全部 shader 走 fp32，decode 掉到 5.5 |
| Vulkan 预填，两族 cooperative 都开 | **145** | `STRATA_VK_F16ACC=1` + `STRATA_VK_F16ACC_DENSE=1`，**输出错**，见 §4.2 |
| **Vulkan 解码，单流** | 10.2 tok/s | `--bench-tg 16`；37.7 GB/s 权重流量 |
| **Vulkan 解码，32 槽** | **26.7 tok/s 聚合**（2.73×） | `--batch-slots 32`；103 GB/s = GPU 上限的 95% |
| **Vulkan 32 槽 + CPU 进程** | **29.1 tok/s 聚合**（25.3 + 3.8） | 两个进程同时跑 |
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
| `--parallel N` / `--batch-slots N` 请求槽 | **正确且有收益：32 槽 2.73×** | `vk-serve.cpp` |
| **companion 包合成**（`--compat FILE` + `strata.companion`） | 逐位不变的往返已验证 | `llama-model-loader.cpp` 三处、`vk-options.*`、`port/tools/gguf-companion.py` |
| **`requant-tensor`** 工具 | 秒级产出重打包 companion | `port/probe/requant-tensor.cpp` |

## 4. 负结果、以及被撤回的结论 / Ruled out, and what was withdrawn

### 4.1 不要再试的方向

| 方向 | 判定 | 数字 / 证据 |
| --- | --- | --- |
| **NPU 放置**（Hexagon HTP） | 关闭 | 每种放置都比 GPU 慢（27B decode 4.1-4.7 对 7.3；35B 10-14 对 39） |
| **raw FastRPC 开 cdsp 用户 PD** | 关闭 | `remote_handle64_open` 返回 `AEE_ERPC 0x200`，三种形状 |
| **f16-accumulate coopmat 作为默认** | 关闭 | 见 §4.2：tile 修好前是零次循环，修好后分组那份写 `-inf` |
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

### 4.2 cooperative matrix：一条"看起来很好"的路，两个阶段都被证伪

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

⇒ **dense cooperative 是对的**（且 **中性偏慢 1.5%**：pp512 43.25/43.58 对默认 44.00/43.91），
**分组（`MUL_MAT_ID`）那份才是 `-inf` 的来源**。`GGML_VK_NAN_SCAN=*` 定位：坏值不是 nan 而是 `-inf`，
恰好是每张量**开头 64 个元素**（一个 `BM` tile = 一个 workgroup 的输出），第一个坏节点是 `blk.3` 的输入残差
（`blk.0`/`blk.1` 干净）⇒ 不是"分组恒错"，而是**跑过若干层之后**才坏，指向有界 scratch 或越界写。
已排除：`n < BN` 的尾部（72-token tile-legal prompt 也失败）、`STRATA_VK_COOP_SHORT=0` 的严格形状门
（仍失败）、scratch 尺寸（`bn` 取自实际选中的管线，自洽）、dense 的 f16 回落（加闸后仍失败）。

## 5. 还剩什么 / What is left

**在 `identical` 前提下，decode 的所有路径都已测过并关闭**：字节（§4.1 两组都破输出）、GPU 单引擎摊销
（95% 已用）、CPU+GPU 并发（+9%，见 §2.2 与下表）、批量槽数（不单调，48 槽是下陷）、NPU（打不开）。

| 配置 | GPU 聚合 | CPU 并发 | 合计 |
| --- | --- | --- | --- |
| GPU 16 槽 + CPU | 14.54 | 8.93 | 23.47 |
| GPU 24 槽 + CPU | 20.90 | 4.43 | 25.33 |
| **GPU 32 槽 + CPU** | 25.32 | 3.48 | **28.80** |
| GPU 64 槽（独占） | 27.09 | — | 27.09 |

CPU 的并发贡献随 GPU 负载单调下降（争同一份带宽），合计压在 **28.4-29.1**。合并流量 112.5 GB/s，
远低于两引擎纯流式的 177 GB/s ⇒ **CPU 那半无法把带宽换成 token**，因为它是 trellis 指令受限。

⇒ 唯一剩下的、有明确做法的一步：**CPU 侧的 trellis 点积**（`ggml_vec_dot_tq_q8_0_impl`，1.78 指令/权重）。
若把它降到 ~1.0-1.2（`tq_state4` 的窗口提取向量化、LUT 与累加结构收紧，**保持整型精确**），CPU 单流
10.2 → 15 左右，并发贡献 3.5 → 5.2，合计 ~30.5。要到 40 需要 CPU 单流到 ~20（2×），而指令数分析只支持
1.4-1.6× ⇒ **40 tok/s 在现约束下不可达**，这一点已在 28-29 处被多方测量确认。

**prefill 侧**还剩一条：修好分组 cooperative 的 staging（§4.2 已把范围收到"越界写"），保守预期 100-150 tok/s
且输出正确；int8（int32 累加、精确）矩阵核心是另一条，需要把创建块搬出 `coopmat_support`。

## 6. 测量纪律 / Measurement discipline

- **成对、交替、同二进制**：A/B 之间只换一个变量；内核级 A/B 直接换 `build\bin\ggml-cpu.dll` 或
  `ggml-vulkan.dll`，把重新构建从测量里去掉。
- **不看单次运行**：中位数；每次跑前杀掉残留的 `strata-vk.exe`。
- **绝对读数不跨时段比**：§1 首段已给漂移范围。
- **正确性先于性能**：任何性能数都要先过 argmax + 相关性；`STRATA_VK_PERF_LOGGER` 的**逐行金额不可信**
  （op 名与时长错位配对：同一张表里有 18 MB 读 3.25 µs 的行），只有 `Total time` 与 dispatch/barrier 计数可信；
  单类价格用 `STRATA_VK_SKIP` 消融，且它的读数是**上界**不是净价。
- **负结果和正结果一样进文档**：推翻一条结论时在原处加撤回注记并指向新证据（§4.2 就是范例）。
