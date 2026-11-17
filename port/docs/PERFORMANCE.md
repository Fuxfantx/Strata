# Performance — what this machine does, why, and what was ruled out

**这台机器上量到的、当前有效的性能结论。** 没有本机测量的数字不写进来；是假设的地方会标出来。
Every number here comes from a reproducible run on this machine; hypotheses are labelled as such.

- 机器：Snapdragon X2 Elite Extreme（18 个 Oryon 核，47.6 GB 统一内存，Adreno X2-90，Vulkan 1.4，驱动
  `30.0.228.10000`），Windows 11 ARM64。
- 目标包：`Gyro-S-TQ1_0.gguf`（54.47 GiB，`qwen4exp`，48 层，512 专家 top-10，d=2560，trellis 专家 24.02 GiB）。
- 方法：同一二进制、成对 ABBA、取中位数；每个数都写清楚它是哪条路径量的。
- 测量工具与开关：[`USAGE.md`](USAGE.md)。设备事实与逐项 Vulkan 测量：[`ADRENO.md`](ADRENO.md)。

---

## 1. 当前数字 / Current numbers

解码 `--bench-tg 16`，预填 `--bench-pp 512`，`--load-mode mmap`，同一二进制（2026-11-18 的构建）。

| 后端 / backend | decode tok/s | prefill pp512 tok/s |
| --- | --- | --- |
| CPU（`--gpu cpu`，大核类 11 线程） | **11.34** | 21.06 |
| Vulkan，f16 shaders（默认） | 10.68 | 19.33 |
| Vulkan，fp32 shaders（`--shader-family fp32`） | 5.48 | **46.63**（用该开关复测 46.09） |
| Vulkan，f16acc coopmat（`STRATA_VK_F16ACC=1`，**输出错**） | 10.90 | 221.20 |

结论：**在这台机器、这个包上，单流最快的是 CPU；GPU 只在"prefill + fp32 shaders"这一格赢（2.2× CPU）**。
decode 上 GPU 输给 CPU，缺的那条路是第 6 节里那条坏掉的内核。

**两个 shader 家族与 `identical` 的关系（实测，2026-11-18）**：两者都不是逐位等于 CPU，但都在 1e-3 以内 ——
35B-A3B 上 `--dump-logits` 的 CPU-vs-Vulkan 相关性 0.999（f16 家族 0.99906、fp32 家族 0.99894），argmax 相同。
差别落在**近似并列**上：同一条模板化 prompt、16 个 token，fp32 家族与 CPU 逐 token 相同，f16 家族在第 14 个
token 上分叉（`13,353,3172` 对 `13,3437,579`）。所以：

- 要**与 CPU 逐 token 相同**：`--shader-family fp32`（prefill 46.6，decode 5.5）。
- 要**decode 最快**：默认 f16 家族（10.7），它与 CPU 的差是一次并列翻转，不是另一个分布（对比 §4.1 的 0.02）。
- 两者都想拿：需要"prefill 用 fp32、decode 用 f16"的逐图切换 —— 见 §5 第 1 项之后的注记，这是当前最划算的
  GPU 方向，但要在后端按图选择 pipeline 家族。

其它包的往返数字（同一构建、同一方法，`port/README.md` 有完整表）：35B-A3B MoE prefill 89.4 tok/s
（fp32 shaders）/ decode 37.8-39.0；27B dense prefill 77.9 / decode 7.0-7.3；tiny dense 118.9 / 126.5。
这些数与"f16acc 打开"的旧表不是一回事：旧表的每一行都来自 §4.1 那条内核。

## 2. 为什么 GPU 没有更快 / Why the GPU is not ahead

一个 GPU decode token 在这台机器上是 94-104 ms，里面有 **2 824 次 dispatch**：矩阵乘本身已经在带宽墙上
（matvec 段 118 GB/s），但整个 token 平均只有 39 GB/s，因为前约 2 000 次 dispatch 是串行延迟。逐族消融
（`STRATA_VK_SKIP`）显示一个 token 的一半时间在 trellis 专家的 matvec 上。

CPU decode 的时间 **96% 在两个 matmul**（稠密 matvec + 专家 trellis matvec），repack 之后这两个已经跑在
它们的指令底上。所以两端的瓶颈是同一个：低位权重的字节数 × 每次触发的次数，而不是"算力不够"。

⇒ 这也解释了为什么"把层搬到别的引擎"没有用（§4）：单 token 内各层串行，UMA 下换引擎不降低字节数。
真正能降的只有两件：**每步摊薄一次权重读取**（批量）和**让每次读取更便宜**（格式/内核）。前者在正确路径上
没有兑现（§4），后者是当前的主线。

## 3. 已经进入代码、并且量到收益的改动 / Shipped, measured

| 改动 | 收益 | 在哪 |
| --- | --- | --- |
| CPU 线程放进**大核类**（6 个 3 072 MHz 的核只增加 barrier 等待） | decode 7.8-8.3 → **11.8-12.1 tok/s（+34%）** | `vk-devices.cpp` `plan_cpu_affinity` |
| CPU 的 repack 门禁按**实际拷贝的字节**算，而不是整包 | CPU prefill **+13%**；CPU 参考流随之改变（已记录） | `vk-engine-load.cpp` |
| CPU prefill 不用全部 18 核（batch pool 取掩码内全部核，decode pool 留一核） | prefill +5.5% | `plan_cpu_affinity` |
| Vulkan **int-dot（dp4a）** mat-mat 路径 | dense 包 prefill 数倍（选择逻辑在 `decide_policy`） | `vk-devices.cpp` |
| dense 包用 **fp32 shaders**（int-dot 的 mat-mat 快 2.2×） | 27B pp512 74.5 → 165.5；Gyro-S prefill 19.3 → **46.6**（`--shader-family fp32`，decode 5.5） | `vk-devices.cpp` `apply_adreno_tuning` |
| flash-attn：`n_ctx ≥ 4096` 且 ≥32 层默认开 | 2048 上下文 4.5× prefill；短位置 decode −32%，所以由调用方决定 | `vk-engine-load.cpp` |
| Vulkan prefill ubatch = 1 024（扫描过 128-2 048） | 默认已是最优（1 024 比 512 快 25%），旋钮关闭 | `kDefaultBatch` |
| KV `q4_0` + `n_ctx = 262 144` | 可用（用户要求）；KV 显存 8× 小 | `--kv q4_0 --max-context 262144` |
| PLE 表 26.8 GiB **本来就是 host-only**（不在 GPU 堆里） | 释放了约 27 GiB 的常驻假设 | `llama-ple-disk.cpp` |
| 专家驻留策略：host 专家 + 设备侧 expert cache | MoE 在 47.6 GiB 上可跑 | `vk-devices.cpp` `--moe-cache-size` |
| 采样器链**每请求一条**（此前缓存了已释放的指针，槽之间共享同一条） | 正确性；默认路径逐 token 不变 | `vk-engine.cpp` `make_sampler` |
| `--parallel N` 请求槽（一份权重、N 条 KV 序列） | 正确（64/64 逐 token）；**无聚合收益**，见 §4 | `vk-serve.cpp` |

## 4. 负结果 / Measured and ruled out

按"结论 + 否掉它的那个数"记。**这些不要再试**，除非上表的机器事实变了。

| 方向 | 判定 | 数字 / 证据 |
| --- | --- | --- |
| **NPU 放置**（Hexagon HTP） | 关闭 | 每种放置都比 GPU 慢：27B decode 4.1-4.7 对 7.3；35B 10-14 对 39；NPU 电源域读到的功耗也不低 |
| **raw FastRPC 开 cdsp 用户 PD** | 关闭（三种形状） | `remote_handle64_open` 返回 `AEE_ERPC 0x200`：映射条目已登记、未映射名字同码、Qualcomm 自家 host 库同样失败；只有新写 QNN 后端才可能再动 |
| **f16-accumulate coopmat** | **默认关闭**（曾默认开） | logits 与 CPU 相关性 **0.02**、max\|Δ\| 21-40、argmax 全错；性能 221 tok/s prefill 是"没做完全部工作"的读数（§5） |
| **host 指针 wrap** | 关闭 | 主机内存能导入，**文件映射不能**；5 秒抛异常；UMA 上不省内存 |
| **`--load-mode import`（一份权重）** | 关闭 | 导入预算实测约 **3 GiB**（host-visible 窗口），堆扩不了；`direct` 峰值 59.25 GiB > 47.6 GiB |
| **两个 context 共享一份权重** | 零增益 | GPU 被单条 decode 完全占满（同后端才共享，跨后端不共享） |
| **静态 CPU/GPU 层切分** | 净损 | 层间串行；越切越慢（`--split-cpu`、`--cpu-moe`） |
| **token 级 CPU+GPU 并行**（三种形态） | 吞吐成立、能耗不成立 | 两进程 +33% 吞吐但 **2.44× J/token**；两 context 0%；`--cpu-moe` −83%（每层交接约 26 ms） |
| **批量（`--parallel` 32 槽）** | 正确但无收益 | 正确路径每步 **3 178 ms ≈ 32 × 99 ms**，聚合 **1.00×**；8 槽是坑（`mul_mat_vec_max_cols = 8`：≤8 走 matvec，≥9 走 matmul） |
| **MTP / NextN 投机解码** | 负收益 | 这个包上 **−31%**（verify pass 比省下的 token 贵） |
| **n-gram lookup 投机** | 中性 | 每次 tg16 都在噪声内；`--lookup` 默认关 |
| **准入的 prefill 切片** | 净亏 | 一个 prefill pass 就要把 3.06 GiB 稠密权重流一遍：48 token 切成 32+16 → prompt 读两遍；默认整段读，旋钮 `STRATA_VK_PREFILL_CHUNK` |
| **Vulkan ubatch 扫描** | 默认已最优 | 1 024 最优，旋钮关闭 |
| **CPU repack 到 trellis** | CPU 大胜、GPU 只有一成 | 同一形状上量类型本身：CPU 17-19 tok/s 级别，GPU +10%；工具留下（`trellis-migration.py`） |
| **能耗计数器的绝对值** | 不可用 | Energy Meter 原始标度约 1e4 单位/W，只有**同一次探针内**的比值有效 |

### 4.1 那些"很好"的 GPU 数字是什么

2026-11-18 之前记录过的 GPU prefill 201-204 tok/s、批量 40.17/40.97 tok/s（4.27×）、每 token 能耗低 4.6-5.1×，
全部来自 f16-accumulate coopmat 那条坏内核：它每行几乎不做完整工作，所以又快又"省电"。它逃过门禁的原因是
**门禁与被测对象之间隔着一次 opt-in** —— `test-backend-ops` 的 MUL_MAT oracle 通过，是因为那条内核只在
`STRATA_VK_F16ACC=1` 时才可达，而 oracle 永远走默认路径。现在的门禁改成 CPU/Vulkan 的 logits 相关性
（§3 最后一行 + `USAGE.md` §5），它不看 opt-in。

### 4.2 精确的 int8 cooperative matrix：硬件有、能接、实测中性

设备能力（ADRENO.md 的 `[coopmat]` 行）：**Sint8×Sint8→Sint32 的 cooperative matrix 是 64×64×32，int32 累加 —— 精确**；
另一条是 f16 累加 64×64×16，而 f32 累加与 dot2 都不存在。所以"精确的矩阵核心"这条路在硬件上是存在的。

upstream 只为 AMD RDNA3/RDNA4 创建 int8 cooperative MMQ 管线（`matmul_q5_k_q8_1` 那一族）。这一轮加了
`STRATA_VK_INT8_COOPMAT=1`，让 Qualcomm 设备也创建它们，然后量：

| 检查 | 结果 |
| --- | --- |
| `test-backend-ops -o MUL_MAT -p type_a=q5_K`（开关下） | **0 失败** |
| 同上 `q6_K` / `q8_0` | **0 失败** |
| Gyro-S pp512 | 16.18 → 15.62 tok/s（**中性，噪声内**） |
| Gyro-S tg16 | 10.37 → 10.31 tok/s（中性） |

即：**精确性有，速度没有**。原因未定 —— 可能是 B=q8_1 的量化决策根本没走到 coopmat 管线（开关只改了创建，
选择条件另有一处），也可能是走到了但在这台设备的形状上不快。开关保留（默认关、逐 token 相同），列入 §5 的 P3。

> **测量噪声的更正**：同一天、同一个二进制，Gyro-S pp512 在 **16.2-19.2 tok/s** 之间漂移（连续几小时 GPU
> 工作之后偏低）。本页别处写的"本机 4% 噪声"只在短会话里成立；引用本页数字按 ±15% 读，或成对重测。

### 4.3 P0 判定：f16-accumulate 是**设计内的近似路**，不是可修的索引 bug

从实现读出来的事实，不需要新的 profiling：

- 生成 f16acc 变体时定义了 `ACC_TYPE = float16_t`、`ACC_TYPE_MAX = float16_t(65504.0)`，shader 在累加之后把结果
  **clamp 到 ±65504**（`mul_mm.comp` 的 `ACC_TYPE_MAX` 段）。上游在 `ggml-vulkan.cpp` 里也写明这条路
  "fails the element-wise oracle ... never the default"。⇒ **f16 部分和 + 饱和是设计行为**，`mm-probe --verify`
  的"全零"和 `test-backend-ops` 的 `ERR = 83…1710` 都与"部分和溢出后被饱和"一致，而不是某个 tile 索引错。
- 这台设备没有 f32 累加的 cooperative matrix、也没有 dot2，所以 float 的矩阵核心**只能** f16 进 f16 出 —— 没有
  "换个累加类型"的修法。

⇒ "修 bug" 的 P0 关闭。要在 identical 前提下还能用它，只剩一条未验证的路（记为 **P0'**）：**staging 之前把 B
乘 1/S、算完再乘回 S**，把部分和压进 f16 的可表示范围。数学恒等，只有舍入变化；若溢出是主因，logits 相关性
会从 0.02 回到 0.99+。这要改 shader（`mul_mm.comp` 的 B staging 与 D store 两处）。

**P0' 的两个前提，这一轮都实测否掉了**（35B-A3B，模板化 prompt，`--dump-logits` 对比 CPU）：

| 假设 | 测法 | 结果 |
| --- | --- | --- |
| 坏的是 flash attention 的 f16acc 变体，不是 matmul | `STRATA_VK_F16ACC=1 --flash-attn off` | corr **0.0196**（FA 打开时 0.0226）⇒ **不是 FA**，matmul 自己就错 |
| 只在小 n / 部分 tile 上错，满 tile（n=64）就对 | 64 token prompt，F16ACC=1 vs 默认 | corr **−0.0128**（默认路径同 prompt **0.9979**）⇒ **满 tile 也错**，不是边界 |

再叠加 §4.3 第一条（部分和量级 ~0.05，远低于 f16 的 65504）：**溢出也不是机制**。⇒ 我上一轮提的 P0'
（B 缩放）**大概率是死路**，不再作为首选；f16acc 的 coopmat 变体是"结构性算错"，下一步只能进 shader 读
`coopMatLoad`/`coopMatMulAdd` 在 `ACC_TYPE=float16_t` 下的用法（变体专属的 spec constant 或 staging 布局），
而不是在数值上做文章。

## 5. 优先级 / Priorities

目标（用户）：**prefill 200 tok/s、decode 40 tok/s**。这两个数目前只有 f16-accumulate 那条内核达到过
（221 / 32 槽聚合 40.97）。要在**输出一致**的前提下拿到它们，按这个顺序做：

| # | 目标 | 为什么排在这 | 第一步 | 现实预期 |
| --- | --- | --- | --- | --- |
| **P0'** | 让 f16acc **不溢出**：staging 前把 B 除以 S、结果乘回 S | §4.3 已证 f16 累加 + 65504 饱和是设计行为，不是索引 bug；这是唯一还能在 identical 前提下用这条 200/40 路径的改法 | 在 `mul_mm.comp` 的 B staging 处加 `/S`、D store 处乘回（S 取让部分和远离 65504 的值）；先过 oracle 再跑 logits 门禁 | 若溢出是主因，corr 从 0.02 回到 0.99+，200/40 才有意义 |
| **P1** | **多行 matmul 的摊薄**（decode 40 的另一半） | 32 槽每步 3 178 ms ≈ 32 × 单条 99 ms，说明 n=32 时权重读取没有被摊薄；即使 P0 修好，批量也要靠这一条 | `mm-probe` 量 q5_K/q6_K 在 n=1..64 的耗时曲线（两个 shader 家族各一遍），看摊薄从哪里开始 | 32 槽回到 4× 聚合 |
| **P2** | **逐图选择 shader 家族** | fp32 家族 prefill 46.6 且 16-token 流与 CPU 相同，f16 家族 decode 10.7；两者都不用重写内核 | 确认 `GGML_VK_PIPELINE_STATS` 里 `matmul_q5_k_q8_1_0/_1` 的后缀是不是 f16 标志（若是，两族都已编译，只差按图选 key） | prefill 46.6 + decode 10.7 同时成立 |
| **P3** | 让精确的 int8 cooperative 真正被**选中** | oracle 已证精确（q5_K/q6_K/q8_0 **0 失败**），但只加创建开关后**中性**（16.18 → 15.62）。实现里没有找到 dispatch 处对 `_cm1` 管线的选择点 —— 很可能**创建了却没被用**，这正好解释中性 | 找到 MMQ 的管线选择函数，确认 `matmul_*_q8_1_cm1` 是否可达；不可达就加一个选择分支 | 若可达，prefill 90-150（精确） |
| **P4** | prefill 的两个慢算子 | 各占正确路径 prefill 的 3-5%，改动局部 | `MUL_MAT f32 m=1 n=512 k=2560`（9.4 ms × 94 次）与 `MUL_MAT_VEC q8_0 m=10240 n=1 k=320`（5.1 ms × 96 次，约 650 MB/s）为什么这么慢 | prefill +5-8% |
| **P5** | 单流 decode 的 94-104 ms | 一个 token ≈ 4.1 GiB / 137 GB/s = **32 ms 带宽地板**，单流上限约 31 tok/s；剩下 3 倍在 2 824 次 dispatch（前 ~2 000 次是串行延迟） | 加 dispatch 已证明不涨时间（§2），所以先找出那 2 000 次里哪些是同一个 kernel 的重复调用 | 单流 decode 15-20 |

**不要做的**：把 f16-accumulate 重新设为默认（输出不一致）；把批路径接进 `serve/server.py`（正确但 1.00×，
除非 P1 成立）。**交互式当下的最优选择**：`--shader-family fp32`（prefill 46.6、decode 5.5）或默认 f16
（prefill 19、decode 10.7），按请求是长文档还是短回复选。

## 6. 测量纪律 / Measurement discipline

- **成对、交替、同二进制**：A/B 之间只换一个变量；内核级 A/B 直接换 `build\bin\ggml-cpu.dll` 或
  `ggml-vulkan.dll`，把重新构建从测量里去掉。
- **不看单次运行**：中位数；每次跑前杀掉残留的 `strata-vk.exe`。
- **90 秒看门狗**：挂住的运行按失败记，不按"慢"记。
- **正确性先于性能**：任何性能数都要先过 logits 相关性门禁（§4.1 那条内核就是反例）。
- **功耗**：`port/tools/energy-ab.py` 采样 Energy Meter，并用引擎自己打印的 `bench window` 裁剪样本；
  只报同一次探针内的比值。
- **文档规则**：推翻一条结论时，在原处加撤回注记并指向新证据；**负结果和正结果一样进文档**。

## 7. 实验日志 / Experiment log

2026-11-18 这一天按轮次记录。每行是"当时的问题 + 结论 + 否掉它的数"；细节都还在 git 历史里。

| 轮 | 结论 |
| --- | --- |
| 1-5 | repack 缺口补上（CPU 稠密 matvec 9×）；GPU 一个 token 104 ms / 2 824 dispatch；GPU 是**忙**的，不是闲的 |
| 6-10 | 逐族消融：一个 token 的一半是 trellis 专家 matvec；LUT 放 shared 是 4.2× 慢；repack 门禁修正 → prefill +13%；CPU 大核类 → decode +34% |
| 11-15 | matvec 工作组 −28%；GPU 不是被 dispatch 数卡住；并发那条线（CPU 副本）端到端打通 |
| 16-20 | trellis 格式迁移：同一形状上量类型本身 → **CPU 大胜、GPU 一成**；接到引擎上是负结果（包比内存大，膨胀 4.9× 就变磁盘读） |
| 21-25 | J/token 基线：CPU 更快但每 token 贵 4.7×；**没有可削减的每 token 固定开销**；CPU prefill 不该用 18 核 |
| 26-30 | 四行 `vec_dot` 的收益上界 <1%；挡住"一份权重"的是 PLE 表；跨引擎并发 +33% 吞吐 / 2.44× 能耗 |
| 31-35 | 静态 CPU/GPU 切分净损；引擎按设计串行处理并发槽；多序列并发做出来了，但不摊薄权重读取 |
| 36-40 | htp_lib_dir 半短路径；NPU 会话开不出（0x200）；`--npu dense` 关闭理由；lookup / resident-budget 中性 |
| 41-45 | 静态切分再确认净损；多序列每多一个 token 付一整步；跨引擎能耗代价；wrap 到驱动层定论（文件映射不能导入） |
| 46-50 | 专家移到 host 两路一起塌；`--load-mode import` 被 3 GiB 预算否决；两个 context 零增益；堆扩不了 |
| 51-55 | 常驻集合不相交被层间交接打死；反向切分下界低于现状；**批量在 ≥16 槽进入 matmul 路径**（当时读数 +154%，后被 §4.1 撤回）；8 槽是坑；阈值是代码事实 `mul_mat_vec_max_cols = 8` |
| 56-60 | 32 槽跑不起来的根因是预热用的抛弃 context（修好后 40.2 tok/s，也是坏内核）；批量接进 `--serve`（40.97 tok/s，坏内核）；准入 prefill 不能切片；批量"第一个 token"复盘 |
| 61-62 | **f16-accumulate coopmat 给出的是另一个分布**（相关性 0.02），默认关闭；正确路径重测：CPU 单流最好、GPU 只赢 prefill/fp32、批量 1.00×（§1、§4.1、§5） |

**未修的已知缺陷 / open defects**（写在这里，避免被当成特性）：

- f16-accumulate coopmat 内核输出错误（默认已关，内核仍在）。
- QSA（`qwen4exp`）在 prompt 短于 compress ratio 时的 tail 掩码路径会在退化输入上给出 `nan` logits；
  模板化 prompt 上未复现，但边界未定。
- `--parallel` 的批路径正确但无收益，服务端未接。
