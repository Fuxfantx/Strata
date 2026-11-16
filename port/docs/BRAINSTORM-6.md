# 头脑风暴（第六轮）：继续榨干本机算力与带宽

目标是 **Prefill / Decode 双双继续上行**，重点 **Q2_0** 与 **Gyro-S**；同时列出 Engine 与上游 llama.cpp 值得重构的地方。
每条都给出**依据（本次或本仓库实测）**、**预期收益**、**风险**与**如何验证**。没有依据的猜测会被标为"待测"。

---

## 0. 账本：现在到底卡在哪

### 0.1 每 token 的真实字节数（新工具 `port/tools/weight-report.py`，按 header 偏移精确取）

| pack | 权重/ token | 实测 decode | 实际带宽 | 本机可读带宽 | 差距 |
| --- | --- | --- | --- | --- | --- |
| **Q2_0**（62 GB，专家在设备上） | **4.37 GiB**（dense 3.76 + 10/512 专家 0.61） | 14.0 tok/s | **66 GB/s** | 137-145 GB/s | **2.1x** |
| **Gyro-S**（54.5 GB，trellis） | **≈4.2 GiB**（dense 3.6 去掉 PLE + 0.47 专家） | 9.7 tok/s | **≈43 GB/s** | 137-145 GB/s | **3.2x** |

> Gyro-S 的 `per_layer_token_embd.weight` 是 **26.8 GiB IQ4_NL**，但每 token 只按层 gather 16 行；把它算进"每 token 权重"会
> 得出 33 GB/token 的荒谬值（321 GB/s），所以上表已剔除。**这 16 行随机读是 Gyro-S 独有的隐藏成本。**

### 0.2 单类型 matvec 的孤立带宽（`mm-probe`，ops=1，冷缓存）

`tqk6 87`、`q6_K 59`、`q4_0 54`、`q5_K 46`、`q8_0 42`、**`q2_0 30`** GB/s。

**结论一**：连最好的 tqk6 也只有墙的 62%；**Q2_0 只有墙的 21%** —— 这是全仓库最大的一块肥肉。
**结论二**：decode 不是"发射受限"（第四阶段命题已被第五阶段实测推翻：砍掉 96 次 dispatch 低于噪声），
而是**内核效率 + host 侧**共同受限。

### 0.3 时间都去哪了（Gyro-S，103 ms/token）

- matvec 部分：4.2 GiB @ 87 GB/s ≈ **50 ms**；
- 其余 ≈ **50 ms**：host 侧 ~30 ms（引擎日志 `cpu ~1.0 s / 32 token`）+ 逐元素尾巴（`SCALE/ADD/norm/GLU` 共约 1 400 次 dispatch）+ PLE 16 次随机读。
- **所以两条并行的主线**：把 matvec 从 87 推向 140（省 ~18 ms），把"其余 50 ms"砍半（省 ~25 ms）。

### 0.4 Prefill

| pack | 现状 | 备注 |
| --- | --- | --- |
| Gyro-S | 212 tok/s @pp512 | f16 累加 coopmat 64x64x16 |
| Q2_0 | **78.0 tok/s @pp512**（本轮新测）、41.5 @pp64 | 512 token / 6 563 ms；说明小 batch 掉得多（pp64 只有 pp512 的 53%） |

**结论三**：Q2_0 的 prefill 不是"5x 落后"，而是 **2.7x**；真正的短板是 **小 batch（pp64 掉一半）** 与 **decode**。

### 0.5 本轮补测的三件事

1. `Q2_0 --bench-pp 512` = **78.0 tok/s**（此前只有 pp64 = 41.5）。
2. 后端**没有**使用 `VK_KHR_push_descriptor`、`VK_KHR_synchronization2`、`VK_EXT_descriptor_indexing`（各 0 处引用），
   但**已用** `VK_EXT_memory_priority`、`VK_KHR_cooperative_matrix`、`VK_EXT_subgroup_size_control`。
   → host 侧与屏障的三个候选**都还没被用上**，这不是"已优化过"的方向。
3. 每 token 权重字节数（`port/tools/weight-report.py`，按 header 偏移精确取）：Q2_0 4.37 GiB、Gyro-S ≈4.2 GiB（已剔除 PLE 表）。

### 0.6 **颠覆性结论：decode 是 CPU 录制受限，不是 GPU 受限**

新增环境开关 `STRATA_VK_TIME=1`（`ggml-vulkan.cpp`，env 门控、零开销关闭）：在 `ggml_backend_vk_graph_compute` 记
"录制+提交"、在 `ggml_vk_submit` 记提交、在 `ggml_backend_vk_synchronize` 记等待，每 token 打一行。

| pack | record（CPU 录制） | submit | wait（等在 GPU 上） | 端到端 |
| --- | --- | --- | --- | --- |
| Q2_0 | **71.1 ms** | 0.90 ms（27 次） | **0.001 ms** | 15.5 tok/s = 64.5 ms/token |
| Gyro-S | **100.7 ms** | 0.93 ms（29 次） | **0.001 ms** | 11.0 tok/s = 91 ms/token |

再用 `GGML_VK_SERIALIZE_SUBMISSIONS=1`（每次提交后强制等待，破坏重叠）测出 **GPU 真实耗时**：

| pack | async | serialized | 差 = GPU 时间 |
| --- | --- | --- | --- |
| Q2_0 | 15.5 tok/s（64.5 ms） | 9.5 tok/s（105 ms） | **≈40 ms** |
| Gyro-S | 11.0 tok/s（91 ms） | 7.7 tok/s（130 ms） | **≈39 ms** |

**所以每个 token 的成本结构是：CPU 录制 65-100 ms、GPU 约 40 ms、等待 ~0。**
即 **GPU 有一半时间是空闲的**，瓶颈在 host 侧的**指令录制**（约 2 700 次 dispatch → **每次 ~26 µs**，远高于上游常见的 1-2 µs）。

这条结论**推翻了"host ~30 ms"的旧估计**，也把优化顺序整个改掉：
原先排第一的 "Q2_0 matvec 内核效率" 仍然值得做（GPU 侧 40 ms 里大部分是它），但**第一优先变成"每 dispatch 的录制成本"**——
只要把它从 26 µs 降到 10 µs，decode 就能从 64.5/91 ms 掉到 ~40/55 ms，即 **+50-60%**。

下一步要定位那 26 µs 花在哪：描述符集写入、`vkCmdPipelineBarrier`（第四阶段实测约 **5 µs/次**）、push constant、
pipeline 绑定，还是 `vkCmdUpdateBuffer`。做法与本次相同：在 `ggml_vk_dispatch_pipeline` 里加同类累加器。

---

## 1. 让 Q2_0 的 matvec 吃到带宽（最高优先级）

现状：in-model 66 GB/s，孤立 30 GB/s，墙 140。Q2_0 权重是 `2 bit + 每块 f16 scale`，每个 2560 维行只需 640 B →
**一行只有 5 个 128 B 扇区**，任何对齐/合并失误都会直接烧掉 20-40% 的带宽。

| # | 想法 | 依据 | 预期 | 风险 | 验证 |
| --- | --- | --- | --- | --- | --- |
| 1.1 | **按 128 B 对齐重排 Q2_0 的专家张量**（load 时一次性 repack：每 4 个 block 一簇，簇内按 lane 交错） | 行宽 640 B = 5 扇区，非整簇访问必然浪费；`q2_0` 孤立 30 GB/s 已说明 | +15-30% decode | 中（需 repack 工具 + 加载期改写，占额外 VRAM 但可原地） | `mm-probe` 前后对比；decode 中位数 |
| 1.2 | **128-bit 向量化 load + 4 block/线程组**，一次读满 128 B 再解 | 与 1.1 配套；当前若按 32 B 粒度读，cache line 利用率 <50% | +10-25% | 低-中 | 同上 |
| 1.3 | **Q2_0 走 int8-dot**（解包成 int8 后 dp4a，而非 f32 累加） | 本设备 int8 dot 可用（`int dot 1`）；coopmat int8 64x64x32 存在 | decode +10-20%，**prefill +100-200%** | 中（精度需 oracle 验证） | `test-backend-ops` + token 相等 |
| 1.4 | **多行共享激活**（把 trellis 已有的"每行解一次、多列复用"结构搬到 Q2_0） | phase 1 在 trellis 上做过；Q2_0 的 matvec 可能还是逐列解包 | MTP/投机解码时才显著（n>1） | 低 | n=4 的 bench |
| 1.5 | **工作点扫描**：rows-per-workgroup、subgroup 64、256 vs 512 线程、是否 `splitting` | 本机 subgroup=64；trellis 的最优点就是扫出来的（rows 40） | +3-10% | 极低 | 自动扫描脚本（每配置 200 ms × 20 配置） |
| 1.6 | **`output.weight` 是 Q5_K 417 MiB**（每 token 必读的 9%）→ 降到 Q4_0/Q3_K 或做 top-k logits | `weight-report.py` 实测 | +5-8% decode | 低（量化）/ 高（top-k） | token 相等（量化会变，需接受） |

**1.7（高风险高收益）"两阶段 logits"**：先用低秩/子采样估出候选集（例如 1 024 个 token），再只对候选算完整 logits。
理论上把 417 MiB 的 `output.weight` 读量降到 ~8 MiB，**decode +8-10%**；风险是采样分布偏移与实现复杂度，建议先做离线
"候选集命中率"实验（用现有 `--dump-logits` 在 200 个 token 上测 top-10 命中率）。

---

## 2. Gyro-S 的 trellis 内核：从 87 GB/s 推向 140

trellis（TQK6/TQK7）已经是本机最快的 matvec，但仍只有墙的 62%。它的成本结构是 **每 4 个权重 8 次 fp16 查表**（不可算术化，
已实测否决）。所以只能从"喂饱流水线"入手：

| # | 想法 | 依据 | 预期 | 风险 |
| --- | --- | --- | --- | --- |
| 2.1 | **软件预取下一段 128 B**（`OpLoad` 提前发出 + 计算与地址解耦），提高 MLP | 62% 的墙说明有 memory-level parallelism 缺口 | +5-15% | 低-中 |
| 2.2 | **把查表结果按 subgroup 广播复用**：同一 block 的 4 个权重被多个 lane 需要时，`subgroupShuffle` 换 LDS 访问 | LDS 已用于暂存，查表是 LDS 带宽密集 | +3-10% | 中 |
| 2.3 | **每 workgroup 处理更多行**（提高 L1 复用：同一激活被多行共享） | 激活 2 560 floats = 10 KB，SLC/L1 能容纳 | +3-8% | 低 |
| 2.4 | **TQK7 与 TQK6 合并成同一 kernel 的模板特化**，减少分支与 pipeline 数量 | 现在 4 个 pipeline（2 类型 × 2 路径） | 间接（dispatch/pipeline 切换） | 低 |
| 2.5 | **PLE 行缓存**：把每层 gather 的 16 行（64 KiB/层，共 ~3 MiB）在 token 间复用 | 同一序列相邻 token 的 per-layer token 命中高度重叠 | **省 16 次随机 DRAM 读**（约 1-3 ms/token） | 中 |

**2.6（高风险）"专家热排布"**：按真实激活频率把 512 个专家重排（前 64 个热专家放一起），让 SLC（本机约 4 MB 级）在
层间/专家间命中。理论收益 5-15%，但需要统计 512×48 的激活频率并重写 GGUF 的专家顺序（**会改变 pack 布局**，风险高）。

---

## 3. 砍掉"其余 50 ms"：host 侧与逐元素尾巴

### 3.1 host 侧（~30 ms/token，占 Gyro-S 的 29%）

| # | 想法 | 依据 | 预期 | 风险 |
| --- | --- | --- | --- | --- |
| 3.1.1 | **图缓存**：decode 图拓扑每 token 完全相同，只换输入指针 → 缓存 `ggml_cgraph` 与 buffer 分配 | 引擎已有 "warmed the decode graph"，但仍每 token 重建；llama.cpp 的 graph 构建 + 分配是纯 host 开销 | **-8-15 ms/token（+8-15%）** | 低-中（llama.cpp 需要 API 支持） |
| 3.1.2 | **描述符集合池化 / `VK_KHR_push_descriptor`**：2 668 次 dispatch × 每次写描述符 | 第四阶段测得"每 dispatch ~5 µs 的 CPU"；**后端目前完全没有用 push descriptor（0 引用）** | **-5-10 ms/token** | 中（需查 ICD 是否支持，未测） |
| 3.1.3 | **提交批处理**：把多层合成更少的 `vkQueueSubmit`，仅在需要回读时 fence | 现在每 token 至少 3 次提交 | -1-3 ms | 低 |
| 3.1.4 | **采样与下一 token 的图构建并行**（worker 线程） | 采样是 host 串行段 | -1-2 ms | 低 |
| 3.1.5 | **线程亲和 + `--perf-cap`** 与 GPU 提交线程分离 | 本机 18 核，CPU 侧 ~1 s/32 token | -2-5% | 低 |
| 3.1.6 | **`--profile` 分阶段计时**（build/submit/wait/sample）先量后改 | 现在只有总 CPU 时间 | 定位用 | 极低 |

### 3.2 逐元素尾巴（约 1 400 次 dispatch/token）

第五阶段已证明"按条数优化"收益低于噪声，**但那是"减去小算子"**；这里的方向不同：**把尾巴从串行变并行**。

| # | 想法 | 依据 | 预期 |
| --- | --- | --- | --- |
| 3.2.1 | **双队列重叠**：matvec 流在 queue0，逐元素尾巴在 queue1（依赖用 semaphore） | 本机单 GPU 双队列可用；尾巴的 kernel 小、利用率低 | +3-8% |
| 3.2.2 | **把尾巴融进 matvec 的 epilogue**（norm+GLU+gate 已经做了 `gated_mul`/`RMS_NORM_MUL`） | 还有 `add_norepeat` 123 次残差 | +2-4% |
| 3.2.3 | **合并 barrier**（用 event + 批量 barrier） | 实测"屏障数"是 CPU 成本，非字节成本 | -2-5 ms/token |

---

## 4. Prefill：Q2_0 是唯一还有 5x 空间的地方

| # | 想法 | 依据 | 预期 | 风险 |
| --- | --- | --- | --- | --- |
| 4.1 | ~~先测 Q2_0 的 pp512~~ **已测：78.0 tok/s** | 本轮实测 | — | — |
| 4.2 | **Q2_0 走 f16 累加 coopmat**（权重解到 tile 内，复用 trellis 的 `mul_mm` 结构） | trellis 同类路径 212 vs Q2_0 78 | **+50-150%** | 中（需要新的 `mul_mm` 分支） |
| 4.3 | **n-tile 256/512 扫描**（trellis 上测过 128→256 无变化，但 Q2_0 的 m 更大） | 未测 | 0-20% | 低 |
| 4.4 | **小 batch 专用路径**：pp64 = 41.5 只有 pp512 的 53%，说明小 batch 的占用/分块没调好（split-K、更小的 n-tile、更多 workgroup） | 本轮两档实测对比 | **+30-80% @pp64** | 中 |
| 4.5 | **prefill 与 decode 的图复用**（同一 session 内避免重复 allocate） | — | -5-10% host | 低 |

---

## 5. MTP / 投机解码：单点收益最大（1.5-2x）

本仓库已有 MTP 嫁接（`llama-ext.h`、`load_mtp`、`embeddings_nextn`）+ 引擎的 `--mtp`，但**没有在 Q2_0 / Gyro-S 上做过接受率实测**。

| # | 想法 | 依据 | 预期 | 风险 |
| --- | --- | --- | --- | --- |
| 5.1 | **测 MTP 在 Gyro-S / Q2_0 上的接受率**（draft k=1..3） | 未测；app 有 "MTP drafts up to N tokens" | 数据 | 低 |
| 5.2 | **verify 批 n=k 的 matvec 复用**（trellis 的多列结构已就绪，Q2_0 需 1.4） | phase 1 已做过 trellis 多列 | 接受率 × 节省 | 低 |
| 5.3 | **自适应 k**（按滑动接受率调 k，避免低接受率时反而变慢） | 投机解码的标准做法 | 稳定 +10-40% | 低 |
| 5.4 | **草稿用更便宜的模型/层子集**（例如只用前 8 层做 draft head） | — | 高 | 高（质量） |

> 注意：第五阶段实测"prompt lookup 投机在混合架构上净亏 8%"（rollback ring 成本），所以 5.x 必须**先量接受率再动手**，
> 且要保证 **token 流与无投机时逐 token 相同**（拒绝时必须严格回退）。

---

## 6. 字节数下降（长期，但收益确定）

| # | 想法 | 依据 | 预期 |
| --- | --- | --- | --- |
| 6.1 | **KV cache 更低比特**（Q2_0 上 KV 占比小，但 64k 上下文时不小） | 未测 | 视上下文 |
| 6.2 | **GDN 状态 / conv 状态的原地写**（已在 overlay 做了 `state_dst`） | 已落地 | 已计入 |
| 6.3 | **expert 权重 2 bit → 1.5 bit**（Q2_0 → PTQ1_0 类） | 需要重量化 | 字节 -25% → decode +25% 上限 |
| 6.4 | **PLE 表在磁盘 + mmap**（已做 `--ngram-on-disk`） | 已落地 | 已计入 |

---

## 7. Engine / 上游 llama.cpp 值得重构的地方

### 7.1 引擎（`port/engine`）

1. **`--profile` 分阶段计时**（build / submit / wait / sample / detokenize）→ 把 30 ms 拆开，否则后续优化都是盲打。
2. **按 pack 的自适应调优表**：加载后跑 200 ms 的微基准（rows-per-wg、split-k、mmq 阈值、f16acc 开关），把最优点缓存到
   `--data-dir` 的同名 `.tune.json`；Q2_0 与 Gyro-S 的最优点并不相同。
3. **图缓存 + buffer 复用**（见 3.1.1）。
4. **多 token 会话的 plan 复用**（`llama_batch` / `llama_memory` 的 per-token 分配）。
5. **KV / state 视图化**（避免 `ggml_cont`/`cpy` 的 37+30 次拷贝——GDN 的 `concat(state, transpose(x))` 可改原地写）。
6. **采样器**: 把 top-k/top-p 的排序改为部分选择（150k vocab 全排序是 host 成本）。

### 7.2 上游 llama.cpp（值得提交/本地维护的）

1. **`ggml_vk_dispatch_pipeline` 的每-dispatch 开销**：描述符集分配与写入、push constant 上传、pipeline 绑定；
   实测每 dispatch ~5 µs CPU（2 668 次 ≈ 13 ms/token）。后端未用 `VK_KHR_push_descriptor`（0 引用），
   也未用 `VK_EXT_descriptor_indexing`（0 引用）——两者都直接对着这项成本。
2. **`ggml_backend_vk_graph_compute` 的循环**：可预取下一节点的依赖、合并 barrier（`VK_KHR_synchronization2` 目前也是 0 引用）。
3. **图分配器**：per-token `ggml_gallocr` 复用（上游有 `ggml_gallocr_reserve`，但 Vulkan 后端仍每 step 重新绑定）。
4. **`ggml_vk_should_use_mmq` 的 per-vendor 阈值**：本机是 Qualcomm，阈值表来自 NVIDIA/AMD 的经验。
5. **`vulkan-shaders-gen` 的变体爆炸**：编译期与 pipeline 数量都影响启动时间（当前 warmup 5.4 s + 2.9 s）。
6. **`third_party/strata-gguf-py` 不认识本移植的类型**（TQK6/TQK7/PTQ1_0 会抛 `ValueError`）—— 这是个真实缺口，
   `serve` 的 GGUF 预检若走 gguf-py 会读不了 trellis pack。建议在 `strata-gguf-py/gguf/constants.py` 补齐 4 个枚举。

---

## 8. 风险 / 收益排序（建议动手顺序）

> **已被 §0.6 修正**：第一优先从"Q2_0 matvec 内核"改为"**每 dispatch 的 CPU 录制成本**"，因为实测每个 token 的
> 录制时间是 GPU 时间的 1.6-2.5 倍，GPU 有一半时间空闲。

| 序 | 项目 | 预期收益 | 风险 | 工作量 |
| --- | --- | --- | --- | --- |
| **0** | **定位并削减每 dispatch 的录制成本**（描述符 / 屏障 / push constant / pipeline 绑定；先加同类累加器定位） | **decode +30-60%（两个 pack）** | 中 | 1-2 轮定位 + 1-2 轮改 |
| 1 | Q2_0 matvec 重排 + 向量化（1.1/1.2）—— GPU 侧 40 ms 里的主体 | decode +10-20%（Q2_0） | 中 | 1-2 轮 |
| 2 | 图缓存 / 减少每 token 的重建（3.1.1） | decode +5-15% | 低 | 1 轮 |
| 3 | Q2_0 的 coopmat f16 prefill（4.2）+ 小 batch 路径（4.4） | prefill +30-150%（Q2_0） | 中 | 2-3 轮 |
| 4 | 描述符池化 / `VK_KHR_push_descriptor`（3.1.2，**是第 0 项的一种实现**） | 含在第 0 项里 | 中 | — |
| 5 | MTP 接受率实测 + 自适应 k（5.1/5.3） | decode +10-40% | 中-高 | 2 轮 |
| 6 | trellis 预取/查表复用（2.1/2.2） | decode +5-15%（Gyro-S） | 中 | 1-2 轮 |
| 7 | Q2_0 int8-dot（1.3） | decode +10-20%、prefill +100% | 中-高 | 2-3 轮 |
| 8 | 双队列重叠尾巴（3.2.1）—— GPU 有空闲时**无效**，先做第 0 项 | 0（当前） | 中 | — |
| 9 | output.weight 降比特（1.6） | +5-8% | 低 | 半天 |
| 10 | 两阶段 logits（1.7）/ 专家热排布（2.6） | +8-15% | 高 | 研究向 |

---

## 9. 先做哪些测量（避免瞎猜）

本轮已完成 3 项（见 §0.5）：Q2_0 的 pp512、扩展使用现状、每 token 字节数。剩下 4 项：

1. `--profile` 分阶段（build/submit/wait/sample）各占多少 ms —— 一次改动就能把 30 ms 拆开。
2. `mm-probe` 对 Q2_0 / tqk6 做 **warm** 的 ops=1/2/4 扫描（区分冷缓存效应）。
3. MTP 接受率（draft k=1..4，三个 pack 各 100 token）。
4. `VK_KHR_push_descriptor` / `synchronization2` 在本机 ICD 上是否可用（后端没用，但 ICD 可能支持；一个 20 行的
   `vkEnumerateDeviceExtensionProperties` 探针即可，或给 `--geometry` 加三行）。

**这 4 项加起来约 1 小时**，能把上面 10 条里"值得做"的收敛到 3-4 条，避免又一轮"砍 dispatch 但收益低于噪声"。

## 10. 用现有工具就能开始的实验（不改引擎）

| 实验 | 命令 | 想回答的问题 |
| --- | --- | --- |
| 小 batch 的占用 | `--bench-pp 64/128/256/512` 四档 | pp64 只有 pp512 的 53%，是 n-tile 还是 occupancy |
| KV 低比特 | `--kv q4_0` vs 默认，64k 上下文 | KV 在 64k 时的占比 |
| f16 累加开关 | `STRATA_VK_F16ACC=0/1` | Q2_0 的 prefill 路径是否已走 coopmat |
| 专家主机/设备 | 27B/35B 的 `--cpu-moe` 对比 | host 路径的带宽下限 |
| MTP 开/关 | `--mtp` + `--n 32` | 接受率与净收益（必须 token 相等） |
