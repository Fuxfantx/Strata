# 第六阶段：代价分解的最终账本，以及被收窄的路线

本轮把"一次 decode 到底花在哪"量到了底，并据此否掉了两个方向。所有数字来自本机
（Snapdragon X2 Elite Extreme / Adreno X2-90 / Windows 11 ARM64），Q2_0 为 62 GB 的 qwen4exp MoE pack。

## 1. 账本（Q2_0，`--tokens 9707 --n 6`，`STRATA_VK_TIME=1`）

| 项 | 时间 | 说明 |
| --- | --- | --- |
| **GPU 真实耗时** | **≈40 ms** | `GGML_VK_SERIALIZE_SUBMISSIONS=1` 对照得出（15.5 → 9.5 tok/s） |
| **进程 CPU** | **≈25 ms** | engine 的 `cpu` 计数；serialize 前后几乎不变，证明它只算真 CPU |
| 可测的 host 分桶 | **≈5.6 ms** | dispatch 1.2 + build 2.8 + submit 0.9 + fusion 0.75 + barrier 0.07 + end 0.015 |
| **记录窗口墙钟** | **≈71 ms** | `record`；≈ 40 + 25 + 少量未重叠部分 |
| 每次 dispatch 的 CPU | **0.49 µs** | 2 512 次 |
| 每个节点的 host 成本 | **≈10 µs** | 2 512 节点 / 25 ms（其中只有 ~2.2 µs 落在可测分桶里） |

**结论**：decode 的 wall ≈ GPU + CPU **几乎没有重叠**，因为 `llama_decode` 在每 token 末尾同步（下一 token 依赖采样结果）。
两者都已被量到，且都难以在 llama.cpp 层面继续压缩：GPU 侧受 2-bit 解包算术限制（见 §3），CPU 侧 25 ms 里只有约 6 ms
落在可测代码里，其余在 Adreno 驱动的内部线程与采样/detokenize。

## 2. 被收窄的路线（如实记录）

| 路线 | 结论 |
| --- | --- |
| 每-dispatch CPU（描述符/屏障/提交） | **否决**：0.49 µs/次，且 1/4/26 次提交吞吐相同 |
| Q2_0 的 128 B 重排 | **否决**：内核不是带宽受限（48.8 GB/s vs q8_0 的 90.6，字节只有 1/3.1 却只快 1.9x） |
| mmvq/mmq 路径切换 | **已最优**：`DISABLE_MMVQ` 15.0 < 默认 15.6；`DISABLE_MMQ` 15.7（噪声内） |
| **MTP / 投机解码** | **这两个 pack 不可用**：Q2_0 与 Gyro-S 都是 48 个 block、**零个 nextn/MTP 张量**，`--mtp` 需要一个单独的 head 文件，本机没有匹配的 |

## 3. 剩下的可做项：prefill 小 batch

Q2_0 的 prefill 随 batch 单调上升：**pp64 40.7 / pp128 53.3 / pp256 66.5 / pp512 77.4 tok/s**。
说明小 batch 吃亏在固定开销与占用率，而不是分块错误。

### 3.1 先校准测量协议（本轮的主要收获）

上一轮我报告"`GGML_VK_DISABLE_MMQ=1` @pp64 → +3.4%"。**这个开关在后端根本不存在**（只有 `GGML_VK_DISABLE_MMVQ`
会被读取，`mmvq_mode`，见 `ggml-vulkan.cpp` L5641）——所以那两次比较是**同一配置对比自己**，+3.4% 是
"第二个运行总是偏快"的位置偏差加漂移。

于是把工具改成带**零对照 + ABBA 交替**的版本（`port/tools/ab_pp.py`）：

| 实验 | 结果 |
| --- | --- |
| **零对照**（同配置跑两遍，3 对） | 中位差 **+0.42%**（2/3 对）⇒ **本机该协议的分辨率 ≈0.5%**，单次运行之间仍可差 ~5% |
| **`GGML_VK_DISABLE_MMVQ=1` @pp64（ABBA，4 对）** | **+0.22%**（3/4 对）⇒ **低于噪声，无效果** |

**结论**：小 batch 的路径选择在本机**不是**杠杆（唯一存在的开关只有 +0.22%），pp64 与 pp512 的差距来自占用率与固定开销。
本轮**不立项**，也不再引用此前那个 +3% 的数字。

### 3.2 之后所有 A/B 的硬性要求

1. 必须带**零对照**（同配置跑两次）测出当轮噪声底；
2. 必须 **ABBA 交替**，避免"第二个总是更快"的位置偏差；
3. 报告的差值必须**大于当轮噪声底**，否则记为"无效果"；
4. 单次运行之间 ±5% 的差异只能用于方向性判断，不能当收益。

## 4. 工具现状（都已入库，env 门控，关闭时零开销）

| 工具 | 用途 |
| --- | --- |
| `STRATA_VK_TIME=1` | 每 token 一行：record/submit/wait/dispatch/barrier/fusion/end/build/dep/mmv/mmvid/cleanup |
| `STRATA_VK_SUBMITS=N` | 限制一张图的提交数（证明提交开销可忽略） |
| `GGML_VK_SERIALIZE_SUBMISSIONS=1` | 每次提交后等待，暴露 GPU 真实耗时 |
| `port/tools/weight-report.py` | 按 header 偏移算每 token 权重字节数（不依赖类型表） |
| `port/tools/ab_pp.py` | ABBA + 零对照的 A/B（`--control` 测噪声底） |
| `port/probe/mm-probe` | 单形状、单 backend 的类型带宽（warm，400 次迭代），不重载 pack |

## 5. Q2_0 的 prefill：先更正一个错误结论，再定位真正的开销

### 5.1 更正

上一轮我根据管线直方图断言"Q2_0 的专家从未进入 coopmat 路径"。**这是错的。**

直方图只统计次数、不带类型归属，而 `--bench-pp N` 的一次运行里既有 prefill 又有随后的 decode 步，两者混在一起。
把 `[node]` 与 `[pipe]` 逐条对齐（`port/tools/zip-node-pipe.py`）后，归属非常清楚：

| 管线 | 次数 | 归属的节点 | 网格 |
| --- | --- | --- | --- |
| `matmul_id_coop_f16_f16acc_aligned_0` | 423 | **`ffn_moe_gate/up/down-<层>`** | 20x160x1 / 80x160x1 |
| `mul_mat_vec_id_q2_0_f32` | 153 | **同一批 `ffn_moe_*` 节点** | 320x10x1 |

**同一批专家节点**：prefill 的那一遍走 `matmul_id_coop`（✓ 已经 tile 化），随后 decode 的每一遍走 `mul_mat_vec_id_q2_0`
（n_tokens ≤ 8，`ggml_vk_use_mul_mat_vec_id` 的阈值）。Gyro-S 在 pp64 的 `matmul_id_coop` 也是 **423 次、同样网格**
（20x160x1 / 80x160x1）——两个 pack 的专家路径**完全一致**。

⇒ 专家路径不是 41 vs 82 tok/s 的原因，上一轮的"缺一条 coopmat 路径"结论作废。

### 5.2 真正的差异：bf16 的密集权重与转换拷贝

同一份 pp256 的 Q2_0 直方图里，占绝对多数的是**转换与 bf16 矩阵乘**：

| 管线 | 次数 | 说明 |
| --- | --- | --- |
| `contig_cpy_f32_bf16` | 1 320 | f32 → bf16 的逐元素转换拷贝 |
| `contig_cpy_f32_f16` | 912 | f32 → f16 的转换拷贝 |
| `matmul_bf16_aligned_1` | 668 | **bf16 权重**的矩阵乘 |
| `matmul_bf16_0` | 426 | 同上 |
| `matmul_bf16_aligned_0` | 334 | 同上 |
| `matmul_coop_f16_f16acc_aligned_*` | 777 | f16 coopmat |
| `scale_f32` / `rms_norm_mul` / `silu` / `gated_mul` / `swiglu` / `add` | 3 300+ | 逐元素尾巴 |

Gyro-S 的同一位置是 `matmul_q5_k_q8_1` / `matmul_q6_k_q8_1` / `matmul_q8_0_q8_1`（int8-dot），而不是 bf16。
**Q2_0 这个 pack 的密集/注意力权重是 BF16（1.37 GiB），于是走了 bf16 矩阵乘，并伴随每 token 上千次 f32↔bf16/f16 的转换拷贝。**

下一轮的候选（按可测性排序）：
1. **把 BF16 权重在加载期转成 F16**（体积不变、本机 f16 coopmat 有硬件路径），消除 bf16 matmul；
2. **减少 `contig_cpy_f32_*` 的转换拷贝**：f16 激活一路保持 f16，而不是 f32→f16→f32；
3. 判据仍是 `--bench-pp 64/256/512` 各 ≥3 次中位数 + 零对照，以及 32 token 与 oracle 不变。

### 5.3 教训（写进纪律）

**直方图必须按 prefill / decode 分开统计，并按节点归属**——`zip-node-pipe.py` 是工具；只看管线名的次数会得出反向结论。

### 5.4 一次不成立的测量（也记下来）

想给"每 token 约 750 次 f32→f16/bf16 转换"定价，就在 `ggml_vk_cpy_to_contiguous` 里加了 `STRATA_VK_SKIP_CASTS=1`
直接跳过这些 dispatch（token 变垃圾、拓扑不变）。结果 **反而更慢且不稳**：

| 配置 | 两次运行 |
| --- | --- |
| 默认 | 15.8 / 15.5 tok/s |
| 跳过转换 | 15.4 / 13.1 tok/s |

原因很可能是**跳过转换后消费者读到未转换的垃圾数据**，产生 NaN/denormal，而 Adreno 的 bf16/f16 通道遇到这些值更慢。
**方法本身污染了被测对象**，所以这个实验**不成立**，开关已移除（避免以后被误用）。

教训：**能"跳过"的测量只适用于产物不影响下游时序的地方**（例如 fwht 那次是纯数值路径）；转换、归一化这类
会改变下游数值分布的算子，不能用跳过法定价。要给它定价只能：给 `ggml_vk_cpy_to_contiguous` 加计时分桶（host 侧），
或用 RenderDoc 抓单次 dispatch 的 GPU 时间。

## 6. Qualcomm 在 decode 上从不走 matvec：`should_use_mmvq` 直接 `return false`

### 6.1 转换的真实归属

把"转换"和它**后面紧跟的那条管线**配对（decode，3 token），归属就确定了：

| 转换 | 次数 | 紧跟的消费者 |
| --- | --- | --- |
| `contig_cpy_f32_bf16` | 440 | `matmul_bf16_aligned_0` 298 + `matmul_bf16_0` 142 |
| `contig_cpy_f32_f16` | 304 | `matmul_coop_f16_f16acc_aligned_0` 210 + `matmul_id_coop_..._0` 94 |

也就是说：**decode（n=1）里仍在用 GEMM（`matmul_*`）**，每个 GEMM 前面还要多一次激活转换。

根因在 `ggml_vk_should_use_mmvq` 的结尾：

```cpp
case VK_VENDOR_ID_QUALCOMM:
    return false;      // → n == 1 的 matmul 全部走 GEMM 路径
default:
    return true;
```

这是上游按别的 Adreno 得出的经验值，本机从未验证过。

### 6.2 用现成开关量它：`GGML_VK_FORCE_MMVQ=1`（上游已有，`mmvq_mode = 1`）

ABBA 两对（`--tokens 9707 --n 8`，steady 值）：

| 顺序 | default | FORCE_MMVQ |
| --- | --- | --- |
| AB | 15.3 | 15.5 |
| BA | 15.2 | 15.4 |

**中位 15.25 → 15.45，+1.3%，两对都赢**（ABBA 已排除位置偏差；高于本轮 ~0.42% 的零对照噪声底）。
但只有两对，**不足以改默认值**。

### 6.3 一个反直觉的细节

开了 `FORCE_MMVQ` 之后，**那 440 + 304 次转换和对应的 `matmul_*` 一次都没少**（次数完全相同）。
说明这些节点根本不受 `mmvq_mode` 影响——它们走的是另一条判据（`ggml_vk_use_mul_mat_vec_id` 的 8-token 阈值，或
attention 侧的 matmul）。+1.3% 来自**另一批**节点切换到了 matvec。

### 6.4 下一轮

1. **5 对 ABBA** 复测 `FORCE_MMVQ`；若仍 ≥1%，把 `VK_VENDOR_ID_QUALCOMM` 的 `return false` 改成按 `n == 1 && k >= 2048` 走 matvec，并记 provenance；
2. 用 `zip-node-pipe.py` 找出**FORCE_MMVQ 下仍在转换的那 744 个节点**是谁（它们不受 mmvq_mode 影响），再决定是改阀值还是改管线选择。

### 6.5 五对复测：**没有效果**，并暴露了"中位数"在漂移下的问题

`GGML_VK_FORCE_MMVQ=1`，ABBA 五对（Q2_0 decode，`--tokens 9707 --n 8`）：

| 对 | default | FORCE_MMVQ |
| --- | --- | --- |
| 1 (AB) | 15.60 | 15.60 |
| 2 (BA) | 14.90 | 14.90 |
| 3 (AB) | 15.60 | 15.40 |
| 4 (BA) | 14.20 | 15.60 |
| 5 (AB) | 14.30 | 14.60 |

中位 **14.90 vs 15.40 = +3.36%**，但**成对比较只有 2/5 赢**。

原因是这一节里机器在**持续变慢**（default 从 15.6 掉到 14.2），而两次运行相隔一分多钟，于是"整段中位数"把**时间漂移**
算进了收益。**成对比较（相邻两次）才是漂移免疫的统计量。**

结论：**`FORCE_MMVQ` 没有可靠效果（2/5）**，此前 2/2 是运气。Qualcomm 的 `return false` 保持原样，不再作为候选。

### 6.6 工具的第三次校准：报告成对统计量

`ab_pp.py` 现在同时输出：整段中位数、成对数（X/N）、以及**成对比值的中位数**（`median(b_i/a_i - 1)`）。
**决定以成对统计量为准**；整段中位数只作为参考。这条与"零对照 + ABBA"一起构成本阶段的三条测量纪律。

## 7. **本阶段唯一的吞吐增益：prompt chunk 512 → 1024（prefill +4.5% ~ +26%）**

### 7.1 发现

在查 prefill 变慢时顺手试了引擎的 `--prefill`（它映射到 llama.cpp 的 `n_batch`/`n_ubatch`）。

| 配置 | pp64 | pp256 | pp512 |
| --- | --- | --- | --- |
| chunk 512（旧默认） | 41.7 | 66.5 | 77.4 |
| **chunk 1024（新默认）** | **45.8** | **83.8** | **96.9** |
| 变化 | +9.8% | +26% | +25% |

Gyro-S 同样受益（`--bench-pp` 实测）：pp64 **82.6 → 91.0（+10.2%）**、pp512 **192.4 → 201.1（+4.5%）**。

### 7.2 正确性

- 文本 prompt 前后 8 token **完全相同**（`1156,4439,883,279,1379,10919,18255,303`）；
- 参考 32 token 流**相同**；
- `test-backend-ops -o MUL_MAT -b Vulkan0`：**0 失败**（3/3 backends）。

### 7.3 代价与上限

`vk-common.h` 原本的注释写明：2048 token 的 batch 会让 Vulkan 后端分配 **3.2 GiB** 的 compute buffer，所以默认一直压得很小。
取 1024（约为其 1/2 的量级）后，两个 60 GB 级 pack 在 47.6 GiB UMA 上都能正常跑完 prefill 与 decode，没有 OOM 或变慢。

**改动**：`port/engine/vk-common.h` 的 `kDefaultBatch` 512 → 1024，注释同步更新为实测数字与理由。
**没有改上游**：这是引擎侧默认值，用户仍可用 `--prefill N` 覆盖。
