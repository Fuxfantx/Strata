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

1. 必须带**零对照**（同配置两次）测出当轮噪声底；
2. 必须 **ABBA 交替**，避免"第二个总是更快"的位置偏差；
3. 报告的差值必须**大于当轮噪声底**，否则记为"无效果"；
4. 单次运行之间的 ±5% 只能用于"方向性判断"，不能当收益。

## 4. 工具现状（都已入库，env 门控，关闭时零开销）

| 工具 | 用途 |
| --- | --- |
| `STRATA_VK_TIME=1` | 每 token 一行，含 record/submit/wait/dispatch/barrier/fusion/end/build/dep/mmv/mmvid/cleanup |
| `STRATA_VK_SUBMITS=N` | 限制一张图的提交数（证明提交开销可忽略） |
| `GGML_VK_SERIALIZE_SUBMISSIONS=1` | 每次提交后等待，暴露 GPU 真实耗时 |
| `port/tools/weight-report.py` | 按 header 偏移算每 token 权重字节数（不依赖类型表） |
| `port/probe/mm-probe` | 单形状、单 backend 的类型带宽（warm，400 次迭代） |
