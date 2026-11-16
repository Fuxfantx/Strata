# 值得从 mainstream Strata 借鉴的策略（按优先级）

方法：直接读 `upstream/main` 的实现与文档 —— `src/ngram/ple_reader.cpp`、`src/prefill/prefill.cpp`、
`src/spec/suffix_drafter.cpp`、`docs/GYRO.md`、`docs/EXPERTDEC.md`、`docs/BATCHED_DMA.md`、`docs/DISJOINT_EXPERT_CACHE.md`。
每条都对照**本机实测**判断是否适用（Vulkan / ARM64 / 单 GPU / 47.6 GiB UMA）。

## 1. 台账

| # | mainstream 的做法 | 原文/位置 | 对本移植是否适用 | 本机依据 | 优先级 |
| --- | --- | --- | --- | --- | --- |
| **M1** | **配套“compat”包**：setup 用 `tools/iq_pack.py --compat-bf16` 把**稠密权重和 Hadamard 表**重打成 1.5 GB 的伴随文件，让它们走引擎自己的快内核 | `docs/GYRO.md` | ✅ **适用**（我们没有重打包步骤，稠密层走的是通用内核） | 稠密占每 token 字节的 **76%**（3.1 GiB / 4.1 GiB），Q5_K 72 GB/s、Q6_K 87 GB/s，而 trellis TQK6 也是 87 GB/s 但**每字节只 2.125 bit** | **P0** |
| **M2** | **后缀查表草稿器**（trigram 索引、4 路、约 20 B/token、无权重、微秒级），命中即一次 verify 接受多 token | `src/spec/suffix_drafter.cpp` | ⚠️ 适用但**已测亏** | 本移植的 prompt-lookup 投机在混合架构上净亏 **8%**（回滚环成本），且"没找到可抄的续写" | P2 |
| **M3** | **PLE 走独立的 direct-I/O 读器**：8 路组相联行缓存 + 异步 job 队列 + 分位数统计 | `src/ngram/ple_reader.cpp` | ✅ 部分已做 | 我们已经照搬**8 路 + mix 散列 + 组内轮转**；但实测 gather 只要 **22-29 µs/token**，异步队列与分位数统计**没有必要** | 完成 |
| **M4** | **专家驻留/决策**（按预算把热专家留在 VRAM，verify 窗口用共享内存暂存激活） | `docs/EXPERTDEC.md`、`DISJOINT_EXPERT_CACHE.md` | ⚠️ 仅概念可借 | EXPERTDEC 是 CUDA kernel（`iq_kernels.cu`）；我们的 trellis matvec 已有"每行解一次、多列复用" | P3 |
| **M5** | **自适应 prefill 分块/切点**（`prefill.cpp` 里一堆规则：switch point、plain floor、staged group…） | `src/prefill/prefill.cpp` | ✅ 适用（我们刚把 chunk 固定成 1024） | 固定 1024 已带来 **pp512 +25%**；自适应可能在小 prompt 上再多一点，但收益上限低 | P2 |
| **M6** | **批量 DMA 提交**（`cudaMemcpyBatchAsync`） | `docs/BATCHED_DMA.md` | ❌ 不适用 | 本机提交 27 次共 **0.85 ms**；上游自己测到"提交延迟 −83%、总传输只 −1.3%、未证明生成变快" | 不做 |
| **M7** | **双卡专家去重**（helper 与 primary 不重复占槽） | `DISJOINT_EXPERT_CACHE.md` | ❌ 不适用 | 单 GPU | 不做 |

## 2. P0 详解：稠密层的配套重打包（M1 的本移植版本）

### 2.1 为什么这是最大的杠杆

Gyro-S 每 token 读 ≈4.1 GiB，其中：

| 部分 | 字节 | 类型 | 孤立带宽 |
| --- | --- | --- | --- |
| 稠密（注意力/shared/输出等） | **≈3.1 GiB** | Q5_K 1.82 + Q6_K 0.50（输出）+ Q8_0 0.39 + 其它 | 72 / 87 / 42 GB/s |
| 稀疏专家（10/512） | 0.47 GiB | TQK6 15.2 + TQK7 8.8（全表） | 87 GB/s |

稠密占了四分之三，却跑在与专家同样的带宽上 ⇒ **把稠密层的字节数压下来是唯一能显著改变 decode 的动作**。

### 2.2 方案（上游思路的本移植版）

上游的 compat 包是**换格式以适配内核**（它们把稠密层放成 bf16 给自己的 GEMM）。本机的等价物是：
**把稠密层重打成 trellis（TQK6/TQK7）**，因为

- 本机 tqk6 的孤立带宽 **87 GB/s**（与 q6_K 并列最高），
- 而 trellis 是 **2.125 bit/权重**，Q5_K 是 5.5 bit ⇒ **同样张量字节数降到 1/2.6**。

理论收益：稠密 3.1 GiB → ≈1.2 GiB，稠密时间 ≈39 ms → ≈15 ms ⇒ token 103 ms → ≈78 ms ⇒ **+30-45% decode**。

### 2.3 代价与边界

- **画质**：稠密层从 5.5 bit 降到 2.125 bit，是有损的。必须给出**质量代理**（同一段文本的困惑度/对数似然对比），
  并把 compat 包做成**可选**（默认不生成、不启用），与上游"setup 里显式准备"的做法一致。
- **工作流整合**（用户要求）：重打包必须进 **CLI 与 desktop app**：
  1. `port/tools/repack-compat.ps1`（或引擎子命令 `strata-vk --repack-compat <in> <out>`）产出伴随文件；
  2. 引擎 `--compat <file>` 加载伴随文件、优先使用其中的稠密权重，缺失则回退原包；
  3. app/CLI 的 setup 步骤调用它（`setup.py` 的 gyro 家族分支 + 一条显式提示）。
- **可回退**：伴随文件不存在或版本不匹配时，一切行为与今天相同（必须保证 token 流在原包上不变）。

### 2.4 实施顺序

1. **选择性重打包工具**：读入 GGUF，只对稠密张量（非 `*_exps`、非 `per_layer_token_embd`）做 trellis 量化，写新 GGUF（保留原 metadata/专家）；
2. **引擎加载伴随文件**：按张量名覆盖稠密权重（内存映射，不复制专家）；
3. **质量代理 + A/B**：同 prompt 的 token 对数似然对比；decode/prefill 的 ABBA 判据；
4. **工作流整合**：CLI 子命令 + `setup.py`/桌面 app 的入口与提示。

## 3. 其余值得做的（P2/P3）

- **P2 M5 自适应 chunk**：固定 1024 已是 +25%，自适应只在"prompt 远小于 1024"时有边际收益；低风险、可顺手做。
- **P2 M2 后缀草稿**：本移植的 lookup 已经亏 8%；除非用上游的"trigram 索引 + 最长匹配 + 4 路候选"重写，否则不立项。
- **P3 M4 专家热集持久化**：跨请求保留热专家（对 Gyro-M 92 GB 这类 >RAM 的包才有意义；Gyro-S 不适用）。

## 4. 结论

**下一阶段的目标定为 M1 的本移植版本：稠密层配套重打包（trellis）+ CLI/app 工作流整合。**
它是唯一一条既能把每 token 字节数砍到 1/2.6、又与本机最快内核（87 GB/s）对齐的路线，代价是稠密层的画质，
因此必须做成**可选伴随包**并提供质量代理与回退。

## 5. 实施进展：伴随包已经能产出（1381 MiB）

### 5.1 命令

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File port/tools/make-compat.ps1 -Pack Gyro-S-TQ1_0.gguf
```

内部就是 `llama-quantize` 的三个开关组合：

```text
--only-tensors '^(?!.*(_exps|per_layer_token_embd)).*$'   # 只写稠密层
--tensor-type ".*=TQK6"                                    # 重打成 trellis
--allow-requantize  in.gguf out.gguf Q8_0                        # 源已是量化格式；Q8_0 是窄张量的兜底
```

### 5.2 实测（Gyro-S，本机）

| 项 | 数值 |
| --- | --- |
| 选中张量 | 1224 |
| 稠密源大小 | 3701 MiB |
| **伴随包** | **1370 MiB（文件 1381 MiB）** |
| 压缩比 | **2.7x** ✓ 与"1/2.6"的推算一致 |
| 走兜底的窄张量 | 98（`ncols` 不能整除 128 ⇒ Q8_0） |
| 耗时 | 约 **17 分钟**（trellis 编码很贵，一次性成本） |

### 5.3 过程中补上的两个真实缺口（都在 overlay 里）

1. **量化器缺少本移植类型的兜底**：`tensor_type_fallback` 对 TQK6/TQK7/TQ2_T/PTQ1_0 没有条目，
   320 宽的稠密张量直接抛 `no tensor type fallback is defined` ⇒ 已加条目，回退到 Q8_0（与 Q6_K 的做法一致）。
2. **行数据校验器不认识 TQK6/TQK7**：`ggml_validate_row_data` 是"逐个类型列 case、default 拒绝"，
   PTQ1_0/TQ2_T 早前加过，TQK6/TQK7 没有 ⇒ 量化后校验报 `invalid type 145`，整次量化失败 ⇒ 已补上两个 case。

### 5.4 还差什么

- **引擎侧 `--compat`**：让 `llama_model_loader` 用伴随包覆盖同名稠密张量的 (file, offset, **type**)，
  这样模型的张量按 TQK6 的形状建立、数据从伴随包读；缺失即回退原包。
- **质量代理**：同段文本的对数似然/困惑度对比。
- **CLI/app 入口**：`strata-vk --compact` 与 `setup.py` 的 gyro 分支提示。

## 6. 伴随包第一次端到端：**更慢、且画质被破坏** —— P0 的原始形式不成立

### 6.1 接通的部分（保留）

`--compat <file>` 已接通：`llama_model_loader` 把伴随包作为**额外文件**登记、映射，
并按张量名用伴随包的条目**替换**主包的同名条目（`get_tensor_meta` 返回权重自带的张量 ⇒ 模型按伴随包的类型建张量、
数据从伴随包读）。缺失时打印 `cannot open the companion pack ...`、形状不符时保留模型自己的张量，默认行为完全不变。

| 检查 | 结果 |
| --- | --- |
| 无伴随包（默认） | **10.0 tok/s**，参考 32 token **逐 token 相同** ✓ |
| 伴随包缺失 | 打印警告、继续用主包 ✓ |
| 伴随包存在（1224 稠密张量 → TQK6） | **7.1 tok/s** ✗ |

### 6.2 结果与解读

1. **更慢**：10.0 → **7.1 tok/s**。稠密层被换成 TQK6 后走的是**trellis 的通用路径**，它在这些形状上比 Q5_K/Q6_K 的
   量化内核慢 —— 本机 trellis 的 87 GB/s 是在**专家 matvec**（大 m、k=2560、专用 kernel）上测的，不能外推到稠密层。
2. **画质被破坏**：输出变成 `17,47,17,594,594,155794,...` 这类退化串。稠密层从 5.5 bit 直接降到 2.125 bit
   **远超**这个模型的稠密层能承受的量化强度（专家的 2.125 bit 是训练时就用 trellis 编码的，稠密层不是）。

### 6.3 结论

**"把稠密层整体重打成 TQK6" 这条路被实测否决**：既慢又毁画质。伴随包机制本身是通用且可用的（保留，默认关闭），
但 P0 需要换形式。可试的三种（按代价从低到高）：

| 形式 | 预期 | 风险 |
| --- | --- | --- |
| **只重打输出层**（`output.weight` Q6_K → Q5_K/Q4_K，占每 token 12%） | +5-8% | 画质轻微 |
| **只重打最大的少数张量**（按字节排序取前 8-16 个，用**轻度**类型如 Q5_K→Q4_K） | +3-6% | 画质轻微 |
| **先修 trellis 的稠密 matvec 路径**（让它在 k=2560、m=2560..10240 上达到专家 kernel 的效率） | 才谈得上 +30-45% | 大工程 |

下一轮从第一种开始（最便宜、可立即量化），并把"伴随包"的定位从"加速"改为"可选的量化/重布局工具"。

## 7. 第二种形式（只重打输出层）：画质温和，但**没有可测的提速**

### 7.1 产物

`--only-tensors '^output\.weight$' --tensor-type '.*=Q5_K'` ⇒ 497.31 MiB → **416.80 MiB**（Q6_K → Q5_K，−16%，文件 427 MiB）。

### 7.2 速率（ABBA 两对，`--tokens 9707 --n 8`）

| 对 | 无伴随包 | 有伴随包 |
| --- | --- | --- |
| 1 | 11.3 | 11.0 |
| 2 | 11.2 | 11.4 |

成对中位差 **−0.45%**（一对 −2.7%、一对 +1.8%）⇒ **低于本会话 ~0.5% 的噪声底，视为无效果**。

### 7.3 质量代理（新工具 `port/tools/logit-diff.py`）

同一 prompt 各 `--dump-logits` 一次（vocab 248 320）：

| 指标 | 结果 |
| --- | --- |
| top-1 | **相同**（220 vs 220）✓ |
| top-10 重合 | **100%** ✓ |
| 对称 KL | **0.0073 nats** |
| max \|Δ\| | 0.5679 logits |

⇒ 只改输出层是**画质温和**的改动，但没有速率收益。

### 7.4 到此为止的结论

| 形式 | 速率 | 画质 |
| --- | --- | --- |
| 全稠密 TQK6（1370 MiB） | **7.1 tok/s（−29%）** | 退化输出 ✗ |
| 只输出层 Q5_K（427 MiB） | **无效果（−0.45%）** | KL 0.0073、top-1 相同 ✓ |

**伴随包机制（工具 + `--compat` + 回退 + 质量代理）已经完整可用，但在本机它带不来吞吐**：
decode 是**内核效率**受限，不是字节数受限 —— 与第五阶段以来的所有测量一致。
因此伴随包保留为**可选的量化/重布局工具**（用户可用它换体积或画质），不再作为加速手段推进；
想真正提速，唯一的路是把 **trellis 的稠密 matvec 路径**做到专家 kernel 的效率（大工程，需要 shader 侧 profiling）。

### 7.5 工作流入口（已完成的部分）

| 入口 | 位置 |
| --- | --- |
| 产出伴随包 | `port/tools/make-compat.ps1`（封装 `llama-quantize --only-tensors/--tensor-type/--allow-requantize`） |
| 使用伴随包 | `strata-vk --compat <file>`（`--help` 里有说明；缺失/形状不符自动回退） |
| 质量代理 | `port/tools/logit-diff.py` + `--dump-logits <file>` |
| `setup.py` / 桌面 app | **未接**：需要 gyro 家族分支里加一步"生成/选择伴随包"，属于产品侧改动，留待确认 |
