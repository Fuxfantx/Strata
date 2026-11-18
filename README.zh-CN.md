<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h3 align="center">Strata for Windows on ARM</h3>
<p align="center">面向 Snapdragon X2 / Adreno X2 的 Vulkan 引擎，正确性优先。</p>

<p align="center">
  <a href="#这个分支是什么">这是什么</a> ·
  <a href="#快速开始">快速开始</a> ·
  <a href="#本机实测">实测</a> ·
  <a href="#状态">状态</a> ·
  <a href="#文档">文档</a> ·
  <a href="#门禁">门禁</a>
</p>

<p align="center"><sub><a href="README.md">English</a> · <b>简体中文</b></sub></p>

**这是 [Strata](https://github.com/Niko1221/Strata) 的 `windows-arm64-vulkan` 分支。** 上游引擎是手写的 CUDA / HIP /
SYCL 内核集，没有一个能在 Adreno GPU 上跑，所以这个分支只加了一样东西：**`strata-vk`** —— 一个说 Strata
`--serve` 协议的 Vulkan 引擎组件，以及为了让 vendored `llama.cpp` 认识这些模型家族和低位类型而做的 overlay。
外围的前端（桌面应用、Terminal、`setup.py`、Python 服务器）是上游的，不用改就能驱动这个引擎。

本分支特有的东西都在 `port/`。移植只在一台机器上开发和测量：一台 Snapdragon X2 Elite Extreme 笔记本。数字属于
那台机器，不是硬件要求。

## 这个分支是什么

| 部分 | 位置 | 做什么 |
| --- | --- | --- |
| 引擎 | `port/engine/` | `strata-vk`：加载与放置、前缀复用、采样、token 循环、`--serve` 行协议、`--parallel` 请求槽 |
| Overlay | `port/llama-overlay-vanilla/` | 相对 pin 住的 vanilla `llama.cpp` 的跟踪差异：模型家族（`qwen35`、`qwen35moe`、`qwen4exp`、trellis 包）、低位类型与它们的 Vulkan 内核、MTP/NextN 嫁接、loader 回退 |
| CPU 与 GPU 同一个构建 | 同一份 | Vulkan 后端与 CPU 后端（含 repack 内核）都在 `strata-vk` 里，`--gpu` 选一个，`test-backend-ops` 能对比两者 |
| 图片编码器 | `port/engine/strata-vision.cpp` | `strata-vision`：llama.cpp 的 mtmd + mmproj，说前端的 `ENC <image> <out>` 行，写出引擎读的 `.sve`；CPU 或 Vulkan，用 `--gpu` 选 |
| 工具 | `port/tools/` | 拉取与构建脚本、测量工具（`batch-serve.py`、`energy-ab.py`、`two-serve.py`、`gguf-tokenizer.py`）、门禁（`check-encoding.py`、`audit-docs.py`、`check-pin.py`） |
| 文档 | `port/docs/` | [索引](port/docs/README.md)：用法、性能、后端选项、维护者笔记、上游补丁、设备事实 |

## 快速开始

```powershell
# 1) 拉取 pin 住的 llama.cpp，并把 overlay 合进去，得到 third_party/llama.cpp
powershell -NoProfile -ExecutionPolicy Bypass -File port/tools/fetch-llama.ps1
# 2) 构建引擎（可选：再构建后端正确性 oracle）
powershell -NoProfile -ExecutionPolicy Bypass -File port/tools/build-engine.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File port/tools/build-engine.ps1 -Target test-backend-ops
# 3) 运行
build\bin\strata-vk.exe --native C:\models\pack.gguf --prompt "Hello" --n 32
```

引擎需要 exe 旁边有 **ARM64** 的 Vulkan loader，`build-engine.ps1` 会放好。缺了它 `ggml-vulkan.dll` 加载失败，
引擎会**静默**只注册 CPU 后端 —— 输出又慢又错。`--geometry` 必须列出 `Vulkan0`。

开关、测量旋钮和完整门禁清单：[port/docs/USAGE.md](port/docs/USAGE.md)。

## 本机实测

Snapdragon X2 Elite Extreme（18 个 Oryon 核、47.6 GB 统一内存、三通道 LPDDR5X 标称 228 GB/s、Adreno X2-90 于
Vulkan 1.4、驱动 `30.0.228.10000`），Windows 11 ARM64。模型 `Gyro-S-TQ1_0.gguf`（54.47 GiB，`qwen4exp`，
trellis 专家；每 token 读 3.86 GB 权重，其中专家只占 0.469 GiB）。解码 `--bench-tg 16`，预填 `--bench-pp 512`。
同一二进制、重复运行的中位数、一台机器。

| 项目 | 实测 | 说明 |
| --- | --- | --- |
| **Vulkan** 预填 pp512 | **43-50 tok/s** | 默认设置；比 2026-11-18 之前的同一构建 **+211%** |
| **Vulkan** 解码（单流） | 10.2 tok/s | 约 37.7 GB/s 权重流量 |
| **Vulkan** 解码（32 请求槽） | **26.7 tok/s 聚合** | 2.73×；**103 GB/s = 本 GPU 108 GB/s 上限的 95%** |
| **CPU**（`--gpu cpu`，大核类） | 解码 10.5-11.3、预填 21 tok/s | 短回复仍是单流最快的后端 |
| **Vulkan + CPU 双进程** | **29.1 tok/s 聚合**（25.3 + 3.8） | CPU 那半塌到独占的 37%：它是指令受限，不是带宽受限 |
| CPU 读带宽 | **154.6 GB/s**（14 线程） | 标称值的 68%；4 线程就到 147 |
| Vulkan 读带宽 | 约 108 GB/s | |
| 两者同时 | 177 GB/s | 标称值的 78% |

这张表说两件事。**预填现在默认就有 CPU 的 2.4 倍**，来源是一处改动：int-dot 的 mat-mat 从 **fp32 模块**取
shader，而 mat-vec 与 elementwise 仍留在 fp16 模块，于是 `--shader-family fp32` 里"预填"那一半是免费的，
而它的解码代价不用付。**批量解码已经贴到内存墙**（103/108 GB/s），所以批量再推不动，更高的数字只能从 CPU
一侧来（见下文"状态"）。

cooperative matrix 这条路**还不是又对又快**，原因都有测量：

- f16-accumulate 的 cooperative tile 曾报 pp512 221 tok/s、logits 与 CPU 相关性 0.02。那颗内核的 `WN` 是 32
  而 `TN` 是 64，`cms_per_col = WN/TN` 整除成 0，MMA 循环、epilogue 与 store 全部编译成空——**它只做 staging**，
  不是"近似"。现在 tile 按本机矩阵形状重建、真的在算，但**分组**（`MUL_MAT_ID`）那份在模型形状上会把
  `-inf` 写进相邻缓冲的头 64 个元素；**dense** 那份是对的（argmax 相同、有限）却**中性**，单独用没有收益。
- `STRATA_VK_INT8_COOPMAT=1` 曾被记作"精确但不更快"，实际上它**从未创建过任何管线**：创建块在
  `coopmat_support` 之下，而本机因为没有 f32 累加矩阵该 flag 为 false。那颗精确的 int8 矩阵核心
  （64x64x32、int32 累加）至今未被使用。

每个数字及其产出命令都在 [PERFORMANCE.md](port/docs/PERFORMANCE.md)，包括已被否掉的方向。

## 状态

- **正确优先。** 默认路径对 CPU 后端验证过：argmax 相同、本包 logits 相关 0.9984、32 槽批量流与单条
  **64/64 逐 token 相同**。下面门禁里写的 0.999 是 35B-A3B 那个包的水平；Gyro-S 上两个后端都在 0.9984，
  argmax 仍然一致。
- **预填**：默认路径是本机最快的正确路径（43-50 tok/s），受 int8-dot tile 限制。cooperative matrix 是越过它的
  办法，现状如上：dense 半对、分组半坏，两边都开时 145 tok/s。
- **解码**：单流受内存与指令延迟限制（10.2 tok/s、37.7 GB/s）；批量到 103 GB/s，即 GPU 上限。再往上每条路都
  测过并关闭了：减字节会改变输出（attention 降 Q4_K 相关性 0.978、第 3 个 token 就变；连"存得过厚"的那些降
  到 Q5_0/Q5_K 也只有 0.991、第 7 个 token 变），CPU+GPU 分工只加 9%（CPU 那半无法把带宽换成 token），
  而 CPU 侧自身的瓶颈是 trellis 点积的 1.78 指令/权重。
- **批量**：`--parallel N` / `--batch-slots N` 在一个进程、一份权重上跑 N 条请求，**确实有收益**：32 槽 2.73×。
  注意 8 槽是坑（`mul_mat_vec_max_cols = 8`）、48 槽实测是下陷，曲线不单调。`serve/server.py` 尚未接上。
- **NPU**：Hexagon 路径能构建，但本驱动上开不出 DSP 会话（每个模块名都返回 `AEE_ERPC 0x200`），此前能测到的
  每种放置都比 GPU 慢。要复活它需要 QNN 后端，不是修 URI。
  见 [NPU-FEASIBILITY.md](port/docs/features/NPU-FEASIBILITY.md)。
- **图片**：通道已建好并端到端跑过——`port/engine/strata-vision.cpp` 是编码器（llama.cpp 的 mtmd + mmproj，CPU 或
  Vulkan），引擎收 `--vision` 和带 embeddings 文件路径的 `GENI` 行，`serve/server.py` 会自己拉起它。**已验证的是
  管道，不是回答**：带图请求返回 200、图片 token 进了 prompt，编码器单图 CPU 5.2 s / GPU 1.6 s、两后端 cosine
  0.99997。**没验证的是"图片被理解"**：这个 pack 的 chat 回答本身就退化，而且 Vulkan 与 CPU 输出逐 token 相同
  （即不是移植的问题），所以图片问答的好坏无法用回答来判。[features/VISION.md](port/docs/features/VISION.md)
  写明了测了什么、没测什么。
- **多槽 + 图片**：`BGENI`（多槽写法）对图片回答 `ERR`；图片只走一次一条的那条路。
- **DeepSeek-V4.1**：本组件不支持，引擎在 serve 行上会明说。

## 文档

[`port/docs/README.md`](port/docs/README.md) 是索引。按你的身份看：

| 你是 | 读 |
| --- | --- |
| **最终用户**：只想跑起来 | 上面的快速开始，然后 [USAGE.md](port/docs/USAGE.md) |
| **贡献者**：想让它更快 | [PERFORMANCE.md](port/docs/PERFORMANCE.md)（当前数字、测过什么、排除了什么）与 [BACKEND-OPTIONS.md](port/docs/BACKEND-OPTIONS.md)（还剩什么） |
| **维护者**：改 vendored 树、overlay、门禁 | [MAINTAINER.md](port/docs/MAINTAINER.md)、[UPSTREAM-PATCHES.md](port/docs/UPSTREAM-PATCHES.md) |
| **想了解这台机器** | [ADRENO.md](port/docs/ADRENO.md)（逐项实测的 Adreno X2-90 事实） |

`port/docs` 里只有一条规则：**没有本机测量的数字不写进去**；是假设的地方会明说。

## 门禁

每次改动提交前全部通过，命令在 [USAGE.md §5](port/docs/USAGE.md)：

1. CPU 与 Vulkan 的 logits 一致：相关性 >= 0.999 且 argmax 相同（两边各 `--dump-logits`）。
2. `test-backend-ops -o MUL_MAT -b Vulkan0`：0 失败。
3. `check-encoding.py`：0 个文件编码损坏。`audit-docs.py`：0 个文档问题。
4. `check-pin.py`：overlay 与 `third_party/llama.cpp` 仍然一致。

[GitHub Sponsors](https://github.com/sponsors/Niko1221) · [Buy me a coffee](https://buymeacoffee.com/strataengine) ·
[MIT license](LICENSE) · [上游 Strata](https://github.com/Niko1221/Strata)
