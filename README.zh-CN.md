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
| 工具 | `port/tools/` | 拉取与构建脚本、测量工具（`batch-serve.py`、`energy-ab.py`、`two-serve.py`）、门禁（`check-encoding.py`、`audit-docs.py`、`check-pin.py`） |
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

Snapdragon X2 Elite Extreme（18 个 Oryon 核、47.6 GB 统一内存、Adreno X2-90 on Vulkan 1.4、驱动
`30.0.228.10000`），Windows 11 ARM64。模型 `Gyro-S-TQ1_0.gguf`（54.47 GiB，`qwen4exp`，trellis 专家）。
decode 用 `--bench-tg 16`，prefill 用 `--bench-pp 512`。同一二进制，多次运行取中位数。

| 后端 | decode | prefill pp512 |
| --- | --- | --- |
| **CPU**（`--gpu cpu`，大核类） | **11.34 tok/s** | 21.06 tok/s |
| **Vulkan**，默认 shader | 10.68 tok/s | 19.33 tok/s |
| **Vulkan**，`--shader-family fp32`（fp32 shader） | 5.48 tok/s | **46.63 tok/s** |

所以在这个包上，单流最好的是 CPU；GPU 只赢 prefill 那一格，用 fp32 shader 家族是 CPU 的 2.2 倍。这张表里每个数
都产生自**正确**的内核组合：Vulkan 的 logits 与 CPU 后端相关性 **0.999**、选同一个 token，这才是真正的门禁
（USAGE §5）。`--shader-family fp32` 还是 35B 包上 16-token 流与 CPU 逐 token 相同的那个家族；f16 家族会在一次
并列上分叉 —— 也就是说"与参考一致"和"prefill 最快"在这里指向同一个设置。

还有一条更快的 Vulkan 路径，**默认不用**：f16-accumulate 的 cooperative matrix 在同一个包上 pp512 是 221 tok/s，
但它的 logits 与 CPU 相关性只有 **0.02**，试过的每个包上都会选到不同的 token。`STRATA_VK_F16ACC=1` 仍然可以
打开它做纯形状基准。数字在 [PERFORMANCE.md §4.1](port/docs/PERFORMANCE.md)；修好那条内核是 GPU 侧最大的
待办。

## 状态

- **正确性优先。** 默认路径都对 CPU 后端验证过：logits 相关性 0.999、argmax 相同，32 槽批量流与单条跑
  64/64 逐 token 相同。
- **Vulkan**：能加载并回答 `qwen35`、`qwen35moe`、`qwen4exp`（含 trellis 的 `Gyro` 包与 `Q2_0` 的 `GSQ-RCO`
  包），以 CPU 为参考后端。decode 在这台机器上还没有比 CPU 快。
- **CPU**：这里单流最快的后端，但只用大核类（`-t 18` 会掉 decode，且没有收益）。
- **批量**：`--parallel N` 在一个进程、一份权重上跑 N 个请求。它是正确的，但在这台机器上不划算：32 槽每步
  3178 ms ≈ 32 个单条步，聚合 **1.00x**。`serve/server.py` 没有接。
- **NPU**：Hexagon 路径能构建，但在这个驱动上开不出 DSP 会话（`remote_handle64_open` 对每个模块名都返回
  `AEE_ERPC 0x200`）；在那之前能测到的每种放置都比 GPU 慢。要复活它得新写 QNN 后端，而不是修 URI。
  [NPU-FEASIBILITY.md](port/docs/features/NPU-FEASIBILITY.md) 有实测。
- **图片**：引擎只读文本；应用的图片路径需要上游的 `strata-vision` 编码器。
- **DeepSeek-V4.1**：这个组件不支持，引擎会在 serve 行上明说。

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
