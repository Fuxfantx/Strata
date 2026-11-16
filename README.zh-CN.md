# Strata on Windows ARM64：Vulkan 引擎

这个分支在上游之上只带一样东西：**`strata-vk`** —— 一个面向 Snapdragon X2 / Adreno X2 的 Vulkan 优先引擎，
以及让 vendored `llama.cpp` 认识相应模型家族与低位类型的 overlay。围绕它的前端（桌面 App、Terminal、`setup.py`
与 Python 服务器）就是上游的那套，**无需改动即可驱动本引擎**（引擎组件的 stdout 契约 10/10 通过）。

本分支特有的内容全部在 `port/` 下，没有其他分叉：App、Terminal、`serve/` 与 `setup.py` 都是上游文件。

## 本机实测

Snapdragon X2 Elite Extreme（18 核、47.6 GB 统一内存、Adreno X2-90），Windows 11 ARM64。
decode 用 `strata-vk --tokens 9707 --n 32`，prefill 用 `--bench-pp`；多次运行取中位数。

| 模型 | decode tok/s | prefill tok/s |
| --- | --- | --- |
| Gyro-S（trellis MoE，54.5 GB） | 9.7 | 212（pp512） |
| Gyro-M（trellis MoE，61.8 GB） | 8.7 | 101（pp64） |
| Q2_0（62.2 GB，专家在设备上） | 14.0 | 41.5（pp64） |
| 27B Q4_0（dense） | 7.8 | 162（pp512） |
| 35B-A3B（MoE） | 43.1 | 613（pp512） |

每个 pack 的 32 token 输出与 CPU 后端**完全一致**。

## 构建与运行

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File port/tools/fetch-llama.ps1   # pin + overlay -> third_party/llama.cpp
powershell -NoProfile -ExecutionPolicy Bypass -File port/tools/build-engine.ps1  # -> build/bin/strata-vk.exe
build\bin\strata-vk.exe --geometry
build\bin\strata-vk.exe --native D:\models\pack.gguf --prompt "The capital of France is" --n 16
build\bin\strata-vk.exe --native D:\models\pack.gguf --bench-pp 512
```

`--geometry` 必须列出 `Vulkan0`。启动时会打印专家放置、f16 累加决策与 warmup 计划；`--dry-run` 会在**不读权重**的情况下
打印这套解析后的策略。

App 与 Terminal 通过 `setup.py` 使用引擎（下载发布版引擎，或自行构建）；构建本分支引擎就是 `build-engine.ps1` 做的事。

## 引擎支持

| pack 族 | 说明 |
| --- | --- |
| trellis `TQK6`、`TQK7`、`TQ2_T`（Gyro） | 2.125 / 1.5 bit 权重，matvec 与 cooperative-matrix 两条路径 |
| ternary `PTQ1_0` | 1.5 bit，同上 |
| `Q2_0`、`Q4_0`、`Q8_0`、`q5_K`、`q6_K` | int8-dot 与 cooperative-matrix 路径 |
| qwen4exp（3.8-Flash-Next） | hyper-connections、gated delta net、稀疏 indexer、逐层 embedding |
| qwen35moe（35B-A3B） | MoE，专家可放设备或 host |
| MTP / NextN 头 | 通过 loader 嫁接，或作为独立 context 加载 |

## 目录

| 路径 | 内容 |
| --- | --- |
| `port/engine/` | `strata-vk` 前端：CLI、session/serve、按模型调优、专家放置、NPU 探测 |
| `port/llama-overlay-vanilla/` | 铺到 pinned `llama.cpp` 上的 60 个文件（按真实路径） |
| `port/tools/` | `fetch-llama.ps1`、`build-engine.ps1`、各门禁、测量工具、`check-pin.py` |
| `port/docs/` | 移植自己的文档：快速开始、维护规则、Adreno 实测、逐补丁 provenance |
| `apps/`、`serve/`、`strata_cli/`、`strata_control/`、`setup.py` | 前端，与上游一致 |

## 下一步读什么

| 你的目的 | 读这篇 |
| --- | --- |
| 运行引擎 | `port/docs/README.md`，然后看 `port/docs/ADRENO.md` 里这台设备相关的开关 |
| 改引擎或 overlay | `port/docs/features/` 与 `port/docs/MAINTAINER.md` |
| 维护 pin | `port/docs/MAINTAINER.md` 与 `port/tools/check-pin.py` |
| 使用 Strata 本身 | `docs/`（仅英语与简体中文） |

MIT 许可，沿用上游条款；移植自己的记录在 `port/docs/`。
