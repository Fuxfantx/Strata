# port/docs — 从这里开始 / Start here

这个目录**只保留当前有效**的内容：能用来构建、运行、测量与维护这个移植的东西，以及"别再试"的实测清单。
被推翻的结论留在原处并加撤回注记（指向新证据），因为它们本身就是测量结果；已经没有读者的过程文档直接删除。

This directory keeps **only what is current**: how to build, run, measure and maintain the port, plus the measured list of
what not to retry. A withdrawn conclusion stays where it is with a correction note pointing at the new evidence, because
the withdrawal is itself a measurement. Documents with no remaining reader are deleted rather than archived.

## 读者路径 / Reading paths

| 你是 / You are | 读这些 / Read |
| --- | --- |
| **最终用户**（只想跑起来） | [`../../README.md`](../../README.md) 的 Quick start，然后 [`USAGE.md`](USAGE.md) |
| **贡献者**（要改性能/内核） | [`PERFORMANCE.md`](PERFORMANCE.md)（当前数字、已进入代码的优化、负结果、还剩什么） |
| **维护者**（改 vendored 树、overlay、门禁） | [`MAINTAINER.md`](MAINTAINER.md)、[`UPSTREAM-PATCHES.md`](UPSTREAM-PATCHES.md) |
| **选后端 / 调旋钮** | [`BACKEND-OPTIONS.md`](BACKEND-OPTIONS.md) |
| **想了解这台机器** | [`ADRENO.md`](ADRENO.md)（逐项实测的 Adreno X2-90 事实） |
| **NPU 那条线为什么关了** | [`features/NPU-FEASIBILITY.md`](features/NPU-FEASIBILITY.md)、[`features/NPU-ASSETS.md`](features/NPU-ASSETS.md) |
| **CPU/GPU/NPU 怎么分工的** | [`features/HYBRID.md`](features/HYBRID.md) |

## 文档清单 / What each file is

| 文件 | 内容 | 状态 |
| --- | --- | --- |
| [`USAGE.md`](USAGE.md) | 构建、运行、全部开关、测量工具、门禁（含 logits 相关性门禁） | 当前 |
| [`PERFORMANCE.md`](PERFORMANCE.md) | **唯一的性能文档**：当前数字、时间去向、已进入代码的优化、逐项负结果、还剩什么、实验日志 | 当前（2026-11-18 重写） |
| [`BACKEND-OPTIONS.md`](BACKEND-OPTIONS.md) | 后端选择与旋钮：Vulkan / CPU / HTP 各自值多少、哪些旋钮真的改数字 | 当前 |
| [`MAINTAINER.md`](MAINTAINER.md) | vendored 树与 overlay 的机制、换 pin、门禁清单 | 当前 |
| [`UPSTREAM-PATCHES.md`](UPSTREAM-PATCHES.md) | overlay 里每个补丁做什么、哪些值得回馈上游（含被撤回的那条） | 当前 |
| [`ADRENO.md`](ADRENO.md) | Adreno X2-90 的设备事实（带宽、Vulkan 特性、逐项测量、f16acc 的教训） | 当前 |
| [`features/HYBRID.md`](features/HYBRID.md) | 一个 decode token 由什么组成、CPU/GPU/NPU 分工的实测 | 记录（结论已并入 PERFORMANCE.md） |
| [`features/NPU-FEASIBILITY.md`](features/NPU-FEASIBILITY.md) | Hexagon 路径：怎么构建、值多少、为什么关 | 记录 |
| [`features/NPU-ASSETS.md`](features/NPU-ASSETS.md) | 本机实际有哪些 NPU/抓帧资产（SDK、QNN 运行时、skel、签名墙） | 记录 |

## 两条规则 / Two rules

1. **没有测量的数字不写进这个目录。** 每个性能数字都来自本机某次可复现的运行；是假设的地方明确标成假设。
   No number without a measurement on this machine; hypotheses are labelled.
2. **负结果和正结果一样进文档，而且写清楚是哪个数否掉了它。** 性能优化最常见的浪费是重试已经试过的方向。
   Negative results are recorded with the number that ruled them out.
