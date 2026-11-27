# port/docs — 从这里开始 / Start here

这个目录**只保留当前有效**的内容：能用来构建、运行、测量与维护这个移植的东西，以及"别再试"的实测清单。
被推翻的结论留在原处并加撤回注记（指向新证据），因为它们本身就是测量结果；已经没有读者的过程文档直接删除。

This directory keeps **only what is current**: how to build, run, measure and maintain the port, plus the measured list of
what not to retry. A withdrawn conclusion stays where it is with a correction note pointing at the new evidence, because
the withdrawal is itself a measurement. Documents with no remaining reader are deleted rather than archived.

## 按读者分的三条路径 / Three reading paths

### 最终用户 / End user — 只想把它跑起来

1. [`../../README.zh-CN.md`](../../README.zh-CN.md)（或 [`../../README.md`](../../README.md)）的 Quick start；
2. [`USAGE.md`](USAGE.md) §1-§3：构建、`--serve`、**全部开关**（含每个开关值多少）；
3. 想知道自己这台机器大概什么速度：README 的"本机实测"表就是唯一一张。

不需要读的：其余所有文档。它们写给改代码的人。

### 任何读者 / Anyone — 想知道什么还不靠谱

[`KNOWN-ISSUES.md`](KNOWN-ISSUES.md)：正确性覆盖、性能上限、设备与环境的已知问题，每条后面跟着说它成立的那个测量。
`--gpu vulkan` 与 `--gpu cpu` 跑同一个 prompt 比 id，是区分"模型本来如此"和"移植改了它"最快的一步。

### 贡献者 / Contributor — 要让它更快

按这个顺序读，前面的给出判据，后面的给出细节：

1. [`PERFORMANCE.md`](PERFORMANCE.md) — **当前结论**：现在的数字、一个 token 的时间去向（CPU 与 GPU
   瓶颈完全不同）、已进入代码的改动、**已被否掉的方向及否掉它的那个数**、还剩什么；
   细节与逐次运行记录在 [`PERFORMANCE-LOG.md`](PERFORMANCE-LOG.md)。
2. [`BACKEND-OPTIONS.md`](BACKEND-OPTIONS.md) — 后端与旋钮：Vulkan / CPU / HTP 各值多少、哪些旋钮真的改数字。
3. [`ADRENO.md`](ADRENO.md) — 这台机器的设备事实：带宽（含为什么到不了标称 228 GB/s）、Vulkan 特性、逐项测量。
4. [`USAGE.md`](USAGE.md) §4 — 测量工具与开关（哪些读数可信、哪些开关改行为、哪些只打印）与 §5 门禁。
5. [`features/`](features/) — 只在需要时读，结论已并入上面几份；图像那条线从
   [`features/VISION.md`](features/VISION.md) 进。
6. [`KNOWN-ISSUES.md`](KNOWN-ISSUES.md) — 动手之前先看：有些"还能再快"的直觉已经在里面被数字否掉了。

**先翻 `PERFORMANCE.md` 的"已关闭路线"再去试任何"显而易见"的优化**：那里每一条都附了否掉它的数字，
细节在 `PERFORMANCE-LOG.md`。

### 维护者 / Maintainer — 要改 vendored 树、overlay 或门禁

1. [`MAINTAINER.md`](MAINTAINER.md) — overlay 与 `third_party/llama.cpp` 的机制、换 pin、门禁清单；
2. [`UPSTREAM-PATCHES.md`](UPSTREAM-PATCHES.md) — overlay 里每个补丁做什么、哪些值得回馈上游；
3. [`USAGE.md`](USAGE.md) §5 — 门禁命令（logits 相关性、`test-backend-ops`、编码、文档、pin）；
4. [`../../port/tools/`](../tools) — 构建、测量、门禁与打包脚本。

## 文档清单 / What each file is

| 文件 | 内容 | 读者 | 状态 |
| --- | --- | --- | --- |
| [`USAGE.md`](USAGE.md) | 构建、运行、全部开关、测量工具与开关、门禁命令 | 全部 | 当前 |
| [`PERFORMANCE.md`](PERFORMANCE.md) | **性能结论**：当前数字、时间去向、已进入代码的改动、已关闭路线及关闭它的数、测量纪律、还剩什么 | 贡献者 | 当前 |
| [`PERFORMANCE-LOG.md`](PERFORMANCE-LOG.md) | 原始测量流水账（追加式，含被推翻的结论）——是证据，不是指南 | 贡献者、维护者 | 记录，仅追溯细节时读 |
| [`BACKEND-OPTIONS.md`](BACKEND-OPTIONS.md) | 后端选择与旋钮：Vulkan / CPU / HTP 各自值多少 | 贡献者、用户 | 当前 |
| [`ADRENO.md`](ADRENO.md) | Adreno X2-90 的设备事实与逐项 Vulkan 测量 | 贡献者 | 当前 |
| [`MAINTAINER.md`](MAINTAINER.md) | vendored 树与 overlay 的机制、换 pin、门禁清单 | 维护者 | 当前 |
| [`UPSTREAM-PATCHES.md`](UPSTREAM-PATCHES.md) | overlay 里每个补丁做什么、哪些值得回馈上游 | 维护者 | 当前 |
| [`SHADER_CONVENTIONS.md`](SHADER_CONVENTIONS.md) | 新写着色器必须遵守的约定（命名、BDA 写法、工作尺寸、ASCII 只读约束） | 贡献者 | 当前 |
| [`features/VISION.md`](features/VISION.md) | 视觉：`ENC`/`.sve` 契约、三个后端、端到端实测与"没测什么" | 贡献者 | 当前 |
| [`KNOWN-ISSUES.md`](KNOWN-ISSUES.md) | 已知问题清单，每条附上让它成立的测量 | 全部 | 当前 |
| [`features/NPU.md`](features/NPU.md) | Hexagon 当前状态（一页）：为什么开不了会话、QNN 那扇门、值多少 | 贡献者、维护者 | 当前 |
| [`features/NPU.md`](features/NPU.md) | Hexagon 路径的完整调查记录（允许列表被证伪、tiled 路径挂死） | 维护者 | 记录 |
| [`features/NPU.md`](features/NPU.md) | 本机有哪些 NPU/抓帧资产（SDK、QNN 运行时、skel、签名墙） | 维护者 | 记录 |
| [`features/HYBRID.md`](features/HYBRID.md) | CPU/GPU/NPU 分工的完整实测矩阵 | 贡献者 | 记录（结论已并入 PERFORMANCE §2、§5） |
| [`features/GDN-FUSION.md`](features/GDN-FUSION.md) | GDN 融合的设计与实测（哪些值得做、哪些不值得） | 贡献者 | 记录 |
| [`features/EXPERT-CACHE.md`](features/EXPERT-CACHE.md) | 专家驻留策略与设备侧 cache 的设计取舍 | 贡献者 | 记录 |

"记录"= 内容仍然正确、但结论已收进上面那几份主文档；只在要追细节时读。

## 两条规则 / Two rules

1. **没有测量的数字不写进这个目录。** 每个性能数字都来自本机某次可复现的运行；是假设的地方明确标成假设。
   No number without a measurement on this machine; hypotheses are labelled.
2. **负结果和正结果一样进文档，而且写清楚是哪个数否掉了它。** 性能优化最常见的浪费是重试已经试过的方向。
   Negative results are recorded with the number that ruled them out.
