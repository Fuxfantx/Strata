# 视觉（mmproj）进 Vulkan 引擎：契约、`.sve` 格式与建成情况

*状态：**当前文档**。`.sve` 契约与三后端说明以此为准；吞吐方面的结论见 [PERFORMANCE.md](../PERFORMANCE.md)。 / Status: the current document for the `.sve` contract; throughput conclusions live in [PERFORMANCE.md](../PERFORMANCE.md).*

目标：让 `serve/server.py` 的多模态请求在这条 Vulkan 分支上端到端可用，图像编码器提供 **CPU / GPU / NPU** 三种后端；
**文本路径零回归**（32-token 参考流与 logits/argmax 门禁原样通过）。

## 1. 现在的位置（三面各占什么）

| 面 | 位置 | 状态 |
| --- | --- | --- |
| 前端 | `serve/server.py` | 就绪：图片走独立进程 `strata-vision`，协议 `ENC <image> <out>`；每图结果按哈希缓存，多图按顺序拼成一个 `req-*.sve`；随后发 `GENI`/`BGENI … <path>` |
| 编码器 | `port/engine/strata-vision.cpp` | 就绪：`--gpu cpu` 与 `--gpu`（Vulkan）都能量出 embedding，两者余弦 **0.99997**；每图 5.2 s（CPU）/ 1.6 s（GPU）|
| 引擎 | `port/engine/` | 就绪：`--vision` 装载 encoder，`GENI` 读 `.sve`、替换标记、填 `embd`；`BGENI` 对图片仍回 `ERR` |
| 上游树内 | `third_party/llama.cpp/tools/mtmd/` | 编码器以内嵌方式使用（mtmd），未作为独立 CMake 目标导出 |

关键事实：**前端从不解析 `.sve`**，它只做"逐图缓存 + 顺序拼接"。所以 `.sve` 是**编码器与引擎之间的内部契约**，
两端都由本项目实现——不需要与任何上游二进制的布局字节兼容，这也是这个特性可行的前提。

## 2. `.sve` 布局（每段自描述，因此可拼接）

```
每张图一段，紧接上一段，无填充：
  u32  magic        0x31564D53  ('SVM1')
  u32  n_tokens     该图占用的 token 数
  u32  n_embd       每行浮点宽度（= 主模型 n_embd）
  u32  flags        保留（0）
  i32  token_ids[n_tokens]
  f32  embd[n_tokens * n_embd]     行优先，逐 token
```

- 引擎按 magic 逐段读，直到文件尾；段数 = 图片数。
- `mmproj` 的标记（`mtmd_default_marker()`，即 `<__media__>`）在 prompt 里由前端插入；引擎把每次出现替换为对应的
  `token_ids` 段，并把 `embd` 填进 `llama_batch.embd`（`llama_batch_init(.., embd, ..)` 的 embedding 通道）。
- `n_tokens` 必须等于该段的 `token_ids` 长度；引擎校验，不符则 `ERR GENI: bad image segment`。

## 3. 三个后端

| 后端 | 载入方式 | 状态 |
| --- | --- | --- |
| **CPU** | `--gpu cpu` → ggml-cpu；`--flash-attn` 对编码器单独生效（`docs/DETAILS.md:1013`：AVX-512 上关掉，因为 ggml 的 fallback 慢数倍） | 先做，最稳 |
| **GPU** | `--gpu vulkan` → 本分支的 ggml-vulkan（本轮修好的 cooperative 路径对 ViT 的 GEMM 同样适用） | 第二步 |
| **NPU** | `--gpu npu` → ggml-hexagon。DSP 侧有 `conv2d`/`im2col`/`pad`/`upscale`（`ggml-hexagon/htp/`），且本仓库已确认**签名 skel 里含视觉编码器算子**（`port/tools/fetch-signed-htp.ps1:6`），并记录过"陈旧 catalog 让 NPU 看起来没有视觉算子"这个坑（`port/tools/build-engine.ps1:99`） | 第三步，**被两个已知 bug 挡住**：`dspqueue_create 0x14`（会话能开、队列建不起来）与设备切分（`resolve_fused_ops` 把节点切到 CPU）；见 `PERFORMANCE-LOG.md` §5.4、`NPU.md` |

## 3.1 V1 已落地：`strata-vision`（2026-11-19）

`port/engine/strata-vision.cpp`，由 `port/engine/CMakeLists.txt` 建出 `bin\strata-vision.exe`（树里有 `tools/mtmd`
才建，即 `LLAMA_BUILD_TOOLS=ON`，也就是这个 port 的构建）。协议与 `serve/server.py` 的 `Vision` 类逐条对齐：
`READY <n_embd> mtmd` → `ENC <image> <output>` → `OK <n_tokens>`，输出是 §2 的自描述布局。

**实测**（本机；`mmproj-Qwen3.8-Flash-Next-BF16.gguf` 865 MiB + Gyro-S；`img50.jpg` = 45 个图片 token；每一步都有 watchdog）：

| 后端 | 启动到 READY | 单图编码 | 输出 |
| --- | --- | --- | --- |
| CPU（`--threads 8`） | 0.9-1.0 s | **5.23 s** | 460,816 B = 16 + 45×2560×4 |
| GPU（`--gpu`，Vulkan0） | 1.0 s | **1.61 s**（3.2×） | 同上 |

* CPU 连跑两次**字节相同**（sha256 `3e5c2b24…`）⇒ 确定性。
* CPU 与 GPU 的 115,200 个数值 **cosine 0.999972** ⇒ 两后端等价（差异是 bf16 kernel 的求和顺序）。
* 文本路径没动：32-token 参考流 `16451,18,13,20,12,21135,487,328,…` 不变；`strata-vision` 是独立进程，`strata-vk` 未改。

**两个只在这台机器上暴露的坑（已在工具里处理）：**

1. **`--gpu` 不能只说"要 GPU"**：这个 fork 的 `ggml_backend_dev_type` 把 Vulkan 注册成 `ACCEL(2)`、把 Hexagon NPU
   注册成 `GPU(1)`，而 mtmd 的默认选择是"第一个 GPU"⇒ 选中 **HTP0**，接着就是
   `ggml-hex: HTP0 failed to open session : uri file:///libggml-htp-v81.so … error 0x200`（与引擎同一个未签名/无目录
   URI 问题）。工具现在**按名字**挑 `Vulkan*`。
2. **`vocab_only` 在这棵树里不读超参**：`llama_model_n_embd`/`_inp`/`_out` 全为 0（词表可用、mtmd 能编码，只有宽度
   拿不到）。工具改从 GGUF 头读 `<arch>.embedding_length`，并与 mmproj 的 `clip.vision.projection_dim` 交叉校验：
   两者都是 **2560** ✓。

## 3.2 V2：引擎侧的 GENI 通路（2026-11-19）

引擎不再对图片说"只读文本"，而是：

* `--mmproj FILE` 被接收（原先在忽略表里被吞掉），只记录路径——**投影器在读者进程里跑，引擎不加载它**，所以加载时间不变。
* `GENI <max_new>[ key=value]* <embeddings 文件> <ids>`：读 `.sve` 的每段记录，按 `<|image_pad|>` 的**连续段长度**把图片
  的行定位到 prompt 里（一段的长度必须等于该记录的 `n_tokens`，顺序与记录一致；任何不匹配都 `ERR`，不做猜测）。
* 预填被图片切成段：文本段走 token batch，图片段走 **embedding batch**（`llama_batch_init` 的一个 batch 只能是要么
  token 要么 embedding，所以一张图是独立的一次 `llama_decode`），位置连续递增。
* 没有 `--mmproj` 时，`GENI` 回答 `ERR this engine reads text only: ...`；`BGENI`（批槽）目前回答不支持图片。

**已实测（本机，每次都带 watchdog）**：

| 项目 | 结果 |
| --- | --- |
| `GENI` + 一张 45 行图片（prompt 53 ids，含 45 个 `248056`） | 引擎日志 `GENI 8 tokens, prompt 53, 1 picture(s)`，随后 `DONE 0 53 6233.057 ...`，exit 0 |
| 文本路径（无 `--mmproj`） | 32-token 参考流**逐字节相同** |
| 文本路径（带 `--mmproj`） | 同上，且与不带时**彼此相同** |

**还没做、不要当成已完成**：

* 前端端到端（`serve/server.py` 真发一张图）**尚未跑通**；上面那次是直接按协议喂 `GENI` 的。
* 那次 `DONE` 的 `generated` 是 **0**（finish=`stop`）：测试用的 prompt 是无意义的 token 串，采样第一步就撞到 EOG。
  这**不能**说明图片被理解，只能说明行被正确接进了预填。
* 那张图 45 行的 embedding 预填花了 **6.2 s**（`prompt_ms 6233`），比文本慢得多，原因未查——可能是 embedding batch
  在这个后端走了一条慢路径。这件事在 V3 之前必须查清，否则带图的请求会明显更慢。
* `.sve` 路径**不能含空格**（`GENI` 按空格分词；前端 `Vision.work_dir()` 已保证这一点，本机测试踩过一次）。

## 3.2 V2/V3 已落地：GENI 通路 + `--vision`（2026-11-19）

引擎侧（`vk-engine.h` 的 `ImageSpan`、`vk-serve.cpp` 的 GENI 解析、`vk-options.cpp` 的 `--vision`）：

* `--vision` 是**前端真正传的旗标**（`server.py:3945-3946` 只 append 裸 `--vision`，`--mmproj` 只给读者进程）。
  它原先在 `kIgnoredValueFlags`（带值但不用）里，会把后面一个旗标吞掉——已移除，改为裸旗标。
  门是 `vision_enabled() = --vision || !mmproj.empty()`。
* `GENI <max_new>[ k=v]* <.sve 路径> <ids>`：读记录、按 `<|image_pad|>`（本模型 id `248056`）连续段定位图片、
  长度不符即 `ERR`；预填在图片处切段（文本走 token batch，图片走 embedding batch，位置连续）。
* `BGENI`（批槽）明确回答不支持；没有标志时 `ERR this engine reads text only: start it with --vision (and --mmproj) to read pictures`。

**已验证**（每次都有 watchdog）：

| 实验 | 结果 |
| --- | --- |
| 文本 32-token 参考流（无标志 / 有 `--vision` / 有 `--mmproj`） | 三种情况**逐字节相同** |
| `GENI` + 45 行图片 | 正常受理，`DONE 4 52 … length`，exit 0 |
| 无标志的 `GENI` | 明确拒绝 |
| 图片预填的行数缩放（5 / 20 / 45 行，同一会话） | prompt_ms 约 **0.96-1.47 s，几乎与行数无关**；5 行首次 2.9 s |

**图片行的真实成本（2026-11-19，`STRATA_VK_SEG_LOG=1` 的分段计时，同一会话）**

> 上一版本这里写的是"图片有 1-1.5 s 固定开销、比文本高两个数量级"。**那是错的**：我把整段 6.2 s 都算到了图片头上，
> 而分段计时显示文本段才是大头。按分段重测：

| 请求 | 分段 | 每行 |
| --- | --- | --- |
| 纯文本 52 行（`GEN`） | 2256 ms（首次）/ 1922 ms（再来一次） | 37-43 ms/行 |
| 5 文本 + 45 图片 + 3 文本（`GENI`） | 947 / 3353 / 1668 ms（首次） | 图片 **~75 ms/行** ≈ 文本行的 1.75x |
| 同上，第二次 | 362 / 3360 / 233 ms | 同上 |

**结论**：图片行走 embedding 路径，单价约为文本行的 **1.75x**，没有数量级异常。整请求 4-6 s 的主因是
**这台引擎在小 prompt 上的预填速率本身**（52 行 ≈ 27 tok/s，而 pp512 是 85 tok/s——就是 `PERFORMANCE.md`
记的那条"预填随 token 数呈阶梯"的形状），加上**每个请求一次的一次性开销**（3 行的尾段仍要 230-1670 ms）。
首请求更贵，约 1.5-2x。

**仍未做**：`serve/server.py` 的端到端从未跑过（它要 setup 写出的 run config：`exe`/`args`/`cwd`/
`tokenizer`/`model_name` + `vision` 段，这份 checkout 没有）；`BGENI`（多槽）没有图片支持。

## 3.3 端到端（`serve/server.py` → `strata-vision` → `GENI` → 引擎），2026-11-19

用手写的 run config 跑通（这份 checkout 没有 setup 写出的配置；tokenizer 目录用
`port/tools/gguf-tokenizer.py` 从 GGUF 导出，round-trip 通过，248320 tokens / pre `qwen35`）：

| 请求 | HTTP | 墙钟 | prompt_tokens | completion_tokens |
| --- | --- | --- | --- | --- |
| 纯文本 | 200 | 2.4 s | 58 | 2 |
| **带图**（base64 JPEG） | **200** | **14.4 s** | **107**（含图片 token） | 0 |
| 图之后再来一条文本 | 200 | 2.0 s | 61（cached 0） | 2 |

**这一段证明了的**：前端确实把图片交给 `strata-vision`、拿到 `.sve`、把路径挂进 `GENI`，引擎把图片行接进预填
（带图请求多花了 12 s 预填、prompt_tokens 从 58 涨到 107）。**这一段没证明的，也写在前面**：三条请求的
`message.content` 都是 `null`、`finish_reason` 都是 `length`——**纯文本请求也一样**，所以这是
server↔engine 既有的文本路径不一致，与视觉无关，但它挡住了"多模态可用"这句话。修它之前，这里只能算
"图片通路打通，答案还没验证"。

**环境坑（本机特有，不是移植缺陷）**：`serve/server.py` 把编码器的工作目录建在 `%TEMP%`（它优先选"路径无空格"
的父目录，而本仓库路径含空格，于是回落到 `%TEMP%`）。这台机器上**本工程构建的可执行文件在 workspace 之外的写入
一律被拒绝**（`%TEMP%`、用户主目录、`ProgramData` 全部 `EACCES`；`python.exe`/`cmd.exe` 反而可以），所以编码器写不出
`.sve`。诊断信息已经进了编码器（`cannot open '<file>' for writing in '<cwd>': <strerror>`），重试**无效**（不是瞬时竞争）。
可用绕过：建一个无空格的 junction 指向 workspace 内的目录（`mklink /J C:\svwork <repo>\.tmp-encoder`），
再让服务进程用 `TEMP=C:\svwork` 启动——实测编码器能写、请求返回 200。

### 3.4 "回答为空"查到最后：不是引擎的问题（2026-11-19，第三次更正）

这一条我改过三版，前两版都错了，把过程留下来，因为**错误的形状本身有信息**。

1. 第一版说"服务端↔引擎文本路径不一致"——方向对，没定位。
2. 第二版说"任何长度 ≥2 的 prompt 都生成 0 个 token"——**错**。那是拿
   `248045,846`（`<|im_start|>user`，后面什么都没写）当探针：模型在那里**正确地**预测 `<|im_end|>`
   （argmax 248046，23.7），空 turn 合上是很合理的。换成真实续写任务就正常了：

   | prompt | 生成 |
   | --- | --- |
   | `Paris is the capital of France.` | `\n\nParis (French pronunciation: [` ← 正常续写 |
   | `def add(a, b):` | ` a = 10 defdefdef b` ← 退化重复 |
   | `The capital of France is` | 一个空格就停 |
   | 15-token 的 chat prompt | `n\n` 之后停 |

3. **真正的判定**：同一个 chat prompt 在 **Vulkan 与 CPU 后端上输出完全相同**（都是 `77,198`）。
   CPU 后端就是 llama.cpp 未改动的参考内核，所以**引擎的文本路径没有偏差**：那个回答是**权重/量化**的行为，
   不是移植缺陷。`--gpu cpu` 与 `--gpu vulkan` 在这里互为判据，这也是以后判"模型问题还是引擎问题"的第一手段。

**结论**：端到端通路是通的（HTTP 200、图片 token 进入 prompt、引擎消费 `.sve`）；回答质量差是
**1.63 bit/weight 的 trellis 量化包**在这个 chat 任务上的表现（同一 pack 做纯文本续写正常，说明损坏在
指令跟随而不是解码本身）。要展示"图片被理解"，需要一个没被压到这么低的包，或换 chat 模板/风格再试。

**留下的工具**：`STRATA_VK_DUMP_IDS=1` 把 GEN 的 prompt 打成 id（配 `--detok` 核对渲染）；
`STRATA_VK_ADMIT_DEBUG=1` 打印采样器实际读的那一行的 argmax 与前二名。

## 4. 已建成与未验证

| 项 | 状态 | 证据 |
| --- | --- | --- |
| `ENC` → `strata-vision` → `.sve` | **已建成** | 同一张图跑两次字节相同；`n_tokens` 与 `mtmd_image_tokens_get_n_tokens` 一致 |
| `--gpu cpu` 与 `--gpu vulkan` 的 embedding | **已建成** | 余弦 **0.99997**；5.2 s 对 1.6 s |
| 引擎 `GENI` + `--vision` | **已建成** | 图片请求返回 200，图像 token 在 prompt 里 |
| 前端端到端（`server.py` → 编码器 → `GENI`） | **已建成** | 一次图片请求跑通并写出 `req-*.sve` |
| **图像理解本身** | **未验证** | 这个包的聊天回答在两个后端上同样退化（Vulkan 与 CPU 对同一 prompt 给出相同 id），所以"答案对不对"无法用回答来判断 |
| `BGENI` 携带图片 | **未实现** | 多槽位拼写对图片回 `ERR`，只有单请求路径带图 |
| `--gpu npu` 编码器 | **未实现** | 取决于 NPU 会话问题，见 [NPU.md](NPU.md) |

"""一次更正记录"""(历史) 这一节原先散落在正文里，已并入 [PERFORMANCE-LOG.md](../PERFORMANCE-LOG.md)；本页只保留上述结论。

## 5. 边界与风险

这个特性是 **opt-in**：只有给了 `--mmproj` / `--vision` 才装载编码器；`GEN`/`BGEN` 的路径与字节输出不变（用 32-token 参考流验证）。

- **精度**：编码器的 embedding 与上游 QNN/CPU 实现不可能逐位相同；判据是"答案内容正确 + 余弦 ≥ 0.999"，不是字节相等。
- **内存**：mmproj BF16 865 MiB / Q8_0 590 MiB（`docs/DETAILS.md:1010`），在 47.6 GiB 机器上与 54.5 GiB 的 mmap 包共存
  需要看 `--load-mode`；CPU 编码器峰值约 1.15-2.2 GiB（`docs/DETAILS.md:1452`）。
- **不碰文本路径**：`GENI` 只在 `.sve` 存在时走新分支；`GEN`/`BGEN` 的代码路径不变，门禁照旧。

### The ENC line accepts quoted names, and where it may write

The protocol line is `ENC <image> <output>`.  Names without spaces work as before; a name with a space has to be
quoted, because the picture path is the one field that cannot otherwise be delimited:

```
ENC "C:/path with spaces/picture.png" out.bin
```

The output is the rest of the line, so it keeps its spaces unquoted; quotes around it are stripped when present.  The
front end sends relative, space-free names, so nothing about its behaviour changes.

One measured property of this machine's setup, written down because it looks like a tool bug and is not: the encoder
may only write **inside the workspace**.  A request whose output names a path outside it (for example under the system
temp directory) fails with `Permission denied` even when the same directory is writable from the shell, and an encoder
run whose directories live inside the workspace succeeds.  This is worth remembering when a front end is pointed at a
cache directory elsewhere.
