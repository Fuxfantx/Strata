# Backends and knobs — what runs where, and what to turn

**一个构建里有两条可用后端，第三个（NPU）能构建但更慢，别的都不在这个组件的范围内。**
This file says what each backend is worth on this machine, how to select it, and which knobs actually change a number.
Every number here is measured on this machine; the method and the full tables are in [`PERFORMANCE.md`](PERFORMANCE.md).

## 1. 后端一览 / The backends

| 后端 | 怎么选 | 当前状态（本机实测，`Gyro-S-TQ1_0`，正确内核） | 结论 |
| --- | --- | --- | --- |
| **Vulkan**（Adreno X2-90） | 默认，或 `--gpu vulkan` | decode 10.25 tok/s，prefill pp512 **85.55**，32 槽 **31.09** | 支持；**默认就是 prefill 优先且 decode 不变的配置**（2026-11-19） |
| **CPU**（18×Oryon，i8mm） | `--gpu cpu` | decode **11.34**，prefill 21.06 | 支持；这台机器单流最快 |
| **HTP**（Hexagon NPU） | `--npu dense` 等 | 能构建、能跑完，但每种放置都更慢（27B decode 4.1-4.7 对 7.3；35B 10-14 对 39） | 默认关闭 |
| OpenCL（同硬件另一驱动） | 无 | 未测 | 不在计划内 |
| D3D12 / QNN / Mesa | 无 | llama.cpp 无 D3D12 后端；QNN 要换运行时与量化格式；Turnip 在 Windows 不可用 | 研究项目，不是工程 |

两后端在同一个 exe 里，`test-backend-ops` 能直接对比它们 —— 这是本项目最重要的门禁工具（见
[`USAGE.md`](USAGE.md) §5 与 [`MAINTAINER.md`](MAINTAINER.md)）。

## 2. Vulkan

**已进入代码的选择逻辑**（`port/engine/vk-devices.cpp`，按 GGUF 头决定，不需要用户操心）：

| 决策 | 规则 | 依据 |
| --- | --- | --- |
| int-dot（dp4a）mat-mat | 默认开；只有 `q2_0`-only 的包关掉（那个 dp4a fp16 shader 在多 token prefill 上会让驱动崩） | prefill 数倍 |
| fp32 shaders | **默认不再需要**：int-dot 的 **mat-mat** 已经固定取自 fp32 模块，而 mat-vec 与 elementwise 留在 f16 模块（2026-11-18，pp512 16.01 → 49.80、tg16 不变）。`--shader-family fp32` 现在只把 elementwise 也换成 fp32，因此是纯损失（decode 减半，prefill 还更慢） | Gyro-S pp512 49.80 对 16.01 |
| flash attention | `n_ctx >= 4096` 且层数 >= 32 默认开；短上下文 decode −32%，所以由调用方决定 | 2048 上下文 4.5× prefill |
| cooperative matrix **f16**-accumulate（分组 = 专家） | **默认开**（trellis 包，按 GGUF 头决定） | 2026-11-19 修好：trellis 类型原先被挂到不能解码它们的通用量化 shader 上，`buf_a` 从不写入 ⇒ 整模型 `-inf`。改成按类型 SPIR-V 后 pp512 42.51 → 85.55、32 槽 25.47 → 31.09，logits corr 0.9982 / argmax 相同 / 32-token 流逐 token 相同（`PERFORMANCE.md` §4.2） |
| cooperative matrix f16-accumulate（dense） | **默认关** | 净损失：pp512 80.98 对 85.55（`STRATA_VK_F16ACC_DENSE=1` 可开） |
| 专家驻留 | host 专家 + 设备侧 `--moe-cache-size` | 47.6 GiB 上跑 54 GiB 包 |

**用户会想动的旋钮**：

```powershell
--gpu vulkan                  # 默认后端
--flash-attn on|off|auto      # 长上下文开；短上下文 decode 会掉
--shader-family f16|fp32      # f16 默认（decode 10.7）；fp32 是 prefill 优先（46.1）且与 CPU 逐 token 相同
--kv q4_0 --max-context 262144   # 用户要的组合，可用（KV 显存 8x 小）
--load-mode mmap|direct       # mmap 是默认；direct 峰值会超过 47.6 GiB
--moe-cache-size MiB          # 设备侧专家缓存
STRATA_VK_F16=off             # 与 --shader-family fp32 等价，留给不碰命令行的 A/B
STRATA_VK_F16ACC=0            # 关掉 cooperative（A/B 用；默认按模型打开）
STRATA_VK_F16ACC_DENSE=1      # 也开 dense 那半：实测更慢，保留给后续调

## 3. CPU

CPU 后端不是备胎：它是这台机器上单流最快的后端，且是 Vulkan 的参考实现。

**已经做了的**：

- **大核类亲和**（`plan_cpu_affinity`）：6 个 3 072 MHz 的核在每个算子的屏障上都在收钱，把它们排除后 decode
  **+34%**。默认线程数是"大核类核数减一"，留一核给主线程/驱动/OS。
- **repack**：低位权重解包成更适合 NEON 的布局，稠密 matvec **9×**；门禁按实际拷贝字节算之后 prefill **+13%**。
- **prefill / decode 两个线程池**：prefill 用掩码内全部核，decode 留一核。prefill +5.5%。
- **trellis 格式迁移**：同一形状上量类型本身 —— CPU 大胜（17-19 tok/s 量级）、GPU 只有一成。

**还剩的**：decode 的 96% 在两个 matmul 上，且都已跑到各自指令底；剩下的只有"更低的位宽/更快的解包"和线程亲和，
不是排期问题。`-t 18`（全部核）在 decode 上更慢，只在 `--bench-pp` 时有意义。

**线程数扫描（2026-11-18，`--gpu cpu --tokens 9707 --n 16`，同一二进制）** —— 默认的 11 线程已经是最优，
"per-op-class 线程数"这条假设就此关闭：

| `-t` | 线程落在哪些核 | decode |
| --- | --- | --- |
| 默认（11） | 大核类的 12 个核里的 11 个 | **10.7 tok/s**（steady 10.8） |
| 11 | 同上（显式指定与默认等价） | 10.7 |
| 12 | 大核类全部 12 个 | 10.4（−3%） |
| 13 | 全部 18 个核 | 9.7（−9%） |

12 个核就已开始掉，13 个（放进小核类）掉 9%，与"异构池在每个算子的屏障上按最慢的核收钱"一致。
绝对读数注意：同一二进制同一天的 CPU decode 在 **10.4-11.3** 之间漂移，跨时段比较没有意义。


## 4. HTP（Hexagon NPU）

- 能构建（Hexagon SDK 6.7.0.0 在本机），但**在当前驱动上开不出 DSP 会话**：`remote_handle64_open` 对每个模块名都返回
  `AEE_ERPC 0x200`（`--npu-probe`、`port/tools/htp_open_probe.py` 都停在这一步）。会话还能打开的那段时间里，
  每种测过的放置都比 GPU 慢，NPU 电源域的功耗也不低。
- **raw FastRPC 开 `cdsp` 用户 PD 在 Windows 驱动 `30.0.228.10000` 上建不起来**：映射条目已登记仍 `0x200`、
  未映射名字同码、Qualcomm 自家 host 库同样失败。三种形状都试过，不再修 URI 或映射；只有新写 QNN 后端才可能再动。
  细节：[`features/NPU-FEASIBILITY.md`](features/NPU-FEASIBILITY.md)、[`features/NPU-ASSETS.md`](features/NPU-ASSETS.md)。

## 5. 一份权重 / Shared weights

在这台机器上想要"内存里只有一份常驻权重、没有多余拷贝"，四条路都实测过：多 context 共享（同后端可以，跨后端不行）、
host-visible import（预算约 3 GiB）、host 指针 wrap（文件映射导不进去）、`--load-mode direct`（峰值 59.25 GiB）。
结论与数字在 [`PERFORMANCE.md`](PERFORMANCE.md) §4。真正能降内存的只有"更小的格式"，不是"更聪明的放置"。
