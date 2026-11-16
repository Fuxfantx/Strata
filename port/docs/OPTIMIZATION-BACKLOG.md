# Measured, and ruled out

Every entry here was tried or measured on this machine and closed. They are kept so the next person does not spend a
campaign re-deriving them; the measurements themselves are in the commits and in `ADRENO.md` where the number is a device
fact rather than a dead end.

## Device and backend

| attempt | outcome |
| --- | --- |
| CPU-side trellis unpack (fold the 2048-entry fp16 codebook lookups into arithmetic) | impossible: `tq2_t_step` is `x = (s & 0x7fff) * 0x9e3779b1` followed by two independent table reads per four weights |
| Repack the trellis types into a smaller or faster layout | lossless layouts cannot remove those lookups; direct indices need 5.5 bit/weight (3.6x larger, 44 GiB of experts, does not fit), 4-bit indices are lossy, fp16 is 13x |
| Trellis MoE through the cooperative matmul path | 3.5x slower and wrong tokens |
| Integerise the `q8_0` matvec on Qualcomm | already the int8 path; the float path decodes it wrongly, which is why the port forces int8 there |
| Split-K for the expert matvec | rows-per-workgroup 40 is the peak; `mm-probe` shows the kernel at its bandwidth wall, not its latency |
| Nested barriers (`STRATA_VK_NO_BARRIER`) | costs 10.4% of a token, but every safe narrowing measured zero: the cost is the number of `vkCmdPipelineBarrier` calls (about 5 us each), not the bytes they cover or the access masks |
| Host-resident experts with a device bank, on a pack larger than RAM | 3.2 tok/s with 49% CPU; keeping the experts on the device and paging the pack (28.76 GiB resident, disk idle) reads 14.0 tok/s |
| Multi-sequence aggregate throughput | the user runs one request 95% of the time and two 5%; the per-request latency work is what matters |
| NPU (Hexagon) placement for the dense tensors | measured slower than the GPU on this machine; `--npu dense` still exists for experiments |

## Prefill

| attempt | outcome |
| --- | --- |
| Wider cooperative tile so the weight tile is unpacked once for several n-tiles | no change: doubling the n-tile from 128 to 256 changed nothing measurable, so the unpack repetition was never the limit |
| The int8-dot path for trellis prefill | 17x slower than the cooperative f16-accumulate path (pp512 12.1 against 211.9 tok/s), so the "4 TFLOP/s int8" road is closed for this format |
| Conclusion | pp512 sits at about 1.0 TFLOP/s of cooperative f16, which is this device's throughput for the path; only a different device or a different format moves it |

## Decode structure

| attempt | outcome |
| --- | --- |
| Fewer dispatches as the lever | 96 of 2 668 dispatches removed (the Hadamard rotations) measured under the machine's 4% noise, and one fused pair of 60 dispatches measured about +1%. Dispatch count is not a linear cost: small in-model dispatches overlap, unlike a single op submitted on its own (41 us of GPU against 139 us of wall) |
| `sigmoid`/`mul` adjacency via operand order | illegal: `ggml_mul`'s second operand must broadcast to the first, and the gate is `[1, T]` |
| `sigmoid`/`mul` adjacency via source order | the graph builder interleaves sibling consumers, so source order does not decide graph order; the pair needs a dedicated op (`ggml_gated_mul`, now in the port) |
| Per-op GPU timings from the backend's timestamp logger | unusable as a budget: the default mode covers about 20% of the time, the concurrent mode overlaps ranges to about 320% |
| RenderDoc capture of a whole decode step | never completes (17 GB resident, ten minutes): one step references a 542 MiB compute buffer, a 32k-token KV cache and every weight. A single-dispatch capture works (3.5 MB in a second) and was used to read real GPU times |
| Prompt lookup speculation | costs 8% on hybrid models (the rollback ring) and the drafter found nothing to copy on the prompts tried; the option stays off |

## What the remaining decode time is

About 30 ms of CPU per token against a 100 ms wall, plus a GPU side that is neither at the memory wall nor at the compute
wall. The honest next target is host side (graph building, sampling, submission and synchronisation), not more kernel
work.
