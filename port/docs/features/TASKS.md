# Where the work stands

This page is a task table. Each row names one piece of work, where it came from, whether it is done, and what the
evidence is. Two facts shape it.

- The Hy4 line has **one** pack, at `C:\Users\Fuxfantx\C-Res\Qwen\Hy4-Flash-Compact\`. The copies under
  `Hy4-Works\out\` are not the reference.
- The Edge0 145B pack is **gone**, lost with a deleted scratch directory. Edge0 work waits for the 120B pack. Do not
  download it yet.

Sources of the rows: the objective, the quantiser's handoff (`Hy4-Works/HY4-COMPACT-HANDOFF.md`), and this tree's own
commits.

## A. The Hy4 pack: read it, prove it, measure it

| id | task | status | evidence | next |
|---|---|---|---|---|
| A1 | Read the pack's one format: file ids 151-154 are the four block types, 148-150 the Engram rows, there is no layout field | done | `03fbdf959`, `7eb56a692` | - |
| A2 | Bind each tensor's codes to its scales. Record the row geometry when binding. Four dequantisers, four dot kernels | done | `19a124124` | - |
| A3 | Register the four types in ggml and on the CPU | done | `4efcf2b2a` | - |
| A4 | Bind every tensor, mapped or copied | partial | `33cc49982`, `7eb26f591` | two tensors still report unbound rows in the perplexity path |
| A5 | Keep one layout only. Old ids are not mapped. The inline-scale types and their code are deleted. One set of names | done | `6206df97d`, `7eb56a692` | - |
| A6 | Make the report side describe the same format | done | `df76d98b0` | - |
| A7 | Load the pack and decode one token | done | reported `context ready in 0.6 s`, exit 0 | - |
| A8 | A gate with a judgeable answer. On the CPU the model repeats tokens, through `--prompt` and through synthetic ids | **open** | gate fails; both layouts failed the same way | confirm the template, BOS/EOG and the CPU parameter set, or find a deeper cause |
| A9 | `llama-perplexity` loads this model | partial | the Engram read is fixed (`48e7fba5c`), the binding is fixed | finish A4, then retest |
| A10 | Perplexity inside this engine | done | `d4040a1f7` (`--ppl`, `--ppl-chunk`, `--ppl-chunks`) | a window of 32 or more still stops the process |
| A11 | The three measurements the handoff asks for: gate, perplexity on a fixed corpus, divergence against an f16-KV run | not started | - | waits for A8 and A9/A10 |

## B. MTP and memory

| id | task | status | evidence | next |
|---|---|---|---|---|
| B1 | The draft block's graph runs: state row, projection, block, shared head, one token out | done | `347c60708` | - |
| B2 | Wire the draft into the engine: a switch, one draft step per decode, a proposed-versus-confirmed count | not started | plan recorded | the probe already does this by hand, and the state link is device-side |
| B3 | Strong lazy load for the weights | not started | - | the handoff measures the pack at 57.8 GB against about 36 GB free memory |
| B4 | A hot-expert cache that works across the tier stacks | not started | the cache steps aside when a layer is tiered, and logs that | teach the cache to hold one tier's stack |

## C. Kernels for the block types

| id | task | status | evidence | next |
|---|---|---|---|---|
| C1 | A CPU dot kernel for each of the four types, so a matmul is not a row-by-row dequantise | done | each type carries `.vec_dot` with an F32 counterpart | - |
| C2 | Vulkan kernels for the same four, to the shader conventions | not started | only `to_float` exists on that side | the largest remaining performance item |
| C3 | Registering the dot kernel changed the logits, and a kernel that was the fallback's own arithmetic changed them too | **open** | recorded earlier, unresolved | retest on the one-layout build; both the names and the fallback changed since |

## D. Q2, from earlier work

| id | task | status | evidence | next |
|---|---|---|---|---|
| D1 | Audit which existing types already have CPU dot acceleration | done | Q2_0, Q3_K, Q4_0, the K-quants, Q8_0, the IQ2 family and F8 all carry a dot kernel. A tiled K-quant matmul covers a large batch, and a per-expert variant covers the mixture | - |
| D2 | Q2-specific tuning | not applicable | this pack holds no Q2_0 tensor | revisit on the Edge0 120B pack if it uses Q2 |

## E. Upstream

| id | task | status | evidence | next |
|---|---|---|---|---|
| E1 | Pin to upstream `b12722`, keep the overlay as the source of truth, regenerate the tree from it | done | the build ritual in this record | - |
| E2 | Confirm that every delivered change is in the overlay and not only in the tree | **needs a pass** | recent work went through the overlay; one diagnostic that did not was reverted | check the tree's `git status` at the end of each session |
| E3 | Build and verify after the pin | partial | the CPU build is green. The Vulkan engine cannot enumerate a device on this machine: `strata-vk --geometry` stops with `terminate: bad allocation` | run it on a machine the docs list |

## F. Edge0

| id | task | status | evidence | next |
|---|---|---|---|---|
| F1 | Get a pack | **blocked** for a local run; the front end is wired and checked | the 145B pack is gone; the 120B folder is published and now pinned in the tables | nothing to download here; the wiring is done and checked |
| F2 | MTP on Edge0 | blocked | - | after F1 |
| F3 | Latent mixture and Q3_K tuning | blocked | - | after F1 |
| F4 | Dense layers by memory, Q4_0 against Q8_0 | blocked | - | after F1 |
| F5 | PLE with affine and GSQ-RCO, to Q4_0 quality | blocked | - | after F1 |
| F6 | ModelScope for the mainland, Hugging Face elsewhere | blocked | - | after F1 |
| F7 | Wire setup, serve, control, cli and desktop | **done for Edge0 and Hy4** | tables, installer, plans, `engine_args` and the app icons are in place and tested (`FRONTEND-WIRING.md`) | keep it in step with each runtime change |

## Order

1. A8 (the gate) and A9/A10 (the perplexity path). Every number after them is unreadable until they work.
2. A11 (the measurements).
3. C2, B3, B4 (performance and memory).
4. B2 (the draft loop).
5. F1 onward, when a pack exists.

## About the numbers on this page

The 57.8 GB pack size, the 36 GB of free memory, the 0.3 to 1.2 s load time and the ~10 s per token prefill are the
quantiser's measurements, on this machine, and are quoted from their handoff. The load time and exit code in A7 were
measured here, on the CPU build. Everything marked done is a commit in this tree. Everything else is a reading of the
code, not a measurement.

## Backends, as the repository states them

The public wording lists inference backends in one table, `docs/AI_REFERENCE.md`, and names the install path for each.
As of this commit:

| Backend | Where the repository documents it |
|---|---|
| NVIDIA RTX / CUDA | `INSTALL.md` |
| AMD Radeon / HIP | `AMD_HIP.md`, with a note that RX 6000 validation differs from newer cards |
| Apple Silicon / Metal | `MAC.md` |
| NVIDIA DGX Spark, Linux ARM64 | `INSTALL.md`; Terminal only, no desktop app |

**Vulkan is not an inference backend in that list.** It appears in `docs/IMAGES.md`, as one of the choices for
`--image-backend cuda|hip|vulkan|cpu`, which is the image path. So the Vulkan engine this port works on is not part of
the documented inference matrix, and no row here should read as if it were.

This machine is Windows on ARM64. That combination is not in any of the lists above: the only ARM64 entry is DGX
Spark, and that one is Linux and CUDA. The CPU path is the only one that works here, which is why the whole record is
measured on it. `strata-vk --geometry` was tried on this machine for this check and stopped with
`terminate: bad allocation` before printing a device.
