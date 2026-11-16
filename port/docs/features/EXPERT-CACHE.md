# How upstream Strata does on-demand expert loading, and what is left for this port

Read from the upstream repository's documentation and pull requests
([Niko1221/Strata](https://github.com/Niko1221/Strata), `docs/HOW_IT_WORKS.md`, `docs/DETAILS.md`, PRs #4117, #547, #1764, #6023,
and the [engine write-up](https://cloud.tencent.com.cn/developer/article/2783023)).

## The upstream design: three tiers, not two

| tier | holds | how it is chosen |
| --- | --- | --- |
| GPU VRAM | attention, routing, shared experts, output head, MTP draft, **and a per-layer expert cache** | sized from free VRAM minus a 700 MiB reserve; about 1 GB of VRAM caches ~700 experts |
| RAM | **all** experts pinned by default | when they fit with 1 GiB to spare (`--resident-experts`) |
| pack on disk | static lookup tables, and the experts themselves in low-RAM mode | read on demand, page cache behind it |

Two behaviours matter for this port:

1. **`ExpertCache`**: GPU slots are split evenly per layer and cold experts are evicted by use frequency, so the cache adapts to
   the conversation. The engine decides the size itself ("free VRAM minus a reserve that keeps the GPU from paging"), and a
   too-large MTP draft head comes out of that reserve.
2. **`--resident-experts` is chosen automatically**: PR #4117 - when the arena cannot fit the RAM available at start with 1 GiB
   to spare, the engine enables the flag and prints one line:
   `the experts (46.8 GiB) do not fit in RAM available now (33.1 GiB, 1.0 GiB kept free). Enabling ...`. Without it the
   process fails at startup (error 1453) on machines where the copy does not fit.
3. On a cache miss the CPU computes the expert **in place, in parallel with the GPU** rather than uploading it.

## What this port already has, and what it does not

This tree's llama.cpp carries the same two mechanisms at the API level, and the engine now exposes them (`--cpu-moe`,
`--n-cpu-moe`, `--moe-cache-size`, `--moe-cache-cpu-share`, `ADRENO.md`): host-resident experts through
`tensor_buft_overrides`, a bounded device expert cache through `llama_context_params::moe_cache_size`, and
`moe_cache_cpu_share` for computing misses on the CPU.

What it does **not** have is upstream's two *policies*:

* `--moe-cache-size auto` - size the cache from free device memory minus a reserve, instead of asking the user;
* automatic host placement - enable the host-resident mode when the expert bytes exceed the RAM available at load, instead of
  letting the user discover it.

Both are small, self-contained engine work: the GGUF tensor table is already parsed for `ModelFacts` (the same pass that
classifies the pack), so the expert/dense byte split is available before the load, and the device's free memory is already
queried for the NPU budget. The rule to implement is upstream's: if `expert_bytes > ram_available - 1 GiB`, place the experts
in host memory and set the device cache to `free_device_memory - 700 MiB` (clamped); otherwise keep today's behaviour with the
cache disabled.

## Why this is worth doing here

On this machine the packs split sharply. 35B-A3B fits (39.7 tok/s all-device; 22.2 with every expert streamed from host memory),
while the 67 GB Q2_0 pack and the 62 GB Gyro-M pack exceed RAM by 20-40%, and their paging is what makes the machine stutter
(`ADRENO.md` has the free-RAM, page-fault and disk measurements). Upstream's answer to exactly that situation is a bounded VRAM
cache plus host-resident experts plus CPU-computed misses - all of which this tree can already run; only the automatic decision
is missing.
