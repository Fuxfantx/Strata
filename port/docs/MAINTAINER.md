# Maintaining the vendored tree, the overlay and the gates

`third_party/llama.cpp` is vanilla `ggml-org/llama.cpp` at the pinned commit with `port/llama-overlay-vanilla/` copied
over it. This file says how that composition is produced, how the fork is synced, and what has to pass before a change
lands.

## Composition

| part | content |
| --- | --- |
| upstream | the pinned commit, fetched by `port/tools/fetch-llama.ps1` |
| overlay | every file the port changes or adds, at its real path; currently 61 files |
| engine | `port/engine/`, built against the composed tree by `port/tools/build-engine.ps1` |

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File port/tools/fetch-llama.ps1   # pin + overlay
powershell -NoProfile -ExecutionPolicy Bypass -File port/tools/fetch-llama.ps1 -Verify
```

`-Verify` reproduces the tree from the pin and checks it byte-for-byte against `third_party/llama.cpp`, then runs the
marker check and a CMake configure. Verified property: the reproduction matches at 4 178 files with zero differing bytes.

The overlay directory must mirror the tree exactly - no extra files, nothing missing. A README inside it would land in
`third_party/llama.cpp` and be the one file the reproduction could not match; the per-file description lives in
`UPSTREAM-PATCHES.md` instead.

## Syncing the fork to a newer upstream

The vendored tree moves by **re-pinning**, not by merging: `fetch-llama.ps1` downloads a new pin, applies the overlay and
the gates decide whether the port still composes. The rules:

1. Change the pin in `fetch-llama.ps1`, fetch, and let the overlay copy fail loudly where upstream moved code the patches
   touch. Resolve each conflict **in the overlay**, never in `third_party/llama.cpp` - a hand edit there is lost on the
   next fetch.
2. A patch that upstream has since implemented is dropped from the overlay and recorded as such in
   `UPSTREAM-PATCHES.md`; a patch that no longer applies is rewritten or, if the feature is gone, removed together with
   its code path.
3. Keep only **topic-relevant** patches: the trellis/ternary types, the MTP/NextN plumbing, the engine hooks, the
   qwen4exp/GDN features and the ARM64 device tuning. Application-side work belongs to the app tree, not to this overlay.
4. **Touch every file after a swap** - `Copy-Item` preserves timestamps, so without

   ```powershell
   (Get-ChildItem third_party\llama.cpp -Recurse -File).LastWriteTime = Get-Date
   ```

   ninja skips objects whose sources look older than the binary and links a mixed tree: it builds cleanly and crashes on
   the first token.
5. Rebuild `ggml-hexagon` **before** the engine; the host backend is not a dependency of the engine target.

## Keeping the pin current

The port follows `ggml-org/llama.cpp` on its own schedule, not on every upstream commit: the tree is verified against the
pin, so a re-pin costs a fetch, the overlay conflicts and the full gate list. Run this to see whether it is time:

```powershell
python port/tools/check-pin.py            # what is pinned, and how far master has moved
python port/tools/check-pin.py --offline  # just the pin, no network
```

It exits 0 when the pin is current, 1 when master is ahead (and prints the leading commits) and 2 when the lookup
failed. A sensible cadence is once a release, or when a change the port wants lands upstream; re-pin when the commits
touch `ggml/`, `src/llama-graph.cpp`, `src/llama-model-loader.*`, `ggml/src/ggml-vulkan/` or the trellis-adjacent type
code, because those are what the overlay patches.

## Gates that must pass

| gate | command | expected |
| --- | --- | --- |
| overlay mirror | `python port/tools/check-marks.py` / the reproduction above | 60 files, zero differing bytes |
| port markers | `python port/tools/check-fork-features.py` | 19/19 markers |
| engine stdout contract | `node port/tests/serve_contract.mjs` | 10/10 |
| encoding | `python port/tools/check-encoding.py` | 0 damaged files |
| documents | `python port/tools/audit-docs.py` | 0 problems |
| kernels | `build\bin\test-backend-ops.exe -o MUL_MAT -b Vulkan0` | 0 failures, and the run covers the trellis types (the 0.2.15 harness no longer matches `-p type_a=`) |
| tokens | `--tokens 9707 --n 32` against the reference stream | identical, and identical to `--gpu cpu` |

Any perf change also has to survive `python port/tools/bench_runs.py <exe> <pack> 3`: this machine's run-to-run spread is
about 4%, so a single run proves nothing.

## Where the rest is written down

| document | content |
| --- | --- |
| `UPSTREAM-PATCHES.md` | every patch, file by file, with the measurement or the bug that justifies it |
| `ADRENO.md` | the device: what it can do, the flags this port sets because of it, and the per-op measurements |
| `OPTIMIZATION-BACKLOG.md` | the campaigns, including what was measured and ruled out |
| `features/` | hyper-connections, the gated delta net, expert placement, the NPU |
