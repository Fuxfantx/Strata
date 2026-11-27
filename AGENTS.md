# AGENTS.md

This repository is **one branch of [Strata](https://github.com/Niko1221/Strata): `windows-arm64-vulkan`**, the
Adreno/Vulkan port for Snapdragon PCs. Upstream's engine is a hand-written CUDA / HIP / SYCL / Metal kernel set and
none of it runs on an Adreno GPU. What this branch carries:

- **`strata-vk`**: a Vulkan engine component (`port/`) that speaks Strata's `--serve` protocol, plus the overlay that
  teaches the vendored `llama.cpp` the model families and the low-bit types these packs use.
- **The Terminal** (`terminal/strata.py`, `terminal/strata_cli/`, `terminal/strata_control/`): the `strata` command,
  ported to install and serve through the engine component instead of compiling per-vendor kernels.
- **The app** (`app/desktop`, the Tauri shell; `app/host`, the Node host and the web UI), with the model catalogue
  wired to the families this branch can actually run.

The public model families here are **Qwen3.8-Flash-Next** (ISTA-DASLab, Unsloth, Swift 1.5, the Coder, Gyro) and
**Qwen3.6-35B-A3B** (and upstream's Ornith-1.5-35B-A3B variant); `terminal/setup.py` also carries the Edge0 packs and
the signed feed's models. Ask `python terminal/setup.py --check --yes` what this PC can run, and
`python terminal/setup.py --help` for the flags. NVIDIA-only, AMD/HIP, SYCL/Intel and Mac/Metal code paths do not
exist in this branch, and the two upstream families Inkling-2-Mini and Hy4-Flash are not carried here.

## Documentation lives in `port/docs/`

`port/docs/` is the only documentation tree. It records **what differs from upstream** and what is measured on this
machine: build, run, measure, maintain (`README.md` is the index), the performance record, the known issues, the
overlay's patch list, and the frontend wiring. Upstream's `docs/` tree is not carried here.

## Working on it

```sh
python port/tools/build-engine.ps1                            # build strata-vk (Vulkan SDK + VS Build Tools)
python terminal/setup.py --check --yes                         # what this PC can run; changes nothing
python terminal/setup.py --yes --family qwen --model Q2_0      # install the engine component + one model, serve it
python tools/check                                            # the fast checks (Python tests, app typecheck/test/build)
```

Gates before a commit, and what each one means, are in `port/docs/USAGE.md` section 5. The short list:
`python tools/check`, `node app/host/scripts/audit-ui.mjs`, `npx tsc --noEmit` in `app/host`,
`python port/tools/check-encoding.py`, `python port/tools/audit-docs.py`.

## Rules that keep this branch buildable

- **Never edit `third_party/` or `port/llama-overlay-vanilla/` in place.** Both are pinned and hash-checked: the
  overlay *is* the patch set applied to the pinned `llama.cpp`. A change to either goes through the overlay workflow in
  `port/docs/MAINTAINER.md` and is written down in `port/docs/UPSTREAM-PATCHES.md`.
- **Read the docs that exist before adding one.** A difference from upstream is recorded in `port/docs/`; a page with
  no remaining reader is deleted, not archived.
- **Tests need no GPU, no model and no download.** Never run a model to make a test pass;
  `terminal/setup.py --check --yes`, `python -m unittest` and the app's UI audits are the safe checks.
- **Measured claims only.** Every number says what it was measured on, and "not measured" is written as such.
- **The branch's history is one commit** on top of upstream `main`. Amend it rather than stacking commits, and push
  with `git push --force-with-lease origin windows-arm64-vulkan`.
- Ignore upstream's release claims (installer names, engine versions for other backends): they belong to a build
  this branch cannot produce.
