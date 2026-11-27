# Front-end wiring for Edge0 and Hy4: the plan

This page fixes the shape of the work before any code changes, because the code is spread over five surfaces
(`setup.py`, `strata_control`, `strata_cli`, `serve`, the desktop app) and each one reads the others' data.

## What already exists, so nothing is built twice

Read from the tree, not assumed:

| piece | where | note |
|---|---|---|
| The model list | `setup.py` `MODELS`, `FAMILIES`, `OWN_PATH_FAMILIES` | `FAMILIES` = packed Flash-Next files; `OWN_PATH_FAMILIES` = a GGUF the engine reads as it is (Qwen3.6 today) |
| Hugging Face | `hf(repo)` = `{HF_ENDPOINT or huggingface.co}/{repo}/resolve/{revision}/` | revision from `HF_REVISIONS`; files verified against `REPO_PINS` / `REV_PINS` |
| ModelScope | `ms_endpoint()`, `ms_url(repo, path)` = `{MS}/models/{repo}/resolve/master/{path}`, `ms_meta(repo, path)` | already used when `model_source() == "modelscope"`; verification uses **ModelScope's own published SHA-256**, so a mirror needs no pin of ours |
| Source choice | `model_source()`, `--source`, `STRATA_SOURCE` | `auto` means Hugging Face today |
| Downloads | `download(url, dst)`, `verify_sha256`, `.done`/`.bad` marks | host-agnostic: any URL works |
| The engine's Edge0 mode | `--pack DIR`, `--expert-cache auto\|MiB`, `--latent-device`, `--ple-io` | the 145B Q3_K pack loads and generates on this machine (0.1 tok/s, about 21.7 s per token) |
| The model's own file concept | every run config carries `--pack <folder>` and `--native <gguf>` in its `args` | `strata_cli/launch/adopt.py` already builds `["--pack", pack, "--native", shards[0]]`; `discovery`, `doctor`, `presets`, `autotune` all read those two flags |
| The engine's server | `strata-vk --serve` with `--host`, `--port`, `--api-key` | started from `serve/server.py` |
| "This model is not supported" | `feed_models.UNAVAILABLE_ENGINE` / `UNAVAILABLE_UNKNOWN` / `unavailable_reason(family)`, and `native.ts`'s `available` / `unavailable_reason` | one sentence a person reads; nothing new is needed for DeepSeek |

## Edge0: a family whose "pack" is the downloaded folder

The pack is a **directory**, not a GGUF (measured sizes from the specification page):

| file | bytes | role |
|---|---:|---|
| `model-skeleton.gguf` | 5,991,569,248 | a `qwen4exp` GGUF, the family this port already loads |
| `experts.bin` | 40,076,574,720 | 48 x 512 native blobs |
| `ple-fp8.gguf` | 51,200,246,112 | the per-layer n-gram table, read by row |
| `latent_dense_q8.bin` | 3,743,416,320 | the per-group D/U rows, Q8_0 |
| `latent_dense_q4.bin` | 1,981,808,640 | the same rows, Q4_0 (the smaller choice) |
| `dense.bin`, `index.txt` | 1,538,625,024 / 117,464 | the pack index and its dense pool |
| `labels.bin` / `profile.bin` | 98,304 / 196,632 | routing group per expert; residency order |
| `mmproj-BF16.gguf` | 907,543,008 | the vision encoder |
| `mtp/` | about 824,000,000 | the draft head and its vocabulary |
| `native_experts.txt` | 1,463 | `v6: latent A-only, Q3_K gu ...` |

The front end therefore needs a family that is **not** a list of GGUF shards:

- `FAMILIES["edge0"]` (or a separate `PACK_FAMILIES`, see below) carries `arch: "qwen4exp"`, `pack: True`, the
  per-size `latent` width (1792 for 145B, 1280 for 120B) and the **file list with sizes**;
- installing it downloads those files into one folder and writes a run config whose args are
  `--pack <folder> --native <folder>/model-skeleton.gguf --expert-cache auto`;
- the D/U choice (`q8` or `q4`) is the family's "size" choice, since both files exist in the pack: the smaller one is
  offered only where memory is short, and the config names it with `--ple-io`/format as the engine expects;
- **verification**: for a ModelScope download, `ms_meta` supplies the published size and SHA-256, so a file is checked
  even when this build holds no pin. For a Hugging Face copy, `REPO_PINS` needs entries - and those need the real
  hashes, which means a download. **Until that download is allowed**, an unpinned source is verified by size only and
  says so in one sentence.

### The source rule

The request is one line: **the mainland reads ModelScope, elsewhere reads Hugging Face**. Today `auto` means Hugging
Face and ModelScope is only ever an explicit choice (`#908`), because ModelScope serves a repository's current files
rather than a pinned revision.

So the rule becomes a property of the **family**, not of the process:

- `sources`: a mapping `{"modelscope": "<owner/repo>", "huggingface": "<owner/repo>"}`, either side optional;
- `model_source()` keeps its meaning for the families that have both, but a family that names only one source is
  downloaded from that source whatever the region says, and the screen says which host it used;
- the region itself is decided once, in one place, from data this machine already has: the time zone and the locale,
  with `STRATA_REGION=cn|intl` as an override for tests and for a user behind a tunnel. **No network probe** decides it,
  because a probe at setup time is a failure mode of its own;
- `--source` still wins over everything, so an explicit choice is never silently changed.

For the 145B pack, the ModelScope folder is known (`Edge0/Edge0-145B-q3k`). The 120B folder is not named in the
specification page, so it is left **unset** and the screen says the pack is not published there yet; nothing is guessed.

## Hy4: import only

`OWN_PATH_FAMILIES["hy4"]`, shaped like `qwen36` but without any download:

- `arch: "hy_v4_flash"` (`strata_control.models.ENGINE_ARCHITECTURES` already knows it);
- `import_only: True`, `models: {}`, **no** `hf`, **no** `sha256`, **no** `repo` - so every download path refuses it
  with one sentence: this model has to be pointed at, not downloaded;
- installing it means: the user chooses an existing GGUF (the app's file dialog, the CLI's `--model-path`), the front
  end validates it with `gguf_preflight` (architecture `hy_v4_flash`, shards complete), and writes a run config whose
  args are `--native <path> --gpu cpu` for now;
- `--gpu cpu` is not a preference: on the Vulkan backend these block types have no pipeline at all, so every matmul
  would fall back to the CPU and pay the transfers (`VULKAN-STATUS.md`). The config says so in one line, and the app's
  engine line shows the same words.

## What the desktop app needs

The app renders `NativeFamily` records and then groups them (`native-catalog.ts` keeps two groups today, so anything
else is invisible). Three additions:

1. a group for **Edge0** (downloadable, pack) and one for **Hy4** (import only), each mapping `sourceFamily`;
2. the import dialog for a family with `import_only`: choose a `.gguf`, then the same "load" path the other models use;
3. for a family with `available: false`, the existing sentence from `unavailable_reason` - which is where **DeepSeek**
   lands without any new code: it is not in the tables, and a file of its architecture is reported unsupported by the
   preflight the app already calls.

## Order of work

1. the source rule and the region decision, with tests that set `STRATA_REGION` and `STRATA_SOURCE`;
2. the `sources` support in the download URL builder, and the family entry for Edge0;
3. the install path for a pack family (download into one folder, write the config, verify);
4. the Hy4 entry, the import path and its refusal of every download;
5. the CLI and the server, which are config-driven: prove it with a `--plan` run;
6. the app: groups, the import dialog, the unsupported sentence;
7. tests: tables, source rule, config shapes, and the import refusal - **no model is run** in any of them.

## What is deliberately not in this plan

- No download of the Edge0 120B pack, and none of the 145B pack.
- No new download machinery: `download`, `verify_sha256`, `ms_meta` and the `.done`/`.bad` marks are reused as they are.
- No new "unsupported" mechanism: DeepSeek uses the sentence the feed already has.
- No removal of Qwen3.6-35B-A3B: it stays an own-path family with its three sizes.

## Branding, taken from the official repositories

The names, by-lines, descriptions and licences a screen shows are read from the models' own repositories rather than
written here, so a screen cannot disagree with the model it offers. What follows is what was read, with its source, and
the one place this build adds a qualifier of its own.

### Edge0

| item | value | source |
|---|---|---|
| title | Edge0-145B / Edge0-120B, build Q3_K | model card heading, `Edge0/Edge0-145B-q3k` |
| by | Edge0 (the `Edge0` organisation) | the repository owner |
| base model | `Qwen/Qwen3.8-Flash-Next` | the card's `base_model` |
| one line | "A latent sparse MoE: 48 layers, 512 experts, 10 active per token, 256K context, trained natively at 4 bits." | the card |
| sizes | Q3_K 40.81 GiB (145B), 27.98 GiB (120B); NVFP4 47.79 GiB (145B), 31.6 GiB (120B) | the card's build table and this record's specification page |
| licence | **Edge0 Community Model License 1.0**, with the Qwen Community License 1.0 also applying | the card's `license_name` / `license_link` and its credits section |
| repositories | Hugging Face `Edge0/Edge0-145B-q3k`, `Edge0/Edge0-120B-q3k`; ModelScope the same paths | the card, and Edge0's own release notes ("from Hugging Face or ModelScope") |

The 120B is published (`Edge0/Edge0-120B-q3k`), so both sizes have both sources. This build adds **no** descriptive
sentence of its own for Edge0: the card's line and its numbers already say what the model is.

### Hy4

**The model's official repository is Tencent's**, not the one whose files this pack happened to be built from. That
matters for a screen: the name a person sees is Tencent's, and the derivative is described as a derivative.

| item | value | source |
|---|---|---|
| title | **Hy4 flash** ("Tencent Hy") | `tencent/Hy4-flash`, the card's heading and its "Tencent Hy Team" footer |
| by | Tencent Hy Team | the card's footer |
| one line | "Hy4 flash is our most cost-effective Hy model to date, featuring 220B total parameters with just 12B active parameters per token. It natively accepts text and image inputs, generates text, and supports a context window of up to 1M tokens." | the card's model introduction |
| architecture | 42 layers (1 dense + 41 MoE), 512 routed experts with 16 active and 1 shared, latent dimension 1536, MLA, hybrid sparse and sliding-window attention, Engram at layers 1, 11, 21 and 31, a 5-layer DFly2 drafter | the card's properties table |
| licence | **Tencent Hy License** | the card's licence section and `license_name` |
| repositories | Hugging Face `tencent/Hy4-flash`, ModelScope **`Tencent-Hunyuan/Hy4-flash`**, AtomGit `tencent_hunyuan/Hy4-flash` | the card's model links |
| the GGUF files Tencent points at | `AngelSlim/Hy4-flash-GGUF` (AngelSlim is Tencent's own compression toolkit) | the card's badges and its quantization section |

Two different repository names for the same model on the two hosts (`tencent/Hy4-flash` against
`Tencent-Hunyuan/Hy4-flash`) is exactly why a family names its source per host instead of deriving one from the other.

**The qualifier this build adds**, and it is the only one: the pack this port reads is a **Compact** repack in this
build's codes-only format, made from an **uncensored derivative** of Hy4 flash that is not Tencent's work. So the screen
shows:

- the title **Hy4 flash**, the by-line **Tencent Hy Team**, and the one line above - the model's own identity;
- beneath it, one qualifier: **"Compact pack of an uncensored derivative (not Tencent's)"**, which is what the file on
  disk actually is.

Nothing else is reworded. The 220B parameters, the 1M context, the MTP drafter and the Engram layers are the card's
claims and they hold for this pack's weights; the uncensored part is the derivative's, and it is labelled as such.

What the screen must **not** claim, because the card does not: a licence name of this build's choosing, a parameter count
for the Compact repack, or any speed number. Speed numbers come from a measurement on the machine that shows them.

## What is wired now

Every claim below was checked without running a model: the tables by unit tests, the plans by `strata launch --plan`, the
server's arguments by calling `serve.server.engine_args` on each config, and the app by `tsc --noEmit` and its own UI
audit.

| surface | Edge0 | Hy4 |
|---|---|---|
| the model list | offered, both builds, resolved by name (`145B-Q3_K`, `120B-Q3_K`) | **absent** until a config exists; then listed as one size |
| the download | a directory pack, each file pinned from its repository's own listing; the Q8 D/U table for the 145B and the Q4 one for the 120B, the other table and the pack's MTP folder left out | none - `--gguf-dir` is the only way in |
| the run config | `--pack <folder> --native <folder>/model-skeleton.gguf --expert-cache auto --ple-io direct`, the four latent environment entries, `--vision` and the reader section | `--native <the trunk> --gpu cpu`, no pack |
| images | on: `mmproj-BF16.gguf` travels in the folder, byte-identical to the projector Qwen3.8-Flash-Next publishes | off (the Compact pack is the two weight files) |
| drafting | off: the pack's draft head is read through `--pack`, so nothing is fetched or looked for | off: the pack's own
 MTP block travels in the trunk |
| the server | `engine_args` keeps the pack, the skeleton, the memory settings and `--vision`; it adds no drafter | keeps `--native` and `--gpu cpu`; adds nothing |
| the app | its own glyph, and Edge0's organisation avatar drawn in place of it | its own glyph, and Tencent's lock-up drawn in place of it |

Two facts are worth keeping beside that table. The picture reader is not a separate download: it is one of the files of
the folder, and the module that writes the config turns images on only when it is there. And the D/U precision is a
property of the build, not of the machine: the table a build does not read is neither downloaded nor named.

## What is left

- The app's file chooser reaches the import through the terminal's `--model-path`; that path was verified on the command
  line, and a click-through in the app was not (no display here).
- Edge0 has not been started from this tree on this machine: the pack is 103.5 GB of downloads and the machine cannot
  hold it resident. What is verified is the wiring, not a run.

## Where the app stands

The terminal side was checked by its own commands; the app side by its own tools, and by one fact that took three rounds
to find because it is invisible from the tables: the app's model library keeps a fixed list of display groups and drops
every family not in it. So a family can be fully wired - downloadable, plannable, startable - and still be missing from
the screen a person looks at.

| piece | state |
|---|---|
| the library's groups | four, in order: Qwen3.8-Flash-Next, Qwen3.6-35B-A3B, **Edge0**, **Hy4 flash**, then the signed feed's models |
| version labels | Edge0's builds read `145B · Q3_K` and `120B · Q3_K`; Hy4's one size reads `Compact` |
| the family mark | `BrandMark` draws the brand's own file when it is in `public/brand/`, else the family's glyph from the sprite; the switch's trigger and its menu rows both use it |
| the icons | two glyphs of this build's drawing (`family-edge0`, `family-hy`) so nothing has to recolour a brand mark; the official files sit beside them and win when present |
| "Use a local GGUF…" | the preview names an import-only family from the header (`architecture_of_an_imported_family`) and returns the installer's own arguments, so the page's install button becomes usable for a Hy4 folder |
| a row with nothing to download | the version row leaves the size cell out for an imported pack instead of writing "0 GB download" |

The test that asserted "only the two requested display families exist" was right while the app offered two; the two extra
families are now requested, so the assertion moved with them and its two index-based checks moved too. The app's own
`scripts/test.mjs` still fails on this checkout for a reason that predates this work: it builds a `file://` URL from the
checkout path and scans the percent-encoded spelling, so any path with a space fails it.

## What the wiring does not do

- It does not make Edge0 runnable here: 103.5 GB (145B) or 89.7 GB (120B) of downloads, on a machine with about 36 GB
  free. What is verified is that every surface agrees on what the pack is and how it would be started.
- It does not quantise the Engram table: the stage exists, the schemes are recorded with the sample that measured them,
  and `apply` refuses anything but "keep" on purpose.
- It does not offer Hy4 for download anywhere, and it hides the family until a config of its own exists.
