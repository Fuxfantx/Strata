# Contributing

**AI assistance is welcome. The contributor is responsible for understanding the change and supplying real evidence.**
You should be able to explain every changed line and answer review questions about it.

This repository is the `windows-arm64-vulkan` branch of [Strata](https://github.com/Niko1221/Strata), not upstream.
Changes to upstream's product belong upstream: the rules, issue and pull-request conventions are in
[upstream's CONTRIBUTING.md](https://github.com/Niko1221/Strata/blob/main/.github/CONTRIBUTING.md). Read them before
you file anything there.

## What a change in this branch looks like

- **One problem per change.** Lead with the problem and the result, then the approach, then the validation (passed,
  failed and not run, kept apart) and the limits.
- **Record the difference from upstream.** A patch to the vendored tree belongs in `port/llama-overlay-vanilla/`
  through the overlay workflow (`port/docs/MAINTAINER.md`) and is described in `port/docs/UPSTREAM-PATCHES.md`. A
  measurement goes in `port/docs/PERFORMANCE-LOG.md` with the machine, the build and the command named.
- **Never submit a measurement, log or reproduction step you did not produce.** A finding from reading code is
  labelled as unconfirmed static analysis.
- **Leave `third_party/` alone.** It is pinned and hash-checked; the overlay is the only supported way to change it.
- **Documentation is `port/docs/` and nothing else.** A page with no remaining reader is deleted, not archived.
- **Tests need no GPU, no model and no download.** State what you ran and what you skipped.

## Before you push

Run the gates in `port/docs/USAGE.md` section 5 - at minimum `python tools/check`,
`node app/host/scripts/audit-ui.mjs`, `npx tsc --noEmit` in `app/host`, `python port/tools/check-encoding.py` and
`python port/tools/audit-docs.py`.

The branch is upstream `main` plus a single port commit: amend that commit rather than stacking more, and push with
`git push --force-with-lease origin windows-arm64-vulkan`.
