# Coding standards

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

What `two-axis-review` checks a change against, beside AGENTS.md §7: Google's style guide for
each language, as the guide stands.

| language | guide |
|---|---|
| C++ | [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) |
| shell | [Google Shell Style Guide](https://google.github.io/styleguide/shellguide.html) |

CUDA (`.cu`, `.cuh`) follows the C++ guide. Formatting and naming follow CUDA_3D's style
instead of the guide's ([ADR-0001](docs/adr/0001-keep-cuda3d-code-style.md)).

A deviation from a guide needs an ADR in `docs/adr/`; a rule stricter than a guide is not a
deviation. This document adds what the guides leave to the project.

## What the guides leave to this tree

- **Floating-point contraction stays on** (C, C++): a fused multiply-add rounds once and is
  the more accurate form. `-ffp-contract=off` goes only on a target whose bit-exact
  reference changes without it, with that measurement written beside the flag.
- **C++17** for C++ and CUDA; CUDA for compute capability 8.7 (`GCA_CUDA_ARCHITECTURES`).
- **Warnings:** the compilers' defaults; no warning set is enforced yet (ROADMAP.md).
- **Include order:** the file's own header, then this tree's headers (`"camera/..."`, from
  `src/`), then the system and library headers, each group sorted.
- **File names:** snake_case; `.cpp`/`.hpp` for host code, `.cu`/`.cuh` for CUDA.

## Tools

`tools/verify.sh` runs each of these, so a point they enforce is no review finding.

| tool | checks | configuration |
|---|---|---|
| clang-format 18.1.8 | formatting of `src/`, `apps/`, `tests/` | `.clang-format` |
| shellcheck 0.10.0 | `tools/*.sh` | defaults |
| ctest | `tests/` | `tests/CMakeLists.txt` |

clang-format and shellcheck come from PyPI into `~/.local/bin`, no root:
`pip3 install --user clang-format==18.1.8 shellcheck-py==0.10.0.1`.

ccache 4.5.1 [M 2026-11-13: `ccache --version`] caches every compile of `tools/build.sh`,
`tools/verify.sh` and `tools/run_mutant.sh`, and each of them stops without it. It comes from
apt, which needs root, so the user installs it: `sudo apt install ccache`. The cache is
ccache's default, `~/.cache/ccache`, at most 5 GB [M 2026-11-13: `ccache -p`]; the tools'
settings for it are in AGENTS.md §4.
