# Coding standards

> KI-generierte Inhalte können fehlerhaft sein; bitte keine personenbezogenen Daten in dieses Repository eingeben. AI-generated content can contain errors; verify against the hardware before trusting a claim here.

What `two-axis-review` checks a change against, beside AGENTS.md §7: Google's style guide for
each language, as the guide stands. File headers and function comments (purpose, every input
and output): AGENTS.md §7 rules 1 and 2.

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
- **C++20** for C++ and CUDA, in the rewrite's code; CUDA for compute capability 8.7
  (`GCA_CUDA_ARCHITECTURES`). Today's build passes no `-std` and so gets the compilers'
  default, gnu++17 [M 2026-11-13: `build/compile_commands.json`; S: GCC 11]; the rewrite's
  first ticket switches it. C++17 is the fallback where C++20 fails on this board's tools;
  such a failure is written here with its measurement. What the tools do with C++20
  [M 2026-11-13, a copy of the tree and a test file outside it, `-std=c++20` for g++ and
  nvcc]:
  - The whole tree builds without a warning and its five tests pass, Eigen 3.4.0, Thrust,
    OpenCV and OrbbecSDK included, once one construct is changed: `src/gpu/cuda_frame.cuh`
    names its constructors `cuda_frame<T, N_CHANNEL>(...)`, which C++20 removed; g++ 11
    refuses it, nvcc takes it. `cuda_frame(...)` is right in both dialects.
  - nvcc 12.6 takes `-std=c++20`: a concept on a `__host__ __device__` function, designated
    initializers and `std::span` compile clean under `-Wall -Wextra`.
  - Device code cannot call a `constexpr` function of the standard library (`std::popcount`):
    nvcc refuses it without the experimental `--expt-relaxed-constexpr`.
  - The dialect is set through `CMAKE_CXX_STANDARD 20` and `CMAKE_CUDA_STANDARD 20`: with
    CMake 4.4.4 ("Tools") the copy builds so without a warning and passes its tests, g++
    getting `-std=gnu++20` and nvcc `-std=c++20` [M 2026-11-14]. CMake knows the CUDA dialect
    from 3.25.2 on, with nvcc 12.0 and above [S: CMake 3.25's release notes, "Updates",
    3.25.2]; apt's 3.22.1 refuses it, also when only `CMAKE_CXX_STANDARD 20` is set
    [M 2026-11-13]. The switch raises `cmake_minimum_required` with it.
- **Warnings:** `-Wall -Wextra` for C and C++, and through `-Xcompiler` for nvcc's host
  compiler; nvcc's own warnings are its defaults, none suppressed (no `-w`, no
  `-diag-suppress`), and `-Wreorder`. The top-level `CMakeLists.txt` sets them on every target;
  `tools/verify.sh` fails on any warning in the configure or build log, also on a ccache hit,
  which replays the compile's warnings. A warning in code that stays is fixed, not silenced.
  nvcc's opt-ins `-Wdefault-stream-launch` and `-Wmissing-launch-bounds` stay off: they are no
  defaults, and in today's kernels they print 4 and 5 warning lines (#20198-D and #20238-D, in
  `gpu_check.cu` and `cuda_point_cloud_factory.cu`) [M 2026-11-13], kernels the rewrite
  replaces; the rewrite can turn them on. `-Wext-lambda-captures-this` cannot fire without
  `--extended-lambda`, which the build does not use.
- **Include order:** the file's own header, then this tree's headers (`"camera/..."`, from
  `src/`), then the system and library headers, each group sorted.
- **File names:** snake_case; `.cpp`/`.hpp` for host code, `.cu`/`.cuh` for CUDA.

## The rewrite's code

CUDA_3D's code is the rewrite's prototype: its methods carry over, its code does not. These
rules hold for every file the rewrite writes; the prototype is replaced, not brought up to
them.

- **Method first, prototype as oracle.** A stage's method (its residuals, Jacobians and
  parameters) is written down before its code. The prototype's implementation stays as the
  test oracle until the new one agrees with it within a stated tolerance, and is deleted then
  (AGENTS.md §7 rule 7).
- **The per-frame path**, everything that runs for each frame of a scan, fits the frame time:
  33 ms at 30 fps [S: the stream's frame rate]. Its buffers exist before the scan starts.
  Every wait for the GPU is a call written in the code, and only what the next step needs
  comes back to the host. So the path holds no allocation, no implicit synchronization and no
  spare readback. Thrust's containers, reductions, scans and compactions do all three
  [S: the audit of CUDA_3D, 2026-11-13] and count as such; Thrust is allowed where a
  measurement shows it costs nothing.
- **Measured stage by stage.** Each stage's time on the board stands in docs/HARDWARE.md.
  A kernel-level choice follows NVIDIA's documentation and the measurement; an optimization
  stays on its measured benefit (AGENTS.md §7 rule 6).
- **A failure is a value.** A function that can fail says so in its return type
  (`std::optional`, a result struct with its status). An empty cloud, a zero count and an
  identity transform mean only themselves. OrbbecSDK's exceptions are caught at the camera
  seam.
- **A registration returns its evidence**: the transform, the fitness, the inlier RMSE, the
  iterations, whether it converged and the conditioning of its system. A degenerate or
  diverged registration is reported as tracking loss (GLOSSARY.md).
- **Numerics.** Normal equations are accumulated in double precision; a covariance is centred
  first; the pose solve is damped and checks for degeneracy; the pose update is the SE(3)
  exponential; residuals carry a robust weight. The tests feed the ill-conditioned cases: a
  single plane, points 2 m away [?: a choice], few correspondences, none.
- **Dependencies** are OrbbecSDK, Eigen and OpenCV, and small libraries for the web page
  ([ADR-0002](docs/adr/0002-own-reconstruction-code.md)). Each has a permissive license,
  named in README.md, "License", and a fixed version that `tools/fetch_deps.sh` fetches.

## Tools

`tools/verify.sh` runs each of these, so a point they enforce is no review finding.

| tool | checks | configuration |
|---|---|---|
| clang-format 18.1.8 | formatting of `src/`, `apps/`, `tests/` | `.clang-format` |
| shellcheck 0.10.0 | `tools/*.sh` | defaults |
| ctest | `tests/` | `tests/CMakeLists.txt` |

clang-format, shellcheck and CMake come from PyPI into `~/.local/bin`, no root:
`pip3 install --user clang-format==18.1.8 shellcheck-py==0.10.0.1 cmake==4.4.4`.

CMake and ctest are 4.4.4 [M 2026-11-14: `cmake --version`]. The tools call `cmake` by name,
and `~/.local/bin` stands before `/usr/bin` on the user's path, so PyPI's CMake is found
before apt's 3.22.1 [M 2026-11-14]. With it `tools/verify.sh` passes on a fresh configure
and build without a warning [M 2026-11-14]. A build directory keeps the CMake that
configured it: `tools/build.sh` went on with 3.22.1 in `build/` until `cmake -S . -B build`
ran once with the new one [M 2026-11-14].

ccache 4.5.1 [M 2026-11-13: `ccache --version`] caches every compile of `tools/build.sh`,
`tools/verify.sh` and `tools/run_mutant.sh`, and each of them stops without it. It comes from
apt, which needs root, so the user installs it: `sudo apt install ccache`. The cache is
ccache's default, `~/.cache/ccache`, at most 5 GB [M 2026-11-13: `ccache -p`]; the tools'
settings for it are in AGENTS.md §4.
