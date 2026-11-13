# SPDX-License-Identifier: 0BSD
#
# build_common.sh - what tools/build.sh, tools/verify.sh and tools/run_mutant.sh
# share: ccache, the configure of a build directory through it, the number of
# parallel compile jobs, and the tree's file set. It is sourced, not run. The
# file set lives here too because it is what a copy of the tree builds from, so
# the three tools source one file.

# shellcheck shell=bash

# The number of parallel compile jobs of every build the three tools run:
# tools/build.sh and tools/verify.sh pass it to cmake --build, and
# tools/run_mutant.sh builds through tools/verify.sh. It is four, not one per
# core (six on this board), the user's decision of 2026-11-13.
# Measured on 2026-11-13 [M 2026-11-13]:
#   - the peak memory of one compiler process: registration/color_icp.cu
#     1554 MB, geometry/cuda_estimate_normals.cu 804 MB,
#     registration/cuda_color_icp_build_least_square.cu 746 MB;
#   - MemAvailable when idle: about 4.3 to 4.5 GB of 7.4 GiB;
#   - at -j3 MemAvailable fell to 1.69 GB at its lowest, with no swap used;
#   - a -j6 clean build of Eigen-heavy CUDA code froze the board at about
#     19:56 (reset reason SYS_RESET_N, no log line, pstore empty).
# Six jobs of such peaks need about 4.5 to 5.5 GB, derived from the peaks
# above, not measured. The kernel has no PSI (CONFIG_PSI is not set), so
# systemd-oomd cannot run; swap was only zram, which lives in the same RAM;
# the user has since added an 8 GB swap file on the NVMe, which backs four jobs.
# Confirmed at four: a cold build at -j4 with an empty ccache had a lowest
# MemAvailable of 1238 MB, swap grew by about 161 MB, and it took 171 s
# [M 2026-11-13].
# shellcheck disable=SC2034  # read by the tools that source this file
readonly BUILD_JOBS=4

#######################################
# Exits when ccache is missing.
# Globals:
#   None
# Arguments:
#   $1: the calling tool's name, for the message.
# Outputs:
#   The missing tool to STDERR.
# Returns:
#   0 when ccache is found; exits 2 otherwise.
#######################################
require_ccache() {
  if ! command -v ccache >/dev/null; then
    echo "$1: ccache missing (CODING_STANDARDS.md, Tools)" >&2
    exit 2
  fi
}

#######################################
# Prints the CMake cache entries that send every C, C++ and CUDA compile of the
# tree in the working directory through ccache (configure_build says how).
# Globals:
#   None
# Arguments:
#   None
# Outputs:
#   One entry per line, CMAKE_<LANG>_COMPILER_LAUNCHER:STRING=<launcher>, to
#   STDOUT.
# Returns:
#   0.
#######################################
launcher_entries() {
  local lang launcher
  launcher="env;CCACHE_BASEDIR=$(pwd -P);CCACHE_NOHASHDIR=1;ccache"
  for lang in C CXX CUDA; do
    echo "CMAKE_${lang}_COMPILER_LAUNCHER:STRING=${launcher}"
  done
}

#######################################
# Configures a build directory of the tree in the working directory: Release,
# every compile through ccache. The launcher (launcher_entries) carries
# ccache's settings, so every build of the directory uses them: the tree's
# root as the base directory, so that paths inside the tree are hashed
# relative to the compile's working directory, and no directory hashing. The
# paths outside the tree (CUDA, the libraries, tools/fetch_deps.sh's
# downloads) stay absolute. A build directory one level below a tree's root,
# as build/ is, therefore hits the entries any other tree's builds wrote for
# the same sources; one at another depth does not, since its relative paths
# differ.
# Globals:
#   None
# Arguments:
#   $1: the build directory, one level below the tree's root.
# Outputs:
#   CMake's output to STDOUT and STDERR.
# Returns:
#   CMake's exit status.
#######################################
configure_build() {
  local entry
  local -a args=(-S . -B "$1" -DCMAKE_BUILD_TYPE=Release)
  while IFS= read -r entry; do
    args+=("-D${entry}")
  done < <(launcher_entries)
  cmake "${args[@]}"
}

#######################################
# Tells whether tools/build.sh's build/ is configured with every entry of
# launcher_entries.
# Globals:
#   None
# Arguments:
#   None
# Outputs:
#   None
# Returns:
#   0 when it is, 1 otherwise.
#######################################
configured_with_ccache() {
  local entry
  while IFS= read -r entry; do
    grep -qxF "${entry}" build/CMakeCache.txt 2>/dev/null || return 1
  done < <(launcher_entries)
}

#######################################
# Prints the tree's files, tracked and untracked but not ignored, as they are
# in the working tree: what tools/run_mutant.sh copies and what
# tools/verify.sh's fingerprint covers. Ignored files (the builds,
# recordings), deleted files and .git are left out. A nested repository or a
# submodule, which git lists as one directory, is expanded into its own
# files by the same rule; an empty submodule directory adds nothing.
# Globals:
#   None
# Arguments:
#   $1: the repository's root.
#   $2: the prefix of the printed paths: absent at the top, the nested
#       repository's path in the recursion.
# Outputs:
#   The paths, NUL-terminated, sorted within each repository, to STDOUT.
# Returns:
#   0, or 1 when git fails.
#######################################
tree_files() {
  local path scratch status=0
  scratch="$(mktemp -d)" || return 1
  if git -C "$1" ls-files -z --cached --others --exclude-standard \
    >"${scratch}/listed" \
    && git -C "$1" ls-files -z --deleted >"${scratch}/deleted"; then
    while IFS= read -r -d '' path; do
      path="${path%/}"
      if [[ -d "$1/${path}" && ! -L "$1/${path}" ]]; then
        if [[ -e "$1/${path}/.git" ]]; then
          tree_files "$1/${path}" "${2:-}${path}/" || status=1
        fi
      else
        printf '%s\0' "${2:-}${path}"
      fi
    done < <(LC_ALL=C comm -z -23 <(LC_ALL=C sort -zu "${scratch}/listed") \
      <(LC_ALL=C sort -zu "${scratch}/deleted"))
  else
    status=1
  fi
  rm -rf "${scratch}"
  return "${status}"
}
