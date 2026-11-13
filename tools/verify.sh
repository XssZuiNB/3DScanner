#!/bin/bash
# SPDX-License-Identifier: TODO(template)
#
# verify.sh - the one offline check: the build, every test and the checkers of
# CODING_STANDARDS.md, "Tools" (AGENTS.md §4).
#
# It touches no hardware and needs no root, exits 0 only when every check
# passes, and writes a failure to STDERR. It passes in a copy of the tree
# without .git too: tools/run_mutant.sh runs it there. The tests need the GPU.
# The build goes to a temporary directory, so a stale build/ cannot hide a
# failure.

set -euo pipefail
cd "$(dirname "$0")/.."

readonly CHECKER_BIN="${HOME}/.local/bin"
readonly CLANG_FORMAT="${CHECKER_BIN}/clang-format"
readonly SHELLCHECK="${CHECKER_BIN}/shellcheck"

# The C++ and CUDA files clang-format holds, and the shell scripts shellcheck
# holds. find, not git ls-files: there may be no .git.
mapfile -t cxx_files < <(find src apps tests -type f \
  \( -name '*.cpp' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' \) | sort)
mapfile -t sh_files < <(find tools -type f -name '*.sh' | sort)

failed=0

#######################################
# Reports a failed check and remembers it.
# Globals:
#   failed: set to 1.
# Arguments:
#   $1: what failed.
# Outputs:
#   The failure to STDERR.
#######################################
fail() {
  echo "verify: FAIL: $1" >&2
  failed=1
}

for tool in "${CLANG_FORMAT}" "${SHELLCHECK}"; do
  [[ -x "${tool}" ]] || { echo "verify: ${tool} missing (CODING_STANDARDS.md, Tools)" >&2; exit 2; }
done

"${CLANG_FORMAT}" --dry-run --Werror --style=file "${cxx_files[@]}" || fail "clang-format"
"${SHELLCHECK}" "${sh_files[@]}" || fail "shellcheck"

build_dir="$(mktemp -d)"
trap 'rm -rf "${build_dir}"' EXIT
if cmake -S . -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release >"${build_dir}/configure.log" 2>&1; then
  if cmake --build "${build_dir}" -j "$(nproc)" >"${build_dir}/build.log" 2>&1; then
    ctest --test-dir "${build_dir}" --output-on-failure || fail "tests"
  else
    tail -40 "${build_dir}/build.log" >&2
    fail "build"
  fi
else
  tail -40 "${build_dir}/configure.log" >&2
  fail "configure"
fi

if ((failed)); then
  exit 1
fi
echo "verify: all checks passed"
