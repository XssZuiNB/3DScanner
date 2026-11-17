#!/bin/bash
# SPDX-License-Identifier: 0BSD
#
# verify.sh - the one offline check: the build, which fails on any warning,
# every test and the checkers of CODING_STANDARDS.md, "Tools" (AGENTS.md §4).
#
#   tools/verify.sh            the checks on a fresh build; records the pass
#   tools/verify.sh --passed   the passed-tree check: whether the working tree
#                              as it is now has passed; exits 0 when it has, 1
#                              when not, 2 when it cannot tell (no git working
#                              tree); builds nothing
#   tools/verify.sh --incremental [--test-timeout <s>]
#                              the checks on build/, rebuilt incrementally,
#                              each test within <s> seconds:
#                              tools/run_mutant.sh's mode in its copy; records
#                              nothing
#
# It touches no hardware and needs no root, exits 0 only when every check
# passes, and writes a failure to STDERR. It passes in a copy of the tree
# without .git too: tools/run_mutant.sh runs it there. The tests need the GPU.
# Every compile goes through ccache (tools/build_common.sh).
#
# The fresh build goes to a new directory beside src/, verify-build.XXXXXX
# (configure_build in tools/build_common.sh says why there), and is removed on
# exit, so a stale build/ cannot hide a failure. A fresh pass in a git working
# tree records the tree's fingerprint (tree_fingerprint) unless the tree
# changed while the checks ran. The fingerprint covers the tree's files, not
# the toolchain or tools/fetch_deps.sh's downloads.

set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=tools/build_common.sh
source tools/build_common.sh

readonly CHECKER_BIN="${HOME}/.local/bin"
readonly CLANG_FORMAT="${CHECKER_BIN}/clang-format"
readonly SHELLCHECK="${CHECKER_BIN}/shellcheck"
# The fingerprints of the trees that passed, one line each: fingerprint, date.
readonly PASSED_LOG="${HOME}/.local/state/3dscanner/verify-passed"
# The incremental mode's build directory, and the file in it that holds the
# configure key (configure_key) of its last successful configure.
readonly INCREMENTAL_DIR=build
readonly KEY_FILE="${INCREMENTAL_DIR}/verify-configure.key"

failed=0
# Whether this run configured its build directory, so that its configure log
# is this run's.
configured=0
# Global, so that the EXIT trap still has it after main returns.
build_dir=''

#######################################
# Reports a failed check and remembers it.
# Globals:
#   failed: set to 1.
# Arguments:
#   $1: what failed.
# Outputs:
#   The failure to STDERR.
# Returns:
#   0.
#######################################
fail() {
  echo "verify: FAIL: $1" >&2
  failed=1
}

#######################################
# Prints the usage and exits.
# Globals:
#   None
# Arguments:
#   None
# Outputs:
#   The usage to STDERR.
# Returns:
#   Exits 2.
#######################################
usage() {
  echo "usage: ${0##*/} [--passed | --incremental [--test-timeout <s>]]" >&2
  exit 2
}

#######################################
# Reads the options.
# Globals:
#   mode: fresh, passed or incremental (set).
#   test_timeout: each test's time limit in seconds, 0 for none (set).
# Arguments:
#   $@: the options.
# Outputs:
#   The usage to STDERR when an option is unknown.
# Returns:
#   0; exits 2 when an option is unknown.
#######################################
parse_options() {
  mode=fresh
  test_timeout=0
  while (($# > 0)); do
    case "$1" in
      --passed) mode=passed ;;
      --incremental) mode=incremental ;;
      --test-timeout)
        if [[ ! "${2:-}" =~ ^[0-9]+$ ]]; then
          usage
        fi
        test_timeout="$2"
        shift
        ;;
      *) usage ;;
    esac
    shift
  done
}

#######################################
# Tells whether the working directory is the root of a git working tree.
# Globals:
#   None
# Arguments:
#   None
# Outputs:
#   None
# Returns:
#   0 when it is, 1 otherwise.
#######################################
in_git_tree() {
  local top
  top="$(git rev-parse --show-toplevel 2>/dev/null)" \
    && [[ "${top}" == "$(pwd -P)" ]]
}

#######################################
# Prints the fingerprint of the tree in the working directory: the SHA-256 of
# a tar of its files (tree_files) in which only the paths, the contents and
# the executable bits vary; the owner, the times and the umask do not change
# it.
# Globals:
#   None
# Arguments:
#   None
# Outputs:
#   The fingerprint, 64 hex digits, to STDOUT.
# Returns:
#   0, or non-zero when a file cannot be listed or read.
#######################################
tree_fingerprint() (
  set -o pipefail
  tree_files . \
    | tar --null --verbatim-files-from --no-recursion -T - --format=gnu \
      --owner=0 --group=0 --numeric-owner --mtime=@0 --mode=u=rwX,go=rX \
      --hard-dereference -cf - \
    | sha256sum | cut -d ' ' -f 1
)

#######################################
# Answers the passed-tree check: whether the working tree as it is now has
# passed a fresh run.
# Globals:
#   PASSED_LOG: the recorded passes (read).
# Arguments:
#   None
# Outputs:
#   The answer to STDOUT; why there is none to STDERR.
# Returns:
#   Exits 0 when the tree has passed, 1 when not, 2 when it cannot tell.
#######################################
report_passed() {
  local fingerprint line
  if ! in_git_tree; then
    echo "verify: not a git working tree, so no fingerprint" >&2
    exit 2
  fi
  if ! fingerprint="$(tree_fingerprint)"; then
    echo "verify: the tree's fingerprint failed" >&2
    exit 2
  fi
  if line="$(grep -m 1 "^${fingerprint} " "${PASSED_LOG}" 2>/dev/null)"; then
    echo "verify: the working tree as it is has passed (${line#* })"
    exit 0
  fi
  echo "verify: the working tree as it is has not passed" \
    "(fingerprint ${fingerprint:0:16}); run tools/verify.sh"
  exit 1
}

#######################################
# Exits when a checker or ccache is missing.
# Globals:
#   CLANG_FORMAT, SHELLCHECK: the checkers (read).
# Arguments:
#   None
# Outputs:
#   The missing tool to STDERR.
# Returns:
#   0; exits 2 when a tool is missing.
#######################################
require_tools() {
  local tool
  for tool in "${CLANG_FORMAT}" "${SHELLCHECK}"; do
    if [[ ! -x "${tool}" ]]; then
      echo "verify: ${tool} missing (CODING_STANDARDS.md, Tools)" >&2
      exit 2
    fi
  done
  require_ccache verify
}

#######################################
# Runs the checkers: clang-format on the C++ and CUDA files and on the page's
# JavaScript (its format only; src/web/page/.clang-format), shellcheck on the
# shell scripts. find, not git ls-files: there may be no .git.
# Globals:
#   CLANG_FORMAT, SHELLCHECK: the checkers (read).
# Arguments:
#   None
# Outputs:
#   The checkers' findings to STDOUT and STDERR.
# Returns:
#   0.
#######################################
run_checkers() {
  local -a format_files sh_files
  mapfile -t format_files < <(find src apps tests -type f \( -name '*.cpp' \
    -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' -o -name '*.js' \) | sort)
  mapfile -t sh_files < <(find tools -type f -name '*.sh' | sort)
  "${CLANG_FORMAT}" --dry-run --Werror --style=file "${format_files[@]}" \
    || fail "clang-format"
  "${SHELLCHECK}" "${sh_files[@]}" || fail "shellcheck"
}

#######################################
# Prints what a configure of the tree depends on, hashed: the list of the
# tree's files (CMake's file(GLOB)) and the content of every CMakeLists.txt
# and *.cmake file. The incremental build directory and .git are left out.
# Globals:
#   INCREMENTAL_DIR: the incremental build directory (read).
# Arguments:
#   None
# Outputs:
#   The key, 64 hex digits, to STDOUT.
# Returns:
#   0, or non-zero when a file cannot be listed or read.
#######################################
configure_key() (
  set -o pipefail
  {
    find . \( -path "./${INCREMENTAL_DIR}" -o -path ./.git \) -prune \
      -o -type f -print0 | LC_ALL=C sort -z
    find . \( -path "./${INCREMENTAL_DIR}" -o -path ./.git \) -prune \
      -o -type f \( -name CMakeLists.txt -o -name '*.cmake' \) -print0 \
      | LC_ALL=C sort -z | xargs -0 -r sha256sum --
  } | sha256sum | cut -d ' ' -f 1
)

#######################################
# Prints the lines of a configure or build log that name a failure: CMake's,
# the compilers' and the linker's errors and fatal errors, without make's own
# lines; the log's last lines when no line names one.
# Globals:
#   None
# Arguments:
#   $1: the log.
# Outputs:
#   At most 20 lines, or the last 40, to STDOUT.
# Returns:
#   0.
#######################################
failure_lines() {
  local lines
  local -r pattern='(^|[^a-z_])(error|fatal)([^a-z_]|$)|undefined reference'
  lines="$(grep -i -E "${pattern}" "$1" \
    | grep -v -E '^g?make(\[[0-9]+\])?: ' | head -n 20)" || true
  echo "${lines:-$(tail -n 40 "$1")}"
}

#######################################
# Configures the build directory when it needs it: a fresh one always; the
# incremental one when it has no configure yet or its configure key changed,
# that is when a file was added, removed or renamed or a CMake file changed.
# A configure over the old CMakeCache.txt would keep the last configure's
# cache variables, so a changed CACHE default or a FORCE would outlive the
# patch that made it: the cache is removed first and configure_build sets
# every option again; the objects stay. The key is written only after a
# configure succeeds, so a failed configure is run again next time.
# Globals:
#   mode: fresh or incremental (read).
#   build_dir: the build directory (read).
#   KEY_FILE: where the incremental build directory's key is kept (read).
#   configured: set to 1 when this run configured.
# Arguments:
#   None
# Outputs:
#   The configure's failure lines (failure_lines) to STDERR when it fails.
# Returns:
#   0 when the directory is configured, 1 when the configure failed.
#######################################
configure_when_needed() {
  local key=''
  if [[ "${mode}" == incremental ]]; then
    key="$(configure_key)"
    if [[ -f "${build_dir}/CMakeCache.txt" \
      && "$(cat "${KEY_FILE}" 2>/dev/null)" == "${key}" ]]; then
      return 0
    fi
    rm -f "${KEY_FILE}" "${build_dir}/CMakeCache.txt"
  fi
  mkdir -p "${build_dir}"
  if ! configure_build "${build_dir}" >"${build_dir}/configure.log" 2>&1; then
    failure_lines "${build_dir}/configure.log" >&2
    fail "configure"
    return 1
  fi
  configured=1
  if [[ -n "${key}" ]]; then
    echo "${key}" >"${KEY_FILE}"
  fi
}

#######################################
# Builds every target in the build directory.
# Globals:
#   build_dir: the build directory (read).
#   BUILD_JOBS: the parallel compile jobs (read).
# Arguments:
#   None
# Outputs:
#   The build's failure lines (failure_lines) to STDERR when it fails.
# Returns:
#   0 when the build succeeds, 1 otherwise.
#######################################
build() {
  if ! cmake --build "${build_dir}" -j "${BUILD_JOBS}" \
    >"${build_dir}/build.log" 2>&1; then
    failure_lines "${build_dir}/build.log" >&2
    fail "build"
    return 1
  fi
}

#######################################
# Prints the lines of the given logs that name a warning, in the forms the
# tools print: "warning:" (g++, the linker, make), "Warning:" (the
# assembler), "warning #NNNNN-D" (nvcc's front end, e.g. #20011-D), "ptxas
# warning" and "CMake Warning" (with or without "(dev)"). The bare word is
# not matched: the configure log prints absolute paths, and a directory
# named, say, no-warning must not fail verify.
# Globals:
#   None
# Arguments:
#   $@: the logs, each of which exists.
# Outputs:
#   The lines to STDOUT.
# Returns:
#   0.
#######################################
warning_lines() {
  local -r pattern='[Ww]arning:|warning #[0-9]+-D|ptxas warning|CMake Warning'
  grep -h -E "${pattern}" "$@" || true
}

#######################################
# Fails on any warning in the logs this run wrote: the build log, and the
# configure log when this run configured. An incremental run that did not
# configure leaves the configure log of an earlier run, which is not read.
# The compiles' warnings are in the build log on a ccache hit too: ccache
# replays them (AGENTS.md §4).
# Globals:
#   build_dir: the build directory (read).
#   configured: whether this run configured (read).
# Arguments:
#   None
# Outputs:
#   At most 20 of the warning lines (warning_lines) to STDERR.
# Returns:
#   0.
#######################################
check_warnings() {
  local lines
  local -a logs=("${build_dir}/build.log")
  if ((configured)); then
    logs=("${build_dir}/configure.log" "${logs[@]}")
  fi
  lines="$(warning_lines "${logs[@]}")"
  if [[ -n "${lines}" ]]; then
    head -n 20 <<<"${lines}" >&2
    fail "warnings"
  fi
}

#######################################
# Prints the failed tests' lines of ctest's output: each failed test's result
# line, which ctest marks with ***, and the test's output after it.
# Globals:
#   None
# Arguments:
#   None
# Inputs:
#   ctest's output (--output-on-failure) on STDIN.
# Outputs:
#   The lines to STDOUT.
# Returns:
#   0.
#######################################
failed_tests() {
  awk '/^ *Start +[0-9]+:/ { next }
       /Test +#[0-9]+:/ { show = /\*\*\*/ }
       /tests passed, / { show = 0 }
       show'
}

#######################################
# Runs the tests, each within the time limit when there is one.
# Globals:
#   build_dir: the build directory (read).
#   test_timeout: each test's time limit in seconds, 0 for none (read).
# Arguments:
#   None
# Outputs:
#   ctest's output to STDOUT; the failed tests' lines to STDERR.
# Returns:
#   0.
#######################################
run_tests() {
  local failures
  local -a args=(--test-dir "${build_dir}" --output-on-failure)
  if ((test_timeout > 0)); then
    args+=(--timeout "${test_timeout}")
  fi
  if ctest "${args[@]}" >"${build_dir}/test.log" 2>&1; then
    cat "${build_dir}/test.log"
    return 0
  fi
  cat "${build_dir}/test.log"
  failures="$(failed_tests <"${build_dir}/test.log")"
  echo "${failures:-$(tail -n 20 "${build_dir}/test.log")}" >&2
  fail "tests"
}

#######################################
# Records a fresh pass, unless the tree changed while the checks ran.
# Globals:
#   PASSED_LOG: the recorded passes (appended to).
# Arguments:
#   $1: the tree's fingerprint before the checks.
# Outputs:
#   What was recorded to STDOUT; a failure to record to STDERR.
# Returns:
#   0; exits 1 when the pass cannot be recorded.
#######################################
record_pass() {
  local after
  if ! after="$(tree_fingerprint)" || [[ "${after}" != "$1" ]]; then
    echo "verify: the tree changed while the checks ran, so the pass is" \
      "not recorded"
    return 0
  fi
  if ! { mkdir -p "${PASSED_LOG%/*}" \
    && echo "${after} $(date -Iseconds)" >>"${PASSED_LOG}"; }; then
    echo "verify: FAIL: the pass cannot be recorded in ${PASSED_LOG}" >&2
    exit 1
  fi
  echo "verify: pass recorded (fingerprint ${after:0:16})"
}

#######################################
# Takes the tree's fingerprint before the checks when this run's pass can be
# recorded, that is in a fresh run in a git working tree.
# Globals:
#   mode: the mode (read).
#   fingerprint: the fingerprint, or empty (set).
#   unrecorded: why a fresh pass will not be recorded, or empty (set).
# Arguments:
#   None
# Outputs:
#   None
# Returns:
#   0.
#######################################
fingerprint_before() {
  fingerprint=''
  unrecorded=''
  if [[ "${mode}" != fresh ]]; then
    return 0
  fi
  if ! in_git_tree; then
    unrecorded="not a git working tree"
  elif ! fingerprint="$(tree_fingerprint)"; then
    fingerprint=''
    unrecorded="the tree's fingerprint failed"
  fi
  return 0
}

#######################################
# Chooses the build directory: a new one beside src/ for a fresh run, removed
# on exit; build/ for an incremental one.
# Globals:
#   mode: the mode (read).
#   build_dir: the build directory (set).
#   INCREMENTAL_DIR: the incremental build directory (read).
# Arguments:
#   None
# Outputs:
#   None
# Returns:
#   0.
#######################################
prepare_build_dir() {
  if [[ "${mode}" == fresh ]]; then
    build_dir="$(mktemp -d "$(pwd -P)/verify-build.XXXXXX")"
    trap 'rm -rf "${build_dir}"' EXIT
  else
    build_dir="${INCREMENTAL_DIR}"
  fi
}

#######################################
# Records a fresh pass (record_pass), or says why it is not recorded.
# Globals:
#   fingerprint, unrecorded: from fingerprint_before (read).
# Arguments:
#   None
# Outputs:
#   What was recorded, or why not, to STDOUT.
# Returns:
#   0; exits 1 when the pass cannot be recorded.
#######################################
record_or_say_why() {
  if [[ -n "${fingerprint}" ]]; then
    record_pass "${fingerprint}"
  elif [[ -n "${unrecorded}" ]]; then
    echo "verify: ${unrecorded}, so the pass is not recorded"
  fi
  return 0
}

#######################################
# Runs every check; records a fresh pass, or answers the passed-tree check.
# Globals:
#   mode, test_timeout: the options (set, then read-only).
#   failed: whether a check failed (read).
# Arguments:
#   $@: the options (usage).
# Outputs:
#   The checks' output to STDOUT; failures to STDERR.
# Returns:
#   Exits 0 when every check passes, 1 when one fails, 2 when a tool is
#   missing or an option unknown. --passed: report_passed.
#######################################
main() {
  parse_options "$@"
  readonly mode test_timeout
  if [[ "${mode}" == passed ]]; then
    report_passed
  fi
  require_tools
  fingerprint_before
  run_checkers
  prepare_build_dir
  if configure_when_needed && build; then
    check_warnings
    run_tests
  fi
  if ((failed)); then
    exit 1
  fi
  record_or_say_why
  echo "verify: all checks passed"
}

main "$@"
