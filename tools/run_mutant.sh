#!/bin/bash
# SPDX-License-Identifier: 0BSD
#
# run_mutant.sh - whether tools/verify.sh catches a deliberate mistake.
#
# A check nobody has seen fail is not a check. A mutant is a patch, as `git
# diff` writes it (paths from the repository's root), that makes the mistake a
# check is meant to catch. This copies the working tree, its uncommitted
# changes and new files included, and requires tools/verify.sh --incremental to
# pass there unpatched: the copy's one full build, into its build/, mostly
# ccache's hits from the working tree's builds (tools/build_common.sh). Then,
# for each patch in turn, it applies the patch, runs tools/verify.sh
# --incremental again (the checkers, a configure on a fresh cache when a build
# file changed, the rebuild of what the patch touched, every test), within
# time limits, reports the mutant caught or survived, and takes the patch back
# out. Taking a patch out writes its files anew, so the next patch's build
# rebuilds whatever the patch left broken before that patch is judged. The
# working tree is only read.

set -u

# A patched run's time limit for each test: this many times the unpatched
# run's time, and at least this many seconds. A mutant can make a test loop
# forever.
readonly LIMIT_FACTOR=3
readonly LIMIT_FLOOR_S=60
# The bound on the rest of a run, the configure and the build above all, which
# a mutant can make hang too: 3 times a verify with an empty ccache
# [M 2026-11-13: 196 s]. The unpatched run's time cannot set it: from ccache's
# hits it says nothing about how long a patched rebuild takes.
readonly BUILD_LIMIT_S=600

# shellcheck source=tools/build_common.sh
source "$(dirname "$0")/build_common.sh" || exit 2

#######################################
# Copies the working tree's files (tree_files) as they are in the working
# tree.
# Globals:
#   None
# Arguments:
#   $1: the repository.
#   $2: the copy's directory.
# Outputs:
#   A failure to STDERR.
# Returns:
#   0 when the copy is made, 1 otherwise.
#######################################
copy_tree() {
  local -a status
  mkdir -p "$2" || return 1
  tree_files "$1" | tar -C "$1" --null --no-recursion -T - -cf - \
    | tar -C "$2" -xf -
  status=("${PIPESTATUS[@]}")
  if ((status[0] != 0 || status[1] != 0 || status[2] != 0)); then
    echo "run_mutant: the working tree could not be copied" >&2
    return 1
  fi
}

#######################################
# Applies a patch to the copy, or takes it back out. git never looks for a
# repository above the copy.
# Globals:
#   tmp: the run's scratch directory, the copy's parent (read).
# Arguments:
#   $1: the patch, an absolute path.
#   $2...: git apply's further options (-R).
# Outputs:
#   git apply's complaint to STDERR.
# Returns:
#   git apply's exit status.
#######################################
apply_in_copy() {
  (cd "${tmp}/tree" && GIT_CEILING_DIRECTORIES="${tmp}" \
    git apply "${@:2}" "$1")
}

#######################################
# Runs tools/verify.sh --incremental in the copy, each test within its limit
# and the whole run within BUILD_LIMIT_S more. timeout(1) runs it in a process
# group of its own and, when the limit expires, signals the whole group; what
# has not ended 10 s after the TERM gets a KILL. The check's scratch files go
# to ${tmp}/scratch, so a check killed there leaves nothing behind.
# Globals:
#   BUILD_LIMIT_S: the bound on the rest of the run (read).
#   tmp: the run's scratch directory (read).
#   group: the running check's process group, while it runs (set).
# Arguments:
#   $1: each test's time limit in seconds; 0 for none.
# Outputs:
#   The check's STDOUT to ${tmp}/verify.log, its STDERR to ${tmp}/verify.err.
# Returns:
#   tools/verify.sh's exit status; 124, or 137 after the KILL, when the whole
#   run's limit expired.
#######################################
verify_in_copy() {
  local status
  (
    cd "${tmp}/tree" || exit 2
    export TMPDIR="${tmp}/scratch" GIT_CEILING_DIRECTORIES="${tmp}"
    exec timeout --kill-after=10 $(($1 + BUILD_LIMIT_S)) \
      tools/verify.sh --incremental --test-timeout "$1"
  ) </dev/null >"${tmp}/verify.log" 2>"${tmp}/verify.err" &
  group=$!
  wait "${group}"
  status=$?
  group=''
  return "${status}"
}

#######################################
# Prints why the last check failed: the first lines it wrote to STDERR (the
# lines that name the failure: the checkers' findings, the configure's or the
# build's errors, a failed test's lines), or, when it wrote none there, the
# last lines of its STDOUT.
# Globals:
#   tmp: the run's scratch directory (read).
# Arguments:
#   None
# Outputs:
#   At most twelve lines, indented, the copy's paths relative, to STDOUT.
# Returns:
#   0.
#######################################
print_failed() {
  local line
  {
    if [[ -s "${tmp}/verify.err" ]]; then
      head -n 12 "${tmp}/verify.err"
    else
      tail -n 12 "${tmp}/verify.log"
    fi
  } | while IFS= read -r line; do
    line="  ${line//"${tmp}/tree/"/}"
    echo "${line:0:160}"
  done
}

#######################################
# Prints the time limit of each test in a patched run.
# Globals:
#   LIMIT_FACTOR, LIMIT_FLOOR_S: the limit's rule (read).
# Arguments:
#   $1: the unpatched run's time in seconds.
# Outputs:
#   The limit in seconds to STDOUT.
# Returns:
#   0.
#######################################
time_limit() {
  local limit=$(($1 * LIMIT_FACTOR))
  echo $((limit < LIMIT_FLOOR_S ? LIMIT_FLOOR_S : limit))
}

#######################################
# Runs one mutant on the copy and reports it.
# Globals:
#   limit: each test's time limit in a patched run (read).
#   BUILD_LIMIT_S: the bound on the rest of the run (read).
#   tmp: the run's scratch directory (read).
# Arguments:
#   $1: the patch, an absolute path.
#   $2: its name, for the report.
# Outputs:
#   The verdict and the lines that failed to STDOUT; a patch that cannot be
#   applied or taken out to STDERR.
# Returns:
#   0 when the mutant is caught, 1 when it survives, 2 when the patch cannot
#   be applied or taken out again.
#######################################
run_one() {
  local verdict=0 status
  if ! apply_in_copy "$1"; then
    echo "run_mutant: $2 does not apply to the copy" >&2
    return 2
  fi
  verify_in_copy "${limit}"
  status=$?
  if ((status == 0)); then
    echo "survived: $2: verify passes with it"
    verdict=1
  elif ((status == 124 || status == 137)); then
    echo "caught by the time limit: $2: verify ran past" \
      "$((limit + BUILD_LIMIT_S)) s"
    print_failed
  elif grep -q -F '***Timeout' "${tmp}/verify.err"; then
    echo "caught by the time limit: $2: a test ran past ${limit} s"
    print_failed
  else
    echo "caught by verify: $2"
    print_failed
  fi
  if ! apply_in_copy "$1" -R; then
    echo "run_mutant: $2 cannot be taken out of the copy again" >&2
    return 2
  fi
  return "${verdict}"
}

#######################################
# Ends a check still running in the copy, its whole process group, and
# removes the scratch directory: the EXIT trap.
# Globals:
#   group: the running check's process group, if one runs (read).
#   tmp: the run's scratch directory (read).
# Arguments:
#   None
# Outputs:
#   None
# Returns:
#   0.
#######################################
clean_up() {
  if [[ -n "${group:-}" ]]; then
    kill -TERM -- "-${group}" 2>/dev/null
    wait "${group}" 2>/dev/null
  fi
  rm -rf "${tmp}"
}

#######################################
# Runs each patch as a mutant on a scratch copy of the working tree.
# Globals:
#   limit: each test's time limit in a patched run (set).
#   group: the running check's process group (set).
#   tmp: the run's scratch directory, removed on exit (set).
# Arguments:
#   $@: the patches.
# Outputs:
#   The unpatched run's time, the time limit and each mutant's verdict to
#   STDOUT; errors to STDERR.
# Returns:
#   Exits 0 when every mutant is caught, 1 when one survives, and 2 when a
#   patch does not apply, a tool is missing, the unpatched copy fails verify,
#   or the run is interrupted.
#######################################
main() {
  local repo patch path survived=0 status start verify_s
  local -a patches=()

  if (($# < 1)); then
    echo "usage: ${0##*/} <patch>..." >&2
    exit 2
  fi
  require_ccache run_mutant
  repo="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 2
  for patch in "$@"; do
    if ! path="$(realpath -e -- "${patch}")" \
      || ! git -C "${repo}" apply --check "${path}"; then
      echo "run_mutant: ${patch} does not apply to the working tree" >&2
      exit 2
    fi
    patches+=("${path}")
  done

  # Global, so that the EXIT trap still has it after main returns.
  tmp="$(mktemp -d)"
  group=''
  trap clean_up EXIT
  trap 'exit 2' INT TERM
  if [[ "${tmp}/" == "${repo}/"* ]]; then
    echo "run_mutant: the scratch directory ${tmp} is inside the" \
      "repository; set TMPDIR elsewhere" >&2
    exit 2
  fi
  mkdir -p "${tmp}/scratch" || exit 2
  copy_tree "${repo}" "${tmp}/tree" || exit 2
  start="${SECONDS}"
  if ! verify_in_copy 0; then
    echo "run_mutant: verify fails in the unpatched copy, so no mutant" \
      "can be judged:" >&2
    print_failed >&2
    exit 2
  fi
  verify_s=$((SECONDS - start))
  limit="$(time_limit "${verify_s}")"
  echo "run_mutant: a copy of the working tree passes verify in" \
    "${verify_s} s; in a patched verify a test may take ${limit} s, the" \
    "whole verify $((limit + BUILD_LIMIT_S)) s"

  for patch in "${patches[@]}"; do
    run_one "${patch}" "${patch##*/}"
    status=$?
    if ((status == 2)); then
      exit 2
    fi
    survived=$((survived + status))
  done
  if ((survived)); then
    exit 1
  fi
}

main "$@"
