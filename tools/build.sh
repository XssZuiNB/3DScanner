#!/bin/bash
# SPDX-License-Identifier: 0BSD
#
# build.sh - the incremental build agents run while they work (AGENTS.md §4):
# configures build/ on first use, and again when it was configured without
# this tree's ccache launcher (tools/build_common.sh), then builds every
# target with warnings, every compile through ccache.
#
# It touches no hardware and needs no root.

set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=tools/build_common.sh
source tools/build_common.sh

#######################################
# Configures build/ when it needs it and builds it.
# Globals:
#   BUILD_JOBS: the parallel compile jobs (read).
# Arguments:
#   None
# Outputs:
#   CMake's and the compilers' output to STDOUT and STDERR.
# Returns:
#   0 when the build succeeds; exits non-zero otherwise.
#######################################
main() {
  require_ccache build
  if ! configured_with_ccache; then
    configure_build build
  fi
  cmake --build build -j "${BUILD_JOBS}"
}

main "$@"
