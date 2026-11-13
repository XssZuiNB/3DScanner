#!/bin/bash
# SPDX-License-Identifier: 0BSD
#
# build.sh - the incremental build agents run while they work (AGENTS.md §4):
# configures build/ on first use, then builds every target with warnings.
#
# It touches no hardware and needs no root.

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f build/CMakeCache.txt ]]; then
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
fi
cmake --build build -j "$(nproc)"
