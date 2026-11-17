#!/bin/bash
# SPDX-License-Identifier: 0BSD
#
# fetch_deps.sh - downloads the dependencies the build takes from outside the
# tree into GCA_DEPS_DIR (default ~/.local/share/3dscanner): OrbbecSDK v2 for
# linux arm64, Eigen 3.4.0 and cpp-httplib v0.66.0. What is already there is
# kept.
#
# It needs the network, no root, no hardware. The udev rule that gives the user
# the camera is a separate root step (AGENTS.md §2).

set -euo pipefail

readonly DEPS_DIR="${GCA_DEPS_DIR:-${HOME}/.local/share/3dscanner}"
readonly OB_VERSION="v2.10.6"
readonly OB_TARBALL="OrbbecSDK_v2.10.6_202609291004_575347c_linux_arm64.tar.gz"
readonly OB_URL="https://github.com/orbbec/OrbbecSDK_v2/releases/download/${OB_VERSION}/${OB_TARBALL}"
readonly EIGEN_URL="https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"
readonly HTTPLIB_URL="https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.66.0.tar.gz"

#######################################
# Downloads a tarball and unpacks it into DEPS_DIR unless its directory exists.
# Globals:
#   DEPS_DIR: the target directory (read).
# Arguments:
#   $1: the URL.
#   $2: the directory the tarball unpacks to, relative to DEPS_DIR.
# Outputs:
#   What it does to STDOUT.
# Returns:
#   0 when the directory is there or the tarball unpacked; exits non-zero
#   otherwise (set -e).
#######################################
fetch() {
  if [[ -d "${DEPS_DIR}/$2" ]]; then
    echo "fetch_deps: $2 present"
    return
  fi
  echo "fetch_deps: downloading $2"
  curl -fsSL "$1" | tar -xz -C "${DEPS_DIR}"
}

#######################################
# Fetches each dependency that is not there yet.
# Globals:
#   DEPS_DIR, OB_URL, OB_TARBALL, EIGEN_URL, HTTPLIB_URL: what to fetch where
#   (read).
# Arguments:
#   None
# Outputs:
#   What it does to STDOUT.
# Returns:
#   0; exits non-zero when a download or an unpack fails.
#######################################
main() {
  mkdir -p "${DEPS_DIR}"
  fetch "${OB_URL}" "${OB_TARBALL%.tar.gz}"
  fetch "${EIGEN_URL}" "eigen-3.4.0"
  fetch "${HTTPLIB_URL}" "cpp-httplib-0.66.0"
}

main "$@"
