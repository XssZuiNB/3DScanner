#!/bin/bash
# SPDX-License-Identifier: TODO(template)
#
# fetch_deps.sh - downloads the dependencies the build takes from outside the
# tree into GCA_DEPS_DIR (default ~/.local/share/3dscanner): OrbbecSDK v2 for
# linux arm64 and Eigen 3.4.0. What is already there is kept.
#
# It needs the network, no root, no hardware. The udev rule that gives the user
# the camera is a separate root step (AGENTS.md §2).

set -euo pipefail

readonly DEPS_DIR="${GCA_DEPS_DIR:-${HOME}/.local/share/3dscanner}"
readonly OB_VERSION="v2.10.6"
readonly OB_TARBALL="OrbbecSDK_v2.10.6_202609291004_575347c_linux_arm64.tar.gz"
readonly OB_URL="https://github.com/orbbec/OrbbecSDK_v2/releases/download/${OB_VERSION}/${OB_TARBALL}"
readonly EIGEN_URL="https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"

#######################################
# Downloads a tarball and unpacks it into DEPS_DIR unless its directory exists.
# Globals:
#   DEPS_DIR: the target directory (read).
# Arguments:
#   $1: the URL.
#   $2: the directory the tarball unpacks to, relative to DEPS_DIR.
# Outputs:
#   What it does to STDOUT.
#######################################
fetch() {
  if [[ -d "${DEPS_DIR}/$2" ]]; then
    echo "fetch_deps: $2 present"
    return
  fi
  echo "fetch_deps: downloading $2"
  curl -fsSL "$1" | tar -xz -C "${DEPS_DIR}"
}

mkdir -p "${DEPS_DIR}"
fetch "${OB_URL}" "${OB_TARBALL%.tar.gz}"
fetch "${EIGEN_URL}" "eigen-3.4.0"
