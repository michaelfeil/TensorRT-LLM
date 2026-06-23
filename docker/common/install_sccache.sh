#!/bin/bash

set -ex

GITHUB_URL="https://github.com"
if [ -n "${GITHUB_MIRROR}" ]; then
    GITHUB_URL=${GITHUB_MIRROR}
fi

ARCH=$(uname -m)
SCCACHE_VERSION="0.15.0"

case "$ARCH" in
  x86_64)
    SCCACHE_TARGET="x86_64-unknown-linux-musl"
    ;;
  aarch64)
    SCCACHE_TARGET="aarch64-unknown-linux-musl"
    ;;
  *)
    echo "Skipping sccache install for unsupported architecture: ${ARCH}"
    exit 0
    ;;
esac

SCCACHE_DIR="sccache-v${SCCACHE_VERSION}-${SCCACHE_TARGET}"
SCCACHE_URL="${GITHUB_URL}/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/${SCCACHE_DIR}.tar.gz"

wget --no-verbose --retry-connrefused --timeout=180 --tries=10 -O "/tmp/${SCCACHE_DIR}.tar.gz" "${SCCACHE_URL}"
tar -xzf "/tmp/${SCCACHE_DIR}.tar.gz" -C /tmp/
cp "/tmp/${SCCACHE_DIR}/sccache" /usr/bin/sccache
chmod +x /usr/bin/sccache
rm -rf "/tmp/${SCCACHE_DIR}" "/tmp/${SCCACHE_DIR}.tar.gz"
