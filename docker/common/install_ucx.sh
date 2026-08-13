#!/bin/bash
set -ex

# Built from the Baseten fork, which carries the rendezvous failover work KV
# transfer depends on: rendezvous admitted under UCP_ERR_HANDLING_MODE_FAILOVER
# for both the get and put schemes, and control-lane selection that avoids a
# device whose lane has already failed. Without it a large KV transfer has no
# failover-capable rendezvous protocol and falls back to eager.
#
# This bump also picks up the fix for a segfault seen in production: the put
# scheme released the peer's remote key before the completion that decides
# whether to restart, so a failover restart dereferenced it. Any build at or
# before the previous pin admits put to failover without that fix and can
# crash the worker on a rail failure mid-write.
UCX_VERSION="master"
UCX_COMMIT="30a2390dfbca3b90d55cca6d518016ee1b384b8f"
UCX_INSTALL_PATH="/usr/local/ucx/"
CUDA_PATH="/usr/local/cuda"
UCX_REPO="https://github.com/basetenlabs/ucx.git"

mkdir -p /third-party-source

rm -rf ${UCX_INSTALL_PATH}
git clone -b ${UCX_VERSION} ${UCX_REPO}
cd ucx
git checkout ${UCX_COMMIT}
cd ..
tar -czf /third-party-source/ucx-${UCX_VERSION}.tar.gz ucx
cd ucx
./autogen.sh
./contrib/configure-release       \
  --prefix=${UCX_INSTALL_PATH}    \
  --enable-shared                 \
  --disable-static                \
  --disable-doxygen-doc           \
  --enable-optimizations          \
  --enable-cma                    \
  --enable-devel-headers          \
  --with-cuda=${CUDA_PATH}        \
  --with-verbs                    \
  --with-dm                       \
  --enable-mt
make install -j$(nproc)
cd ..
if [ "${KEEP_SOURCE:-1}" = "1" ]; then
    # Preserve pristine source so downstream dev images can bind-mount
    # or symlink it. Recursive .git removal — UCX submodules carry
    # nested .git dirs. `make distclean` wipes the autotools build
    # artifacts (.libs/, *.o, generated configure outputs — typically
    # ~half the post-install tree) and leaves the source ready to
    # re-run ./autogen.sh in-pod.
    ( cd ucx && make distclean >/dev/null 2>&1 || true )
    mkdir -p /opt/src && mv ucx /opt/src/ucx
    find /opt/src/ucx -name .git -prune -exec rm -rf {} +
else
    rm -rf ucx  # Remove UCX source to save space
fi
echo "export LD_LIBRARY_PATH=${UCX_INSTALL_PATH}/lib:\$LD_LIBRARY_PATH" >> "${ENV}"
