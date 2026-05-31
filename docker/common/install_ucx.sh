#!/bin/bash
set -ex

UCX_VERSION="v1.21.x"
UCX_COMMIT="167a4c6a311d9a42e30a37dcc01b8a3e73ea2826"
UCX_INSTALL_PATH="/usr/local/ucx/"
CUDA_PATH="/usr/local/cuda"
UCX_REPO="https://github.com/openucx/ucx.git"

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
