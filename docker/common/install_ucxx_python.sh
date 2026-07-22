#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

set -ex

# basetenlabs/ucxx fork: rapidsai/ucxx v0.50.00 (cf34a4e) + the AM
# receiver-callback Python binding and opt-in UCP failover error-handling
# mode that the B10 AM KV-transfer wire plane depends on. See
# basetenlabs/ucxx#1 (branch wilson/am-receiver-callback, base release-0.50).
# Cloned from github.com directly, NOT via GITHUB_MIRROR: the mirror only
# carries upstream (rapidsai/nvidia) repos, not this fork. The rapids-cmake
# fetch below still honors GITHUB_MIRROR.
UCXX_REPO="${UCXX_REPO:-https://github.com/basetenlabs/ucxx.git}"
UCXX_COMMIT="6c6d7015c4bb08949c294c61e7c3e715adb43fd9"
UCXX_INSTALL_PREFIX="${UCXX_INSTALL_PREFIX:-/usr/local}"
UCX_INSTALL_PATH="${UCX_INSTALL_PATH:-/usr/local/ucx}"
UCXX_SOURCE_PATH="${UCXX_SOURCE_PATH:-/tmp/ucxx-python-src}"
UCXX_BUILD_PATH="${UCXX_BUILD_PATH:-/tmp/ucxx-python-build}"

python3 -m pip install --no-cache-dir \
    --index-url https://pypi.nvidia.com \
    --extra-index-url https://pypi.org/simple \
    "cython>=3.2.2" \
    "rapids-build-backend>=0.4.0,<0.5.0" \
    "scikit-build-core[pyproject]>=0.11.0" \
    "rmm-cu13==26.6.0" \
    "librmm-cu13==26.6.0"

rm -rf "${UCXX_SOURCE_PATH}" "${UCXX_BUILD_PATH}"
git clone --filter=blob:none "${UCXX_REPO}" "${UCXX_SOURCE_PATH}"
cd "${UCXX_SOURCE_PATH}"
git checkout "${UCXX_COMMIT}"
if [ -n "${GITHUB_MIRROR:-}" ] && [ -f fetch_rapids.cmake ]; then
    sed -i \
        "s#https://raw.githubusercontent.com/rapidsai/rapids-cmake#${GITHUB_MIRROR}/rapidsai/rapids-cmake/raw/refs/heads#g" \
        fetch_rapids.cmake
fi

GLIBCXX_USE_CXX11_ABI="$(python3 - <<'PY'
import torch

print(int(torch._C._GLIBCXX_USE_CXX11_ABI))
PY
)"

export INSTALL_PREFIX="${UCXX_INSTALL_PREFIX}"
export CONDA_PREFIX="${UCXX_INSTALL_PREFIX}"
export CMAKE_PREFIX_PATH="${UCX_INSTALL_PATH}:${UCXX_INSTALL_PREFIX}:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="${UCX_INSTALL_PATH}/lib:${LD_LIBRARY_PATH:-}"
export LIB_BUILD_DIR="${UCXX_BUILD_PATH}/libucxx"
export PYTHON_BUILD_DIR="${UCXX_BUILD_PATH}/libucxx_python"

./build.sh libucxx libucxx_python ucxx \
    "--cmake-args=\"-DBUILD_SHARED_LIBS=ON -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}\""

ldconfig
python3 - <<'PY'
import torch  # noqa: F401
import ucxx

print(f"UCXX Python import OK: ucxx={ucxx.__version__}, ucx={ucxx.get_ucx_version()}")
PY

rm -rf "${UCXX_SOURCE_PATH}" "${UCXX_BUILD_PATH}"
