#!/usr/bin/env bash

set -eu

export BUILDX_NO_DEFAULT_ATTESTATIONS=1

commit_sha="$(git rev-parse HEAD)"
image_tag="baseten/tensorrt_llm-release:${commit_sha:0:10}"

CUDA_ARCHS_AMD64="${CUDA_ARCHS_AMD64:-80-real;86-real;90-real;100-real;103-real}"
CUDA_ARCHS_ARM64="${CUDA_ARCHS_ARM64:-100-real;103-real}"
BUILD_WHEEL_JOBS="${BUILD_WHEEL_JOBS:-16}"

build_wheel_opts="-j ${BUILD_WHEEL_JOBS} -D CMAKE_CXX_COMPILER_LAUNCHER=sccache -D CMAKE_CUDA_COMPILER_LAUNCHER=sccache"
docker_build_opts="--pull --push --platform linux/amd64,linux/arm64"
docker_build_args="--secret id=SCCACHE_WEBDAV_TOKEN,env=SCCACHE_WEBDAV_TOKEN"
docker_build_args="${docker_build_args} --build-arg CUDA_ARCHS_AMD64=\"${CUDA_ARCHS_AMD64}\""
docker_build_args="${docker_build_args} --build-arg CUDA_ARCHS_ARM64=\"${CUDA_ARCHS_ARM64}\""

env -u CUDA_ARCHS -u CUDA_VERSION -u CUDNN_VERSION -u NCCL_VERSION -u CUBLAS_VERSION make \
    -C docker \
    release_build \
    "IMAGE_WITH_TAG=${image_tag}" \
    "PLATFORM=multi" \
    "BUILD_WHEEL_OPTS=${build_wheel_opts}" \
    "DOCKER_BUILD_OPTS=${docker_build_opts}" \
    "DOCKER_BUILD_ARGS=${docker_build_args}" \
    "DOCKER_PROGRESS=plain"
