#!/usr/bin/env bash

set -eu

case "$(uname -m)" in
    x86_64 | amd64)
        arch="amd64"
        ;;
    aarch64 | arm64)
        arch="arm64"
        ;;
    *)
        echo "ERROR: unsupported host architecture: $(uname -m)" >&2
        exit 1
        ;;
esac

commit_sha="$(git rev-parse HEAD)"
IMAGE_TAG="baseten/tensorrt_llm-release:${commit_sha:0:10}"
CUDA_ARCHS="${CUDA_ARCHS:-80-real;86-real;90-real;100-real}"
PLATFORM="${PLATFORM:-${arch}}"
BUILD_WHEEL_OPTS=${BUILD_WHEEL_OPTS:-"--clean --use_ccache"}
DOCKER_BUILD_OPTS=${DOCKER_BUILD_OPTS:-"--pull --push"}

env -u CUDA_VERSION -u CUDNN_VERSION -u NCCL_VERSION -u CUBLAS_VERSION make \
    -C docker \
    release_build \
    "IMAGE_WITH_TAG=${IMAGE_TAG}-${arch}" \
    "PLATFORM=${PLATFORM}" \
    "CUDA_ARCHS=${CUDA_ARCHS}" \
    "BUILD_WHEEL_OPTS=${BUILD_WHEEL_OPTS}" \
    "DOCKER_BUILD_OPTS=${DOCKER_BUILD_OPTS}" \
    "DOCKER_BUILD_ARGS=${DOCKER_BUILD_ARGS:-}" \
    "DOCKER_PROGRESS=plain"
