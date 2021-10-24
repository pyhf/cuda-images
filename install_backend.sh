#!/bin/bash

# CUDA_VERSION already exists as ENV variable in the base image

function install_jax_backend {
    local jaxlib_version
    local cuda_version

    python -m pip --no-cache-dir install --upgrade jax jaxlib
    jaxlib_version=$(python -c 'import jaxlib; print(jaxlib.__version__)')
    # shellcheck disable=SC2153
    cuda_version=$(echo "${CUDA_VERSION}" | cut -d . -f -2 | sed 's/\.//')
    # Determine CuDNN version number
    CUDNN_MAJOR="$(cat /usr/include/x86_64-linux-gnu/cudnn_v*.h | grep '#define CUDNN_MAJOR' | awk '{print $NF}')"
    CUDNN_MINOR="$(cat /usr/include/x86_64-linux-gnu/cudnn_v*.h | grep '#define CUDNN_MINOR' | awk '{print $NF}')"
    CUDNN_PATCHLEVEL="$(cat /usr/include/x86_64-linux-gnu/cudnn_v*.h | grep '#define CUDNN_PATCHLEVEL' | awk '{print $NF}')"
    cudnn_version="${CUDNN_MAJOR}${CUDNN_MINOR}${CUDNN_PATCHLEVEL}"
    echo "jaxlib version: ${jaxlib_version}"
    echo "jaxlib+cuda version: ${jaxlib_version}+cuda${cuda_version}.cudnn${cudnn_version}"
    python -m pip --no-cache-dir install --upgrade jax jaxlib=="${jaxlib_version}+cuda${cuda_version}.cudnn${cudnn_version}" --find-links https://storage.googleapis.com/jax-releases/jax_releases.html
}

function install_pytorch_backend {
    local torch_version=1.9.0
    local cuda_version

    cuda_version=$(echo "${CUDA_VERSION}" | cut -d . -f -2 | sed 's/\.//')
    echo "torch+cuda version: ${torch_version}+cu${cuda_version}"
    python -m pip --no-cache-dir install torch=="${torch_version}+cu${cuda_version}" --find-links https://download.pytorch.org/whl/torch_stable.html
}


function main() {
    # 1: pyhf backend name

    pyhf_backend_name=""
    if [[ $# -gt 0 ]];then
        pyhf_backend_name=$(echo "${1}" | awk '{print tolower($0)}')
    fi

    if [[ "${pyhf_backend_name}" = "jax" ]]; then
        install_jax_backend
    elif [[ "${pyhf_backend_name}" =~ ^("pytorch"|"torch")$ ]]; then
        install_pytorch_backend
    elif [[ "${pyhf_backend_name}" =~ ^("tensorflow"|"tf")$ ]]; then
        # TODO: Impliment
        # install_tensorflow_backend
        exit 1
    fi
}

main "$@" || exit 1
