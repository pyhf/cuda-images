#!/bin/bash


function main() {
    # 1: pyhf backend name

    pyhf_backend_name=""
    if [[ $# -gt 0 ]];then
        pyhf_backend_name=$(echo "${1}" | awk '{print tolower($0)}')
    fi

    if [[ ! -d nvidia-gpu-ml-library-test ]];then
        git clone --depth 1 https://github.com/matthewfeickert/nvidia-gpu-ml-library-test.git
    fi
    cd nvidia-gpu-ml-library-test || exit

    printf "\n# nvidia-smi --list-gpus\n"
    nvidia-smi --list-gpus

    if [[ "${pyhf_backend_name}" = "jax" ]]; then
        printf "\n# python jax_detect_GPU.py\n"
        python jax_detect_GPU.py
        printf "\n# python jax_MNIST.py\n"
        python jax_MNIST.py
    elif [[ "${pyhf_backend_name}" =~ ^("pytorch"|"torch")$ ]]; then
        printf "\n# python torch_detect_GPU.py\n"
        python torch_detect_GPU.py
        printf "\n# python torch_MNIST.py --epochs 5\n"
        python torch_MNIST.py --epochs 5
    fi
}

main "$@" || exit 1
