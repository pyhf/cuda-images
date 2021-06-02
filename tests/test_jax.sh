#!/bin/bash

if [[ ! -d nvidia-gpu-ml-library-test ]];then
    git clone --depth 1 https://github.com/matthewfeickert/nvidia-gpu-ml-library-test.git
fi
cd nvidia-gpu-ml-library-test || exit

printf "\n# nvidia-smi --list-gpus\n"
nvidia-smi --list-gpus
printf "\n# python jax_detect_GPU.py\n"
python jax_detect_GPU.py
printf "\n# python jax_MNIST.py\n"
python jax_MNIST.py
