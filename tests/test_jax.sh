#!/bin/bash

if [[ ! -d nvidia-gpu-ml-library-test ]];then
    git clone --depth 1 https://github.com/matthewfeickert/nvidia-gpu-ml-library-test.git
fi
cd nvidia-gpu-ml-library-test || exit
python jax_detect_GPU.py
python jax_MNIST.py
