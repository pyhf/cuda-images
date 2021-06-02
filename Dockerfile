ARG BASE_IMAGE=nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04
# ARG BASE_IMAGE=nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
# ARG BASE_IMAGE=nvidia/cuda:10.1-base-ubuntu18.04
FROM ${BASE_IMAGE} as base

FROM base as builder
# hadolint ignore=DL3003,SC2102
RUN apt-get -qq -y update && \
    apt-get -qq -y install --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        git && \
    apt-get -y autoclean && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/*
# Try 0.1.57 as for 0.1.67
# Starting training...
# 2021-06-01 01:58:28.798857: E external/org_tensorflow/tensorflow/stream_executor/cuda/cuda_blas.cc:226] failed to create cublas handle: CUBLAS_STATUS_NOT_INITIALIZED
# 2021-06-01 01:58:28.798884: F external/org_tensorflow/tensorflow/compiler/xla/service/gpu/gemm_algorithm_picker.cc:113] Check failed: stream->parent()->GetBlasGemmAlgorithms(&algorithms)
# Aborted (core dumped)
    # python3 -m pip --no-cache-dir install jax jaxlib && \
    # python3 -m pip --no-cache-dir install --upgrade jax==0.2.7 jaxlib==0.1.57+cuda101 --find-links https://storage.googleapis.com/jax-releases/jax_releases.html && \
RUN python3 -m pip --no-cache-dir install --upgrade pip setuptools wheel && \
    python3 -m pip --no-cache-dir install --upgrade jax jaxlib==0.1.67+cuda111 --find-links https://storage.googleapis.com/jax-releases/jax_releases.html && \
    python3 -m pip list
RUN git clone https://github.com/matthewfeickert/nvidia-gpu-ml-library-test.git && \
    echo '' >> ~/.bashrc && \
    echo 'alias python=$(command -v python3)' >> ~/.bashrc
