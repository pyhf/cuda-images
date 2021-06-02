ARG BASE_IMAGE=nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04
# Thanks Anish (@trickarcher)
FROM ${BASE_IMAGE} as base

ARG PYHF_VERSION=0.6.1
# CUDA_VERSION is already set in the base image
# hadolint ignore=DL3003,SC2102
RUN apt-get -qq -y update && \
    apt-get -qq -y install --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        git && \
    apt-get -y autoclean && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m pip --no-cache-dir install --upgrade pip setuptools wheel && \
    python3 -m pip --no-cache-dir install "pyhf[xmlio,contrib]==${PYHF_VERSION}" && \
    python3 -m pip --no-cache-dir install --upgrade jax jaxlib && \
    export jaxlib_version=$(python3 -c 'import jaxlib; print(jaxlib.__version__)') && \
    export cuda_version=$(echo ${CUDA_VERSION} | cut -d . -f -2 | sed 's/\.//') && \
    echo "jaxlib version: ${jaxlib_version}" && \
    echo "jaxlib+cuda version: ${jaxlib_version}+cuda${cuda_version}" && \
    python3 -m pip --no-cache-dir install --upgrade jax jaxlib=="${jaxlib_version}+cuda${cuda_version}" --find-links https://storage.googleapis.com/jax-releases/jax_releases.html && \
    python3 -m pip list && \
    echo '' >> ~/.bashrc && \
    echo 'alias python=$(command -v python3)' >> ~/.bashrc
RUN git clone https://github.com/matthewfeickert/nvidia-gpu-ml-library-test.git
