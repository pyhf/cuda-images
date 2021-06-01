ARG BASE_IMAGE=nvidia/cuda:10.1-base-ubuntu18.04
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
RUN python3 -m pip --no-cache-dir install --upgrade pip setuptools wheel && \
    python3 -m pip --no-cache-dir install jax jaxlib && \
    python3 -m pip install --upgrade jaxlib==0.1.67+cuda101 --find-links https://storage.googleapis.com/jax-releases/jax_releases.html && \
    python3 -m pip list
