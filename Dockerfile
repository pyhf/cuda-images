ARG BASE_IMAGE=nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04
FROM ${BASE_IMAGE} as base

WORKDIR /home/data

ARG PYHF_VERSION=0.6.1
# Set PATH to pickup virtualenv when it is unpacked
ENV PATH=/usr/local/venv/bin:"${PATH}"
# CUDA_VERSION already exists as ENV variable in the base image
# hadolint ignore=DL3003,SC2102
RUN apt-get -qq -y update && \
    apt-get -qq -y install --no-install-recommends \
        python3 \
        python3-dev \
        curl \
        git && \
    apt-get -y autoclean && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/* && \
    curl --silent --location --remote-name https://bootstrap.pypa.io/virtualenv.pyz && \
    python3 virtualenv.pyz /usr/local/venv && \
    rm virtualenv.pyz && \
    python -m pip --no-cache-dir install --upgrade pip setuptools wheel && \
    python -m pip --no-cache-dir install "pyhf[xmlio,contrib]==${PYHF_VERSION}" && \
    python -m pip --no-cache-dir install --upgrade jax jaxlib && \
    export jaxlib_version=$(python -c 'import jaxlib; print(jaxlib.__version__)') && \
    export cuda_version=$(echo ${CUDA_VERSION} | cut -d . -f -2 | sed 's/\.//') && \
    echo "jaxlib version: ${jaxlib_version}" && \
    echo "jaxlib+cuda version: ${jaxlib_version}+cuda${cuda_version}" && \
    python -m pip --no-cache-dir install --upgrade jax jaxlib=="${jaxlib_version}+cuda${cuda_version}" --find-links https://storage.googleapis.com/jax-releases/jax_releases.html && \
    python -m pip list

ENTRYPOINT ["/bin/bash", "-l", "-c"]
CMD ["/bin/bash"]
