ARG BASE_IMAGE=nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04
FROM ${BASE_IMAGE} as base

SHELL [ "/bin/bash", "-c" ]

WORKDIR /home/data

ARG PYHF_VERSION=0.6.1
ARG PYHF_BACKEND=jax
# Set PATH to pickup virtualenv when it is unpacked
ENV PATH=/usr/local/venv/bin:"${PATH}"
COPY install_backend.sh /tmp/install_backend.sh
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
    . /tmp/install_backend.sh "${PYHF_BACKEND}" && \
    python -m pip list

ENTRYPOINT ["/bin/bash", "-l", "-c"]
CMD ["/bin/bash"]
