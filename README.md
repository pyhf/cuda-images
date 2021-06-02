# `pyhf` NVIDIA CUDA enabled Docker images


[`pyhf`](https://pyhf.readthedocs.io/) Docker images built on the [NVIDIA CUDA enabled images](https://github.com/NVIDIA/nvidia-docker) for runtime use with the the NVIDIA Container Toolkit.

[![Docker Images](https://github.com/pyhf/cuda-images/actions/workflows/docker.yml/badge.svg?branch=main)](https://github.com/pyhf/cuda-images/actions/workflows/docker.yml?query=branch%3Amain)
[![Docker Pulls](https://img.shields.io/docker/pulls/pyhf/cuda.svg)](https://hub.docker.com/r/pyhf/cuda/)


## Installation

- Make sure that you have the [`nvidia-container-toolkit`](https://github.com/NVIDIA/nvidia-docker) installed on the host machine
- Check the [list of available tags on Docker Hub](https://hub.docker.com/r/pyhf/cuda/tags?page=1) to find the tag you want
- Use `docker pull` to pull down the image corresponding to the tag

Example:

```
docker pull pyhf/cuda:0.6.1-jax-cuda-11.1
```

## Use

To check that NVIDIA GPUS are being properly detected run

```
docker run --rm --gpus all pyhf/cuda:0.6.1-jax-cuda-11.1 'nvidia-smi'
```

and check if the [`nvidia-smi`](https://developer.nvidia.com/nvidia-system-management-interface) output appears correctly.

To run (interactively) using GPUs on the host machine:

```
docker run --rm -ti --gpus all pyhf/cuda:0.6.1-jax-cuda-11.1
```

## Tests

To verify things are working on your host machine you can run

```
docker run --rm --gpus all -v $PWD:$PWD -w $PWD pyhf/cuda:0.6.1-jax-cuda-11.1 'bash tests/test_jax.sh'
```
