default: image

all: image

image:
	docker pull nvidia/cuda:10.1-base-ubuntu18.04
	docker pull nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04
	docker build . \
	-f Dockerfile \
	-t pyhf/cuda:jax-debug-local
