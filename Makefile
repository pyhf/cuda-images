default: image

all: image

image:
	docker pull nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04
	docker build . \
	-f Dockerfile \
	--build-arg BASE_IMAGE=nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04 \
	--build-arg PYHF_VERSION=0.6.3 \
	--build-arg PYHF_BACKEND=jax \
	-t pyhf/cuda:jax-debug-local \
	-t pyhf/cuda:0.6.3-jax-cuda-11.1-debug-local
	docker system prune -f

image-torch:
	docker pull nvidia/cuda:11.1-base-ubuntu20.04
	docker build . \
	-f Dockerfile \
	--build-arg BASE_IMAGE=nvidia/cuda:11.1-base-ubuntu20.04 \
	--build-arg PYHF_VERSION=0.6.3 \
	--build-arg PYHF_BACKEND=torch \
	-t pyhf/cuda:0.6.3-torch-cuda-11.1-debug-local
	docker system prune -f

image-cuda-101:
	docker pull nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
	docker build . \
	-f Dockerfile \
	--build-arg BASE_IMAGE=nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04 \
	--build-arg PYHF_VERSION=0.5.4 \
	--build-arg PYHF_BACKEND=jax \
	-t pyhf/cuda:0.5.4-jax-cuda-10.1-debug-local
	docker system prune -f
