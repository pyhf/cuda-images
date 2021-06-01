default: image

all: image

image:
	docker build . \
	-f Dockerfile \
	-t pyhf/cuda:jax-debug-local
