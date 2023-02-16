#!/bin/bash

# Show how to manually build & Push a docker image

ARCH=${ARCH:-rocm}
TAG=${TAG:-nightly} 
USER=${USER:-} 
TOKEN =${TOKEN:-} 

build_docker {
    # Build docker
    sudo docker build                            \
        -t milabench:${ARCH}-${TAG}              \
        --build-arg ARCH=${ARCH}                 \
        --build-arg CONFIG=standard-${ARCH}.yaml \
        .
}

push_docker {
    # Push the image to github
    echo $TOKEN | docker login ghcr.io -u $USER --password-stdin 
    docker image tag milabench:${ARCH}-${TAG} ghcr.io/$USER/milabench:${ARCH}-${TAG}
    docker push ghcr.io/$USER/milabench:${ARCH}-${TAG}
}
