#!/bin/bash


build_docker {
    # Build docker
    sudo docker build                           \
        -t milabench:nightly-rocm               \
        --build-arg ARCH=rocm                   \
        --build-arg CONFIG=standard-rocm.yaml   \
        .
}

push_docker {
    # Push the image to github
    echo $TOKEN | docker login ghcr.io -u $USER --password-stdin 
    docker image tag milabench:nightly-rocm ghcr.io/$USER/milabench:nightly-rocm
    docker push ghcr.io/$USER/milabench:nightly-cuda
}
