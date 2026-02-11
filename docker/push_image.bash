#!/bin/bash

# Show how to manually build & Push a docker image

ARCH=${ARCH:-rocm}
TAG=${TAG:-nightly} 
GHUSER=${USER:-} 
TOKEN =${TOKEN:-} 

build_docker () {
    # Build docker
    sudo docker build                            \
        -t milabench:${ARCH}-${TAG}              \
        --build-arg ARCH=${ARCH}                 \
        --build-arg CONFIG=standard-${ARCH}.yaml \
        .
}

push_docker () {
    # Push the image to github
    echo $TOKEN | docker login ghcr.io -u $GHUSER --password-stdin 
    docker image tag milabench:${ARCH}-${TAG} ghcr.io/$GHUSER/milabench:${ARCH}-${TAG}
    docker push ghcr.io/$GHUSER/milabench:${ARCH}-${TAG}
}



run_docker() {
    export MILABENCH_DEV=/home/ciara/benchdevenv
    export MILABENCH_BASE=$MILABENCH_DEV/output
    export MILABENCH_IMAGE=milabench:cuda-cuda-nightly
    export SSH_KEY_FILE=$HOME/.ssh/id_rsa

    sudo docker run --rm --ipc=host --network=host                   \
            --device nvidia.com/gpu=all                           \
            --security-opt=label=disable                          \
            -e HF_TOKEN=$MILABENCH_HF_TOKEN                       \
            -e MILABENCH_HF_TOKEN=$MILABENCH_HF_TOKEN             \
            -v $SSH_KEY_FILE:/root/.ssh/id_rsa:Z                  \
            -v $MILABENCH_BASE/runs:/milabench/envs/runs          \
            -v $MILABENCH_BASE/data:/milabench/envs/data          \
            -v $MILABENCH_BASE/cache:/milabench/envs/cache        \
            $MILABENCH_IMAGE                                      \
            /milabench/.env/bin/milabench run $MILABENCH_ARGS || :
}