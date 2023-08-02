#!/bin/bash


set -m

#
#
#

echo ">> Configure the benchmark"
echo "=========================="


#
# Tweak the values to fit your system
#

USERNAME=${USER:-"mila"}
SSH_KEY_FILE=$HOME/.ssh/id_rsa
ARCH="cuda"
WORKER_0="cn-d003"
WORKER_1="cn-d004"



# Derived
VERSION="v0.0.7"
IMAGE="ghcr.io/mila-iqia/milabench:$ARCH-$VERSION"


# Create the config file
cat >overrides.yaml <<EOL
opt-6_7b-multinode:
  docker_image: "$IMAGE"

  worker_user: "$USERNAME"
  manager_addr: "$WORKER_0"
  worker_addrs:
    - "$WORKER_1"

  num_machines: 2
  capabilities:
    nodes: 2

opt-1_3b-multinode:
  docker_image: "$IMAGE"

  worker_user: "$USERNAME"
  manager_addr: "$WORKER_0"
  worker_addrs:
    - "$WORKER_1"

  num_machines: 2
  capabilities:
    nodes: 2

EOL

echo "<< ======================="
echo ""
echo ">> Prepare docker images"
echo "========================"

ssh $USERNAME@$WORKER_0 "docker pull $IMAGE"&
ssh $USERNAME@$WORKER_1 "docker pull $IMAGE"&
fg
fg

echo "<< ====================="
echo ""

#
#
#

echo ">> Run milabench"
echo "================"

if [ "$ARCH" = "cuda" ]; then 
    docker run -it --rm --gpus all --network host --ipc=host --privileged           \
        -v $SSH_KEY_FILE:/milabench/id_milabench                                    \
        -v $(pwd)/results:/milabench/envs/runs                                      \
        $IMAGE                                                                      \
        milabench run --override "$(cat overrides.yaml)" 

elif [ "$ARCH" = "rocm" ]; then 
    docker run -it --rm --network host --ipc host --privileged                      \
        --security-opt seccomp=unconfined --group-add video                         \
        -v /opt/amdgpu/share/libdrm/amdgpu.ids:/opt/amdgpu/share/libdrm/amdgpu.ids  \
        -v /opt/rocm:/opt/rocm                                                      \
        -v $(pwd)/results:/milabench/envs/runs                                      \
        $IMAGE                                                                      \
        milabench run --override "$(cat overrides.yaml)"
fi

echo "<< ============="
echo ""

#
#
#

echo ">> Print report"
echo "==============="
docker run -it --rm                                                                 \
    -v $(pwd)/results:/milabench/envs/runs                                          \
    $IMAGE                                                                          \
    milabench report --runs /milabench/envs/runs

echo "<< ============"
