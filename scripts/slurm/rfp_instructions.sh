#!/bin/bash

set -ex

# ===
OUTPUT_DIRECTORY=$(scontrol show job "$SLURM_JOB_ID" --json | jq -r '.jobs[0].standard_output' | xargs dirname)
mkdir -p $OUTPUT_DIRECTORY/meta
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
touch $SLURM_SUBMIT_DIR/.no_report
# ===

# >>>>>>>>>>>>>>>>>>
# Instruction Starts
# >>>>>>>>>>>>>>>>>>

# Dependencies
#
#   Software
#    1. Podman
#    2. NVIDIA driver 
#
#   Data
#    1. Request Access to HuggingFace Gated Repos
#        - https://huggingface.co/meta-llama/Llama-2-7b/tree/main
#        - https://huggingface.co/meta-llama/Llama-3.1-8B
#        - https://huggingface.co/meta-llama/Llama-3.1-70B
#
#   2. Create a Hugging Face Access Token
#       - https://huggingface.co/settings/tokens/new?tokenType=read
#       - export MILABENCH_HF_TOKEN={$your_token}
#
#   3. Now you are ready to run milabench
#

#
#
#  milabench_run HF_TOKEN BASE SSH_KEY FIRST_NODE SECOND_NODE USERNAME MILABENCH_VERSION
# 
milabench_run () {

  # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  #           MODIFY ME
  # ----------------------------------
  export MILABENCH_MODE=$1
  export MILABENCH_HF_TOKEN=$2
  export MILABENCH_BASE=$3
  export SSH_KEY_FILE=$4

  # Node we are running milabench from
  export FIRST_NODE_IP=$5
  export SECOND_NODE_IP=dummy
  export USERNAME=$7
  export MILABENCH_VERSION=$8
  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


  export MILABENCH_SYSTEM="$MILABENCH_BASE/runs/system.yaml"
  export MILABENCH_GPU_ARCH=cuda
  export MILABENCH_IMAGE=ghcr.io/mila-iqia/milabench:$MILABENCH_VERSION
  export MILABENCH_ARGS=$9

#
#   4. Configure the system to run on your nodes
#       Specify the the list of nodes
#
mkdir -p $MILABENCH_BASE/runs
cat > $MILABENCH_SYSTEM << EOF
system:
  # podman/docker config 
  # This is used to spawn the worker node when needed
  docker:
    executable: podman
    image: $MILABENCH_IMAGE
    base: $MILABENCH_BASE
    args: [
       --rm, --ipc=host, --network=host,
       --device, nvidia.com/gpu=all, 
       --security-opt=label=disable,
       -e, HF_TOKEN=$MILABENCH_HF_TOKEN,
       -e, MILABENCH_HF_TOKEN=$MILABENCH_HF_TOKEN,
       -v, "$SSH_KEY_FILE:/root/.ssh/id_rsa:Z",
       -v, "$MILABENCH_BASE/data:/milabench/envs/data",
       -v, "$MILABENCH_BASE/cache:/milabench/envs/cache",
       -v, "$MILABENCH_BASE/runs:/milabench/envs/runs",
    ]
EOF

  if [[ "$MILABENCH_MODE" == "single" ]]; then
    NODE_LIST=("$FIRST_NODE_IP")
    cat >> $MILABENCH_SYSTEM << EOF
  # Nodes list
  nodes:
    - name: main
      ip: $FIRST_NODE_IP
      main: true
      user: $USERNAME
EOF
  else
    NODE_LIST=("$FIRST_NODE_IP" "$SECOND_NODE_IP")
    cat >> $MILABENCH_SYSTEM << EOF
  # Nodes list
  nodes:
    - name: main
      ip: $FIRST_NODE_IP
      main: true
      user: $USERNAME
    - name: worker
      ip: $SECOND_NODE_IP
      main: false
      user: $USERNAME
EOF
  fi

  #
  #   Download the data on both nodes
  #
  for node in "${NODE_LIST[@]}"; do
      ssh -oCheckHostIP=no -oStrictHostKeyChecking=no "$USERNAME@$node" podman pull $MILABENCH_IMAGE &

      ssh -oCheckHostIP=no -oStrictHostKeyChecking=no "$USERNAME@$node" mkdir -p $MILABENCH_BASE/runs
      ssh -oCheckHostIP=no -oStrictHostKeyChecking=no "$USERNAME@$node" mkdir -p $MILABENCH_BASE/data
      ssh -oCheckHostIP=no -oStrictHostKeyChecking=no "$USERNAME@$node" mkdir -p $MILABENCH_BASE/cache
  done
  wait

  for node in "${NODE_LIST[@]}"; do
      ssh -oCheckHostIP=no -oStrictHostKeyChecking=no "$USERNAME@$node" podman run --rm --ipc=host --network=host   \
          --device nvidia.com/gpu=all                                   \
          --security-opt=label=disable                                  \
          -e HF_TOKEN=$MILABENCH_HF_TOKEN                               \
          -e MILABENCH_HF_TOKEN=$MILABENCH_HF_TOKEN                     \
          -v "$SSH_KEY_FILE:/root/.ssh/id_rsa:Z"                        \
          -v "$MILABENCH_BASE/runs:/milabench/envs/runs"                \
          -v "$MILABENCH_BASE/data:/milabench/envs/data"                \
          -v "$MILABENCH_BASE/cache:/milabench/envs/cache"              \
          "$MILABENCH_IMAGE"                                            \
          /milabench/.env/bin/milabench prepare $MILABENCH_ARGS &
  done
  wait

  # Make sure the data is there on both nodes
  # rsync -a "$MILABENCH_BASE" "$USERNAME@$SECOND_NODE_IP:$MILABENCH_BASE"

  #
  #   Run milabench
  #
  podman run --rm --ipc=host --network=host                   \
        --device nvidia.com/gpu=all                           \
        --security-opt=label=disable                          \
        -e HF_TOKEN=$MILABENCH_HF_TOKEN                       \
        -e MILABENCH_HF_TOKEN=$MILABENCH_HF_TOKEN             \
        -v $SSH_KEY_FILE:/root/.ssh/id_rsa:Z                  \
        -v $MILABENCH_BASE/runs:/milabench/envs/runs          \
        -v $MILABENCH_BASE/data:/milabench/envs/data          \
        -v $MILABENCH_BASE/cache:/milabench/envs/cache        \
        $MILABENCH_IMAGE                                      \
        /milabench/.env/bin/milabench run --system /milabench/envs/runs/system.yaml $MILABENCH_ARGS || :


  # Provide this zipped folder
  PWD=$(pwd)
  (cd $MILABENCH_BASE && zip -r $PWD/runs.zip runs/)
}


export HF_TOKEN=t
export BASE=/tmp/
export SSH_KEY=$HOME/.ssh/id_ed25519
export FIRST_NODE=$HOSTNAME
export SECOND_NODE_IP=dummy
export USERNAME=$USER
export VERSION=cuda-stacc-v3


# milabench_run single "$@"

milabench_run single $HF_TOKEN $BASE $SSH_KEY $FIRST_NODE $SECOND_NODE_IP $USERNAME $VERSION

rsync -az $BASE/runs $OUTPUT_DIRECTORY
