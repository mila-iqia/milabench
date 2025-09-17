#!/bin/bash

set -ex

export MILABENCH_GPU_ARCH=cuda
export MILABENCH_ARGS="--select diffusion-nodes"
export MILABENCH_IMAGE=ghcr.io/mila-iqia/milabench:${MILABENCH_GPU_ARCH}-nightly

# ---
export MILABENCH_WORDIR=/tmp/
export MILABENCH_BASE="$MILABENCH_WORDIR"
export MILABENCH_SYSTEM="$MILABENCH_BASE/runs/system.yaml"

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
first_node=${nodes[0]}
second_node=${nodes[1]}

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
#       - export MILABENCH_HF_TOKEN={your_token}
#
#   3. Now you are ready to run milabench
#

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#           MODIFY ME
# ----------------------------------
export MILABENCH_HF_TOKEN="-"
export SSH_KEY_FILE=$HOME/.ssh/id_rsa

# Node we are running milabench from
export FIRST_NODE_IP=$first_node
export SECOND_NODE_IP=$second_node
export USERNAME=$USER
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

mkdir -p $MILABENCH_BASE/runs
mkdir -p $MILABENCH_BASE/data
mkdir -p $MILABENCH_BASE/cache

#
#   4. Configure the system to run on your nodes
#       Specify the the list of nodes
#
cat > $MILABENCH_SYSTEM << EOF
system:
  # sshkey used in remote milabench operations
  sshkey: /milabench/id_milabench

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

  # podman/docker config 
  # This is used to spawn the worker node
  docker:
    executable: podman
    image: $MILABENCH_IMAGE
    base: $MILABENCH_BASE
    args: [
       --rm, --ipc=host, --network=host,
       --device, nvidia.com/gpu=all, 
       --security-opt=label=disable,
       -e, MILABENCH_HF_TOKEN=$MILABENCH_HF_TOKEN,
       -v, "$SSH_KEY_FILE:/root/.ssh/id_rsa:Z",
       -v, "$MILABENCH_BASE/data:/milabench/envs/data",
       -v, "$MILABENCH_BASE/cache:/milabench/envs/cache",
       -v, "$MILABENCH_BASE/runs:/milabench/envs/runs",
    ]
EOF

#
#   Download the data on both nodes
#
for node in "$FIRST_NODE_IP" "$SECOND_NODE_IP"; do
    ssh -oCheckHostIP=no -oStrictHostKeyChecking=no "$USERNAME@$node" podman pull $MILABENCH_IMAGE &
done
wait

for node in "$FIRST_NODE_IP" "$SECOND_NODE_IP"; do
    ssh -oCheckHostIP=no -oStrictHostKeyChecking=no "$USERNAME@$node" podman run --rm --ipc=host --network=host   \
        --security-opt=label=disable                                  \
        -v "$SSH_KEY_FILE:/root/.ssh/id_rsa:Z"                        \
        -v "$MILABENCH_BASE/runs:/milabench/envs/runs"                \
        -v "$MILABENCH_BASE/data:/milabench/envs/data"                \
        -v "$MILABENCH_BASE/cache:/milabench/envs/cache"              \
        "$MILABENCH_IMAGE"                                            \
        /milabench/.env/bin/milabench prepare $MILABENCH_ARGS &
done
wait

# ===
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
# ===
