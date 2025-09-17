#!/bin/bash

export MILABENCH_GPU_ARCH=cuda
export MILABENCH_ARGS="--select diffusion-nodes"
export MILABENCH_IMAGE=ghcr.io/mila-iqia/milabench:${MILABENCH_GPU_ARCH}-nightly

# ---
export MILABENCH_WORDIR="/tmp/" 
export MILABENCH_BASE="$MILABENCH_WORDIR"
export MILABENCH_SYSTEM="$MILABENCH_BASE/runs/system.yaml"


# >>>>>>>>>>>>>>>>>>
# Instruction Starts
# >>>>>>>>>>>>>>>>>>

# 1. Request Access to HuggingFace Gated Repos
#   - 
#   - 
#   - 
#
# 2. Create a Hugging Face Access Token
#   - MILABENCH_HUGGING_FACE
#
# 3. Now you are ready to run milabench

export MILABENCH_HF_TOKEN="-"


podman pull $MILABENCH_IMAGE

mkdir -p $MILABENCH_BASE/runs
mkdir -p $MILABENCH_BASE/data
mkdir -p $MILABENCH_BASE/cache


#
#   Configure the system to run on your nodes
#
cat > $MILABENCH_SYSTEM << EOF
system:
  # sshkey used in remote milabench operations
  # sshkey: ~/.ssh/id_ed25519

  # podman/docker config 
  # This is used to spawn the worker node
  docker:
    executable: podman
    image: $MILABENCH_IMAGE
    base: $MILABENCH_BASE
    args: [
       --rm, 
       --ipc=host,
       --device, nvidia.com/gpu=all, 
       --security-opt=label=disable,
       --network=host,
       -e, MILABENCH_HF_TOKEN=$MILABENCH_HF_TOKEN,
       -v, "$MILABENCH_BASE/data:/milabench/envs/data",
       -v, "$MILABENCH_BASE/cache:/milabench/envs/cache",
       -v, "$MILABENCH_BASE/runs:/milabench/envs/runs",
    ]

  # Nodes list
  nodes:
    - name: main
      ip: cn-d003
      main: true
      # user: username      # If different from current user

    - name: worker
      ip: cn-d004
      main: false
      # user: username      # If different from current user
EOF


#
#   Download the data on both nodes
#
podman run --rm --ipc=host                                  \
      --device nvidia.com/gpu=all                           \
      --security-opt=label=disable                          \
      --network=host                                        \
      -v $MILABENCH_BASE/runs:/milabench/envs/runs          \
      -v $MILABENCH_BASE/data:/milabench/envs/data          \
      -v $MILABENCH_BASE/cache:/milabench/envs/cache        \
      $MILABENCH_IMAGE                                      \
      /milabench/.env/bin/milabench prepare --system /milabench/envs/runs/system.yaml $MILABENCH_ARGS || :


#
#   Run milabench
#
podman run --rm --ipc=host                                  \
      --device nvidia.com/gpu=all                           \
      --security-opt=label=disable                          \
      --network=host                                        \
      -v $MILABENCH_BASE/runs:/milabench/envs/runs          \
      -v $MILABENCH_BASE/data:/milabench/envs/data          \
      -v $MILABENCH_BASE/cache:/milabench/envs/cache        \
      $MILABENCH_IMAGE                                      \
      /milabench/.env/bin/milabench run --system /milabench/envs/runs/system.yaml $MILABENCH_ARGS || :
    

# Provide this zipped folder
# zip -r runs.zip $MILABENCH_BASE/runs/

# <<<<<<<<<<<<<<<<<<<
# Instruction Ends
# <<<<<<<<<<<<<<<<<<<

rsync -az $MILABENCH_WORDIR/results/runs $OUTPUT_DIRECTORY


# ===
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
# ===


