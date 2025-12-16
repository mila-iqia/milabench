#!/bin/bash

mkdir -p /tmp/workspace
cd /tmp/workspace

export MILABENCH_BRANCH=realtime_tracking
export PYTHON_VERSION=3.12
export MILABENCH_GPU_ARCH=cuda
export PYTHONUNBUFFERED=0
export MILABENCH_ARGS=""
export MILABENCH_CONFIG_NAME=inference
export MILABENCH_REPO=https://github.com/milabench/milabench.git
export MILABENCH_SHARED="$HOME/scratch/shared"

export MILABENCH_WORDIR="$(pwd)"  
export MILABENCH_ENV="$MILABENCH_WORDIR/.env/$PYTHON_VERSION/"
export MILABENCH_BASE="$MILABENCH_WORDIR/results"
export MILABENCH_SIZER_SAVE="$MILABENCH_WORDIR/results/runs/scaling.yaml"
export BENCHMARK_VENV="$MILABENCH_WORDIR/results/venv/torch"
export MILABENCH_SOURCE="$HOME/milabench"
export MILABENCH_CONFIG="$MILABENCH_WORDIR/milabench/config/$MILABENCH_CONFIG_NAME.yaml"
export MILABENCH_HF_TOKEN=$HF_TOKEN

mkdir -p $MILABENCH_WORDIR
cd $MILABENCH_WORDIR

ln -s $HOME/milabench
ln -s $HOME/stat
mkdir -p /tmp/workspace/flashinfer
(cd /home/mila/d/delaunap/.cache/ && rm -rf flashinfer && ln -s /tmp/workspace/flashinfer /home/mila/d/delaunap/.cache/flashinfer)

# git clone $MILABENCH_REPO -b $MILABENCH_BRANCH
conda create --prefix $MILABENCH_ENV python=$PYTHON_VERSION -y
conda activate $MILABENCH_ENV

pip install -e $MILABENCH_SOURCE[$MILABENCH_GPU_ARCH]
pip install psycopg2-binary


unset HF_DATASETS_CACHE
unset HF_HOME
module load cuda/12.6.0

pip install torch
milabench pin --variant cuda
milabench slurm_system > $MILABENCH_WORDIR/system.yaml
milabench install --system $MILABENCH_WORDIR/system.yaml --force
milabench prepare --system $MILABENCH_WORDIR/system.yaml
milabench run --system $MILABENCH_WORDIR/system.yaml


milabench run --config "$MILABENCH_WORDIR/milabench/config/vllm.yaml"



milabench sharedsetup --network $MILABENCH_SHARED --local $MILABENCH_BASE

(
    cd $MILABENCH_BASE
    cd cache/huggingface
    rm -rf hub
    ln -s ../../data/hub hub
)

milabench slurm_system > $MILABENCH_WORDIR/system.yaml
rm -rf $MILABENCH_WORDIR/results/venv

module load cuda/12.6.0

pip install torch
milabench pin --variant cuda
milabench install --system $MILABENCH_WORDIR/system.yaml $MILABENCH_ARGS

ln -s $HOME/milabench
ln -s $HOME/stat
mkdir -p /tmp/workspace/flashinfer
(cd /home/mila/d/delaunap/.cache/ && rm -rf flashinfer && ln -s /tmp/workspace/flashinfer /home/mila/d/delaunap/.cache/flashinfer)

unset HF_DATASETS_CACHE
unset HF_HOME

milabench prepare --system $MILABENCH_WORDIR/system.yaml
milabench run --system $MILABENCH_WORDIR/system.yaml


export BENCH="txt-to-image-gpus"
export BENCH="whisper-transcribe-single"
export BENCH="vllm-dense-physics-gpus"

export BENCH="vllm-moe-code-gpus"

export BENCH="whisper-transcribe-single"
milabench prepare --system $MILABENCH_WORDIR/system.yaml --select $BENCH > $HOME/stat/${BENCH}_prepare.txt
find $MILABENCH_BASE/data -type f | sort > $HOME/stat/${BENCH}_prepare_files.txt

milabench run --select $BENCH  > $HOME/stat/${BENCH}_run.txt
find $MILABENCH_BASE/data -type f | sort > $HOME/stat/${BENCH}_run_files.txt


export BENCH="whisper-transcribe-single"
milabench prepare --system $MILABENCH_WORDIR/system.yaml --select $BENCH
milabench run --system $MILABENCH_WORDIR/system.yaml --select $BENCH

export BENCH="vllm-dense-physics-gpus"
milabench prepare --system $MILABENCH_WORDIR/system.yaml --select $BENCH
milabench run --system $MILABENCH_WORDIR/system.yaml --select $BENCH

export BENCH="vllm-moe-code-gpus"
milabench prepare --system $MILABENCH_WORDIR/system.yaml --select $BENCH
milabench run --system $MILABENCH_WORDIR/system.yaml --select $BENCH

milabench prepare --system $MILABENCH_WORDIR/system.yaml
milabench run --system $MILABENCH_WORDIR/system.yaml



rsync -az $MILABENCH_WORDIR/results/runs $OUTPUT_DIRECTORY

# ===
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
# ===

kill $BEEGFS_PID
wait $BEEGFS_PID 2>/dev/null || :

kill $TUNNEL_PID
wait $TUNNEL_PID 2>/dev/null || :





# flashinfer
./results/venv/torch/bin/pip install flashinfer-python flashinfer-cubin==0.5.2
# JIT cache package (replace cu129 with your CUDA version: cu128, cu129, or cu130)
./results/venv/torch/bin/pip install flashinfer-jit-cache==0.5.2 --index-url https://flashinfer.ai/whl/cu129




# export MILABENCH_SIZER_CONFIG="$MILABENCH_WORDIR/milabench/config/scaling/inference.yaml"
# export MILABENCH_SIZER_SAVE="$MILABENCH_WORDIR/milabench/config/scaling/inference.yaml"
# export MILABENCH_SIZER_AUTO=1
# export MILABENCH_SIZER_BATCH_SIZE=2
# milabench run --system $MILABENCH_WORDIR/system.yaml
# export MILABENCH_SIZER_BATCH_SIZE=4
# milabench run --system $MILABENCH_WORDIR/system.yaml
# export MILABENCH_SIZER_BATCH_SIZE=8
# milabench run --system $MILABENCH_WORDIR/system.yaml
# export MILABENCH_SIZER_BATCH_SIZE=16
# milabench run --system $MILABENCH_WORDIR/system.yaml

# export MILABENCH_SIZER_BATCH_SIZE=160
# milabench run --system $MILABENCH_WORDIR/system.yaml
# export MILABENCH_SIZER_BATCH_SIZE=256
# milabench run --system $MILABENCH_WORDIR/system.yaml