#!/bin/bash

. ~/anaconda3/etc/profile.d/conda.sh
conda activate mlperf

export PATH=$PATH:"/root/anaconda3/envs/mlperf/bin/"

SCRIPT_PATH=$(dirname "$0")
$SCRIPT_PATH/docker_cgroup_setup.sh

export MILARUN_DATAROOT=~/milabench/data/
export MILARUN_OUTROOT=~/milabench/

exec $@
