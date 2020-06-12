#!/bin/bash

set -e

. ~/anaconda3/etc/profile.d/conda.sh
conda activate mlperf

mount -t tmpfs cgroup_root /sys/fs/cgroup

mkdir /sys/fs/cgroup/cpuset
mount -t cgroup cpuset -o cpuset /sys/fs/cgroup/cpuset

mkdir /sys/fs/cgroup/memory
mount -t cgroup memory -o memory /sys/fs/cgroup/memory

SCRIPT_PATH=$(dirname "$0")
$SCRIPT_PATH/cgroup_setup.sh
