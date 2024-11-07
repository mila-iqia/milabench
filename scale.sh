#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=7-0:0:0
#SBATCH --mem=350000
#SBATCH --output=%x.%j.out

exit_script() {
	echo "Preemption signal $1, saving myself ${SLURM_JOB_ID}"
	trap - $1 # clear the trap
	# Optional: sends SIGTERM to child/sub processes
	kill -s $1 -- -$$ &
	sleep 5
	scontrol requeue ${SLURM_JOB_ID}
}

kill_script() {
	echo "Kill signal $1"
}

trap "exit_script SIGTERM" SIGTERM
trap "exit_script SIGUSR1" SIGUSR1

export MILABENCH_BASE="${SCRATCH}/data/milabench"
export MILABENCH_CONFIG="${PWD}/config/standard.yaml"
export MILABENCH_SIZER_SAVE="${PWD}/config/scaling.yaml"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="upb"
SELECT=
EXCLUDE="--exclude multinode"

MILABENCH_GPU="${1}" python3 scale.py hatch run milabench run $SELECT $EXCLUDE
