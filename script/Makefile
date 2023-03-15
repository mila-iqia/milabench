
schedule:
	rm -rf ${SCRATCH}/milabench.out
	touch ${SCRATCH}/milabench.out

	sbatch -o ${SCRATCH}/milabench.out -e ${SCRATCH}/milabench.out slurm.bash

	tail -f ${SCRATCH}/milabench.out
