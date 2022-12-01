



tail-output:
	tail -f ${SCRATCH}/milabench.err ${SCRATCH}/milabench.out

schedule:
	rm -rf ${SCRATCH}/milabench.out
	rm -rf ${SCRATCH}/milabench.err

	touch ${SCRATCH}/milabench.out
	touch ${SCRATCH}/milabench.err

	sbatch slurm.bash

	tail -f ${SCRATCH}/milabench.err ${SCRATCH}/milabench.out
