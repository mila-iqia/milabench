
schedule:
	rm -rf ${SCRATCH}/milabench.out
	touch ${SCRATCH}/milabench.out

	sbatch slurm.bash --output=${SCRATCH}/milabench.out  --error=${SCRATCH}/milabench.out

	tail -f ${SCRATCH}/milabench.out
