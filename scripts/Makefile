





schedule:
	rm -rf /network/scratch/d/delaunap/milabench.out
	rm -rf /network/scratch/d/delaunap/milabench.err

	touch /network/scratch/d/delaunap/milabench.out
	touch /network/scratch/d/delaunap/milabench.err

	sbatch slurm.bash

	tail -f /network/scratch/d/delaunap/milabench.err
