




vlt-all:
	rm -rf runs/
	milabench install  config/standard.yaml --select vit_l_32
	milabench prepare  config/standard.yaml --select vit_l_32
	milabench run config/standard.yaml --select vit_l_32



vlt:
	export MILABENCH_BASE=/data/delaunap/milabench/runs
	milabench run config/standard.yaml --select vit_l_32 --sync


run:
	milabench run config/standard.yaml --sync

