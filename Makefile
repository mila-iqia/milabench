


MILABENCH_BASE=/data/delaunap/milabench/runs

vlt-all:
	rm -rf runs/
	milabench install  config/standard.yaml --select vit_l_32
	milabench prepare  config/standard.yaml --select vit_l_32
	milabench run config/standard.yaml --select vit_l_32



vlt:
	milabench run config/standard.yaml --select vit_l_32 --sync --base $(MILABENCH_BASE)


run:
	milabench run config/standard.yaml --sync --base $(MILABENCH_BASE)


ppo-all:
	milabench install  config/standard.yaml --select ppo --base $(MILABENCH_BASE)
	milabench prepare  config/standard.yaml --select ppo --base $(MILABENCH_BASE)
	milabench run config/standard.yaml --select ppo --sync --base $(MILABENCH_BASE)

td3:
	milabench run config/standard.yaml --select td3 --sync --base $(MILABENCH_BASE)


all:
	milabench install config/standard.yaml --base $(MILABENCH_BASE)
	milabench prepare config/standard.yaml --base $(MILABENCH_BASE)
	milabench run config/standard.yaml --base $(MILABENCH_BASE)
	milabench summary runs/data/
	milabench summary runs/data/ -o summary.json
