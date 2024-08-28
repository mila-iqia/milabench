# Use global base if possible
ifndef MILABENCH_BASE
	MILABENCH_BASE="base"
endif

export MILABENCH_BASE
export MILABENCH_GPU_ARCH=cuda

BENCH_NAME=geo_gnn
MILABENCH_CONFIG=dev.yaml
MILABENCH_ARGS=--config $(MILABENCH_CONFIG) --base $(MILABENCH_BASE)

all:
	install prepare single gpus nodes

install:
	milabench install $(MILABENCH_ARGS) --update

prepare:
	milabench prepare $(MILABENCH_ARGS)

tests: # install 
	milabench run $(MILABENCH_ARGS) --select pna

single:
	milabench run $(MILABENCH_ARGS) --select $(BENCH_NAME)-single

gpus:
	milabench run $(MILABENCH_ARGS) --select $(BENCH_NAME)-gpus

nodes:
	milabench run $(MILABENCH_ARGS) --select $(BENCH_NAME)-nodes

pin:
	milabench pin $(MILABENCH_ARGS)
