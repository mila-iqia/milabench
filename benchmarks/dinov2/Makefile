# Use global base if possible
ifndef MILABENCH_BASE
	MILABENCH_BASE="base"
endif

export MILABENCH_BASE

BENCH_NAME=dinov2-giant
MILABENCH_CONFIG=dev.yaml
MILABENCH_ARGS=--config $(MILABENCH_CONFIG) --base $(MILABENCH_BASE)

all:
	install prepare single gpus nodes

install:
	milabench install $(MILABENCH_ARGS) --force

prepare:
	milabench prepare $(MILABENCH_ARGS)

tests: install prepare
	milabench run $(MILABENCH_ARGS)

single:
	CUDA_VISIBLE_DEVICES=0 milabench run $(MILABENCH_ARGS) --select $(BENCH_NAME)

gpus:
	milabench run $(MILABENCH_ARGS) --select $(BENCH_NAME)

nodes:
	milabench run $(MILABENCH_ARGS) --select $(BENCH_NAME)
