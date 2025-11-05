
CONDA_ENV=py312
MILABENCH=conda run -n $(CONDA_ENV) milabench
ACTIVATE=. $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(CONDA_ENV)

# Use global base if possible
ifndef MILABENCH_BASE
	MILABENCH_BASE="/tmp/milabench"
endif

export MILABENCH_BASE

BENCH_NAME=fno_benchmark
MILABENCH_CONFIG=dev.yaml
MILABENCH_ARGS=--use-current-env --config $(MILABENCH_CONFIG) --base $(MILABENCH_BASE)

.PHONY: init

init:
	mkdir -p $(MILABENCH_BASE)

all:
	install prepare single gpus nodes

install:
	$(ACTIVATE) && milabench install $(MILABENCH_ARGS)

prepare:
	$(ACTIVATE) && milabench prepare $(MILABENCH_ARGS)

tests: install prepare
	$(ACTIVATE) && milabench run $(MILABENCH_ARGS)

single:
	$(ACTIVATE) && milabench run $(MILABENCH_ARGS) --select pic1d

gpus:
	$(ACTIVATE) && milabench run $(MILABENCH_ARGS) --select $(BENCH_NAME)-gpus

nodes:
	$(ACTIVATE) && milabench run $(MILABENCH_ARGS) --select $(BENCH_NAME)-nodes
