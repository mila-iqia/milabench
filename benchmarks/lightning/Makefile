
tests:
	milabench install --config dev.yaml --base base
	milabench prepare --config dev.yaml --base base
	milabench run --config dev.yaml --base base


gpus:
	milabench run --config dev.yaml --base base --select lightning-gpus

nodes:
	milabench run --config dev.yaml --base base --select lightning-nodes