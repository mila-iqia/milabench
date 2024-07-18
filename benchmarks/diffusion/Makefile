
install:
	milabench install --config dev.yaml --base base --force

tests:
	milabench install --config dev.yaml --base base
	milabench prepare --config dev.yaml --base base
	milabench run --config dev.yaml --base base

single:
	milabench run --config dev.yaml --base base --select diffusion-single

gpus:
	milabench run --config dev.yaml --base base --select diffusion-gpus

nodes:
	milabench run --config dev.yaml --base base --select diffusion-nodes