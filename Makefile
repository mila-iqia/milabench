



run:
	# rm -rf _dev_*
	milabench install benchmarks/stable_baselines3/dev.yaml --base .
	milabench run benchmarks/stable_baselines3/dev.yaml --base .
