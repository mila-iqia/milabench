from milabench.pack import Package


class RWKVBenchmark(Package):
    base_requirements = "requirements.in"
    prepare_script = "prepare.py"
    main_script = "rwkv-v4neo/train.py"


__pack__ = RWKVBenchmark
