from milabench.pack import Package


class LightningBenchmark(Package):
    base_requirements = "requirements.in"
    prepare_script = "prepare.py"
    main_script = "main.py"


__pack__ = LightningBenchmark
