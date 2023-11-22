from milabench.pack import Package


class MLPerfBenchmark(Package):
    base_requirements = "requirements.in"
    main_script = "main.py"


__pack__ = MLPerfBenchmark

