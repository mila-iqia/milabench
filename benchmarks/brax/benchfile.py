from milabench.pack import Package


class BraxBenchmark(Package):
    base_requirements = "requirements.in"
    main_script = "main.py"


__pack__ = BraxBenchmark
