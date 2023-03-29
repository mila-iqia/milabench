from milabench.pack import Package


class StarganBenchmark(Package):
    base_requirements = "requirements.in"
    main_script = "stargan/main.py"


__pack__ = StarganBenchmark
