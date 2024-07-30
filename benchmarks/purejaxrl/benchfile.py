from milabench.pack import Package


class PureJaxRLBenchmark(Package):
    base_requirements = "requirements.in"
    main_script = "main.py"


__pack__ = PureJaxRLBenchmark()
