from milabench.pack import Package


class TransformerBenchmark(Package):
    base_requirements = "requirements.in"
    main_script = "main.py"


__pack__ = TransformerBenchmark
