from milabench.pack import Package


class TransformerBenchmark(Package):
    base_requirements = "requirements.in"
    main_script = "bench"


__pack__ = TransformerBenchmark
