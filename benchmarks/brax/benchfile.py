from milabench.pack import Package


class BraxBenchmark(Package):
    base_requirements = "requirements.in"
    main_script = "main.py"

    def make_env(self):
        env = super().make_env()
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
        return env
    
__pack__ = BraxBenchmark
