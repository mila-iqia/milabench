from milabench.pack import Package


class StarganBenchmark(Package):
    base_requirements = "requirements.in"
    main_script = "stargan/main.py"

    @property
    def working_directory(self):
        return self.dirs.code / "stargan"


__pack__ = StarganBenchmark
