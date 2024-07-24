from milabench.pack import Package


class SuperSlomoPack(Package):
    base_requirements = "requirements.in"
    prepare_script = "prepare.py"
    main_script = "slomo/train.py"

    @property
    def working_directory(self):
        return self.dirs.code / "slomo"

__pack__ = SuperSlomoPack
