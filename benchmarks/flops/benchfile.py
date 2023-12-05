from milabench.pack import Package


class FlopsBenchmarch(Package):
    base_requirements = "requirements.in"
    prepare_script = "prepare.py"
    main_script = "main.py"

    def build_run_plan(self) -> "execs.Command":
        import milabench.executors as execs

        main = self.dirs.code / self.main_script
        pack = execs.PackCommand(self, *self.argv, lazy=True)
        # pack = execs.VoirCommand(pack, cwd=main.parent)
        pack = execs.ActivatorCommand(pack, use_stdout=True)
        return pack


__pack__ = FlopsBenchmarch
