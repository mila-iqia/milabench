from milabench.pack import Package


class FlopsBenchmarch(Package):
    base_requirements = "requirements.in"
    prepare_script = "prepare.py"
    main_script = "main.py"

    def build_run_plan(self) -> "Command":
        import milabench.commands as cmd

        pack = cmd.PackCommand(self, *self.argv, lazy=True)
        # pack = cmd.VoirCommand(pack, cwd=main.parent)
        pack = cmd.ActivatorCommand(pack)
        return pack.use_stdout()


__pack__ = FlopsBenchmarch
