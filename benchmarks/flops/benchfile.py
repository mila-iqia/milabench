from milabench.pack import Package


class FlopsBenchmarch(Package):
    base_requirements = "requirements.in"
    prepare_script = "prepare.py"
    main_script = "main.py"

    def build_run_plan(self) -> "Command":
        import milabench.commands as cmd
        main = self.dirs.code / self.main_script
        pack = cmd.PackCommand(self, *self.argv, lazy=True)
            
        use_stdout = True
        
        if use_stdout:
            return pack.use_stdout()
        else:
            pack = cmd.VoirCommand(pack, cwd=main.parent)
            return pack

__pack__ = FlopsBenchmarch
