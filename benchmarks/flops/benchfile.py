from milabench.pack import Package


class FlopsBenchmarch(Package):
    base_requirements = "requirements.in"
    prepare_script = "prepare.py"
    main_script = "main.py"
    
    def build_run_plan(self) -> "execs.Executor":
        import milabench.executors as execs
        
        main = self.dirs.code / self.main_script
        pack = execs.PackExecutor(self, *self.argv, lazy=True)
        # pack = execs.VoirExecutor(pack, cwd=main.parent)
        pack = execs.ActivatorExecutor(pack)
        return pack
    

__pack__ = FlopsBenchmarch
