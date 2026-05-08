from milabench.pack import Package


class Grpo(Package):
    base_requirements = "requirements.in"
    prepare_script = "prepare.py"
    main_script = "main.py"

    def make_env(self):
        return super().make_env()

    async def install(self):
        await super().install()

    async def prepare(self):
        await super().prepare()

    def build_run_plan(self):
        from milabench.commands import PackCommand, AccelerateAllNodes

        main = self.dirs.code / self.main_script
        plan = PackCommand(self, *self.argv, lazy=True)

        if False:
            plan = VoirCommand(plan, cwd=main.parent)

        return AccelerateAllNodes(plan).use_stdout()


__pack__ = Grpo
