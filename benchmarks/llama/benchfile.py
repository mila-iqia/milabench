from milabench.commands import CmdCommand
from milabench.pack import Package


class LLAMA(Package):
    base_requirements = "requirements.in"
    main_script = "main.py"

    async def install(self):
        await super().install()

    def build_prepare_plan(self):
        return CmdCommand(
            self,
            "python",
            str(self.dirs.code / "main.py"),
            *self.argv,
            "--prepare",
            "--cache",
            str(self.dirs.cache),
        )

    def build_run_plan(self):
        return CmdCommand(
            self,
            "python",
            str(self.dirs.code / "main.py"),
            *self.argv,
            "--cache",
            str(self.dirs.cache),
            use_stdout=True,
        )


__pack__ = LLAMA
