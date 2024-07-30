from milabench.pack import Package


from milabench.commands import AccelerateAllNodes
from milabench.pack import BasePackage
from milabench.commands import CmdCommand


class Torchtune(CmdCommand):
    def __init__(self, pack: BasePackage, lazy=True, **kwargs):
        super().__init__(pack, **kwargs)
        self.lazy = lazy

    def command_arguments(self, **kwargs):
        if self.lazy:
            return self.pack.argv
        return super()._argv(**kwargs)

    def _argv(self, **kwargs):
        # tune run [TORCHRUN-OPTIONS] <recipe> --config <config> [RECIPE-OPTIONS]
        return [
            "-m", "torchtune._cli.tune", "run", "--"
        ] + self.command_arguments()



class Llm(Package):
    # Requirements file installed by install(). It can be empty or absent.
    base_requirements = "requirements.in"

    # The preparation script called by prepare(). It must be executable,
    # but it can be any type of script. It can be empty or absent.
    prepare_script = "prepare.py"

    async def install(self):
        await super().install()  # super() call installs the requirements

    # def build_prepare_plan(self) -> "cmd.Command":
    #     from milabench.commands import PackCommand, VoidCommand

    #     if self.prepare_script is not None:
    #         prep = self.dirs.code / self.prepare_script
    #         if prep.exists():
    #             print(self.argv)
    #             return PackCommand(
    #                 self, prep, *self.argv, env=self.make_env(), cwd=prep.parent
    #             )
    #     return VoidCommand(self)

    def build_run_plan(self):
        from milabench.commands import VoirCommand
        # python -m torchtune._cli.tune run ...
        pack = Torchtune(self, lazy=True)
        # voir ... -m torchtune._cli.tune run ...
        return VoirCommand(pack, cwd=self.dirs.code)


__pack__ = Llm
