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
        return [
            "-m", "torchtune._cli.tune", "run", 
        ] + self.command_arguments()



class Llm(Package):
    # Requirements file installed by install(). It can be empty or absent.
    base_requirements = "requirements.in"

    # The preparation script called by prepare(). It must be executable,
    # but it can be any type of script. It can be empty or absent.
    prepare_script = "prepare.py"

    async def install(self):
        await super().install()  # super() call installs the requirements

    async def prepare(self):
        await super().prepare()  # super() call executes prepare_script

    def build_run_plan(self):
        from milabench.commands import VoirCommand
        # python -m torchtune._cli.tune run ...
        pack = Torchtune(self, *self.argv, lazy=True)
        # voir ... -m torchtune._cli.tune run ...
        return VoirCommand(pack, cwd=self.dirs.code)


__pack__ = Llm
