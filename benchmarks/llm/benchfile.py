from milabench.pack import Package


from milabench.commands import TorchrunAllGPU
from milabench.pack import BasePackage
from milabench.commands import SimpleCommand


class Torchtune(TorchrunAllGPU):
    @property
    def executable(self):
        return f"{self.binfolder}/bin/tune"

    #def should_wrap(self):
    #    return True

    def __init__(self, pack: BasePackage, *torchrun_args, **kwargs):
        super().__init__(pack, *torchrun_args, module=False, **kwargs)


class Llm(Package):
    # Requirements file installed by install(). It can be empty or absent.
    base_requirements = "requirements.in"

    # The preparation script called by prepare(). It must be executable,
    # but it can be any type of script. It can be empty or absent.
    prepare_script = "prepare.py"

    async def install(self):
        await super().install()  # super() call installs the requirements

    def build_run_plan(self):
        exec = SimpleCommand(self)
        return Torchtune(exec, "run").use_stdout()


__pack__ = Llm
