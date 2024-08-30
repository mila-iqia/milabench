from milabench.pack import Package


from milabench.commands import TorchrunAllGPU, TorchrunAllNodes, ForeachNode
from milabench.pack import BasePackage
from milabench.commands import SimpleCommand


URL = "https://github.com/pytorch/torchtune.git"
BRANCH = "a83eeff0079a73ee04a11e8fc2573ed8f671b231"


class Torchtune(TorchrunAllGPU):
    @property
    def executable(self):
        return f"{self.binfolder}/bin/tune"

    #def should_wrap(self):
    #    return True

    def __init__(self, pack: BasePackage, *torchrun_args, **kwargs):
        super().__init__(pack, "run", *torchrun_args, module=False, **kwargs)


class TorchtuneAllNodes(TorchrunAllNodes):
    def __init__(self, executor, *args, **kwargs) -> None:
        base_exec = TorchrunAllNodes.make_base_executor(
            Torchtune, 
            executor,
            *args, 
            **kwargs
        )
        ForeachNode.__init__(self, base_exec)


class Llm(Package):
    # Requirements file installed by install(). It can be empty or absent.
    base_requirements = "requirements.in"

    # The preparation script called by prepare(). It must be executable,
    # but it can be any type of script. It can be empty or absent.
    prepare_script = "prepare.py"

    async def install(self):
        await super().install()  # super() call installs the requirements

        # Clone the right version of torchtune
        tune = self.dirs.code / "tune"
        if not tune.exists():
            tune.clone_subtree(URL, BRANCH)

        # make an editable install
        await self.pip_install("-e", str(tune))

    def build_run_plan(self):
        exec = SimpleCommand(self)
        return TorchtuneAllNodes(exec).use_stdout()


__pack__ = Llm
