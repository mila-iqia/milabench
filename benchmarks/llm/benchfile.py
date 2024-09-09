import tempfile
from milabench.fs import XPath
from milabench.pack import Package


from milabench.commands import TorchrunAllGPU, TorchrunAllNodes, ForeachNode
from milabench.pack import BasePackage
from milabench.commands import SimpleCommand


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
        llama3_dir = XPath(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = XPath(tmp_dir)
            tmp_dir.clone_subtree(
                "https://github.com/meta-llama/llama3.git",
                "11817d47e1ba7a4959b025eb1ca308572e0e3963",
            )
            tmp_dir.merge_into(
                llama3_dir,
                manifest="\n".join(
                    [
                        "/llama/",
                        "/requirements.txt",
                    ]
                )
            )
        # Fix conflict with tiktoken. As we only need llama/model.py, we don't
        # need to care about a compatible tiktoken for the llama3 module
        requirements = (llama3_dir / "requirements.txt").read_text().splitlines()
        requirements = [l for l in requirements if not l.startswith("tiktoken==")]
        (llama3_dir / "requirements.txt").write_text("\n".join(requirements))

        await super().install()  # super() call installs the requirements

    def build_run_plan(self):
        exec = SimpleCommand(self)
        return TorchtuneAllNodes(exec).use_stdout()


__pack__ = Llm
