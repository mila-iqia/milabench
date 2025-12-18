import tempfile
from milabench.fs import XPath
from milabench.pack import Package


from milabench.commands import TorchrunAllGPU, TorchrunAllNodes, ForeachNode
from milabench.pack import BasePackage
from milabench.commands import SimpleCommand, WorkingDir


class Torchtune(TorchrunAllGPU):
    @property
    def executable(self):
        return f"{self.binfolder}/bin/python"

    def should_wrap(self):
       # Always wrap inside torchtune even if the bench is single device
       return True

    def __init__(self, pack: BasePackage, *torchrun_args, **kwargs):
        super().__init__(pack, "-m", "torchtune._cli.tune", "run", *torchrun_args, module=False, **kwargs)


class TorchtuneAllNodes(TorchrunAllNodes):
    def __init__(self, executor, *args, **kwargs) -> None:
        base_exec = TorchrunAllNodes.make_base_executor(
            Torchtune,
            executor,
            *args,
            **kwargs
        )
        ForeachNode.__init__(self, base_exec)

    def make_new_node_executor(self, rank, node, base):
        """Make a new environment and create a new executor for the node"""
        executor = WorkingDir(super().make_new_node_executor(rank, node, base))
        return executor


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


def main():
    config = {
        "name": "llm",
        "definition": ".",
        "plan": {
            "method": "njobs",
            "n": 1
        },
        "num_machines": 2,
        "dirs": {
            "extra": "extra",
            "cache": "cache",
            "venv": "env",
            "base": "base"
        },
        "tag": [],
        "system": {
            "self": {
                    "name": "n1",
                    "ip": "n1",
                    "main": True,
                    "user": "user",
                    "hostname": "n1"
            },
            "nodes": [
                {
                    "name": "n3",
                    "ip": "n1",
                    "main": True,
                    "user": "user",
                    "hostname": "n1"
                },
                {
                    "name": "n2",
                    "ip": "n2",
                    "user": "user",
                    "hostname": "n2"
                }
            ],
            "docker": {
                "executable": "podman",
                "image": "$MILABENCH_IMAGE",
                "base": "$MILABENCH_BASE",
                "args": [
                    "--rm", "--ipc=host", "--network=host"
                ]
            }
        }
    }

    bench = Llm(config)

    plan = bench.build_run_plan()

    print(plan)

    for pack, argv, _ in plan.commands():
        print(" ".join(argv))


if __name__ == "__main__":
    main()
