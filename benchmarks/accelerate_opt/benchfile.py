from milabench.commands import (
    AccelerateAllNodes,
    AccelerateLaunchCommand,
    CmdCommand,
    DockerRunCommand,
    ListCommand,
    SSHCommand,
)
from milabench.pack import Package
from milabench.utils import select_nodes

class AccelerateBenchmark(Package):
    base_requirements = "requirements.in"

    def build_prepare_plan(self):
        return CmdCommand(
            self,
            "accelerate",
            "launch",
            "--mixed_precision=bf16",
            "--num_machines=1",
            "--dynamo_backend=no",
            "--num_processes=1",
            "--num_cpu_threads_per_process=8",
            str(self.dirs.code / "main.py"),
            *self.argv,
            "--prepare_only",
            "--cache",
            str(self.dirs.cache)
        )

    def build_run_plan(self): 
        from milabench.commands import PackCommand, VoirCommand
        main = self.dirs.code / self.main_script
        plan = PackCommand(self, *self.argv, lazy=True)

        if False:
            plan = VoirCommand(pack, cwd=main.parent)

        return AccelerateAllNodes(plan).use_stdout()


__pack__ = AccelerateBenchmark
