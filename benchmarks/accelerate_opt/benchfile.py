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

    def make_env(self):
        env = super().make_env()
        value = self.resolve_argument("--cpus_per_gpu", 8)
        env["OMP_NUM_THREADS"] = str(value)
        return env

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
        plan = super().build_run_plan().use_stdout()
        return AccelerateAllNodes(plan)


__pack__ = AccelerateBenchmark
