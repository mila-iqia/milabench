from milabench.executors import AccelerateLaunchExecutor, AccelerateLoopExecutor, CmdExecutor, DockerRunExecutor, ListExecutor, SCPExecutor, SSHExecutor, SequenceExecutor, VoidExecutor
from milabench.pack import Package


class AccelerateBenchmark(Package):
    base_requirements = "requirements.in"

    def make_env(self):
        env = super().make_env()
        env["OMP_NUM_THREADS"] = str(self.config["cpus_per_gpu"])
        return env

    def build_docker_prepare_remote_plan(self):
        executors = []
        docker_pull_exec = CmdExecutor(
            self,
            "docker",
            "pull",
            self.config["system"].get("docker_image", None)
        )
        for node in self.config["system"]["nodes"][1:]:
            executors.append(
                SSHExecutor(
                    docker_pull_exec,
                    node["ip"]
                )
            )
        return ListExecutor(*executors)

    def build_prepare_plan(self):
        prepare = [
            CmdExecutor(
                self,
                "accelerate",
                "launch",
                "--mixed_precision=fp16",
                "--num_machines=1",
                "--dynamo_backend=no",
                "--num_processes=1",
                "--num_cpu_threads_per_process=8",
                str(self.dirs.code / "main.py"),
                env={"MILABENCH_PREPARE_ONLY": "1"},
            )
        ]
        docker_image = self.config["system"].get("docker_image", None)
        if docker_image:
            prepare.append(self.build_docker_prepare_remote_plan())
        else:
            remote_prepare = []
            for node in self.config["system"]["nodes"][1:]:
                for d in self.dirs:
                    remote_prepare.append(
                        SCPExecutor(self, node["ip"], str(d))
                    )
            if remote_prepare:
                prepare.append = ListExecutor(*remote_prepare)

        return SequenceExecutor(*prepare)

    def build_run_plan(self):
        # XXX: this doesn't participate in the process timeout
        return AccelerateLoopExecutor(
            AccelerateLaunchExecutor(self),
            SSHExecutor(
                DockerRunExecutor(
                    AccelerateLoopExecutor.PLACEHOLDER,
                    self.config["system"].get("docker_image", None)
                ),
                None
            )
        )

__pack__ = AccelerateBenchmark
