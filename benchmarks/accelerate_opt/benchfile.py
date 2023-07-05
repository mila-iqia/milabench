from milabench.executors import (
    AccelerateLaunchExecutor,
    CmdExecutor,
    DockerRunExecutor,
    ListExecutor,
    SCPExecutor,
    SSHExecutor,
    SequenceExecutor,
    VoidExecutor,
)
from milabench.pack import Package


class AccelerateBenchmark(Package):
    base_requirements = "requirements.in"

    def make_env(self):
        env = super().make_env()
        env["OMP_NUM_THREADS"] = str(self.config["argv"]["--cpus_per_gpu"])
        return env

    def build_docker_prepare_remote_plan(self):
        executors = []
        docker_pull_exec = CmdExecutor(
            self, "docker", "pull", self.config["system"].get("docker_image", None)
        )
        for node in self.config["system"]["nodes"]:
            if node["main"]:
                continue
            executors.append(SSHExecutor(docker_pull_exec, node["ip"]))
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
                *self.argv,
                "--prepare_only",
                "--cache",
                str(self.dirs.cache)
            )
        ]
        
        docker_image = self.config["system"].get("docker_image", None)
        if docker_image:
            prepare.append(self.build_docker_prepare_remote_plan())

        return SequenceExecutor(*prepare)

    def build_run_plan(self):
        plans = []

        rank = 1
        for node in self.config["system"]["nodes"]:
            host = node["ip"]
            user = node["user"]
            options = dict()

            assigned_rank = rank
            if node["main"]:
                assigned_rank = 0
                options = dict(
                    setsid=True,
                    use_stdout=True,
                )
            else:
                rank += 1

            pack = self.copy({"tag": [*self.config["tag"], node["name"]]})
            worker = SSHExecutor(
                host=host,
                user=user,
                executor=DockerRunExecutor(
                    AccelerateLaunchExecutor(pack, rank=assigned_rank),
                    None,
                ),
                **options
            )
            plans.append(worker)

        return ListExecutor(*plans)


__pack__ = AccelerateBenchmark
