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
from milabench.utils import select_nodes


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
        
        max_num = self.config["num_machines"]
        nodes = select_nodes(self.config["system"]["nodes"], max_num)

        for rank, node in enumerate(nodes):
            host = node["ip"]
            user = node["user"]
            options = dict()

            if rank == 0:
                options = dict(
                    setsid=True,
                    use_stdout=True,
                )
                
            tags = [*self.config["tag"], node["name"]]
            if rank != 0:
                # Workers do not send training data
                # tag it as such so we validation can ignore this pack
                tags.append("nolog")

            pack = self.copy({"tag": tags})
            worker = SSHExecutor(
                host=host,
                user=user,
                executor=DockerRunExecutor(
                    AccelerateLaunchExecutor(pack, rank=rank),
                    self.config["system"].get("docker_image"),
                ),
                **options
            )
            plans.append(worker)

        return ListExecutor(*plans)


__pack__ = AccelerateBenchmark
