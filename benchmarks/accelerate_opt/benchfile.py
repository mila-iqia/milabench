from milabench.commands import (
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
        # FIXME: or AccelerateAllNodes
        plans = []

        max_num = self.config["num_machines"]
        nodes = select_nodes(self.config["system"]["nodes"], max_num)
        key = self.config["system"].get("sshkey")

        for rank, node in enumerate(nodes):
            host = node["ip"]
            user = node["user"]
            port = node.get("port", 22)
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
            worker = SSHCommand(
                host=host,
                user=user,
                key=key,
                port=port,
                executor=DockerRunCommand(
                    AccelerateLaunchCommand(pack, rank=rank),
                    self.config["system"].get("docker_image"),
                ),
                **options
            )
            plans.append(worker)

        return ListCommand(*plans)


__pack__ = AccelerateBenchmark
