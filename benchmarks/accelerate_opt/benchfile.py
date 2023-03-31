from milabench.pack import Package
import asyncssh
import asyncio


class AccelerateBenchmark(Package):
    base_requirements = "requirements.in"

    def make_env(self):
        env = super().make_env()
        env["OMP_NUM_THREADS"] = str(self.config["cpus_per_gpu"])
        return env

    async def prepare(self):
        self.phase = "prepare"
        await self.execute(
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

    async def run_remote_command(self, host, command):
        print(f"Connecting to worker: {host}")
        async with asyncssh.connect(host,
                                    # disable host key validation
                                    known_hosts=None,
                                    # use our key to authenticate
                                    client_keys=[asyncssh.read_private_key(self.dirs.code / "id_milabench")],
                                    username="milabench",
        ) as conn:
            print(f"Running command: {command}")
            path = str(self.dirs.runs / self.config['run_name'] / host)
            return await conn.run(command, stderr=path + ".stderr",
                                  stdout=path + ".stdout")


    def accelerate_command(self, rank):
        nproc = len(self.config.get("devices", []))
        return [
            "accelerate",
            "launch",
            "--mixed_precision=fp16",
            "--dynamo_backend=no",
            f"--machine_rank={rank}",
            f"--num_machines={self.config['num_machines']}",
            "--use_deepspeed",
            "--deepspeed_multinode_launcher=standard",
            f"--gradient_accumulation_steps={self.config['gradient_accumulation_steps']}",
            "--zero_stage=2",
            f"--num_cpu_threads_per_process={self.config['cpus_per_gpu']}",
            f"--main_process_ip={self.config['manager_addr']}",
            f"--main_process_port={self.config['manager_port']}",
            f"--num_processes={nproc}",
            str(self.dirs.code / "main.py"),
        ]

    async def run(self):
        self.phase = "run"
        futs = []
        futs.append(self.execute(
            *self.accelerate_command(rank=0),
            setsid=True,
            use_stdout=True,
        ))
        # XXX: this doesn't participate in the process timeout
        for i, worker in enumerate(self.config['worker_addrs']):
            command = ["docker", "run", "-i", "--rm",
                       "--network", "host",
                       "--gpus", "all"]
            env = self.make_env()
            for var in ('MILABENCH_CONFIG', 'XDG_CACHE_HOME', 'OMP_NUM_THREADS'):
                command.append("--env")
                command.append(f"{var}='{env[var]}'")
            command.append(f"{self.config['docker_image']}")
            command.append(f"{self.dirs.code / 'activator'}")
            command.append(f"{self.dirs.venv}")
            command.extend(self.accelerate_command(rank=i + 1))
            futs.append(self.run_remote_command(worker, " ".join(command)))
        await asyncio.gather(*futs)

__pack__ = AccelerateBenchmark
