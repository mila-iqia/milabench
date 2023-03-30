from milabench.pack import Package
import asyncssh


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
        async with asyncssh.connect(host) as conn:
            return await conn.run(command,
                                  # disable host key validation
                                  known_hosts=None,
                                  # use our key to authenticate
                                  client_keys=[asyncssh.read_private_key(self.dirs.code / "id_milabench")],
                                  username="milabench",
            )

    async def run(self):
        self.phase = "run"
        nproc = len(self.config.get("devices", []))
        futs = []
        futs.append(self.execute(
            "accelerate",
            "launch",
            "--mixed_precision=fp16",
            "--dynamo_backend=no",
            f"--num_machines={self.config['num_machines']}",
            "--use_deepspeed",
            f"--gradient_accumulation_steps={self.config['gradient_accumulation_steps']}",
            "--zero_stage=2",
            f"--num_cpu_threads_per_process={self.config['cpus_per_gpu']}",
            f"--main_process_ip={self.config['manager_addr']}",
            f"--main_process_port={self.config['manager_port']}",
            f"--num_processes={nproc}",
            str(self.dirs.code / "main.py"),
            setsid=True,
            use_stdout=True,
        ))
        # XXX: this doesn't participate in the process timeout
        for worker in self.config['worker_addrs']:
            futs.append(self.run_remote_command(
                worker,
                f"docker run --rm --network host --gpus all "
                f"{self.config['docker_image']} "
                # XXX: figure out the right path
                f"milabench run /standard.yaml "
                f"--select {self.config['name']} "
                f"--override manager_addr='{repr(self.config['manager_addr'])}' "
                f"--override manager_port='{repr(self.config['manager_port'])}'"
                )
            )
        _, pending = await asyncio.wait(futs, return_when=asyncio.FIRST_EXCEPTION)
        # XXX: what to do with the pendings?

__pack__ = AccelerateBenchmark
