from milabench.pack import Package


class AccelerateBenchmark(Package):
    base_requirements = "requirements.in"

    def make_env(self):
        env = super().make_env()
        env["OMP_NUM_THREADS"] = str(self.config["cpus_per_gpu"])
        return env

    async def prepare(self):
        self.phase = "prepare"
        nproc = len(self.config.get("devices", []))
        await self.execute(
            "accelerate",
            "launch",
            "--mixed_precision=fp16",
            "--num_machines=1",
            "--dynamo_backend=no",
            f"--num_processes={nproc}",
            str(self.dirs.code / "main.py"),
            env={"MILABENCH_PREPARE_ONLY": "1"},
        )

    async def run(self):
        self.phase = "run"
        nproc = len(self.config.get("devices", []))
        await self.execute(
            "accelerate",
            "launch",
            "--mixed_precision=fp16",
            "--dynamo_backend=no",
            "--num_machines=1",
            "--use_deepspeed",
            "--deepspeed_multinode_launcher=standard",
            f"--gradient_accumulation_steps={self.config['gradient_accumulation_steps']}",
            "--zero_stage=2",
            f"--num_cpu_threads_per_process={self.config['cpus_per_gpu']}",
            f"--main_process_ip={self.config['manager_addr']}",
            f"--main_process_port={self.config['manager_port']}",
            # f"--num_processes={self.config['num_processes']}",
            f"--num_processes={nproc}",
            str(self.dirs.code / "main.py"),
            setsid=True,
            use_stdout=True,
        )


__pack__ = AccelerateBenchmark
