from milabench import gpu
from milabench.pack import Package

import json, os

# This is called to ensure the arch variable is set
gpu.get_gpu_info()


class TheBenchmark(Package):
    if gpu.arch == "cuda":
        requirements_file = "requirements.nvidia.txt"
    elif gpu.arch == "rocm":
        requirements_file = "requirements.amd.txt"
    else:
        raise ValueError(f"Unsupported arch: {gpu.arch}")

    def make_env(self):
        env = super().make_env()
        env["MILABENCH_CONFIG"] = json.dumps(self.config)
        env["HF_HOME"] = self.config['dirs']['data']
        env["OMP_NUM_THREADS"] = str(self.config["cpus_per_gpu"])
        return env

    def prepare(self):
        env = self.make_env()
        env["MILABENCH_PREPARE_ONLY"] = "1"
        self.execute(
            "accelerate",
            "launch",
            f"--config_file={self.dirs.code / self.config['accelerate_config']}",
            os.path.join(self.dirs.code / "main.py"),
            env=env,
        )

    def run(self, args, voirargs, env):
        return self.execute(
            "accelerate",
            "launch",
            #"--machine_rank=0",
            f"--config_file={self.dirs.code / self.config['accelerate_config']}",
            f"--num_cpu_threads_per_process={self.config['cpus_per_gpu']}",
            f"--main_process_ip={self.config['manager_addr']}",
            f"--main_process_port={self.config['manager_port']}",
            #f"--num_processes={self.config['num_processes']}",
            os.path.join(self.dirs.code / "main.py"),
            env=self.make_env(),
        )


__pack__ = TheBenchmark
