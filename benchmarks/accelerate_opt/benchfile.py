from milabench import gpu
from milabench.pack import Package

import subprocess
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
            "--mixed_precision=fp16",
            "--num_machines=1",
            "--dynamo_backend=no",
            "--num_processes=8",
            os.path.join(self.dirs.code / "main.py"),
            env=env,
        )

    def run(self, args, voirargs, env):
        proc = subprocess.Popen(
            ["accelerate",
             "launch",
             "--multi_gpu",
             "--mixed_precision=fp16",
             "--dynamo_backend=no",
             f"--num_cpu_threads_per_process={self.config['cpus_per_gpu']}",
             f"--main_process_ip={self.config['manager_addr']}",
             f"--main_process_port={self.config['manager_port']}",
             f"--num_processes={self.config['num_processes']}",
             str(self.dirs.code / "main.py")],
            env={"PYTHONUNBUFFERED": "1", **self._nox_session.env, **self.make_env()},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=self.dirs.code,
            preexec_fn=os.setsid,
        )
        proc.did_setsid = True
        return proc


__pack__ = TheBenchmark
