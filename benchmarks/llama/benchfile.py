import uuid

from milabench.executors import TorchRunExecutor
from milabench.pack import Package


class LLAMA(Package):
    base_requirements = "requirements.in"
    main_script = "main.py"

    def make_env(self):
        return {
            **super().make_env(),
            "OMP_NUM_THREADS": str(self.config.get("cpus_per_gpu", 8))
        }


__pack__ = LLAMA
