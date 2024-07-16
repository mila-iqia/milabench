from milabench.commands import TorchRunCommand
from milabench.pack import Package


class LightningBenchmark(Package):
    base_requirements = "requirements.in"
    prepare_script = "prepare.py"
    main_script = "main.py"

    def make_env(self):
        return {
            **super().make_env(),
            "OMP_NUM_THREADS": str(self.config.get("cpus_per_gpu", 8)),
        }

    def build_run_plan(self):
        # self.config is not the right config for this
        plan = super().build_run_plan()
        return TorchRunCommand(plan, use_stdout=True)


__pack__ = LightningBenchmark
