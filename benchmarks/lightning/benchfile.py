from milabench.commands import TorchrunAllNodes, TorchrunAllGPU, ListCommand
from milabench.pack import Package


class LightningBenchmark(Package):
    base_requirements = "requirements.in"
    prepare_script = "prepare.py"
    main_script = "main.py"

    def build_run_plan(self):
        # self.config is not the right config for this
        plan = super().build_run_plan()

        # if self.config.get("num_machines", 1) <= 1:
        #    return ListCommand(TorchrunAllGPU(plan, use_stdout=True))
    
        return TorchrunAllNodes(plan).use_stdout()


__pack__ = LightningBenchmark
