from collections import defaultdict

from .validation import ValidationLayer
from voir.instruments.gpu import get_gpu_info


class _Layer(ValidationLayer):
    """Makes sure the events we are receiving are consistent with the planning method

    Notes
    -----
    Check that we are receiving loss from the right number of processes

    """

    def __init__(self) -> None:
        gpus = get_gpu_info().values()

        self.gpus = len(gpus)
        self.configs = defaultdict(lambda: defaultdict(int))

    def on_event(self, pack, run, tag, keys, data):
        cfg = pack.config
        plan = cfg["plan"]
        method = plan["method"].replace("-", "_")

        self.configs[tag]["method"] = method
        self.configs[tag]["njobs"] = plan.get("n", 0)

        assert method in ("per_gpu", "njobs")

        # Counts the number of loss we are receiving
        loss = data.get("loss")
        if loss is not None:
            self.configs[tag]["loss"] += 1

    def report(self, summary, **kwargs):
        failed = False
        config_count = len(self.configs)

        for bench, config in self.configs.items():
            with summary.section(bench):
                method = config["method"]
                njobs = config["njobs"]

                if method == "njobs" and config_count != njobs:
                    summary.add(f"* Wrong number of configs")
                    failed = True

                if method == "per_gpu" and config_count != self.gpus:
                    summary.add(f"* Wrong number of configs")
                    failed = True

        return failed
