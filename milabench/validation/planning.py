from collections import defaultdict
from dataclasses import dataclass, field

from .validation import ValidationLayer
import voir.instruments.gpu


@dataclass
class Planning:
    errors: int = 0
    method: str = None
    njobs: int = None
    loss: dict = field(default_factory=dict)


class _Layer(ValidationLayer):
    """Makes sure the events we are receiving are consistent with the planning method

    Notes
    -----
    Check that we are receiving loss from the right number of processes

    """

    def __init__(self, **kwargs) -> None:
        gpus = voir.instruments.gpu.get_gpu_info()["gpus"]
        self.gpus = len(gpus)
        self.configs = defaultdict(Planning)

    def on_event(self, entry):
        if entry.pipe != "data":
            return

        tag = entry.tag
        benchname = entry.tag.split(".")[0]

        cfg = entry.pack.config
        plan = cfg["plan"]
        method = plan["method"].replace("-", "_")
        njobs = plan.get("n", 0)

        p = self.configs[benchname]

        if p.njobs is None:
            p.njobs = njobs

        if p.method is None:
            p.method = method

        p.errors += int(p.method != method)
        p.errors += int(p.njobs != njobs)
        p.errors += int(method not in ("per_gpu", "njobs"))

        # Counts the number of loss we are receiving
        if entry.data:
            loss = entry.data.get("loss")
            if loss is not None:
                p.loss[tag] = True

    def report(self, summary, **kwargs):
        failed = 0

        if len(self.configs) == 0:
            summary.add("* no data received")
            failed += 1

        for bench, config in self.configs.items():
            with summary.section(bench):
                count = len(config.loss)

                if config.method is None or config.njobs is None or count == 0:
                    summary.add("* no data received")
                    failed += 1

                if config.method == "njobs" and count != config.njobs:
                    summary.add(
                        f"* Wrong number of configs; expected (njobs: {config.njobs}) but got (config: {count})"
                    )
                    failed += 1

                if config.method == "per_gpu" and count != self.gpus:
                    summary.add(
                        f"* Wrong number of configs; expected (ngpus: {self.gpus}) but got (config: {count})"
                    )
                    failed += 1

                if config.errors > 0:
                    summary.add("* Config is inconsistent")
                    failed += 1

        self.set_error_code(failed)
        return failed
