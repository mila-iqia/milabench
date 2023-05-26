from collections import defaultdict
from dataclasses import dataclass, field

from .validation import ValidationLayer


def defaultfloatdict():
    return field(default_factory=lambda: defaultdict(float))


@dataclass
class UsageCheck:
    avg_load: dict = defaultfloatdict()
    avg_mem: dict = defaultfloatdict()
    count: dict = defaultfloatdict()
    max_load: dict = defaultfloatdict()
    max_mem: dict = defaultfloatdict()
    config: dict = None

    def get_config(self):
        if self.config is None:
            return {}

        return self.config.get("validations", {}).get("usage", {})

    @property
    def gpu_mem_threshold(self):
        return self.get_config().get("gpu_mem_threshold", 0.5)

    @property
    def gpu_load_threshold(self):
        return self.get_config().get("gpu_load_threshold", 0.5)


class Layer(ValidationLayer):
    """Checks that GPU utilisation and memory is above a given threshold.

    Notes
    -----
    This is a sanity check that only runs when enabled

    """

    def __init__(self, **kwargs) -> None:
        self.warnings = defaultdict(UsageCheck)
        self.devices = set()
        self.count = 0

    def on_event(self, entry):
        if entry.pipe != "data":
            return

        if entry.data is None:
            return

        data = entry.data
        tag = entry.tag
        stats = self.warnings[tag]
        gpudata = data.get("gpudata")

        if gpudata is not None:
            for device, data in gpudata.items():
                self.devices.add(device)
                usage, total = data.get("memory", [0, 1])
                load = data.get("load", 0)

                usage = usage / total

                stats.avg_load[device] += load
                stats.avg_mem[device] += usage
                stats.count[device] += 1
                stats.max_load[device] = max(load, stats.max_load[device])
                stats.max_mem[device] = max(usage, stats.max_mem)

    def report(self, summary, **kwargs):
        failed = 0
        warn = 0

        for bench, warnings in self.warnings.items():
            with summary.section(bench):
                for device in self.devices:
                    load = warnings.avg_load.get(device, None)
                    mem = warnings.avg_mem.get(device, None)
                    mxmem = warnings.max_mem.get(device, None)
                    mxload = warnings.max_load.get(device, None)
                    count = warnings.count.get(device, 0)

                    if load is not None and load / count < warnings.gpu_load_threshold:
                        summary.add(
                            f"* Device {device} loads is below threshold "
                            f"{load / count:5.2f} < {warnings.gpu_load_threshold:5.2f} (max load: {mxload})"
                        )
                        failed += 1

                    if mem is not None and mem / count < warnings.gpu_mem_threshold:
                        summary.add(
                            f"* Device {device} used memory is below threshold "
                            f"{mem / count:5.2f} < {warnings.gpu_mem_threshold:5.2f} (max use: {mxmem})"
                        )
                        warn += 1

        self.set_error_code(failed)
        return failed
