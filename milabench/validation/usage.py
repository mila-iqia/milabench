from collections import defaultdict

from .validation import ValidationLayer


class _Layer(ValidationLayer):
    """Checks that GPU utilisation is > 0.01 and that memory used is above 50%.

    Notes
    -----
    This is a sanity check that only runs when enabled

    """

    def __init__(self, **kwargs) -> None:
        self.warnings = defaultdict(lambda: defaultdict(float))

        self.devices = set()
        self.count = 0
        self.mem_threshold = 0.50
        self.load_threshold = 0.01

    def on_event(self, entry):
        if entry.pipe != "data":
            return

        data = entry.data
        tag = entry.tag

        gpudata = data.get("gpudata")

        if gpudata is not None:
            for device, data in gpudata.items():
                loadkey = f"{device}-load_avg"
                memkey = f"{device}-mem_avg"
                countkey = f"{device}-count"

                stats = self.warnings[tag]
                self.devices.add(device)
                usage, total = data.get("memory", [0, 1])

                stats[loadkey] += data.get("load", 0)
                stats[memkey] += usage / total
                stats[countkey] += 1

    def report(self, summary, **kwargs):
        failed = 0

        for bench, warnings in self.warnings.items():
            with summary.section(bench):
                for device in self.devices:
                    loadkey = f"{device}-load_avg"
                    memkey = f"{device}-mem_avg"
                    countkey = f"{device}-count"
                    count = warnings[countkey]

                    load = warnings.get(loadkey, None)
                    mem = warnings.get(memkey, None)

                    if load and load / count < self.load_threshold:
                        summary.add(
                            f"* Device {device} loads is below threshold {load / count:5.2f} < {self.load_threshold:5.2f}"
                        )
                        failed += 1

                    if mem and mem / count < self.mem_threshold:
                        summary.add(
                            f"* Device {device} used memory is below threshold {mem / count/ count:5.2f} < {self.mem_threshold:5.2f}"
                        )
                        failed += 1

        self.set_error_code(failed)
        return failed
