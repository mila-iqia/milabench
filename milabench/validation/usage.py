from collections import defaultdict

from .validation import ValidationLayer


class _Layer(ValidationLayer):
    """Checks that GPU utilisation is > 0.01 and that memory used is above 50%.

    Notes
    -----
    This is a sanity check that only runs when enabled

    """

    def __init__(self) -> None:
        self.warnings = defaultdict(lambda: defaultdict(float))

        self.devices = set()
        self.count = 0
        self.mem_threshold = 0.50
        self.load_threshold = 0.01

    def on_event(self, pack, run, tag, keys, data):
        gpudata = data.get("gpudata")

        if gpudata is not None:

            for device, data in gpudata.items():
                self.devices.add(device)
                self.warnings[tag][f"{device}-load_avg"] += data.get("load", 0)

                usage, total = data.get("memory", [0, 1])
                self.warnings[tag][f"{device}-mem_avg"] += usage / total

            self.warnings[tag]["count"] += 1

    def report(self, summary, **kwargs):
        failed = False

        for bench, warnings in self.warnings.items():

            with summary.section(bench):
                count = warnings["count"]

                for device in self.devices:
                    load = warnings[f"{device}-load_avg"] / count
                    mem = warnings[f"{device}-mem_avg"] / count

                    if load < self.load_threshold:
                        summary.add(
                            f"* Device {device} loads is below threshold {load} < {self.load_threshold}"
                        )
                        failed = True

                    if mem < self.mem_threshold:
                        summary.add(
                            f"* Device {device} used memory is below threshold {mem} < {self.mem_threshold}"
                        )
                        failed = True

        return failed
