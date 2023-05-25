from collections import defaultdict
import math

from .validation import ValidationLayer


class _Layer(ValidationLayer):
    """Makes sures the loss we receive is not Nan.

    Notes
    -----
    Show a warning if the loss is not decreasing.

    """

    def __init__(self, **kwargs) -> None:
        self.previous_loss = dict()
        self.warnings = defaultdict(lambda: defaultdict(int))
        self.nan_count = 0
        self.increasing_loss = 0

    def on_event(self, entry):
        if entry.pipe != "data":
            return

        tag = entry.tag
        loss = entry.data.get("loss")

        if loss is not None:
            prev = self.previous_loss.get(tag)
            self.previous_loss[tag] = loss

            if prev is not None:
                latest = int(math.isnan(loss))
                self.warnings[tag]["nan_count"] += latest
                self.nan_count += latest

                if loss > prev:
                    self.warnings[tag]["increasing_loss"] += 1

    def report(self, summary, **kwargs):
        for bench, warnings in self.warnings.items():
            with summary.section(bench):
                nan_counts = warnings["nan_count"]
                loss_inc = warnings["increasing_loss"]

                if nan_counts > 0:
                    summary.add(f"* Loss was Nan {nan_counts} times")

                if loss_inc > 0:
                    summary.add(f"* Loss increased {loss_inc} times")

        self.set_error_code(self.nan_count)
        return self.nan_count