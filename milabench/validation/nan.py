from collections import defaultdict
import math

from .validation import ValidationLayer


class _Layer(ValidationLayer):
    """Makes sures the loss we receive is not Nan.

    Notes
    -----
    Show a warning if the loss is not decreasing.

    """

    def __init__(self) -> None:
        self.previous_loss = dict()
        self.warnings = defaultdict(lambda: defaultdict(int))
        self.nan_count = 0
        self.increasing_loss = 0

    def on_event(self, pack, run, tag, keys, data):
        loss = data.get("loss")

        if loss is not None:
            prev = self.previous_loss.get(tag)
            self.previous_loss[tag] = loss

            if prev is not None:
                self.warnings[tag]["nan_count"] += int(math.isnan(loss))
                if loss > prev:
                    self.warnings[tag]["increasing_loss"] += 1

    def report(self, summary, **kwargs):
        for bench, warnings in self.warnings.items():
            with summary.section(bench):
                summary.add(f'Loss was Nan {warnings["nan_count"]} times')
                summary.add(f'Loss increased {warnings["increasing_loss"]} times')

        return self.nan_count == 0
