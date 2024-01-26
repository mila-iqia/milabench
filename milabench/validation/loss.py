import math
from collections import defaultdict
from dataclasses import dataclass

from .validation import ValidationLayer


@dataclass
class LossError:
    """Loss tracking state"""

    nan_count: int = 0
    loss_count: int = 0
    increased_count: int = 0
    prev_loss: float = None
    first_loss: float = None

    @property
    def overall_loss_change(self):
        return self.first_loss - self.prev_loss


class Layer(ValidationLayer):
    """Makes sures the loss we receive is not Nan.

    Notes
    -----
    Show a warning if the loss is not decreasing.

    """

    def __init__(self, **kwargs) -> None:
        self.previous_loss = dict()
        self.warnings = defaultdict(LossError)
        self.nan_count = 0
        self.increasing_loss = 0

    def on_data(self, entry):
        if entry.data is None:
            return

        tag = entry.tag
        loss = entry.data.get("loss")
        warning = self.warnings[tag]

        if loss is not None:
            prev = self.previous_loss.get(tag)
            self.previous_loss[tag] = loss

            if prev is not None:
                if warning.first_loss is None:
                    warning.first_loss = prev

                latest = int(math.isnan(loss))

                warning.nan_count += latest
                warning.loss_count += 1
                self.nan_count += latest

                if loss >= prev:
                    warning.increased_count += 1

    def report(self, summary, **kwargs):
        for bench, warnings in self.warnings.items():
            with summary.section(bench):
                nan_counts = warnings.nan_count
                loss_inc = warnings.increased_count
                loss_count = warnings.loss_count

                if loss_count == 0:
                    summary.add("* No loss was found")

                if nan_counts > 0:
                    summary.add(f"* Loss was Nan {nan_counts} times")

                if loss_inc > 0:
                    summary.add(f"* Loss increased {loss_inc} times")

        self.set_error_code(self.nan_count)
        return self.nan_count
