import math
from collections import defaultdict
from dataclasses import dataclass

from .validation import ValidationLayer, group_by_benchname


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
        grouped = group_by_benchname(self.warnings)

        with summary.section("Loss Tracking"):
            for bench, warnings in grouped.items():
                with summary.section(bench):
                    no_loss = []
                    loss_was_nan = []
                    loss_increased = []

                    for tag, warning in warnings:
                        nan_counts = warning.nan_count
                        loss_inc = warning.increased_count
                        loss_count = warning.loss_count

                        if loss_count == 0:
                            no_loss.append(tag)
                        
                        if nan_counts > 0:
                            loss_was_nan.append(tag)

                        if loss_inc > 0:
                            loss_increased.append(tag)

                    if len(no_loss):
                        summary.add(f"* {len(no_loss)} x {bench} No loss was found ({', '.join(no_loss)})")
                
                    if len(loss_was_nan):
                        summary.add(f"* {len(loss_was_nan)} x {bench} Loss was NaN ({', '.join(loss_was_nan)})")
                        
                    if len(loss_increased):
                        summary.add(f"* {len(loss_increased)} x {bench} Loss increased ({', '.join(loss_increased)})")

        self.set_error_code(self.nan_count)
        return self.nan_count
