from collections import defaultdict
from dataclasses import dataclass

from .validation import ValidationLayer


@dataclass
class VersionCheck:
    has_meta: bool = False
    has_version: bool = False


class Layer(ValidationLayer):
    """Makes sure milabench version info is available"""

    def __init__(self, **kwargs) -> None:
        self.errors = defaultdict(VersionCheck)

    def on_data(self, entry):
        # Record that the bench exists but we have not received metadata
        if entry.tag not in self.errors:
            self.errors[entry.tag] = VersionCheck()

    def on_meta(self, entry):
        self.errors[entry.tag].has_meta = True

        has_version = entry.data.get("milabench", {}).get("tag", "<tag>") != "<tag>"
        self.errors[entry.tag].has_version = has_version

    def report(self, summary, short=True, **kwargs):
        fatal = 0

        for tag, error in self.errors.items():
            with summary.section(tag):
                if not error.has_meta:
                    summary.add("* no metadata was received")
                    fatal += 1

                if not error.has_version:
                    summary.add("* no milabench version was saved")

        self.set_error_code(fatal)
        return fatal
