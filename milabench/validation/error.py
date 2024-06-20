from collections import defaultdict
from dataclasses import dataclass, field
from typing import List

from .validation import ValidationLayer


@dataclass
class PackError:
    """Error messages received from a running pack"""

    stderr: List[str] = field(default_factory=list)
    code: int = 0
    message: str = None
    early_stop: bool = False
    trace: str = None


@dataclass
class ParsedTraceback:
    lines: list[str]

    def find_raise(self):
        import re

        pattern = re.compile(r"raise (.*)\(")
        for i, line in enumerate(self.lines):
            if m := pattern.search(line):
                return i, m.group(1)

        return None, None

    def raised_exception(self):
        raised_idx, _ = self.find_raise()

        if raised_idx is not None:
            return self.lines[min(raised_idx + 1, len(self.lines))]

        return self.lines[-1]


def _extract_traceback(lines) -> list[ParsedTraceback]:
    output = []
    traceback = None

    for line in lines:
        line = line.rstrip()

        if "During handling of the above exception" in line:
            # The exceptions that happened afterwards are not relevant
            break

        if "Traceback" in line:
            if traceback is not None:
                output.append(traceback)

            traceback = ParsedTraceback([])

        if traceback and line != "":
            traceback.lines.append(line)

    if traceback is not None:
        output.append(traceback)

    return output


class Layer(ValidationLayer):
    """Capture all error event and save them to generate a summary"""

    def __init__(self, **kwargs) -> None:
        self.errors = defaultdict(PackError)

    def on_stop(self, entry):
        error = self.errors[entry.tag]
        error.early_stop = True

    def on_line(self, entry):
        error = self.errors[entry.tag]
        if entry.pipe == "stderr":
            error.stderr.append(entry.data)

    def on_error(self, entry):
        error = self.errors[entry.tag]

        if error.code == 0:
            error.code = 1

        info = entry.data
        error.message = f'{info["type"]}: {info["message"]}'
        error.trace = info.get("trace")

    def on_end(self, entry):
        error = self.errors[entry.tag]
        info = entry.data
        error.code = info["return_code"]

    def report_exceptions(self, summary, error, short):
        if error.trace:
            exceptions = [ParsedTraceback(error.trace.splitlines())]
        else:
            exceptions = _extract_traceback(error.stderr)

        if len(exceptions) == 0:
            summary.add("* No traceback info about the error")
            return

        summary.add(f"* {len(exceptions)} exceptions found")

        grouped = defaultdict(list)
        for exception in exceptions:
            grouped[exception.raised_exception()].append(exception)

        for k, exceptions in grouped.items():
            summary.add(f"  * {len(exceptions)} x {k}")

            if not short:
                selected = exceptions[0]
                for line in selected.lines:
                    summary.add(f"      | {line}")

    def report(self, summary, short=False, **kwargs):
        """Print an error report and exit with an error code if any error were found"""

        failures = 0
        success = 0

        for name, error in self.errors.items():
            if error.code == 0:
                success += 1
                continue

            if error.code == 0:
                success += 1
                continue

            with summary.section(name):
                if not error.early_stop:
                    failures += 1
                else:
                    summary.add("* early stopped")
                    continue

                summary.add(f"* Error code = {error.code}")

                self.report_exceptions(summary, error, short)

        self.set_error_code(failures)
        return failures
