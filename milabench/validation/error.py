from collections import defaultdict
from dataclasses import dataclass, field
from typing import List

from .validation import ValidationLayer, BenchLogEntry


@dataclass
class PackError:
    """Error messages received from a running pack"""

    stderr: List[str] = field(default_factory=list)
    code: int = 0
    message: str = None
    early_stop: bool = False
    trace: str = None


def _extract_traceback(lines):
    output = []
    traceback = False

    for line in lines:
        line = line.rstrip()

        if "During handling of the above exception" in line:
            # The exceptions that happened afterwards are not relevant
            break

        if "Traceback" in line:
            traceback = True

        if traceback and line != "":
            output.append(line)

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

    def report(self, summary, short=True, **kwargs):
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

                if error.trace:
                    tracebacks = error.trace.splitlines()
                else:
                    tracebacks = _extract_traceback(error.stderr)
                summary.add(f"* Error code = {error.code}")

                if len(tracebacks) != 0:
                    if short:
                        summary.add("* " + tracebacks[-1])
                    else:
                        summary.add("* " + tracebacks[0])
                        for line in tracebacks[1:]:
                            summary.add("  " + line)
                else:
                    summary.add(f"* No traceback info about the error")

        self.set_error_code(failures)
        return failures
