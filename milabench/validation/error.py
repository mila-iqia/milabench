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


class _Layer(ValidationLayer):
    """Capture all error event and save them to generate a summary"""

    def __init__(self, **kwargs) -> None:
        self.errors = defaultdict(PackError)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def on_event(self, entry: BenchLogEntry):
        error = self.errors[entry.tag]

        if entry.event == "line" and entry.pipe == "stderr":
            error.stderr.append(entry.data)

        elif entry.event == "error":
            info = entry.data
            error.message = f'{info["type"]}: {info["message"]}'

        elif entry.event == "end":
            info = entry.data
            if not self.early_stop:
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
                failures += 1
                tracebacks = _extract_traceback(error.stderr)

                if len(tracebacks) != 0:
                    if short:
                        summary.add("* " + tracebacks[-1])
                    else:
                        summary.add("* " + tracebacks[0])
                        for line in tracebacks[1:]:
                            summary.add("  " + line)
                else:
                    summary.add("No traceback info about the error")

        self.set_error_code(failures)
        return failures
