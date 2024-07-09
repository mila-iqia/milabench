from collections import defaultdict
from dataclasses import dataclass, field
from typing import List

from .validation import ValidationLayer


@dataclass
class PackError:
    """Error messages received from a running pack"""

    stderr: List[str] = field(default_factory=list)
    code: List[int] = field(default_factory=list)
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


def _extract_traceback(lines, is_install) -> list[ParsedTraceback]:
    output = []
    traceback = None
    parsing_pip_error = False

    def push_trace():
        nonlocal parsing_pip_error, traceback, output
        if traceback is not None:
            if parsing_pip_error:
                traceback.lines.append("PipInstallError")
                parsing_pip_error = False
            output.append(traceback)
  
    # Only extract pip error during installs
    # to avoid false positive
    def is_pip_error(line):
        nonlocal is_install

        if not is_install:
            return False
        
        return "ERROR" in line and not parsing_pip_error

    for line in lines:
        line = line.rstrip()

        if "During handling of the above exception" in line:
            # The exceptions that happened afterwards are not relevant
            break

        # "ERROR" comes from pip install
        if "Traceback" in line or is_pip_error(line):
            # New traceback push old one
            push_trace()

            if is_pip_error(line):
                parsing_pip_error = True
        
            traceback = ParsedTraceback([])

        if traceback and line != "":
            traceback.lines.append(line)

    push_trace()

    return output


class Layer(ValidationLayer):
    """Capture all error event and save them to generate a summary"""

    def __init__(self, **kwargs) -> None:
        self.errors = defaultdict(PackError)
        self.is_prepare = False
        self.is_install = False

    def on_stop(self, entry):
        error = self.errors[entry.tag]
        error.early_stop = True

    def on_config(self, entry):
        # config is not passed for install & prepare run
        self.is_prepare = False
        self.is_install = False

    def on_start(self, entry):
        cmd = entry.data.get("command", ["nothing"])[0]
        self.is_install = cmd == "pip"

    def on_line(self, entry):
        error = self.errors[entry.tag]
        if entry.pipe == "stderr":
            error.stderr.append(entry.data)

    def on_error(self, entry):
        error = self.errors[entry.tag]
        error.code.append(1)
        info = entry.data
        error.message = f'{info["type"]}: {info["message"]}'
        error.trace = info.get("trace")

    def on_end(self, entry):
        error = self.errors[entry.tag]
        info = entry.data
        error.code.append(info["return_code"])

    def report_exceptions(self, summary, error, short):
        if error.trace:
            exceptions = [ParsedTraceback(error.trace.splitlines())]
        else:
            exceptions = _extract_traceback(error.stderr, self.is_install)

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

        items = sorted(self.errors.items(), key=lambda item: str(item[0]))

        for name, error in items:
            total = sum(error.code)

            if total == 0:
                success += 1
                continue

            with summary.section(name):
                if not error.early_stop:
                    failures += 1
                else:
                    summary.add("* early stopped")
                    continue

                summary.add(f"* Error codes = {', '.join(str(c) for c in error.code)}")

                self.report_exceptions(summary, error, short)

        self.set_error_code(failures)
        return failures
