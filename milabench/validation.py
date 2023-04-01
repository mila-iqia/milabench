from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class PackError:
    """Error messages received from a running pack"""

    stderr: list[str] = field(default_factory=list)
    code: int = 0
    message: str = None


class ErrorValidation:
    """Capture all error event and save them to generate a summary"""

    def __init__(self, short=True) -> None:
        self.short = short
        self.errors = defaultdict(PackError)
        self.failed = False
        self.early_stop = False

    def __call__(self, data):
        pack = data.pack
        run = pack.config

        tg = ".".join(run["tag"]) if run else pack.config["name"]
        error = self.errors[tg]

        if data.event == "line" and data.pipe == "stderr":
            error.stderr.append(data.data)

        elif data.event == "stop":
            self.early_stop = True

        elif data.event == "error":
            info = data.data
            error.message = f'{info["type"]}: {info["message"]}'

        elif data.event == "end":
            info = data.data
            if not self.early_stop:
                error.code = info["return_code"]
                self.failed = self.failed or error.code != 0

    def end(self):
        """Print an error report and exit with an error code if any error were found"""

        report = [
            "",
            "Error Report",
            "------------",
            "",
        ]
        indent = "    "

        failures = 0
        success = 0

        for name, error in self.errors.items():
            traceback = False
            output = []

            for line in error.stderr:
                line = line.rstrip()

                if "During handling of the above exception" in line:
                    # The exceptions that happened afterwards are not relevant
                    break

                if "Traceback" in line:
                    traceback = True

                if traceback and line != "":
                    output.append(line + "\n")

            if error.code != 0:
                # Traceback
                failures += 1

                if self.short:
                    traceback = output[-1] if output else "No traceback found"
                else:
                    traceback = "".join(output).replace("\n", "\n    ")

                report.append(name)
                report.append("^" * len(name))
                report.append(indent + traceback)
            else:
                success += 1

        if failures > 0:
            report.extend(
                [
                    "Summary",
                    "-------",
                    f"{indent}Success: {success}",
                    f"{indent}Failures: {failures}",
                ]
            )

            print("\n".join(report))

        return -1 if self.failed else 0
