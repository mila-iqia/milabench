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

    def __init__(self, gv) -> None:
        self.errors = defaultdict(PackError)
        self.failed = False
        gv.subscribe(self.on_event)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def on_event(self, data):
        data = dict(data)
        run = data.pop("#run", None)
        pack = data.pop("#pack", None)

        if pack is None:
            return

        tg = ".".join(run["tag"]) if run else pack.config["name"]

        ks = set(data.keys())
        error = self.errors[tg]

        if ks == {"#stderr"}:
            txt = str(data["#stderr"])
            error.stderr.append(txt)

        elif ks == {"#error"}:
            error.message = data

        elif ks == {"#end", "#return_code"}:
            error.code = data["#return_code"]
            self.failed = self.failed or error.code != 0

    def report(self, short=True):
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
                line = line.strip()

                if "During handling of the above exception" in line:
                    # The exceptions that happened afterwards are not relevant
                    break

                if "Traceback" in line:
                    traceback = True

                if traceback and line != "":
                    output.append(line + "\n")

            if error.code != 0:
                # Tracback
                failures += 1

                if short:
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
