import os
import subprocess
import warnings


ROOT = os.path.join(os.path.dirname(__file__), "..")


def _exec(cmd, default):
    try:
        return subprocess.check_output(
            cmd.split(" "), encoding="utf-8", cwd=ROOT
        ).strip()
    except subprocess.CalledProcessError:
        warnings.warn("out of tree; milabench could not retrieve version info")
        return default


version_info = {
    "tag": _exec("git describe --tags", "<tag>"),
    "commit": _exec("git rev-parse HEAD", "<commit>"),
    "date": _exec("git show -s --format=%ci", "<date>"),
}


with open(os.path.join(ROOT, "milabench", "_version.py"), "w") as file:
    for key, data in version_info.items():
        file.write(f'__{key}__ = "{data}"\n')
