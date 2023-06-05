"""Use to retrieve GIT version info, this file cannot import milabench modules
as it is executed as part of the installation process"""
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


def retrieve_git_versions(tag="<tag>", commit="<commit>", date="<date>"):
    return {
        "tag": _exec("git describe --tags", tag),
        "commit": _exec("git rev-parse HEAD", commit),
        "date": _exec("git show -s --format=%ci", date),
    }


def update_version_file():
    version_info = retrieve_git_versions()

    with open(os.path.join(ROOT, "milabench", "_version.py"), "w") as file:
        for key, data in version_info.items():
            file.write(f'__{key}__ = "{data}"\n')


if __name__ == "__main__":
    update_version_file()
