"""Use to retrieve GIT version info, this file cannot import milabench modules
as it is executed as part of the installation process"""

import os
import subprocess
import warnings

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


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
        "tag": _exec("git describe --always --tags", tag),
        "commit": _exec("git rev-parse HEAD", commit),
        "date": _exec("git show -s --format=%ci", date),
    }


def version_file():
    return os.path.join(ROOT, "milabench", "_version.py")

def read_previous():
    info = ["<tag>", "<commit>", "<date>"]
    
    if not os.path.exists(version_file()):
        return info
    
    with open(version_file(), "r") as file:
        for line in file.readlines():
            if "tag" in line:
                _, v = line.split("=")
                info[0] = v.strip()

            if "commit" in line:
                _, v = line.split("=")
                info[1] = v.strip()

            if "date" in line:
                _, v = line.split("=")
                info[2] = v.strip()

    return info


def update_version_file():
    version_info = retrieve_git_versions(*read_previous())

    with open(version_file(), "w") as file:
        file.write('"""')
        file.write("This file is generated, do not modify")
        file.write('"""\n\n')

        for key, data in version_info.items():
            file.write(f'__{key}__ = "{data}"\n')


if __name__ == "__main__":
    update_version_file()
