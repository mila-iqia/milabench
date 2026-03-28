"""Print the names of packages installed in the system site-packages.

In a --system-site-packages venv, importlib.metadata.distributions() returns
both venv-local and system packages.  This script filters out any distribution
whose files live inside the venv, so only true system packages are printed.

Output is one package name per line, sorted and deduplicated.

Usage from pack.py::

    subprocess.run([venv_python, path/to/system_packages.py], ...)
"""

import importlib.metadata
import sys
from pathlib import Path


def list_system_packages():
    if sys.prefix == sys.base_prefix:
        return sorted({dist.metadata["Name"] for dist in importlib.metadata.distributions()})

    venv_prefix = str(Path(sys.prefix).resolve())

    packages = set()
    for dist in importlib.metadata.distributions():
        dist_location = str(Path(dist._path).parent.resolve())
        if not dist_location.startswith(venv_prefix):
            packages.add(dist.metadata["Name"])

    return sorted(packages)


if __name__ == "__main__":
    for name in list_system_packages():
        print(name)
