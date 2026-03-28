"""Uninstall locally-installed packages that shadow system-provided ones.

In a --system-site-packages venv (e.g. NGC containers), `milabench install`
may pull in packages like torch that are already provided by the base image.
The local copies shadow the system ones, defeating the purpose of using
the container's optimized builds.

This command detects those duplicates and removes the local copies so
the system versions become active again.
"""

import subprocess
import sys
from pathlib import Path

from coleo import Option, tooled


PROTECTED = frozenset({
    "pip", "setuptools", "wheel", "pkg_resources",
    "milabench", "benchmate", "virtualenv",
})


def _normalize(name):
    return name.lower().replace("-", "_").replace(".", "_")


def find_system_duplicates(protected=PROTECTED):
    """Return a dict of locally-installed packages that also exist in system site-packages.

    Keys are pip-compatible package names, values are dicts with local/system versions.
    """
    if sys.prefix == sys.base_prefix:
        return {}

    venv_prefix = str(Path(sys.prefix).resolve())

    from importlib.metadata import distributions

    local = {}
    system = {}

    for dist in distributions():
        raw_name = dist.metadata["Name"]
        norm = _normalize(raw_name)
        version = dist.metadata["Version"]
        dist_location = str(Path(dist._path).parent.resolve()) 

        if dist_location.startswith(venv_prefix):
            local[norm] = {"name": raw_name, "version": version}
        else:
            system.setdefault(norm, {"name": raw_name, "version": version})

    protected_norm = {_normalize(p) for p in protected}
    duplicates = {}
    for norm in sorted(set(local) & set(system) - protected_norm):
        duplicates[local[norm]["name"]] = {
            "local": local[norm]["version"],
            "system": system[norm]["version"],
        }

    return duplicates


@tooled
def cli_prefer_system():
    """Uninstall local packages that shadow system-provided ones."""

    # Show what would be done without actually uninstalling
    dry_run: Option & bool = False

    if sys.prefix == sys.base_prefix:
        print("Not running inside a virtual environment, nothing to do.")
        return 0

    duplicates = find_system_duplicates()

    if not duplicates:
        print("No local packages shadowing system packages.")
        return 0

    print(f"Found {len(duplicates)} local package(s) shadowing system packages:")
    for name, versions in duplicates.items():
        print(f"  {name}: {versions['local']} (local) -> {versions['system']} (system)")

    if dry_run:
        print("\nDry run, no changes made.")
        return 0

    to_uninstall = list(duplicates.keys())
    result = subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y"] + to_uninstall,
    )

    if result.returncode == 0:
        print(f"\nUninstalled {len(to_uninstall)} local package(s). System versions now active.")
    else:
        print(f"\npip uninstall exited with code {result.returncode}", file=sys.stderr)

    return result.returncode


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without uninstalling")
    args = parser.parse_args()

    if sys.prefix == sys.base_prefix:
        print("Not running inside a virtual environment, nothing to do.")
        sys.exit(0)

    duplicates = find_system_duplicates()

    if not duplicates:
        print("No local packages shadowing system packages.")
        sys.exit(0)

    print(f"Found {len(duplicates)} local package(s) shadowing system packages:")
    for name, versions in duplicates.items():
        print(f"  {name}: {versions['local']} (local) -> {versions['system']} (system)")

    if args.dry_run:
        print("\nDry run, no changes made.")
        sys.exit(0)

    to_uninstall = list(duplicates.keys())
    result = subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y"] + to_uninstall,
    )

    if result.returncode == 0:
        print(f"\nUninstalled {len(to_uninstall)} local package(s). System versions now active.")
    else:
        print(f"\npip uninstall exited with code {result.returncode}", file=sys.stderr)

    sys.exit(result.returncode)
