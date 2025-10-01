from dataclasses import dataclass
from pathlib import Path
import shutil

from coleo import Option, tooled


def find_site_packages(venv_path: Path) -> Path:
    """Guess site-packages path inside a venv."""

    if not venv_path.exists():
        raise FileNotFoundError(f"{venv_path} does not exist")

    # Handle typical venv layout
    for sub in ["lib/python*/site-packages", "Lib/site-packages"]:  # unix, windows
        for p in venv_path.glob(sub):
            return p

    raise RuntimeError(f"Could not locate site-packages under {venv_path}")


@dataclass
class Arguments:
    venv: str 

@tooled
def arguments() -> Arguments:
    venv: Option & str

    return Arguments(venv)


@tooled
def cli_global_patch(args: Arguments = None):
    if args is None:
        args = arguments()
    
    import benchmate.progress

    source_file = Path(benchmate.progress.__file__).resolve()

    site_packages = find_site_packages(Path(args.venv))
    target_file = site_packages / "sitecustomize.py"

    # Copy our patch file into place
    shutil.copy2(source_file, target_file)


if __name__ == "__main__":
    cli_global_patch()
