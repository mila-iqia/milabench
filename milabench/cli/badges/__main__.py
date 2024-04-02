import subprocess
import sys


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    try:
        import pybadges as _
    except ImportError:
        from ..utils import run_in_module_venv
        check_if_module = "import pybadges"
        return run_in_module_venv(__file__, check_if_module, argv)

    return subprocess.run([
        sys.executable,
        "-m",
        "pybadges",
        *argv
    ], check=True).returncode


if __name__ == "__main__":
    sys.exit(main())
