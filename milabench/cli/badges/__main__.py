import pathlib
import subprocess
import sys


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    try:
        import pybadges as _
    except ImportError:
        module = pathlib.Path(__file__).resolve().parent
        cache_dir = pathlib.Path(f"/tmp/milabench/{module.name}_venv")
        python3 = str(cache_dir / "bin/python3")
        check_module = "import pybadges"
        try:
            subprocess.run([python3, "-c", check_module], check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            cache_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run([sys.executable, "-m", "virtualenv", str(cache_dir)], check=True)
            subprocess.run([python3, "-m", "pip", "install", "-U", "pip"], check=True)
            subprocess.run([
                python3,
                "-m",
                "pip",
                "install",
                "-r",
                str(module / "requirements.txt")
            ], check=True)
            subprocess.run([python3, "-c", check_module], check=True)
        return subprocess.call(
            [python3, __file__, *argv],
        )
    
    return subprocess.run([
        sys.executable,
        "-m",
        "pybadges",
        *argv
    ], check=True).returncode


if __name__ == "__main__":
    sys.exit(main())
