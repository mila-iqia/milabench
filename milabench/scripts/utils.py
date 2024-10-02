import getpass
import json
import pathlib
import subprocess
import sys


def get_venv(venv:pathlib.Path) -> dict:
    activate = venv / "bin/activate"
    if not activate.exists():
        raise FileNotFoundError(str(activate))
    env = subprocess.run(
        f". '{activate}' && python3 -c 'import os ; import json ; print(json.dumps(dict(os.environ)))'",
        shell=True,
        capture_output=True
    ).stdout
    return json.loads(env)


def get_module_venv(module_main:str, check_if_module:str):
    module = pathlib.Path(module_main).resolve().parent
    cache_dir = pathlib.Path(f"/tmp/{getpass.getuser()}/milabench/{module.name}_venv")
    python3 = str(cache_dir / "bin/python3")
    try:
        subprocess.run([python3, "-c", check_if_module], check=True,
                       stdout=sys.stderr)
    except (FileNotFoundError, subprocess.CalledProcessError):
        cache_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "virtualenv", str(cache_dir)],
                       check=True, stdout=sys.stderr)
        subprocess.run([python3, "-m", "pip", "install", "-U", "pip"],
                       check=True, stdout=sys.stderr)
        subprocess.run([
            python3,
            "-m",
            "pip",
            "install",
            "-r",
            str(module / "requirements.txt")
        ], stdout=sys.stderr, check=True)
        subprocess.run([python3, "-c", check_if_module], check=True, stdout=sys.stderr)
    return python3, get_venv(cache_dir)


def run_in_module_venv(module_main:str, check_if_module:str, argv:list=None):
    python3, env = get_module_venv(module_main, check_if_module)
    return subprocess.call([python3, module_main, *argv], env=env)