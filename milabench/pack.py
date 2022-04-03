import json
import os
import subprocess
from argparse import Namespace as NS
from sys import version_info as pyv

from nox.sessions import Session, SessionRunner

from .fs import XPath


class BasePackage:
    def __init__(self, config):
        self.config = config
        self.pack_path = XPath(config["definition"])

        self.dirs = NS(**{name: XPath(d) for name, d in config["dirs"].items()})

        reuse = True
        self._nox_runner = SessionRunner(
            name=self.dirs.venv.name,
            signatures=[],
            func=NS(
                python=f"{pyv.major}.{pyv.minor}.{pyv.micro}",
                venv_backend="virtualenv",
                venv_params=[],
                reuse_venv=reuse,
            ),
            global_config=NS(
                posargs=[],
                envdir=self.dirs.venv.parent,
                force_venv_backend=False,
                default_venv_backend=None,
                reuse_existing_virtualenvs=reuse,
                no_install=False,
                error_on_external_run=False,
                install_only=False,
            ),
            manifest=None,
        )
        self._nox_session = Session(self._nox_runner)
        self._nox_runner._create_venv()

    def make_env(self):
        return {}

    def checked_install(self, force=False):
        sentinel = self.dirs.code / "installed"
        if not force and sentinel.exists():
            name = self.config["name"]
            print(f"Benchmark {name} is already installed")
            return

        if self.dirs.code == self.pack_path:
            print(f"Cannot install if the code destination is the same as the source")
            return

        self.install_code()
        self.install_milabench()
        self.install()
        sentinel.touch()

    def install_code(self):
        if self.dirs.code.exists():
            print(f"Clearing existing data in {self.dirs.code}")
            self.dirs.code.rm()
        self.pack_path.merge_into(self.dirs.code, self.pack_path / "manifest")

    def install_milabench(self):
        devreqs = os.environ.get("MILABENCH_DEVREQS", None)
        if devreqs:
            self.pip_install("-r", devreqs)
        else:
            self.pip_install("milabench", "voir")

    def pip_install(self, *args, **kwargs):
        args = [str(x) for x in args]
        return self._nox_session.install(*args, **kwargs, silent=False)

    def conda_install(self, *args, **kwargs):
        args = [str(x) for x in args]
        return self._nox_session.conda_install(*args, **kwargs, silent=False)

    def execute(self, *args, cwd=None, **kwargs):
        args = [str(x) for x in args]
        curdir = os.getcwd()
        if cwd is None:
            cwd = self.dirs.code
        try:
            os.chdir(cwd)
            return self._nox_session.run(*args, **kwargs, external=True, silent=False)
        finally:
            os.chdir(curdir)

    def python(self, *args, cwd=None, **kwargs):
        self.execute("python", *args, **kwargs, cwd=cwd)

    def launch(self, script, args=(), voirargs=(), env={}):
        if not XPath(script).is_absolute():
            script = str(self.dirs.code / script)

        command = ["voir", *voirargs, script, *args]
        process = subprocess.Popen(
            command,
            env={"PYTHONUNBUFFERED": "1", **self._nox_session.env, **env},
            stdout=subprocess.PIPE,
            # NOTE: the forward instrumenter will tag stderr lines
            # on stdout, so we don't really lose information by
            # forwarding stderr to stdout here
            stderr=subprocess.STDOUT,
            cwd=self.dirs.code,
        )
        return process


class Package(BasePackage):
    main_script = "main.py"
    prepare_script = "prepare.py"
    requirements_file = "requirements.txt"

    def make_env(self):
        env = {
            f"MILABENCH_DIR_{name.upper()}": path
            for name, path in self.config["dirs"].items()
        }
        env["MILABENCH_CONFIG"] = json.dumps(self.config)
        return env

    def install(self):
        if self.requirements_file is not None:
            reqs = self.dirs.code / self.requirements_file
            if reqs.exists():
                self.pip_install("-r", reqs)

    def prepare(self):
        if self.prepare_script is not None:
            prep = self.dirs.code / self.prepare_script
            if prep.exists():
                self.execute(prep, json.dumps(self.config), env=self.make_env())

    def run(self, args, voirargs, env):
        main = self.dirs.code / self.main_script
        if not main.exists():
            raise FileNotFoundError(f"Cannot run main script because it does not exist: {main}")

        env.update(self.make_env())
        return self.launch(self.main_script, args=args, voirargs=voirargs, env=env)
