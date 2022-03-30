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

    def do_install(self, force=False):
        sentinel = self.dirs.code / "installed"
        if not force and sentinel.exists():
            name = self.config["name"]
            print(f"Benchmark {name} is already installed")
            return

        if self.dirs.code == self.path_pack:
            return

        if self.dirs.code.exists():
            self.dirs.code.rm()
        self.pack_path.merge_into(self.dirs.code, self.pack_path / "manifest")

        devreqs = os.environ.get("MILABENCH_DEV", None)
        if devreqs:
            self.install("-r", devreqs)
        else:
            self.install("milabench", "voir")
        self.setup()
        sentinel.touch()

    def install(self, *args, **kwargs):
        args = [str(x) for x in args]
        return self._nox_session.install(*args, **kwargs, silent=False)

    def conda_install(self, *args, **kwargs):
        args = [str(x) for x in args]
        return self._nox_session.conda_install(*args, **kwargs, silent=False)

    def run(self, *args, cwd=None, **kwargs):
        args = [str(x) for x in args]
        curdir = os.getcwd()
        if cwd is None:
            cwd = self.dirs.code
        try:
            os.chdir(cwd)
            return self._nox_session.run(*args, **kwargs, external=True, silent=False)
        finally:
            os.chdir(curdir)

    def launch_script(self, script, args=(), voirargs=(), env={}):
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
    prepare_script = "prepare.py"

    def setup(self):
        reqs = self.dirs.code / "requirements.txt"
        if reqs.exists():
            self.install("-r", reqs)

    def prepare(self):
        self.run(
            "python",
            self.dirs.code / self.prepare_script,
            "--bench-config",
            json.dumps(self.config),
        )
