"""This module defines the Package class from which new benchmarks inherit.

A ``Package`` must define three methods corresponding to milabench's three main
commands: ``install``, ``prepare`` and ``run``. The :class:`~milabench.pack.Package`
class defines good default behavior.
"""

import json
import os
import subprocess
import re
from argparse import Namespace as NS
from sys import version_info as pyv

from nox.sessions import Session, SessionRunner

from .fs import XPath


class BasePackage:
    """Base package, with no behavior defined for install/prepare/run.

    Attributes:
        config: The configuration dictionary as defined in the benchmark
            config for the benchmark.
        pack_path: The path to the package file (same as ``config["definition"]``)
        dirs: A Namespace object with important paths:

            * ``code``: The code directory for this benchmark
            * ``venv``: The virtual environment for this benchmark
            * ``data``: The data directory (shared)
            * ``runs``: The runs directory (shared)
    """

    def __init__(self, config):
        self.config = config
        self.pack_path = XPath(config["definition"])

        self.dirs = NS(**{name: XPath(d) for name, d in config["dirs"].items()})

        constraints = self.config.get("pip", {}).get("constraints", None)
        if constraints:
            self.constraints = self.dirs.code / "pip-constraints.txt"
        else:
            self.constraints = None

        reuse = True
        self._nox_runner = SessionRunner(
            name=self.dirs.venv.name,
            signatures=[],
            func=NS(
                python=f"{pyv.major}.{pyv.minor}.{pyv.micro}",
                venv_backend=self.config["venv"]["type"],
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
        self.code_mark_file = self.dirs.code / ".mark"
        ig = self.config.get("install_group", self.config["name"])
        self.install_mark_file = self.dirs.code / f".mark_{ig}"

    def make_env(self):
        """Return a dict of environment variables to use for prepare/run."""
        return {}

    def checked_install(self, force=False, sync=False):
        """Entry method to install the benchmark.

        * Check if the benchmark is installed.
        * :meth:`~milabench.pack.BasePackage.install_code`
        * :meth:`~milabench.pack.BasePackage.install_milabench`
        * :meth:`~milabench.pack.BasePackage.install`

        Arguments:
            force: Whether to force installation if the benchmark
                is already installed.
            sync: Whether to only sync changed files in the manifest.
        """
        if not force and not sync and self.install_mark_file.exists():
            name = self.config["name"]
            print(f"Benchmark {name} is already installed")
            return

        if self.dirs.code == self.pack_path:
            print(f"Cannot install if the code destination is the same as the source")
            return

        if force or sync or not self.code_mark_file.exists():
            self.install_code(sync=sync)
        if sync:
            return
        self.install_milabench()
        self.install()
        self.install_mark_file.touch()
        self.code_mark_file.touch()

    def install_code(self, sync=False):
        """Copy the contents of the manifest into ``self.dirs.code``.

        If the directory already exists, it is cleared out beforehand,
        unless ``sync == True``.

        Arguments:
            sync: Whether we are performing a simple sync, in which case
                existing code is not cleared.
        """
        if sync:
            print(f"Syncing changes into {self.dirs.code}")
        elif self.dirs.code.exists():
            print(f"Clearing existing data in {self.dirs.code}")
            self.dirs.code.rm()
        self.pack_path.merge_into(
            self.dirs.code, self.pack_path / "manifest", readonly=True
        )
        self.config.setdefault("pip", {})
        if self.constraints:
            self.constraints.write_text("\n".join(self.config["pip"]["constraints"]))

    def install_milabench(self):
        """Install milabench in the virtual environment.

        Essentially runs ``pip install milabench``.

        The ``$MILABENCH_DEVREQS`` environment variable can be set to point
        to a requirements file to use instead of the standard command.
        """
        devreqs = os.environ.get("MILABENCH_DEVREQS", None)
        # Make sure pip is recent enough
        self.pip_install("pip", "-U")
        if devreqs:
            self.pip_install("-r", devreqs)
        else:
            # Install as editable if we see the pyproject file in
            # the parent directory of milabench
            import milabench

            mb_parent = XPath(milabench.__file__).parent.parent
            if (mb_parent / "pyproject.toml").exists():
                self.pip_install("-e", mb_parent)
            else:
                self.pip_install("milabench")

    def pip_install(self, *args, **kwargs):
        """Install a package in the virtual environment.

        The arguments are given to ``pip install`` verbatim, so you can
        e.g. do ``self.pip_install("-r", filename)`` to install a list
        of requirements.
        """
        args = [str(x) for x in args]
        if self.constraints:
            args += ["-c", str(self.constraints)]
        for line in self.config.get("pip", {}).get("args", []):
            args += line.split(" ")
        return self._nox_session.install(*args, **kwargs, silent=False)

    def conda_install(self, *args, **kwargs):
        """Install a package using conda."""
        args = [str(x) for x in args]
        return self._nox_session.conda_install(*args, **kwargs, silent=False)

    def execute(self, *args, cwd=None, **kwargs):
        """Run a command in the virtual environment.

        Unless specified otherwise, the command is run with
        ``self.dirs.code`` as the cwd.

        Arguments:
            args: The arguments to the command
            cwd: The cwd to use (defaults to ``self.dirs.code``)
        """
        args = [str(x) for x in args]
        curdir = os.getcwd()
        if cwd is None:
            cwd = self.dirs.code
        try:
            os.chdir(cwd)
            return self._nox_session.run(*args, **kwargs, external=True, silent=False)
        finally:
            os.chdir(curdir)

    def python(self, *args, **kwargs):
        """Run a Python script.

        Equivalent to:

        .. code-block:: python

            self.execute("python", *args, **kwargs)
        """
        self.execute("python", *args, **kwargs)

    def launch(self, script, args=(), voirargs=(), env={}):
        """Launch a script using ``voir``.

        This runs:

        .. code-block::

            voir {*voirargs} {self.dirs.code / self.main_script} {*args}

        Using ``self.dirs.code`` as the current working directory.

        .. note::
            stderr is piped to stdout in the process

        Arguments:
            args: A list of arguments to the program.
            voirargs: A list of arguments to ``voir``.
            env: Environment variables to set for the process.

        Returns:
            A subprocess.Popen instance representing the running process.
        """

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
            preexec_fn=os.setsid,
        )
        process.did_setsid = True
        return process


class Package(BasePackage):
    """Package class with default behaviors for install/prepare/run.

    See:

    * :meth:`~milabench.pack.Package.install`
    * :meth:`~milabench.pack.Package.prepare`
    * :meth:`~milabench.pack.Package.run`
    """

    main_script = "main.py"
    prepare_script = "prepare.py"
    requirements_file = "requirements.txt"

    def make_env(self):
        """Return a dict of environment variables to use for prepare/run.

        By default, ``make_env()`` will return:

        .. code-block:: python

            {
                "MILABENCH_DIR_CODE": self.dirs.code,
                "MILABENCH_DIR_DATA": self.dirs.data,
                "MILABENCH_DIR_VENV": self.dirs.venv,
                "MILABENCH_DIR_RUNS": self.dirs.runs,
                "MILABENCH_CONFIG": json.dumps(self.config),
            }
        """
        env = {
            f"MILABENCH_DIR_{name.upper()}": path
            for name, path in self.config["dirs"].items()
        }
        env["MILABENCH_CONFIG"] = json.dumps(self.config)
        return env

    def install(self):
        """Install the benchmark.

        By default, this installs the requirements file pointed to by the
        instance or class attribute ``self.requirements_file``, which is set
        to ``"requirements.txt"`` by default. That path is understood to be
        relative to self.dirs.code. In other words, if ``self.dirs.code == /blah``
        and ``self.requirements_file == "requirements.txt"``, ``self.install()``
        executes:

        .. code-block::

            pip install -r /blah/requirements.txt``

        .. note::
            The main method ``milabench install`` calls is
            :meth:`~milabench.pack.BasePackage.checked_install` which takes
            care of checking if the install already occurred, copying over
            the manifest's contents to ``self.dirs.code``, installing
            milabench in the venv, and then calling this method.
        """
        if self.requirements_file is not None:
            reqs = self.dirs.code / self.requirements_file
            if reqs.exists():
                self.pip_install("-r", reqs)

    def pin(self, *pip_compile_args, requirements_file=None, cwd=None):
        """Pin versions to requirements file.
        """
        if requirements_file is None:
            requirements_file = self.requirements_file
        if cwd is None:
            cwd = self.pack_path
        if requirements_file is not None:
            if self.pack_path != cwd and (self.pack_path / requirements_file).exists():
                (self.pack_path / requirements_file).copy(cwd)
            if (cwd / re.sub(".txt$", ".in", requirements_file)).exists():
                pip_compile_args = (*pip_compile_args, re.sub(".txt$", ".in", requirements_file))
            self.pip_install("pip-tools")
            self.execute("python3", "-m", "piptools", "compile", "--resolver",
                "backtracking", "--output-file", requirements_file,
                *pip_compile_args, cwd=cwd)
            if self.pack_path != cwd:
                (cwd / requirements_file).copy(self.pack_path)

    def prepare(self):
        """Prepare the benchmark.

        By default, this executes ``self.dirs.code / self.prepare_script``,
        which should be an executable (python, bash, etc.)

        The environment variables from :meth:`~milabench.pack.BasePackage.make_env` are set for that
        invocation, so the script can use e.g. ``$MILABENCH_DIR_DATA`` to
        access the data directory for the benchmark.

        The default value of ``self.prepare_script`` is ``"prepare.py"``.
        """
        if self.prepare_script is not None:
            prep = self.dirs.code / self.prepare_script
            if prep.exists():
                self.execute(prep, json.dumps(self.config), env=self.make_env())

    def run(self, args, voirargs, env):
        """Start the benchmark and return the running process.

        By default, this runs:

        .. code-block::

            voir {*voirargs} {self.dirs.code / self.main_script} {*args}

        The environment variables from :meth:`~milabench.pack.BasePackage.make_env` are set for that
        invocation, so the script can use e.g. ``$MILABENCH_DIR_DATA`` to
        access the data directory for the benchmark.

        The default value of ``self.main_script`` is ``"main.py"``.

        Arguments:
            args: A list of arguments to the program.
            voirargs: A list of arguments to ``voir``.
            env: Environment variables to set for the process.

        Returns:
            A subprocess.Popen instance representing the running benchmark.
        """
        main = self.dirs.code / self.main_script
        if not main.exists():
            raise FileNotFoundError(
                f"Cannot run main script because it does not exist: {main}"
            )

        env.update(self.make_env())
        return self.launch(self.main_script, args=args, voirargs=voirargs, env=env)
