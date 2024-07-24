"""This module defines the Package class from which new benchmarks inherit.

A ``Package`` must define three methods corresponding to milabench's three main
commands: ``install``, ``prepare`` and ``run``. The :class:`~milabench.pack.Package`
class defines good default behavior.
"""

import json
import os
import traceback
from argparse import Namespace as NS
from sys import version_info as pyv
from typing import Sequence

from nox.sessions import Session, SessionRunner

from . import commands as cmd
from .alt_async import run, send
from .fs import XPath
from .merge import merge
from .structs import BenchLogEntry
from .utils import (
    assemble_options,
    deprecated,
    make_constraints_file,
    relativize,
)


def is_editable_install():
    import json
    import subprocess

    try:
        output = subprocess.check_output(["pip", "list", "-e", "--format", "json"])
        editable_package = json.loads(output)

        for p in editable_package:
            if p["name"] == "milabench":
                return True
        return False
    except:
        return False


async def install_benchmate(pack):
    milabench = os.path.dirname(__file__)
    benchmate = os.path.join(milabench, "..", "benchmate")
    await pack.pip_install("-e", benchmate)


class PackageCore:
    def __init__(self, config):
        self.pack_path = XPath(config["definition"])
        self.dirs = NS(**{name: XPath(d) for name, d in config["dirs"].items()})
        self.dirs.code = self.pack_path

        constraints = config.get("pip", {}).get("constraints", None)
        os.makedirs(self.dirs.extra, exist_ok=True)

        if constraints:
            self.constraints = self.dirs.extra / "pip-constraints.txt"
        else:
            self.constraints = None

        reuse = True
        self._nox_runner = SessionRunner(
            name=self.dirs.venv.name,
            signatures=[],
            func=NS(
                python=f"{pyv.major}.{pyv.minor}.{pyv.micro}",
                venv_backend=config.get("venv", {"type": "virtualenv"})["type"],
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
        grp = config.get("group", config["name"])
        ig = config.get("install_group", grp)
        self.install_mark_file = self.dirs.extra / f"mark_{ig}"


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

    def __init__(self, config, core=None):
        if not core:
            core = PackageCore(config)

        self.__dict__.update(core.__dict__)
        self.core = core
        self.config = config
        self.phase = None
        self.processes = []

    def copy(self, config):
        return type(self)(config=merge(self.config, config))

    @property
    def argv(self):
        return assemble_options(self.config.get("argv", []))

    @property
    def tag(self):
        return ".".join(self.config["tag"]) if self.config else self.config["name"]

    @property
    def logdir(self):
        run_name = self.config["run_name"]
        if run_name and run_name[0] in ("/", ".", "~"):
            return XPath(run_name).expanduser().absolute()
        else:
            return self.dirs.runs / run_name

    def logfile(self, extension):
        return self.logdir / (".".join(self.config["tag"]) + f".{extension}")

    def make_env(self):
        """Return a dict of environment variables to use for prepare/run."""
        return {}

    async def message(self, *message_parts):
        message = " ".join(map(str, message_parts))
        await send(
            BenchLogEntry(
                event="message",
                data={"message": message},
                pack=self,
            )
        )

    async def message_error(self, exc):
        await send(
            BenchLogEntry(
                event="error",
                data={
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "trace": traceback.format_exc(),
                },
                pack=self,
            )
        )

    async def send(self, **kwargs):
        await send(
            BenchLogEntry(
                **kwargs,
                pack=self,
            )
        )

    async def checked_install(self):
        """Entry method to install the benchmark.

        * Check if the benchmark is installed.
        * :meth:`~milabench.pack.BasePackage.install`
        """
        if self.install_mark_file.exists():
            name = self.config["name"]
            await self.message(f"Benchmark {name} is already installed")
            return

        await self.install()
        self.install_mark_file.touch()

    async def pip_install(self, *args):
        """Install a package in the virtual environment.

        The arguments are given to ``pip install`` verbatim, so you can
        e.g. do ``self.pip_install("-r", filename)`` to install a list
        of requirements.
        """
        args = [str(x) for x in args]
        if self.constraints:
            self.constraints.write_text("\n".join(self.config["pip"]["constraints"]))
            args += ["-c", str(self.constraints)]
        for line in self.config.get("pip", {}).get("args", []):
            args += line.split(" ")
        await run(
            ["pip", "install", *args],
            info={"pack": self},
            env={
                **self.core._nox_session.env,
                **self.make_env(),
                **self.config.get("env", {}),
            },
            constructor=BenchLogEntry,
        )

    def conda_install(self, *args, **kwargs):
        """Install a package using conda."""
        args = [str(x) for x in args]
        return self._nox_session.conda_install(*args, **kwargs, silent=False)

    @deprecated
    async def execute(self, *args, cwd=None, env={}, external=False, **kwargs):
        """Run a command in the virtual environment.

        Unless specified otherwise, the command is run with
        ``self.dirs.code`` as the cwd.

        Arguments:
            args: The arguments to the command
            cwd: The cwd to use (defaults to ``self.dirs.code``)
        """
        from .executors import execute

        return execute(self, *args, cwd=cwd, env=env, external=external, **kwargs)

    async def python(self, *args, **kwargs):
        """Run a Python script.

        Equivalent to:

        .. code-block:: python

            self.execute("python", *args, **kwargs)
        """
        return await self.execute("python", *args, **kwargs)

    async def voir(self, script, args=(), wrapper=[], cwd=None, **kwargs):
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
        from . import commands as cmd

        if isinstance(script, list):
            executor = cmd.CmdCommand(self, *script, *args)
        else:
            if not XPath(script).is_absolute():
                script = str(self.dirs.code / script)
            executor = cmd.PackCommand(self, script, *args)

        voir = cmd.VoirCommand(executor, cwd=cwd, **kwargs)
        wrapper = cmd.WrapperCommand(voir, *wrapper)
        return await wrapper.execute()


class Package(BasePackage):
    """Package class with default behaviors for install/prepare/run.

    See:

    * :meth:`~milabench.pack.Package.install`
    * :meth:`~milabench.pack.Package.prepare`
    * :meth:`~milabench.pack.Package.run`
    """

    main_script = "main.py"
    prepare_script = "prepare.py"
    base_requirements = "requirements.in"

    def requirements_map(self, variant=None):
        if variant is None:
            variant = self.config.get("install_variant", None)
        base_reqs = (
            [self.base_requirements]
            if isinstance(self.base_requirements, str)
            else self.base_requirements
        )
        base_reqs = [self.dirs.code / req for req in base_reqs]
        if variant == "unpinned":
            return {req: req for req in base_reqs}
        suffix = f".{variant}.txt" if variant else ".txt"
        return {req: req.with_suffix(suffix) for req in base_reqs}

    def requirements_files(self, variant=None):
        return list(self.requirements_map(variant).values())

    @property
    def working_directory(self):
        return self.dirs.code
    
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
        if self.phase == "prepare" or self.phase == "run":
            # XDG_CACHE_HOME controls basically all caches (pip, torch, huggingface,
            # etc.). HOWEVER, we do not want pip's cache to be in self.dirs.cache,
            # but we do want torch, huggingface, etc. to download configurations
            # and models in there so that we can bundle it in a Docker image.
            # Therefore, we set this variable for prepare and run, but not for
            # install or pin. (We could also specifically clear out cache/pip when
            # building an image, but it is overall nicer for development to use
            # the default cache).
            env["XDG_CACHE_HOME"] = str(self.dirs.cache)
        return env

    def full_env(self, env={}):
        return {
            **self.core._nox_session.env,
            **self.make_env(),
            **self.config.get("env", {}),
            **env,
        }

    async def install(self):
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
        assert self.phase == "install"

        for reqs in self.requirements_files(self.config.get("install-variant", None)):
            if reqs.exists():
                await self.pip_install("-r", reqs)
            else:
                raise FileNotFoundError(f"Requirements file not found: {reqs}")

        if is_editable_install():
            await install_benchmate(self)

    async def pin(
        self,
        clear_previous: bool = True,
        pip_compile_args: Sequence = tuple(),
        input_files: Sequence = tuple(),
        constraints: Sequence = tuple(),
        working_dir=None,
    ):
        """Pin versions to requirements file.

        Arguments:
            *pip_compile_args: `python3 -m piptools compile` extra arguments
            requirements_file: The output requirements file
            input_files: A list of inputs to piptools compile
            constraint: The constraint file
        """
        ivar = self.config.get("install_variant", None)

        if ivar == "unpinned":
            raise Exception("Cannot pin the 'unpinned' variant.")

        # assert self.phase == "pin"
        for base_reqs, reqs in self.requirements_map().items():
            if not base_reqs.exists():
                raise FileNotFoundError(
                    f"Cannot find base requirements file: {base_reqs}"
                )

            if clear_previous and reqs.exists():
                await self.message(f"Clearing out existing {reqs}")
                reqs.rm()

            grp = self.config["group"]
            constraint_path = XPath(".pin") / f"tmp-constraints-{ivar}-{grp}.txt"
            constraint_files = make_constraints_file(
                constraint_path, constraints, working_dir
            )
            current_input_files = constraint_files + (base_reqs, *input_files)

            await self.exec_pip_compile(
                reqs,
                current_input_files,
                argv=pip_compile_args,
                working_dir=working_dir,
            )

            # Add previous requirements as inputs
            input_files = (reqs, *input_files)

    async def exec_pip_compile(
        self, requirements_file: XPath, input_files: XPath, argv=[], working_dir=None
    ):
        input_files = [relativize(inp, working_dir) for inp in input_files]
        from . import commands as cmd

        return await cmd.CmdCommand(
            self,
            "python3",
            "-m",
            "piptools",
            "compile",
            "--resolver",
            "backtracking",
            "--output-file",
            relativize(requirements_file, working_dir),
            *argv,
            *input_files,
            cwd=working_dir,
            external=True,
        ).execute()

    async def prepare(self):
        """Prepare the benchmark.

        By default, this executes ``self.dirs.code / self.prepare_script``,
        which should be an executable (python, bash, etc.)

        The environment variables from :meth:`~milabench.pack.BasePackage.make_env` are set for that
        invocation, so the script can use e.g. ``$MILABENCH_DIR_DATA`` to
        access the data directory for the benchmark.

        The default value of ``self.prepare_script`` is ``"prepare.py"``.
        """
        assert self.phase == "prepare"
        return await self.build_prepare_plan().execute()

    def build_prepare_plan(self) -> "cmd.Command":
        if self.prepare_script is not None:
            prep = self.dirs.code / self.prepare_script
            if prep.exists():
                return cmd.PackCommand(
                    self, prep, *self.argv, env=self.make_env(), cwd=prep.parent
                )
        return cmd.VoidCommand(self)

    async def run(self):
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
        """
        assert self.phase == "run"
        return await self.build_run_plan().execute()

    def build_run_plan(self) -> "cmd.Command":
        main = self.dirs.code / self.main_script
        pack = cmd.PackCommand(self, *self.argv, lazy=True)
        return cmd.VoirCommand(pack, cwd=main.parent)

    def resolve_argument(self, name, default):
        """Resolve as single placeholder argument"""
        placeholder = self.config.get("argv", {}).get(name)
        if placeholder:
            return self.resolve_placeholder(placeholder)
        return default

    def resolve_placeholder(self, placeholder):
        """Resolve as single placeholder argument

        Examples
        --------
        >>> resolve_placeholder("auto({n_worker}, 8)")
        16

        """
        from milabench.sizer import resolve_argv
        return str(resolve_argv(self, [str(placeholder)])[0])
