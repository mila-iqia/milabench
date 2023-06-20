"""This module defines the Package class from which new benchmarks inherit.

A ``Package`` must define three methods corresponding to milabench's three main
commands: ``install``, ``prepare`` and ``run``. The :class:`~milabench.pack.Package`
class defines good default behavior.
"""

import json
import os
from argparse import Namespace as NS
from sys import version_info as pyv
from typing import Sequence

from nox.sessions import Session, SessionRunner

from .alt_async import send
from .fs import XPath
from .merge import merge
from .structs import BenchLogEntry
from .utils import assemble_options
from .utils import make_constraints_file, relativize


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
                data={"type": type(exc).__name__, "message": str(exc)},
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

    def conda_install(self, *args, **kwargs):
        """Install a package using conda."""
        args = [str(x) for x in args]
        return self._nox_session.conda_install(*args, **kwargs, silent=False)


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
        
    def pip_install(self, *args):
        args = [str(x) for x in args]

        if self.constraints:
            self.constraints.write_text("\n".join(self.config["pip"]["constraints"]))
            args += ["-c", str(self.constraints)]
            
        for line in self.config.get("pip", {}).get("args", []):
            args += line.split(" ")
            
        return ["pip", "install", *args]
    
    def install(self):
        cmd = self.pip_install()
        
        for reqs in self.requirements_files(self.config.get("install_variant", None)):
            if reqs.exists():
                cmd += ["-r", reqs]
            else:
                raise FileNotFoundError(f"Requirements file not found: {reqs}")
        
        return cmd
    
    def prepare(self):
        assert self.phase == "prepare"
        
        if self.prepare_script is not None:
            prep = self.dirs.code / self.prepare_script
            return [prep, *self.argv]
        
        return []
    
    def pip_compile(self, requirements_file: XPath, input_files: XPath, argv=[]):
        input_files = [relativize(inp) for inp in input_files]
        return [
            "python3",
            "-m",
            "piptools",
            "compile",
            "--resolver",
            "backtracking",
            "--output-file",
            relativize(requirements_file),
            *argv, 
            *input_files,
        ]
        
    def pin(self,
            base_reqs, 
            reqs,
            pip_compile_args: Sequence = tuple(),
            input_files: Sequence = tuple(),
            constraints: Sequence = tuple()):
        
        ivar = self.config.get("install_variant", None)
        
        grp = self.config["group"]
        constraint_path = XPath(".pin") / f"tmp-constraints-{ivar}-{grp}.txt"
        constraint_files = make_constraints_file(constraint_path, constraints)
        current_input_files = constraint_files + (base_reqs, *input_files)

        return self.pip_compile(
            reqs, current_input_files, argv=pip_compile_args
        )
