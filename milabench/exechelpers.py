"""Execute commands inside a pack context"""

import os
from hashlib import md5
from typing import Sequence

from .alt_async import run
from .fs import XPath
from .structs import BenchLogEntry
from .utils import make_constraints_file, relativize


async def run_command(pack, *args, cwd=None, env={}, external=False, **kwargs):
    """Run a command in the virtual environment.

    Unless specified otherwise, the command is run with
    ``pack.dirs.code`` as the cwd.

    Arguments:
        args: The arguments to the command
        cwd: The cwd to use (defaults to ``pack.dirs.code``)
    """
    args = [str(x) for x in args]
    
    if cwd is None:
        cwd = pack.dirs.code
    
    return await run(
        args,
        **kwargs,
        info={"pack": pack},
        env=pack.full_env(env) if not external else {**os.environ, **env},
        constructor=BenchLogEntry,
        cwd=cwd,
        process_accumulator=pack.processes,
    )


async def python(pack, *args, **kwargs):
    """Run a Python script.

    Equivalent to:

    .. code-block:: python

        pack.execute("python", *args, **kwargs)
    """
    return await run_command(pack, "python", *args, **kwargs)


async def run_pip_install(pack, *args):
    """Install a package in the virtual environment.

    The arguments are given to ``pip install`` verbatim, so you can
    e.g. do ``pack.pip_install("-r", filename)`` to install a list
    of requirements.
    """
    args = [str(x) for x in args]

    if pack.constraints:
        pack.constraints.write_text("\n".join(pack.config["pip"]["constraints"]))
        args += ["-c", str(pack.constraints)]
        
    for line in pack.config.get("pip", {}).get("args", []):
        args += line.split(" ")
        
    await run(
        ["pip", "install", *args],
        info={"pack": pack},
        env={
            **pack.core._nox_session.env,
            **pack.make_env(),
            **pack.config.get("env", {}),
        },
        constructor=BenchLogEntry,
    )


async def install(pack):
    """Install the benchmark.

    By default, this installs the requirements file pointed to by the
    instance or class attribute ``pack.requirements_file``, which is set
    to ``"requirements.txt"`` by default. That path is understood to be
    relative to pack.dirs.code. In other words, if ``pack.dirs.code == /blah``
    and ``pack.requirements_file == "requirements.txt"``, ``pack.install()``
    executes:

    .. code-block::

        pip install -r /blah/requirements.txt``

    .. note::
        The main method ``milabench install`` calls is
        :meth:`~milabench.pack.BasePackage.checked_install` which takes
        care of checking if the install already occurred, copying over
        the manifest's contents to ``pack.dirs.code``, installing
        milabench in the venv, and then calling this method.
    """
    assert pack.phase == "install"
    for reqs in pack.requirements_files(pack.config.get("install_variant", None)):
        if reqs.exists():
            await pack.pip_install("-r", reqs)
        else:
            raise FileNotFoundError(f"Requirements file not found: {reqs}")


async def pin(
    pack,
    clear_previous: bool = True,
    pip_compile_args: Sequence = tuple(),
    input_files: Sequence = tuple(),
    constraints: Sequence = tuple(),
):
    """Pin versions to requirements file.

    Arguments:
        *pip_compile_args: `python3 -m piptools compile` extra arguments
        requirements_file: The output requirements file
        input_files: A list of inputs to piptools compile
        constraint: The constraint file
    """
    ivar = pack.config.get("install_variant", None)
    
    if ivar == "unpinned":
        raise Exception("Cannot pin the 'unpinned' variant.")
    
    assert pack.phase == "pin"
    
    for base_reqs, reqs in pack.requirements_map().items():
        if not base_reqs.exists():
            raise FileNotFoundError(
                f"Cannot find base requirements file: {base_reqs}"
            )

        if clear_previous and reqs.exists():
            await pack.message(f"Clearing out existing {reqs}")
            reqs.rm()

        grp = pack.config["group"]
        constraint_path = XPath(".pin") / f"tmp-constraints-{ivar}-{grp}.txt"
        constraint_files = make_constraints_file(constraint_path, constraints)
        current_input_files = constraint_files + (base_reqs, *input_files)

        await pack.exec_pip_compile(
            reqs, current_input_files, argv=pip_compile_args
        )

        # Add previous requirements as inputs
        input_files = (reqs, *input_files)


async def exec_pip_compile(
    pack, requirements_file: XPath, input_files: XPath, argv=[]
):
    input_files = [relativize(inp) for inp in input_files]
    return await pack.execute(
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
        cwd=XPath(".").absolute(),
        external=True,
    )


async def prepare(pack):
    """Prepare the benchmark.

    By default, this executes ``pack.dirs.code / pack.prepare_script``,
    which should be an executable (python, bash, etc.)

    The environment variables from :meth:`~milabench.pack.BasePackage.make_env` are set for that
    invocation, so the script can use e.g. ``$MILABENCH_DIR_DATA`` to
    access the data directory for the benchmark.

    The default value of ``pack.prepare_script`` is ``"prepare.py"``.
    """
    assert pack.phase == "prepare"
    
    if pack.prepare_script is not None:
        prep = pack.dirs.code / pack.prepare_script
        
        if prep.exists():
            await pack.execute(
                prep, *pack.argv, env=pack.make_env(), cwd=prep.parent
            )
