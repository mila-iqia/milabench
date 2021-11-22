import ast
import importlib
import os
import runpy
import sys
from types import ModuleType

from coleo import Option
from coleo import config as configuration
from coleo import default, run_cli

from .bench import BenchmarkRunner


def split_script(script):
    """Split off the part of the script that tests for __name__ == '__main__'.

    Essentially, we want to be able to instrument functions in the main script, which
    requires evaluating the functions, but we want to do this before executing the main
    code. So we split off the if __name__ == '__main__' part so that we can evaluate
    the module and then evaluate that code separately.
    """

    code = open(script).read()
    tree = ast.parse(code, mode="exec")
    found = None
    for stmt in tree.body:
        if isinstance(stmt, ast.If):
            test = stmt.test
            is_entry_statement = (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq)
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == "__main__"
            )
            if is_entry_statement:
                found = stmt
                break

    new_body = [entry for entry in tree.body if entry is not found]
    if found is not None:
        found = ast.copy_location(
            ast.Module(
                body=[found],
                type_ignores=[],
            ),
            found,
        )

    new_module = ast.copy_location(
        ast.Module(
            body=new_body,
            type_ignores=tree.type_ignores,
        ),
        tree,
    )

    return new_module, found


def resolve(mod, default_field=None):
    if ":" in mod:
        mod, field = mod.split(":", 1)
    elif default_field is not None:
        field = default_field
    else:
        mod, field = "milabench.instrument", mod
    return mod, field


def fetch(mod, default_field=None):
    mod, field = resolve(mod, default_field)

    if os.path.exists(mod):
        glb = runpy.run_path(mod)
    else:
        mod_obj = importlib.import_module(mod)
        glb = vars(mod_obj)

    return glb[field]


def cli():
    runner = run_cli(main)
    runner()


def _runmain(script, node, glb):
    code = compile(node, script, "exec")
    return lambda: exec(code, glb, glb)


def main():
    # Instrumenting functions
    # [alias: -i]
    # [action: append]
    instrumenter: Option = default([])

    # Bridge
    # [alias: -b]
    bridge: Option = default(None)

    # Configuration
    # [alias: -c]
    config: Option & configuration = default(None)

    # Path to the script
    # [positional]
    script: Option

    # Arguments to the script
    # [positional: --]
    args: Option

    script, field = resolve(script, "__main__")

    node, mainsection = split_script(script)
    mod = ModuleType("__main__")
    glb = vars(mod)
    glb["__file__"] = script
    sys.modules["__main__"] = mod
    code = compile(node, script, "exec")
    exec(code, glb, glb)
    glb["__main__"] = _runmain(script, mainsection, glb)

    sys.argv = [script, *args]
    sys.path.insert(0, os.path.abspath(os.curdir))

    return BenchmarkRunner(
        fn=glb[field],
        config=config,
        bridge=bridge and fetch(bridge),
        instruments=[fetch(inst) for inst in instrumenter],
    )
