import ast
import importlib
import json
import os
import runpy
from functools import partial


class Named:
    """A named object.
    This class can be used to construct objects with a name that will be used
    for the string representation.
    """

    def __init__(self, name):
        """Construct a named object.
        Arguments:
            name: The name of this object.
        """
        self.name = name

    def __repr__(self):
        """Return the object's name."""
        return self.name


MISSING = Named("MISSING")


def resolve(mod, default_field=None, may_have_arg=True):
    if ":" in mod:
        mod, field = mod.split(":", 1)
    elif default_field is not None:
        field = default_field
    else:
        mod, field = "milabench.instrument", mod
    if may_have_arg and "=" in field:
        field, arg = field.split("=", 1)
        arg = json.loads(arg)
    else:
        arg = MISSING
    return mod, field, arg


def fetch(mod, default_field=None, arg=MISSING):
    if arg is MISSING:
        mod, field, arg = resolve(mod, default_field, may_have_arg=True)
    else:
        mod, field, _ = resolve(mod, default_field)

    if os.path.exists(mod):
        glb = runpy.run_path(mod)
    else:
        mod_obj = importlib.import_module(mod)
        glb = vars(mod_obj)

    if arg is not MISSING:
        return partial(glb[field], arg=arg)
    else:
        return glb[field]


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


def exec_node(script, node, glb):
    code = compile(node, script, "exec")
    return lambda: exec(code, glb, glb)
