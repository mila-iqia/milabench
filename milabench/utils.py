import itertools
import os
from contextlib import contextmanager, ExitStack
import random
import sys
import tempfile
import importlib
import pkgutil
import traceback
from functools import wraps
from typing import Any

import yaml
from ovld import ovld

from milabench.fs import XPath
import milabench.validation
from milabench.validation.validation import Summary
from .slurm import build_system_config


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


vowels = list("aeiou")
consonants = list("bdfgjklmnprstvz")
syllables = ["".join(letters) for letters in itertools.product(consonants, vowels)]


def blabla(n=4):
    return "".join([random.choice(syllables) for _ in range(n)])


def error_guard(default_return):
    def deco(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception:
                print("=" * 80, file=sys.stderr)
                print("A non-fatal error happened", file=sys.stderr)
                print("=" * 80, file=sys.stderr)
                traceback.print_exc()
                return (
                    default_return(*args, **kwargs)
                    if callable(default_return)
                    else default_return
                )

        return wrapped

    return deco


@ovld
def assemble_options(options: list):
    return options


@ovld
def assemble_options(options: dict):
    args = []
    for k, v in options.items():
        if v is None:
            continue
        elif v is True:
            args.append(k)
        elif k == "--":
            args.extend(v)
        elif v is False:
            raise ValueError("Use null to cancel an option, not false")
        else:
            args.append(k)
            args.append(",".join(map(str, v)) if isinstance(v, list) else str(v))
    return args


def relativize(pth):
    pth = XPath(pth)
    if pth.is_absolute():
        return pth.relative_to(XPath(".").absolute())
    else:
        return pth


def make_constraints_file(pth, constraints):
    if constraints:
        os.makedirs(XPath(pth).parent, exist_ok=True)
        with open(pth, "w") as tfile:
            # We prefix the constraint with ../ because we are creating a constraint
            # file in ./.pin/,but containing constraints with paths relative to ./
            tfile.write("\n".join([f"-c ../{relativize(c)}" for c in constraints]))
        return (pth,)
    else:
        return ()


def discover_validation_layers(module):
    """Discover validation layer inside the milabench.validation module"""
    path = module.__path__
    name = module.__name__

    layers = {}

    for _, layerpath, _ in pkgutil.iter_modules(path, name + "."):
        layername = layerpath.split(".")[-1]
        layermodule = importlib.import_module(layerpath)

        if hasattr(layermodule, "Layer"):
            layers[layername] = layermodule.Layer

        if hasattr(layermodule, "__layers__"):
            layers.update(layermodule.__layers__)

    return layers


VALIDATION_LAYERS = discover_validation_layers(milabench.validation)


def available_layers():
    return VALIDATION_LAYERS.keys()


def validation_layers(*layer_names, **kwargs):
    """Initialize a list of validation layers"""
    layers = []

    for layer_name in layer_names:
        layer = VALIDATION_LAYERS.get(layer_name)

        if layer is not None:
            layers.append(layer())
        else:
            names = list(VALIDATION_LAYERS.keys())
            raise RuntimeError(f"Layer `{layer_name}` does not exist: {names}")

    return layers


class MultiLogger:
    def __init__(self, funs) -> None:
        self.funs = funs

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        for _, fun in self.funs.items():
            try:
                fun(*args, **kwds)

            except Exception:
                logger_name = getattr(fun, "__name__", fun)
                print(f"Error happened in logger {logger_name}", file=sys.stderr)
                print("=" * 80, file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                print("=" * 80, file=sys.stderr)

    def result(self):
        """Combine error codes"""
        rc = 0
        for _, fun in self.funs.items():
            if hasattr(fun, "error_code") and fun.error_code is not None:
                rc = rc | fun.error_code

        return rc

    def report(self, **kwargs):
        """Generate a full report containing warnings from all loggers"""
        summary = Summary()
        for _, layer in self.funs.items():
            if hasattr(layer, "report"):
                layer.report(summary, **kwargs)
        summary.show()


@contextmanager
def multilogger(*logs, **kwargs):
    """Combine loggers into a single context manager"""
    results = dict()

    with ExitStack() as stack:
        for log in logs:
            results[type(log)] = stack.enter_context(log)

        multilog = MultiLogger(results)
        yield multilog

    multilog.report(**kwargs)


def select_nodes(nodes, n):
    """Select n nodes, main node is always first"""
    ranked = []

    for node in nodes:
        if node["main"]:
            ranked.insert(0, node)
        else:
            ranked.append(node)

    return ranked[: max(1, min(n, len(ranked)))]


def enumerate_rank(nodes):
    rank = 1
    for node in nodes:
        if node["main"]:
            yield 0, node
        else:
            yield rank, node
            rank += 1


def find_private_ssh_key():
    homessh = os.path.expanduser("~/.ssh/")
    key_path = None

    for key in os.listdir(homessh):
        if key.endswith(".pub"):
            continue

        if key.startswith("id_"):
            key_path = os.path.join(homessh, key)
            break

    assert key_path is not None, "No ssh key found"
    return key_path


def command_generator(node_range, arch, version):
    output = os.path.join(os.getcwd(), "results")
    system = os.path.join(output, "system.yaml")

    # Make output folder
    os.makedirs("results", exist_ok=True)

    # Find a ssh private key
    key_path = find_private_ssh_key()

    # make system file
    if os.path.exists(system):
        print("System file already exists, skipping")
    else:
        with open(system, "w") as file:
            system_conf = build_system_config(node_range)
            system_conf["system"]["sshkey"] = "/milabench/id_milabench"

            formatted = yaml.dump(system_conf, indent=2)
            print(formatted)
            file.write(formatted)

    # At that point we might as well run it for them ?
    print()
    print("Running milabench")
    print()
    print(f"    output: {output}")
    print(f"    system: {system}")
    print()
    print("Command:")
    print()

    cmd = [
        (f"  docker run -it --rm --gpus all --network host --ipc=host --privileged"),
        (f"    -v {key_path}:/milabench/id_milabench                              "),
        (f"    -v {output}:/milabench/envs/runs                                   "),
        (f"    ghcr.io/mila-iqia/milabench:{arch}-{version}                        "),
        (f"    milabench run --system /milabench/envs/runs/system.yaml"),
    ]

    length = 0
    for line in cmd:
        line = line.strip()
        length = max(len(line) + 4, length)

    formatted = []
    for i, line in enumerate(cmd):
        line = line.strip()
        idt_size = ((i > 0) + 1) * 2
        idt = " " * idt_size

        formatted.append(f"{idt}{line:<{length - idt_size}}")

    print("\\\n".join(formatted))
