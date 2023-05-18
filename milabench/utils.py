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

from ovld import ovld

from milabench.fs import XPath
import milabench.validation
from milabench.validation.validation import Summary


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
            tfile.write("\n".join([f"-c {relativize(c)}" for c in constraints]))
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

        if hasattr(layermodule, "_Layer"):
            layers[layername] = layermodule._Layer

    return layers



VALIDATION_LAYERS = discover_validation_layers(milabench.validation)

class _LayerProxy:
    def __init__(self, funs) -> None:
        self.funs = funs

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        for _, fun in self.funs.items():
            fun(*args, **kwds)
            

@contextmanager
def validation(*layer_names, short=True):
    """Combine validation layers into a single context manager"""
    results = dict()

    with ExitStack() as stack:

        for layer_name in layer_names:
            layer = VALIDATION_LAYERS.get(layer_name)

            if layer is not None:
                results[layer_name] = stack.enter_context(layer())
            else:
                names = list(VALIDATION_LAYERS.keys())
                raise RuntimeError(f"Layer `{layer_name}` does not exist: {names}")

        yield _LayerProxy(results)

        summary = Summary()

        for _, layer in results.items():
            layer.report(summary, short=short)

        summary.show()
        return ()
   

class _LoggerProxy:
    def __init__(self, funs) -> None:
        self.funs = funs

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        for _, fun in self.funs.items():
            try:
                fun(*args, **kwds)
            
            except Exception:
                logger_name = getattr(fun, "__name__", fun)
                print(f"Error happened in logger {logger_name}", file=sys.stderr)
                print("=" * 80)
                traceback.print_exc()
                print("=" * 80)
                
    def result(self):
        """Combine error codes"""
        rc = 0
        for _, fun in self.funs.items():
            rc = rc | fun._rc
        return rc
        

@contextmanager
def multilogger(*logs):
    """Combine loggers into a single context manager"""
    results = dict()

    with ExitStack() as stack:

        for log in logs:
            results[type(log)] = stack.enter_context(log)

        yield _LoggerProxy(results)

    return None