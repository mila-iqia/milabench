import itertools
import random
import sys
import tempfile
import traceback
from functools import wraps

from ovld import ovld

from milabench.fs import XPath


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


def make_constraints_file(constraints):
    if constraints:
        tf = tempfile.NamedTemporaryFile(delete=False)
        with open(tf.name, "w") as tfile:
            tfile.write("\n".join([f"-c {XPath(c).absolute()}" for c in constraints]))
        return (tf.name,)
    else:
        return ()
