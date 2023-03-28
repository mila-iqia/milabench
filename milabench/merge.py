"""Utilities to merge dictionaries and other data structures."""


from collections import deque
from functools import reduce
from typing import Union

import yaml
from ovld import ovld

from .utils import Named

# Use in a merge to indicate that a key should be deleted
DELETE = Named("DELETE")


###########
# cleanup #
###########


@ovld
def cleanup(value: object):
    """Clean up work structures and values."""
    return value


@ovld  # noqa: F811
def cleanup(d: dict):
    return type(d)({k: cleanup(v) for k, v in d.items() if v is not DELETE})


@ovld  # noqa: F811
def cleanup(xs: Union[tuple, list, set, frozenset]):
    return type(xs)(cleanup(x) for x in xs)


#########
# merge #
#########


@ovld  # noqa: F811
def merge(d1: dict, d2):
    rval = type(d1)()
    for k, v in d1.items():
        if k in d2:
            v2 = d2[k]
            if v2 is DELETE:
                pass
            else:
                rval[k] = merge(v, v2)
        else:
            rval[k] = v
    for k, v in d2.items():
        if k not in d1:
            rval[k] = cleanup(v)
    return rval


@ovld  # noqa: F811
def merge(l1: list, l2: list):
    return l2


@ovld  # noqa: F811
def merge(l1: list, d: dict):
    if "append" in d:
        return l1 + d["append"]
    else:
        raise TypeError("Cannot merge list and dict unless dict has 'append' key")


@ovld  # noqa: F811
def merge(a: object, b):
    if hasattr(a, "__merge__"):
        return a.__merge__(b)
    else:
        return cleanup(b)


##############
# self_merge #
##############


@ovld
def self_merge(self, d: dict):
    d = {k: self(v) for k, v in d.items()}
    if "<<<" in d:
        m = d.pop("<<<")
        d = merge(m, d)
    return d


@ovld
def self_merge(self, li: list):
    return list(map(self, li))


@ovld
def self_merge(self, x):
    return x


#############################
# YAML parser modifications #
#############################


def _tweak(d):
    parts = []
    todo = deque(d.items())
    while todo:
        k, v = todo.popleft()
        if isinstance(k, str) and "." in k:
            k, rest = k.split(".", 1)
            parts.append({k: _tweak({rest: v})})
        elif k == "<<<":
            todo.extendleft(v.items())
        else:
            parts.append({k: v})
    return reduce(merge, parts)


def tweak(loader, node):
    orig = loader.construct_mapping(node, deep=True)
    return _tweak(orig)


yaml.SafeLoader.add_constructor("!expand-dot", tweak)
yaml.SafeLoader.add_constructor("!delete", lambda loader, node: DELETE)
