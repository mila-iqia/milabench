import copy
import json
import os
import sys
from datetime import datetime

from milabench.utils import REAL_STDOUT

from .fs import XPath
from .log import blabla


def get_rundir(rundir, runname=None):
    now = str(datetime.today()).replace(" ", "_")
    if runname is None:
        bla = blabla()
        runname = f"{bla}.{now}"

    rundir = XPath(rundir) / runname
    if XPath(rundir).exists():
        print(f"Rundir {rundir} already exists", file=sys.stderr)
        sys.exit(1)

    os.makedirs(rundir, exist_ok=True)
    return rundir


def flatten(dictionary):
    """Turn all nested dict keys into a {key}.{subkey} format"""

    def _flatten(dictionary):
        if dictionary == {}:
            return dictionary

        key, value = dictionary.popitem()
        if not isinstance(value, dict) or not value:
            new_dictionary = {key: value}
            new_dictionary.update(flatten(dictionary))
            return new_dictionary

        flat_sub_dictionary = flatten(value)
        for flat_sub_key in list(flat_sub_dictionary.keys()):
            flat_key = key + "." + flat_sub_key
            flat_sub_dictionary[flat_key] = flat_sub_dictionary.pop(flat_sub_key)

        new_dictionary = flat_sub_dictionary
        new_dictionary.update(flatten(dictionary))
        return new_dictionary

    return _flatten(copy.deepcopy(dictionary))


class RawStreamSaver:
    def __init__(self, gv, rundir, runname=None) -> None:
        self.rundir = get_rundir(rundir, "abc")

        self.file = open(self.rundir / "raw.json", "w", encoding="utf-8")
        self.file.write("[\n")

        self.firstline = True
        gv.subscribe(self.on_event)
        print(f"[BEGIN] Reports directory: {self.rundir}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.file.write("\n]\n")
        self.file.flush()
        self.file.close()
        print(f"[DONE] Reports directory: {self.rundir}")

    def on_event(self, data):
        data = dict(data)

        _ = data.pop("#pack")

        self.write(data)

    def write(self, data):
        if not self.firstline:
            self.file.write(",\n")

        self.file.write(json.dumps(data))
        self.firstline = False
