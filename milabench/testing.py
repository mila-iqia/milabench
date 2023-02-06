import os
import sys

from coleo import run_cli


from milabench.fs import XPath
from milabench.cli import Main


path = XPath(os.path.dirname(__file__))
configfiles = path / ".." / "tests" / "config"
assert configfiles.exists()


def milabench_cmd(*args):
    """Run milabench command line with specific arguments"""
    sys.path.insert(0, os.path.abspath(os.curdir))
    run_cli(Main, argv=[str(arg) for arg in args])


def config(name):
    """Returns a testing configuration"""
    return str(configfiles / (name + ".yaml"))
