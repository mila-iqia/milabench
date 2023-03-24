import os

from milabench.cli import main
from milabench.fs import XPath

path = XPath(os.path.dirname(__file__))
configfiles = path / ".." / "tests" / "config"
result = path / ".." / "tests" / "results"

assert configfiles.exists()


def milabench_cmd(*args):
    """Run milabench command line with specific arguments"""
    main(args)


def config(name):
    """Returns a testing configuration"""
    return str(configfiles / (name + ".yaml"))


def resultfolder():
    """Returns the result folder example"""
    return str(result)


def has_result_folder():
    return result.exists()
