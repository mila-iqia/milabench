#!/usr/bin/env python3

import subprocess
import sys


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from ...utils import get_module_venv
    from .. import __main__
    check_if_module = "import covalent"
    python3, env = get_module_venv(__main__.__file__, check_if_module)

    return subprocess.call([python3, *argv], env=env)


if __name__ == "__main__":
    sys.exit(main())
