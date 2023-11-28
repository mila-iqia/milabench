#!/usr/bin/env python

import json
import os
import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    cfg = json.loads(os.environ["MILABENCH_CONFIG"])
    extra = Path(cfg["dirs"]["extra"])
    os.makedirs(extra, exist_ok=True)
    prepfile = extra / "prepared"
    if prepfile.exists():
        print("rwkv was already prepared")
        sys.exit(0)

    argv = sys.argv[1:]

    os.chdir("rwkv-v4neo")

    print("Run the training process, but only one step.")
    print("This will compile the appropriate torch extensions.")
    print("=" * 80)
    result = subprocess.run(
        ["voir", "--no-dash", "--interval", "1", "--stop", "1", "train.py", *argv]
    )
    print("=" * 80)
    print("Done")

    prepfile.touch()
