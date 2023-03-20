#!/usr/bin/env python

import importlib
import json
import sys


if __name__ == "__main__":
    config = json.loads(sys.argv[1])
    model = config["model"]
    # Use 'ModelWrapper' instead of 'Model' for these models
    if model in {"Super_SloMo"}:
        module = importlib.import_module(f".models.{model}", package="torchbenchmark")
        getattr(module, "ModelWrapper", None)()
