#!/usr/bin/env python

import os

from pathlib import Path

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "gflownet", "src"))


if __name__ == "__main__":
    from gflownet.models.bengio2021flow import load_original_model
    
    # If you need the whole configuration:
    # config = json.loads(os.environ["MILABENCH_CONFIG"])
    print("+ Full environment:\n{}\n***".format(os.environ))

    #milabench_cfg = os.environ["MILABENCH_CONFIG"]
    #print(milabench_cfg)

    xdg_cache = os.environ["XDG_CACHE_HOME"]

    print("+ Loading proxy model weights to MILABENCH_DIR_DATA={}".format(xdg_cache))
    _ = load_original_model(
        cache=True,
        location=Path(os.path.join(xdg_cache, "bengio2021flow_proxy.pkl.gz")),
        )

