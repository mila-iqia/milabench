#!/usr/bin/env python

import os
from gflownet.models.bengio2021flow import load_original_model
from pathlib import Path


if __name__ == "__main__":
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

