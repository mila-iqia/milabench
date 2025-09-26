#!/usr/bin/env python

import os

from pathlib import Path

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "gflownet", "src"))




def parser():
    import argparse

    parser = argparse.ArgumentParser(description="Recurson gfn")
    parser.add_argument(
        "--data",
        type=str,
        default=os.getenv("MILABENCH_DIR_DATA", None),
        help="Dataset path",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=os.getenv("XDG_CACHE_HOME", None),
        help="Dataset path",
    )
    return parser


if __name__ == "__main__":
    from gflownet.models.bengio2021flow import load_original_model
    
    # If you need the whole configuration:
    # config = json.loads(os.environ["MILABENCH_CONFIG"])
    print("+ Full environment:\n{}\n***".format(os.environ))

    #milabench_cfg = os.environ["MILABENCH_CONFIG"]
    #print(milabench_cfg)

    args, _ = parser().parse_known_args()

    print("+ Loading proxy model weights to MILABENCH_DIR_DATA={}".format(args.data))
    _ = load_original_model(
        cache=True,
        location=Path(os.path.join(args.data, "bengio2021flow_proxy.pkl.gz")),
    )

