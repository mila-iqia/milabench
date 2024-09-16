#!/usr/bin/env python
import argparse
import os

from pcqm4m_subset import PCQM4Mv2Subset
from torch_geometric.datasets import QM9


def parser():
    parser = argparse.ArgumentParser(description="Geometric GNN")
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Number of samples to process in the dataset",
        default=100000,
    )
    parser.add_argument(
        "--root",
        type=str,
        default=os.environ["MILABENCH_DIR_DATA"],
        help="Dataset path",
    )
    return parser


if __name__ == "__main__":
    args, _ = parser().parse_known_args()

    dataset = PCQM4Mv2Subset(args.num_samples, root=args.root)
