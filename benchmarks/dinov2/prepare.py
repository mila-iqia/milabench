#!/usr/bin/env python

import os
from benchmate.datagen import generate_fakeimagenet, device_count


if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(__file__) + "/src/")

    if job_id := os.getenv("SLURM_JOB_ID"):
        del os.environ["SLURM_JOB_ID"]

    from argparse import Namespace
    from dinov2.train.train import setup, get_args_parser

    args = get_args_parser(add_help=True).parse_args()
    cfg = setup(args)

    args = Namespace(
        batch_size=cfg["train"]["batch_size_per_gpu"],
        batch_count=60,
        device_count=device_count(),
        device=None,
        image_size=[3, 384, 384],
        val=0.1,
        test=0.1
    )
    # 
    generate_fakeimagenet(args)
