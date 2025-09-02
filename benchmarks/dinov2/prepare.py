#!/usr/bin/env python

from pathlib import Path
import os
from benchmate.datagen import generate_fakeimagenet, device_count
from tqdm import tqdm


def loop_on(iterable:list):
    while 1:
        yield from iterable


if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(__file__) + "/src/")

    if job_id := os.getenv("SLURM_JOB_ID"):
        del os.environ["SLURM_JOB_ID"]

    from argparse import Namespace
    from dinov2.data.loaders import ImageNet, _parse_dataset_str
    from dinov2.train.train import get_args_parser
    from dinov2.utils.config import get_cfg_from_args, apply_scaling_rules_to_cfg

    args = get_args_parser(add_help=True).parse_args()

    cfg = get_cfg_from_args(args)
    os.makedirs(args.output_dir, exist_ok=True)
    apply_scaling_rules_to_cfg(cfg)

    args = Namespace(
        batch_size=cfg["train"]["batch_size_per_gpu"],
        batch_count=60,
        device_count=device_count(),
        device=None,
        image_size=[3, 384, 384],
        val=0.1,
        test=0.1,
        output=os.getenv("MILABENCH_DIR_DATA", None),
    )
    # 
    generate_fakeimagenet(args)

    # Generate metadata
    class_, kwargs = _parse_dataset_str(cfg.train.dataset_path)
    dataset = class_(**kwargs)
    root = Path(dataset.root)
    for split in class_.Split:
        dirs = sorted(entry for entry in root.glob(f"{split.value}/*/") if entry.is_dir())
        first_files = [
            sorted(entry for entry in _dir.glob(f"*") if not entry.is_dir())[0]
            for _dir in dirs
        ]
        files_cnt = len([entry for entry in root.glob(f"{split.value}/*/*") if not entry.is_dir()])
        missings_cnt = split.length - files_cnt

        for linkname, first_file in zip(
            (
                root / split.get_image_relpath(i, _dir.name)
                for i, _dir in zip(
                    tqdm(range(split.length), total=split.length),
                    loop_on(dirs),
                )
            ),
            loop_on(first_files)
        ):
            if missings_cnt <= 0:
                break
            if linkname.exists():
                continue
            linkname.hardlink_to(first_file)
            missings_cnt -= 1

    dataset.dump_extra()
