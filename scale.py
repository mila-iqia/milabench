from datetime import datetime
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import FrameType
import yaml

logging.basicConfig(level=logging.DEBUG)

gpu = subprocess.run(
    [
        "bash", "-c", 'nvidia-smi -i 0 -q | grep "Product Name" | cut -d":" -f2'
    ],
    capture_output=True,
    check=True,
    encoding="utf8"
).stdout.strip()
gpu = gpu.replace(" ", "-")
gpu = gpu.replace("_", "-")

if not gpu:
    gpu = gpu or os.environ["MILABENCH_GPU"]
    logging.warning(f"Could not find gpu using nvidia-smi. Using {gpu}")


def run(name:str, batch_size:int):
    run_dir = Path(os.environ["MILABENCH_BASE"]) / f"runs/{gpu}_{name}_{batch_size}"
    staging_dir = run_dir.parent / f"{run_dir.name}.staging"

    argv = sys.argv[1:] + [
        "--select", name, "--run-name", staging_dir.name,
        "--override", f"{name}.plan.method=njobs",
        "--override", f"{name}.plan.n=1"
    ]

    if run_dir.exists():
        logging.info(
            f"({batch_size}) Found existing run dir {run_dir}. Not executing "
            f"{argv}"
        )
        return True

    failed = [str(_d) for _d in run_dir.parent.glob(f"{run_dir.name}.failed_*")]
    if len(failed) > 2:
        logging.warning(
            f"({batch_size}) Failed more than 2 times ({len(failed)}) {failed}. "
            f"Not executing {argv}"
        )
        return False

    returncode = None
    cleanup = None
    try:
        logging.debug(
            f"({batch_size}) Executing {argv}"
        )
        p = subprocess.Popen(
            args=argv,
            env={
                **os.environ,
                "MILABENCH_SIZER_BATCH_SIZE": str(batch_size),
                "MILABENCH_SIZER_SAVE": f"config/scaling_{gpu}.yaml",
            },
        )

        def cleanup(signum: int, frame: FrameType | None):
            if signum is not None:
                logging.info(f"Received signal {signum}")
                p.send_signal(signum)

            if p.poll() == 0:
                logging.info(f"Execution succeeded {argv}")
                if staging_dir.exists():
                    staging_dir.rename(run_dir)            
                    logging.info(f"Renamed {staging_dir} to {run_dir}")
            else:
                logging.error(f"Execution failed {argv}")
                if staging_dir.exists():
                    staging_dir.rename(staging_dir.with_suffix(f".failed_{datetime.now()}".replace(" ", "-")))

        # signal.signal(signal.SIGTERM, cleanup)
        # signal.signal(signal.SIGUSR1, cleanup)

        p.wait()

        returncode = p.poll()

    finally:
        cleanup(None, None)
    
    return returncode == 0


def check_batch_size(name:str, batch_size:int, scaling:Path, force=False):
    scaling = get_scaling(scaling).get(name, {})

    if "arg" not in scaling:
        logging.warning(
            f"({batch_size}) No scaling argument found for {name}. Not "
            f"executing {sys.argv[1:]}"
        )
        return False
    
    if not force and batch_size in scaling.get("model", {}):
        logging.info(
            f"({batch_size}) Found batch size {batch_size} in {name}'s scaling "
            f"config. Not executing {sys.argv[1:]}"
        )
        return True

    return run(name, batch_size) or batch_size in scaling.get("model", {})


def round_even(number:int):
    # Round to next even number
    if int(number + 0.5) == 1:
        return 1
    _ = int(number / 2 + 0.5) - int(number / 2)
    return (int(number / 2) + _) * 2


def test_round_even():
    for i in range(100):
        print([(i, i:=round_even(i / 2)) for _ in range(100) if i > 1])


def get_scaling(scaling_file:Path):
    start_scaling = Path("config/scaling.yaml")

    retries = [True] * 5
    scaling = None
    if not scaling_file.exists() and scaling_file != start_scaling:
        start_scaling = get_scaling(start_scaling)
        for name, conf in start_scaling.items():
            if isinstance(conf, dict) and "model" in conf:
                first = sorted(conf["model"].items())[:1]
                del conf["model"]
                conf["model"] = {
                    k: v for k, v in first
                }
        if start_scaling:
            scaling_file.write_text(yaml.dump(start_scaling))

    while retries and not scaling:
        logging.debug(f"Scaling in {scaling_file} is {scaling}")
        time.sleep(5)
        retries.pop()
        scaling = yaml.safe_load(scaling_file.read_text()) or {}
    return scaling


enabled = set()
for name, conf in yaml.safe_load(Path("config/standard.yaml").read_text()).items():
    if isinstance(conf, dict) and conf["enabled"]:
        enabled.add(name)

for name, conf in yaml.safe_load(Path("config/base.yaml").read_text()).items():
    if name in enabled and "multinode" in conf.get("tags", []):
        enabled.remove(name)

for name, conf in get_scaling(Path("config/scaling.og.yaml")).items():
    if name in enabled:
        lower_batch_size = 1
        for batch_size in conf.get("model", {}):
            while lower_batch_size * 2 <= batch_size:
                lower_batch_size = lower_batch_size * 2

        # MILABENCH_BASE="${MILABENCH_BASE:-$SCRATCH/data/milabench}" MILABENCH_CONFIG="${PWD}/config/standard.yaml" MILABENCH_SIZER_BATCH_SIZE=180 MILABENCH_SIZER_SAVE=config/scaling.yaml hatch run milabench run --system config/cloud-system.yaml.slurm__a100l_x4 --select bert-fp16

        # find lower bound batch_size
        # as long as the test passes, try to find a bigger batch_size
        batch_size = None
        while (
            check_batch_size(name, lower_batch_size * 2, Path(f"config/scaling_{gpu}.yaml"))
        ):
            batch_size = lower_batch_size
            lower_batch_size = lower_batch_size * 2

        logging.info(f"Found lower bound {lower_batch_size} for {name} in scaling")

        # In case the gpu doesn't support the highest recorded batch_size, find
        # the lower bound for the current gpu
        while (
            not batch_size and
            not check_batch_size(name, lower_batch_size, Path(f"config/scaling_{gpu}.yaml")) and
            lower_batch_size > 1
        ):
            lower_batch_size = round_even(lower_batch_size / 2)

        upper_batch_size = lower_batch_size * 2

        while (
            upper_batch_size > lower_batch_size and
            # If we get a 5% difference between lower and upper we consider to
            # have explored all possible scaling batch size
            (upper_batch_size - lower_batch_size) / upper_batch_size >= 0.025
        ):
            logging.info(f"Sweeping for upper bound between {lower_batch_size} and {upper_batch_size} for {name}")

            batch_size = round_even(lower_batch_size + (upper_batch_size - lower_batch_size) / 2) 
            result = check_batch_size(name, batch_size, Path(f"config/scaling_{gpu}.yaml"))

            if result:
                lower_batch_size = batch_size
            elif upper_batch_size == batch_size:
                assert upper_batch_size - lower_batch_size <= 2
                break
            else:
                upper_batch_size = batch_size

        if not lower_batch_size:
            logging.warning(f"Failed to find a batch size for bench {name}")
            continue

        upper_batch_size = lower_batch_size
        lower_batch_size = lower_batch_size / 2

        while lower_batch_size <= upper_batch_size:
            logging.info(f"Sweeping batch sizes from {lower_batch_size} to {upper_batch_size} for {name}")

            lower_batch_size = round_even(lower_batch_size)

            check_batch_size(name, lower_batch_size, Path(f"config/scaling_{gpu}.yaml"))

            if lower_batch_size == upper_batch_size:
                break

            lower_batch_size += int((upper_batch_size - lower_batch_size) / 2)

            if (upper_batch_size - lower_batch_size) / upper_batch_size <= 0.025:
                lower_batch_size = upper_batch_size
