# This is the script run by milabench run (by default)

# It is possible to use a script from a GitHub repo if it is cloned using
# clone_subtree in the benchfile.py, in which case this file can simply
# be deleted.

import time
import random

import torchcompat.core as accelerator


def criterion(*args, **kwargs):
    return random.normalvariate(0, 1)


def prepare_voir():
    from benchmate.observer import BenchObserver
    from benchmate.monitor import bench_monitor

    observer = BenchObserver(
        accelerator.Event, 
        earlystop=65,
        batch_size_fn=lambda x: len(x[0]),
        raise_stop_program=False,
        stdout=True,
    )

    return observer, bench_monitor


def main():
    from vllm.benchmarks.serve import main as benchmark_run
    from vllm.entrypoints.cli.server import ServeSubcommand

    def run_server():
        ServeSubcommand.cmd(args)

    def run_benchmark():
        benchmark_run()

    # vllm serve <your_model> <engine arguments>

    # vllm bench serve                                      \
    #     --backend openai                                  \
    #     --label milabench                                 \
    #     --model <your_model>                              \
    #     --dataset-name <dataset_name. Default 'random'>   \
    #     --request-rate inf                                \
    #     --num-prompts 1000


if __name__ == "__main__":
    main()
