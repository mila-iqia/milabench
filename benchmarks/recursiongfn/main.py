# This is the script run by milabench run (by default)
# It is possible to use a script from a GitHub repo if it is cloned using
# clone_subtree in the benchfile.py, in which case this file can simply
# be deleted.

import datetime
import os
from pathlib import Path
from typing import Callable

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "gflownet", "src"))

import numpy as np
import torch.nn as nn
import torchcompat.core as accelerator
from gflownet.config import Config, init_empty
from gflownet.models import bengio2021flow
from gflownet.tasks.seh_frag import SEHFragTrainer, SEHTask
from gflownet.utils.conditioning import TemperatureConditional
from gflownet.utils.misc import get_worker_device
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from benchmate.observer import BenchObserver


class SEHFragTrainerMonkeyPatch(SEHFragTrainer):
    def __init__(self, data, *args, **kwargs):
        self.data = data
        super().__init__(*args, **kwargs)
        self.batch_size_in_nodes = []

        def batch_size(x):
            """Measures the batch size as the sum of all nodes in the batch."""
            return self.batch_size_in_nodes.pop()

        self.observer = BenchObserver(
            accelerator.Event,
            earlystop=65,
            batch_size_fn=batch_size,
            raise_stop_program=False,
            stdout=False,
        )

    def _maybe_resolve_shared_buffer(self, *args, **kwargs):
        batch = super()._maybe_resolve_shared_buffer(*args, **kwargs)

        # Accumulate the size of all graphs in the batch measured in nodes.
        acc = 0
        n = len(batch)
        for i in range(n):
            elem = batch[i]
            acc += elem.x.shape[0]

        self.batch_size_in_nodes.append(acc)
        return batch

    def step(self, loss: Tensor):
        original_output = super().step(loss)
        self.observer.record_loss(loss)
        return original_output

    def build_training_data_loader(self) -> DataLoader:
        original_output = super().build_training_data_loader()
        return self.observer.loader(original_output)

    def setup_task(self):
        self.task = SEHTaskMonkeyPatch(
            data=self.data,
            dataset=self.training_data,
            cfg=self.cfg,
            rng=self.rng,
            wrap_model=self._wrap_for_mp,
        )


class SEHTaskMonkeyPatch(SEHTask):
    """Allows us to specify the location of the original model download."""

    def __init__(
        self,
        data: str,
        dataset: Dataset,
        cfg: Config,
        rng: np.random.Generator = None,
        wrap_model: Callable[[nn.Module], nn.Module] = None,
    ):
        self._wrap_model = wrap_model
        self.rng = rng
        self.models = self._load_task_models(data)
        self.dataset = dataset
        self.temperature_conditional = TemperatureConditional(cfg, rng)
        self.num_cond_dim = self.temperature_conditional.encoding_size()

    def _load_task_models(self, data):
        xdg_cache = os.environ["XDG_CACHE_HOME"]
        model = bengio2021flow.load_original_model(
            cache=True,
            location=Path(os.path.join(xdg_cache, "bengio2021flow_proxy.pkl.gz")),
        )
        model.to(get_worker_device())
        model = self._wrap_model(model)
        return {"seh": model}


def main(
    data:str, batch_size: int, num_workers: int, num_steps: int, layer_width: int, num_layers: int
):
    # This script runs on an A100 with 8 cpus and 32Gb memory, but the A100 is probably
    # overkill here. VRAM peaks at 6Gb and GPU usage peaks at 25%.

    config = init_empty(Config())
    config.print_every = 1
    config.log_dir = f"./logs/debug_run_seh_frag_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    config.device = accelerator.fetch_device(0)  # This is your CUDA device.
    config.overwrite_existing_exp = True

    config.num_training_steps = num_steps  # Change this to train for longer.
    config.checkpoint_every = 5  # 500
    config.validate_every = 0
    config.num_final_gen_steps = 0
    config.opt.lr_decay = 20_000
    config.opt.clip_grad_type = "total_norm"
    config.algo.sampling_tau = 0.9
    config.cond.temperature.sample_dist = "constant"
    config.cond.temperature.dist_params = [64.0]
    config.replay.use = False

    # Things it may be fun to play with.
    config.num_workers = num_workers
    config.model.num_emb = layer_width
    config.model.num_layers = num_layers
    batch_size = batch_size

    if config.replay.use:
        config.algo.num_from_policy = 0
        config.replay.num_new_samples = batch_size
        config.replay.num_from_replay = batch_size
    else:
        config.algo.num_from_policy = batch_size

    # This may need to be adjusted if the batch_size is made bigger
    config.mp_buffer_size = 32 * 1024**2  # 32Mb
    trial = SEHFragTrainerMonkeyPatch(data, config, print_config=False)
    trial.run()
    trial.terminate()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("-b", "--batch_size", help="Batch Size", default=128)
    parser.add_argument("-n", "--num_workers", help="Number of Workers", default=8)
    parser.add_argument(
        "-s", "--num_steps", help="Number of Training Steps", default=100
    )
    parser.add_argument(
        "-w", "--layer_width", help="Width of each policy hidden layer", default=128
    )
    parser.add_argument("-l", "--num_layers", help="Number of hidden layers", default=4)
    parser.add_argument(
        "--data",
        type=str,
        default=os.getenv("MILABENCH_DIR_DATA", None),
        help="Dataset path",
    )
    args = parser.parse_args()

    main(
        args.data,
        args.batch_size,
        args.num_workers,
        args.num_steps,
        args.layer_width,
        args.num_layers,
    )
