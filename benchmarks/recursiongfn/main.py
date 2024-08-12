# This is the script run by milabench run (by default)
# It is possible to use a script from a GitHub repo if it is cloned using
# clone_subtree in the benchfile.py, in which case this file can simply
# be deleted.

import datetime
import os
import random
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def batch_size(x):
            """Needs to convert x into correct format to access batch_size attr."""
            x = self._maybe_resolve_shared_buffer(x, self.train_dl)
            return x.batch_size

        self.observer = BenchObserver(
            accelerator.Event,
            earlystop=65,
            batch_size_fn=batch_size,
            raise_stop_program=False,
            stdout=True,
        )

        # Need to cache result.
        self.train_dl = self.build_training_data_loader()

    def step(self, loss: Tensor):
        original_output = super().step(loss)
        self.observer.record_loss(original_output)
        return original_output

    def build_training_data_loader(self) -> DataLoader:
        original_output = super().build_training_data_loader()
        return self.observer.loader(original_output)

    def setup_task(self):
        self.task = SEHTaskMonkeyPatch(
            dataset=self.training_data,
            cfg=self.cfg,
            rng=self.rng,
            wrap_model=self._wrap_for_mp,
        )


class SEHTaskMonkeyPatch(SEHTask):
    """Allows us to specify the location of the original model download."""

    def __init__(
        self,
        dataset: Dataset,
        cfg: Config,
        rng: np.random.Generator = None,
        wrap_model: Callable[[nn.Module], nn.Module] = None,
    ):
        self._wrap_model = wrap_model
        self.rng = rng
        self.models = self._load_task_models()
        self.dataset = dataset
        self.temperature_conditional = TemperatureConditional(cfg, rng)
        self.num_cond_dim = self.temperature_conditional.encoding_size()

    def _load_task_models(self):
        xdg_cache = os.environ["XDG_CACHE_HOME"]
        model = bengio2021flow.load_original_model(
            cache=True,
            location=Path(os.path.join(xdg_cache, "bengio2021flow_proxy.pkl.gz")),
        )
        model.to(get_worker_device())
        model = self._wrap_model(model)
        return {"seh": model}


def main():
    # This script runs on an A100 with 8 cpus and 32Gb memory, but the A100 is probably
    # overkill here. VRAM peaks at 6Gb and GPU usage peaks at 25%.

    config = init_empty(Config())
    config.print_every = 1
    config.log_dir = f"./logs/debug_run_seh_frag_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    config.device = accelerator.fetch_device(0)  # This is your CUDA device.
    config.overwrite_existing_exp = True

    config.num_training_steps = 10  # 1000 # Change this to train for longer
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
    config.num_workers = 8
    config.model.num_emb = 128
    config.model.num_layers = 4
    batch_size = 64

    if config.replay.use:
        config.algo.num_from_policy = 0
        config.replay.num_new_samples = batch_size
        config.replay.num_from_replay = batch_size
    else:
        config.algo.num_from_policy = batch_size

    # This may need to be adjusted if the batch_size is made bigger
    config.mp_buffer_size = 32 * 1024**2  # 32Mb

    trial = SEHFragTrainerMonkeyPatch(config, print_config=False)
    trial.run()
    trial.terminate()


if __name__ == "__main__":
    main()
