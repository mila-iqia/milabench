import argparse
from contextlib import nullcontext

import torch
from torch import optim
from torch.utils.data import DataLoader
import torchcompat.core as accelerator

import transformers
from benchmate.observer import BenchObserver

from .models import models
from .synth import SyntheticData, generators


def is_tf32_allowed(args):
    return "tf32" in args.precision


def is_fp16_allowed(args):
    return "fp16" in args.precision or "bf16" in args.precision


def float_dtype(precision):
    if "fp16" in precision:
        if torch.cuda.is_available():
            return torch.float16
        else:
            return torch.bfloat16
        
    if "bf16" in precision:
        return torch.bfloat16
        
    return torch.float

class NoScale:
    def scale(self, loss):
        return loss
    def step(self, optimizer):
        optimizer.step()
    def update(self):
        pass

class Runner:
    def __init__(self, args):
        accelerator.set_enable_tf32(is_tf32_allowed(args))

        accelerator.manual_seed(args.seed)
        transformers.set_seed(args.seed)

        self.device = accelerator.fetch_device(0)
        self.batch_size = args.batch_size
        info = models[args.model]()
        self.model = info.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        # this cause the bench to fail for one model (reformer)
        # dtype=float_dtype(args.precision)
        self.model, self.optimizer = accelerator.optimize(self.model, optimizer=self.optimizer)

        self.data = SyntheticData(
            n=args.batch_size,
            repeat=100000,
            generators=generators[info.category](info),
        )
        self.loader = DataLoader(
            self.data, batch_size=args.batch_size, num_workers=args.num_workers
        )

        self.amp_scaler = NoScale()
        if torch.cuda.is_available():
            self.amp_scaler = accelerator.amp.GradScaler(enabled=is_fp16_allowed(args))

        if is_fp16_allowed(args):
            self.amp_context = lambda: accelerator.amp.autocast(dtype=float_dtype(args.precision))
        else:
            self.amp_context = nullcontext

    def step(self, data):
        with self.amp_context():
            outputs = self.model(**data)

        loss = outputs.loss

        self.amp_scaler.scale(loss).backward()
        accelerator.mark_step()

        self.amp_scaler.step(self.optimizer)
        accelerator.mark_step()

        self.amp_scaler.update()
        return loss

    def train(self):
        def batch_size(bs):
            # whisper: ['input_features', 'labels']
            # bert   : ['input_ids', 'labels']
            input_ids = bs.get("labels")
            if input_ids is not None:
                return input_ids.shape[0]

            print(list(bs.keys()))
            raise RuntimeError("Batch size unknown")
        
        observer = BenchObserver(
            event_fn=accelerator.Event, 
            batch_size_fn=batch_size
        )
        loader = observer.loader(self.loader)
    
        for data in loader:
            data = {k: v.to(self.device) for k, v in data.items()}
            loss = self.step(data)

            observer.record_loss(loss)


def parser():
    parser = argparse.ArgumentParser(description="Transformers models")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        metavar="N",
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--model", type=str, help="Transformer model name", required=True
    )
    # parser.add_argument(
    #     "--epochs",
    #     type=int,
    #     default=10,
    #     metavar="N",
    #     help="number of epochs to train (default: 10)",
    # )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        metavar="S",
        help="random seed (default: 1234)",
    )
    # parser.add_argument("--data", type=str, help="data directory")
    # parser.add_argument(
    #     "--synthetic-data", action="store_true", help="whether to use synthetic data"
    # )
    # parser.add_argument(
    #     "--fixed-batch", action="store_true", help="use a fixed batch for training"
    # )
    # parser.add_argument(
    #     "--with-amp",
    #     action="store_true",
    #     help="whether to use mixed precision with amp",
    # )
    # parser.add_argument(
    #     "--no-tf32",
    #     dest="allow_tf32",
    #     action="store_false",
    #     help="do not allow tf32",
    #     default=True,
    # )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp16", "fp32", "tf32", "tf32-fp16", "bf16"],
        default="fp32",
        help="Precision configuration",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="number of workers for data loading",
    )
    # parser.add_argument(
    #     "--no-stdout",
    #     action="store_true",
    #     help="do not display the loss on stdout",
    # )
    return parser


def main():
    args = parser().parse_args()
    runner = Runner(args)
    runner.train()


if __name__ == "__main__":
    main()
