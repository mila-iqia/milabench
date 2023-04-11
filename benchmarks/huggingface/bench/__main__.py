import argparse
from contextlib import nullcontext

import torch
import transformers
import voir
from giving import give
from torch import optim
from torch.utils.data import DataLoader

from .models import models
from .synth import SyntheticData, generators


def is_tf32_allowed(args):
    return "tf32" in args.precision


def is_fp16_allowed(args):
    return "fp16" in args.precision


class Runner:
    def __init__(self, args):
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        if use_cuda:
            if is_tf32_allowed(args):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            torch.cuda.manual_seed(args.seed)
            transformers.set_seed(args.seed)
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.batch_size = args.batch_size
        info = models[args.model]()
        self.model = info.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        self.data = SyntheticData(
            n=args.batch_size,
            repeat=100000,
            generators=generators[info.category](info),
        )
        self.loader = DataLoader(self.data, batch_size=args.batch_size)

        self.amp_scaler = torch.cuda.amp.GradScaler(enabled=args.with_amp)
        if is_fp16_allowed(args):
            self.amp_context = lambda: torch.cuda.amp.autocast(dtype=torch.float16)
        else:
            self.amp_context = nullcontext

    def step(self, data):
        with self.amp_context():
            outputs = self.model(**data)

        loss = outputs.loss

        self.amp_scaler.scale(loss).backward()
        self.amp_scaler.step(self.optimizer)
        self.amp_scaler.update()

        give(loss=loss.item())

    def train(self):
        for data in voir.iterate(
            "train", self.loader, report_batch=True, batch_size=self.batch_size
        ):
            data = {k: v.to(self.device) for k, v in data.items()}
            self.step(data)


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
    parser.add_argument(
        "--fixed-batch", action="store_true", help="use a fixed batch for training"
    )
    parser.add_argument(
        "--with-amp",
        action="store_true",
        help="whether to use mixed precision with amp",
    )
    parser.add_argument(
        "--no-tf32",
        dest="allow_tf32",
        action="store_false",
        help="do not allow tf32",
        default=True,
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp16", "fp32", "tf32", "tf32-fp16"],
        default="fp32",
        help="Precision configuration",
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
