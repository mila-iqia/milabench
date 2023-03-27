import argparse
from contextlib import nullcontext

import torch
import transformers
import voir
from giving import give
from models import models
from torch import optim


class SyntheticData:
    def __init__(self, device, vocab_size, sequence_length, batch_size, n, fixed_batch):
        self.n = n
        self.device = device
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.inputs = self.gen()
        self.labels = self.gen()
        self.fixed_batch = fixed_batch

    def gen(self):
        return torch.randint(
            0, self.vocab_size, (self.batch_size, self.sequence_length)
        ).to(self.device)

    def __iter__(self):
        inp, out = self.inputs, self.labels
        for i in range(self.n):
            if not self.fixed_batch:
                inp = self.gen()
                out = self.gen()
            yield (inp, out)

    def __len__(self):
        return self.n


class Runner:
    def __init__(self, args):
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        if use_cuda:
            if args.allow_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            torch.cuda.manual_seed(args.seed)
            transformers.set_seed(args.seed)
        self.device = torch.device("cuda" if use_cuda else "cpu")
        info = models[args.model]()
        self.model = info.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.data = SyntheticData(
            device=self.device,
            vocab_size=info.config.vocab_size,
            sequence_length=info.train_length,
            batch_size=args.batch_size,
            n=100000,
            fixed_batch=args.fixed_batch,
        )
        if args.with_amp:
            self.amp_context = lambda: torch.cuda.amp.autocast(dtype=torch.float16)
        else:
            self.amp_context = nullcontext

    def step(self, inputs, labels):
        with self.amp_context():
            outputs = self.model(input_ids=inputs, labels=labels)
        loss = outputs.loss
        give(loss=loss.item())
        loss.backward()
        self.optimizer.step()

    def train(self):
        for inputs, labels in voir.iterate("train", self.data, True):
            self.step(inputs, labels)


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
