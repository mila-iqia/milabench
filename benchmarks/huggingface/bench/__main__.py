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
    return "fp16" in args.precision or "bf16" in args.precision


def has_xpu():
    try:
        import intel_extension_for_pytorch as ipex
        return torch.xpu.is_available()
    except ImportError as err:
        return True
    


device_interface = None
backend_optimizer = lambda x, y: (x, y)
device = "cpu"
if has_xpu():
    device = "xpu"
    device_interface = torch.xpu
    backend_optimizer = device_interface.optimize

if torch.cuda.is_available():
    device = "cuda"
    device_interface = torch.cuda


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
        if torch.cuda.is_available():
            if is_tf32_allowed(args):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        if torch.xpu.is_available():
            import intel_extension_for_pytorch as ipex
            if is_tf32_allowed(args):
                ipex.set_fp32_math_mode(device="xpu", mode=ipex.FP32MathMode.TF32)
            else:
                ipex.set_fp32_math_mode(device="xpu", mode=ipex.FP32MathMode.FP32)

        device_interface.manual_seed(args.seed)
        transformers.set_seed(args.seed)

        self.device = torch.device(device)
        self.batch_size = args.batch_size
        info = models[args.model]()
        self.model = info.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        # this cause the bench to fail for one model (reformer)
        # dtype=float_dtype(args.precision)
        self.model, self.optimizer = backend_optimizer(self.model, optimizer=self.optimizer)

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
            self.amp_scaler = device_interface.amp.GradScaler(enabled=is_fp16_allowed(args))

        if is_fp16_allowed(args):
            self.amp_context = lambda: device_interface.amp.autocast(dtype=float_dtype(args.precision))
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
