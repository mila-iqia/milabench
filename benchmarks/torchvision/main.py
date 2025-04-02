import argparse
import contextlib

import torch
import torch.cuda.amp
import torch.nn as nn
import torchvision.models as tvmodels
import torchcompat.core as accelerator

from benchmate.dataloader import imagenet_dataloader, dataloader_arguments
from benchmate.metrics import StopProgram


def is_tf32_allowed(args):
    return "tf32" in args.precision


def is_fp16_allowed(args):
    return "fp16" in args.precision or "bf16" in args.precision


def float_dtype(precision):
    if "fp16" in precision:
        if accelerator.device_type == "cuda":
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


@contextlib.contextmanager
def scaling(enable, dtype):
    if enable:
        with accelerator.amp.autocast(dtype=dtype):
            yield
    else:
        yield


def model_optimizer(args, model, device):
    if hasattr(model, "train"):
        model.train()

    if "channel_last" in args.optim:
        model = model.to(memory_format=torch.channels_last)

    if "trace" in args.optim:
        input = torch.randn((args.batch_size, 3, 224, 224)).to(device)
        model = torch.jit.trace(model, input)
        return model, model.parameters()

    # if "inductor" in args.optim:
    #     from functorch import make_functional_with_buffers
    #     from functorch.compile import make_boxed_func

    #     model, params, buffers = make_functional_with_buffers(model)

    #     model = make_boxed_func(model)

    #     # backend , nvprims_nvfuser, cnvprims_nvfuser
    #     model = torch.compile(model, backend="inductor")

    #     def forward(*args):
    #         return model((params, buffers, *args))

    #     return forward, params

    model = accelerator.compile(model)

    return model, model.parameters()

def main():
    from voir.phase import StopProgram

    try:
        _main()
    except StopProgram:
        raise

def _main():
    parser = argparse.ArgumentParser(description="Torchvision models")

    dataloader_arguments(parser)

    parser.add_argument(
        "--optim", 
        type=str, 
        default="",
        nargs="+",
        choices=["trace", "inductor", "script", "channel_last"],
        help="Optimization to enable",
    )
    parser.add_argument(
        "--model", type=str, help="torchvision model name", required=True
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
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
    parser.add_argument(
        "--no-stdout",
        action="store_true",
        help="do not display the loss on stdout",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp16", "fp32", "tf32", "tf32-fp16"],
        default="fp32",
        help="Precision configuration",
    )
    parser.add_argument(
        "--iobench",
        action="store_true",
        default=False,
        help="Precision configuration",
    )

    args = parser.parse_args()

    if args.iobench:
        iobench(args)
    else:
        trainbench(args)

def iobench(args):
    device = accelerator.fetch_device(0)
    model = getattr(tvmodels, args.model)()
    model.to(device)

    loader = imagenet_dataloader(args, model)
    dtype = float_dtype(args.precision)

    for _ in range(args.epochs):
        for inp, target in loader:
            inp = inp.to(device, dtype=dtype)
            target = target.to(device)


def train_epoch(args, model, criterion, optimizer, loader, device, dtype, scaler=None):
    if hasattr(model, 'train'):
        model.train()

    transform = dict(device=device, dtype=dtype)
    if "channel_last" in args.optim:
        transform["memory_format"] = torch.channels_last

    for inp, target in loader:
        inp = inp.to(**transform)
        target = target.to(device)
        optimizer.zero_grad()

        with scaling(scaler is not None, dtype):
            output = model(inp)
            loss = criterion(output, target)
    
            scaler.scale(loss).backward()
            accelerator.mark_step()

            scaler.step(optimizer)
            accelerator.mark_step()

            scaler.update()

def trainbench(args):
    torch.manual_seed(args.seed)

    accelerator.set_enable_tf32(is_tf32_allowed(args))
    device = accelerator.fetch_device(0)

    model = getattr(tvmodels, args.model)()
    model.to(device)

    model, params = model_optimizer(args, model, device)

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(params, args.lr)

    model, optimizer = accelerator.optimize(model, optimizer=optimizer, dtype=float_dtype(args.precision))

    train_loader = imagenet_dataloader(args, model)

    scaler = NoScale()
    if torch.cuda.is_available():
        scaler = accelerator.amp.GradScaler(enabled=is_fp16_allowed(args))

    for _ in range(args.epochs):
        train_epoch(
            args,
            model, 
            criterion, 
            optimizer, 
            train_loader, 
            device, 
            scaler=scaler, 
            dtype=float_dtype(args.precision),
        )


if __name__ == "__main__":
    try:
        main()
    except StopProgram:
        print("Early stopped")