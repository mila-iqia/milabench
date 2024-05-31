import argparse
import contextlib
import os

import torch
import torch.cuda.amp
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as tvmodels
import torchvision.transforms as transforms
import torchcompat.core as accelerator

import voir
from voir.wrapper import DataloaderWrapper, StopProgram
from giving import give, given

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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


data_transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
)


@contextlib.contextmanager
def scaling(enable, dtype):
    if enable:
        with accelerator.amp.autocast(dtype=dtype):
            yield
    else:
        yield


class SyntheticData:
    def __init__(self, model, device, batch_size, n, fixed_batch):
        self.n = n
        self.inp = torch.randn((batch_size, 3, 224, 224)).to(device)
        self.out = torch.rand_like(model(self.inp))
        self.fixed_batch = fixed_batch

    def __iter__(self):
        inp, out = self.inp, self.out
        for i in range(self.n):
            if not self.fixed_batch:
                inp = torch.rand_like(self.inp)
                out = torch.rand_like(self.out)
            yield (inp, out)

    def __len__(self):
        return self.n


def dali(args, images_dir):
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
    from nvidia.dali.plugin.pytorch import DALIGenericIterator

    @pipeline_def(num_threads=args.num_workers, device_id=0)
    def get_dali_pipeline():
        images, labels = fn.readers.file(
            file_root=images_dir, 
            random_shuffle=True, 
            name="Reader",
        )
        # decode data on the GPU
        images = fn.decoders.image_random_crop(
            images, 
            device="mixed", 
            output_type=types.RGB,
        )
        # the rest of processing happens on the GPU as well
        images = fn.resize(images, resize_x=256, resize_y=256)
        images = fn.crop_mirror_normalize(
            images,
            crop_h=224,
            crop_w=224,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            mirror=fn.random.coin_flip()
        )
        return images, labels

    train_data = DALIGenericIterator(
        [get_dali_pipeline(batch_size=args.batch_size)],
        ['data', 'label'],
        reader_name='Reader'
    )

    def iter():
        for _ in range(args.epochs):
            for data in train_data:
                x, y = data[0]['data'], data[0]['label']
                yield x, torch.squeeze(y, dim=1).type(torch.LongTensor)

    yield from iter()


def dataloader(args, model):
    if args.loader == "dali":
        return dali(args, args.data)

    if args.fixed_batch:
        args.synthetic_data = True

    if args.synthetic_data:
        args.data = None
    else:
        if not args.data:
            data_directory = os.environ.get("MILABENCH_DIR_DATA", None)
            if data_directory:
                args.data = os.path.join(data_directory, "FakeImageNet")

    if args.data:
        train = datasets.ImageFolder(os.path.join(args.data, "train"), data_transforms)
        return torch.utils.data.DataLoader(
            train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=True,
            # The dataloader needs a warmup sometimes
            # by avoiding to go through too many epochs
            # we reduce the standard deviation
            # sampler=torch.utils.data.RandomSampler(
            #     train, 
            #     replacement=True, 
            #     num_samples=len(train) * args.epochs
            # )
        )
    
    return SyntheticData(
        model=model,
        device=accelerator.fetch_device(0),
        batch_size=args.batch_size,
        n=1000,
        fixed_batch=args.fixed_batch,
    )

def model_optimizer(args, model, device):
    model.train()

    if "channel_last" in args.optim:
        model = model.to(memory_format=torch.channels_last)

    if "trace" in args.optim:
        input = torch.randn((args.batch_size, 3, 224, 224)).to(device)
        model = torch.jit.trace(model, input)
        return model, model.parameters()

    if "inductor" in args.optim:
        from functorch import make_functional_with_buffers
        from functorch.compile import make_boxed_func

        model, params, buffers = make_functional_with_buffers(model)

        model = make_boxed_func(model)

        # backend , nvprims_nvfuser, cnvprims_nvfuser
        model = torch.compile(model, backend="inductor")

        def forward(*args):
            return model((params, buffers, *args))

        return forward, params

    return model, model.parameters()

def main():
    from voir.phase import StopProgram

    try:
        _main()
    except StopProgram:
        raise

def _main():
    parser = argparse.ArgumentParser(description="Torchvision models")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        metavar="N",
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--loader", type=str, help="Dataloader implementation", default="pytorch"
    )
    parser.add_argument(
        "--optim", 
        type=str, 
        default="",
        nargs="+",
        choices=["trace", "inductor", "script", "channel_last", "set_grad_none"],
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
        "--num-workers",
        type=int,
        default=8,
        help="number of workers for data loading",
    )
    parser.add_argument("--data", type=str, help="data directory")
    parser.add_argument(
        "--synthetic-data", action="store_true", help="whether to use synthetic data"
    )
    parser.add_argument(
        "--fixed-batch", action="store_true", help="use a fixed batch for training"
    )
    # parser.add_argument(
    #     "--with-amp",
    #     action="store_true",
    #     help="whether to use mixed precision with amp",
    # )
    parser.add_argument(
        "--no-stdout",
        action="store_true",
        help="do not display the loss on stdout",
    )
    # parser.add_argument(
    #     "--no-tf32",
    #     dest="allow_tf32",
    #     action="store_false",
    #     help="do not allow tf32",
    # )
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
    data_directory = os.environ.get("MILABENCH_DIR_DATA", None)
    if args.data is None and data_directory:
        args.data = os.path.join(data_directory, "FakeImageNet")
        
    loader = dataloader(args)
    device = accelerator.fetch_device(0)
    dtype = float_dtype(args.precision)

    for _ in range(args.epochs):
        for inp, target in loader:
            inp = inp.to(device, dtype=dtype)
            target = target.to(device)


def train_epoch(args, model, criterion, optimizer, loader, device, dtype, scaler=None):
    model.train()

    set_to_none = "set_grad_none" in args.optim

    transform = dict(device=device, dtype=dtype)
    if "channel_last" in args.optim:
        transform["memory_format"] = torch.channels_last

    for inp, target in loader:
        inp = inp.to(**transform)
        target = target.to(device)
        optimizer.zero_grad(set_to_none=set_to_none)

        with scaling(scaler is not None, dtype):
            output = model(inp)
            loss = criterion(output, target)
            loader.add_loss(loss)

        if scaler:
            scaler.scale(loss).backward()
            accelerator.mark_step()

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            accelerator.mark_step()
            optimizer.step()

        accelerator.mark_step()

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

    train_loader = dataloader(args, model)

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