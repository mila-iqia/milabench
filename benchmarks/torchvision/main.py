import argparse
import contextlib
import os

import torch
import torch.cuda.amp
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as tvmodels
import torchvision.transforms as transforms
import voir
from giving import give, given

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def is_tf32_allowed(args):
    return "tf32" in args.precision


def is_fp16_allowed(args):
    return "fp16" in args.precision


data_transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
)


@contextlib.contextmanager
def scaling(enable):
    if enable:
        with torch.cuda.amp.autocast():
            yield
    else:
        yield


def train_epoch(model, criterion, optimizer, loader, device, scaler=None):
    model.train()
    for inp, target in voir.iterate("train", loader, True):
        inp = inp.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        with scaling(scaler is not None):
            output = model(inp)
            loss = criterion(output, target)
            give(loss=loss.item())

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()


def model_optimizer(model, args):
    return args


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


def dataloader(args, model, device):    
    if args.loader == "dali":
        return dali(args, args.data)
    
    if args.data:
        train = datasets.ImageFolder(os.path.join(args.data, "train"), data_transforms)
        return torch.utils.data.DataLoader(
            train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
    else:
        return SyntheticData(
            model=model,
            device=device,
            batch_size=args.batch_size,
            n=1000,
            fixed_batch=args.fixed_batch,
        )


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


def main():
    parser = argparse.ArgumentParser(description="Torchvision models")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        metavar="N",
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--loader", 
        type=str, 
        default="pytorch",
        choices=["pytorch", "dali"],
        help="Dataloader backend",
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

    args = parser.parse_args()
    if args.fixed_batch:
        args.synthetic_data = True

    if args.synthetic_data:
        args.data = None
    else:
        if not args.data:
            data_directory = os.environ.get("MILABENCH_DIR_DATA", None)
            if data_directory:
                args.data = os.path.join(data_directory, "FakeImageNet")

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if use_cuda:
        if is_tf32_allowed(args):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    model = getattr(tvmodels, args.model)()
    model.to(device)
    
    model = model_optimizer(model)

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr)

    train_loader = dataloader(args, model, device)

    if is_fp16_allowed(args):
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    with given() as gv:
        if not args.no_stdout:
            gv.where("loss").display()

        for epoch in voir.iterate("main", range(args.epochs)):
            if not args.no_stdout:
                print(f"Begin training epoch {epoch}/{args.epochs}")
            train_epoch(
                model, criterion, optimizer, train_loader, device, scaler=scaler
            )


if __name__ == "__main__":
    main()
