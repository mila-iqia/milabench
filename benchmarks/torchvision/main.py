import argparse
import contextlib
import os

import torch
import torch.cuda.amp
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as tvmodels
import torchvision.transforms as transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


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
    for inp, target in loader:
        inp = inp.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        with scaling(scaler is not None):
            output = model(inp)
            loss = criterion(output, target)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()


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



def pytorch_dataloader(args):
    train = datasets.ImageFolder(os.path.join(args.data, "train"), data_transforms)
    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    return train_loader


def dali_dataloader(args):
    # pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110
    # workers
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    import os

    @pipeline_def(num_threads=args.workers, device_id=0)
    def get_dali_pipeline():
        images, labels = fn.readers.file(
            file_root=os.path.join(args.data, "train"), 
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
            mirror=fn.random.coin_flip(),
        )
        return images, labels


    train_data = DALIGenericIterator(
        [get_dali_pipeline(batch_size=args.batch_size)],
        ['data', 'label'],
        reader_name='Reader'
    )  
    
    class Iter:
        def __init__(self, loader) -> None:
            self.loader = loader
            
        def __next__(self):
            data = next(self.loader)
                        
            # x: (16, x, 224, 224)
            # y: (16, 1)
            x, y = data[0]['data'], data[0]['label']
            y = torch.squeeze(y, dim=1).type(torch.LongTensor)

            return x, y
            
    class Adapter:
        def __init__(self, loader) -> None:
            self.loader = loader
            
        def __iter__(self):
            return Iter(self.loader)

    return Adapter(train_data)


def dataloader(args):
    try: 
        print("Using DALI")
        return dali_dataloader(args)
    except ImportError:
        return pytorch_dataloader(args)

def main():
    parser = argparse.ArgumentParser(description="Torchvision models")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 64)",
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
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
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
    parser.add_argument("--data", type=str, help="data directory")
    parser.add_argument(
        "--synthetic-data", action="store_true", help="whether to use synthetic data"
    )
    parser.add_argument(
        "--fixed-batch", action="store_true", help="use a fixed batch for training"
    )
    parser.add_argument(
        "--with-amp",
        action="store_true",
        help="whether to use mixed precision with amp",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of workers",
    )

    args = parser.parse_args()
    
    if args.fixed_batch:
        # avoid unexpected side effects unless explicit
        assert args.synthetic_data, "Fixed batch needs synthetic data"

    if args.synthetic_data:
        args.data = None
    else:
        if not args.data:
            data_directory = os.environ.get("MILABENCH_DIR_DATA", None)
            assert data_directory is not None, "No data folder specified"
            args.data = os.path.join(data_directory, "FakeImageNet")

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    model = getattr(tvmodels, args.model)()
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr)

    if args.data:
        train_loader = dataloader(args)
    else:
        train_loader = SyntheticData(
            model=model,
            device=device,
            batch_size=args.batch_size,
            n=1000,
            fixed_batch=args.fixed_batch,
        )

    if args.with_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    for epoch in range(args.epochs):
        train_epoch(model, criterion, optimizer, train_loader, device, scaler=scaler)


if __name__ == "__main__":
    main()
