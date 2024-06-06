import argparse
import os

import torch
import torch.cuda.amp
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchcompat.core as accelerator


def generate_tensors(batch_size, shapes, device):
    """
    Examples
    --------

    >>> generate_tensors(128, [(3, 224, 224), (1000)], "cuda")
    [Tensor((128, 3, 224, 223)), Tensor(128, 1000)]

    >>> generate_tensors(128, [("x", (3, 224, 224)), ("y", (1000,))], "cuda")
    {"x": Tensor((128, 3, 224, 223))  "y": Tensor(128, 1000)}
    """
    tensors = []
    if len(shapes[0]) == 2:
        tensors = dict()
    
    for kshape in shapes:
        if len(kshape) == 2:
            key, shape = kshape
            tensors[key] = torch.randn((batch_size, *shape), device=device)
        else:
            tensors.append(torch.randn((batch_size, *kshape), device=device)) 
    
    return tensors


def generate_tensor_classification(model, batch_size, in_shape, device):
    model = model.to(device)
    inp = torch.randn((batch_size, *in_shape), device=device)
    out = torch.rand_like(model(inp))
    return inp, out


class SyntheticData:
    def __init__(self, tensors, n, fixed_batch):
        self.n = n
        self.tensors = tensors
        self.fixed_batch = fixed_batch

    def __iter__(self):
        if self.fixed_batch:
            for _ in range(self.n):
                yield self.tensors
        
        else:
            for _ in range(self.n):
                yield [torch.rand_like(t) for t in self.tensors]

    def __len__(self):
        return self.n


def dali(folder, batch_size, num_workers):
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
    from nvidia.dali.plugin.pytorch import DALIGenericIterator

    @pipeline_def(num_threads=num_workers, device_id=0)
    def get_dali_pipeline():
        images, labels = fn.readers.file(
            file_root=folder, 
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
        [get_dali_pipeline(batch_size=batch_size)],
        ['data', 'label'],
        reader_name='Reader'
    )

    class Adapter:
        def __init__(self, iter):
            self.iter = iter

        def __len__(self):
            return len(self.iter)
        
        def __iter__(self):
            for data in self.iter:
                x, y = data[0]['data'], data[0]['label']
                yield x, torch.squeeze(y, dim=1).type(torch.LongTensor)

    return Adapter(train_data)


def pytorch(folder, batch_size, num_workers):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    data_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
        
    train = datasets.ImageFolder(os.path.join(folder, "train"), data_transforms)
    return torch.utils.data.DataLoader(
        train,
        batch_size=batch_size,
        num_workers=num_workers,
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


def synthetic(model, batch_size, fixed_batch):
    return SyntheticData(
        tensors=generate_tensor_classification(
            model, 
            batch_size, 
            (3, 244, 244), 
            device=accelerator.fetch_device(0)
        ),
        n=1000,
        fixed_batch=fixed_batch,
    )

def synthetic_fixed(*args):
    return synthetic(*args, fixed_batch=True)


def synthetic_random(*args):
    return synthetic(*args, fixed_batch=False)


def dataloader_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--loader", type=str,  help="Dataloader implementation", 
        default="pytorch"
    )
    parser.add_argument(
        "--num-workers", type=int, default=8,
        help="number of workers for data loading",
    )
    parser.add_argument( 
        "--data", type=str, default=os.environ.get("MILABENCH_DIR_DATA", None),
        help="data directory"
    )
    parser.add_argument(
        "--synthetic-data", action="store_true", 
        help="whether to use synthetic data"
    )
    parser.add_argument(
        "--fixed-batch", action="store_true", 
        help="use a fixed batch for training"
    )

def imagenet_dataloader(args, model):
    if not args.data:
        data_directory = os.environ.get("MILABENCH_DIR_DATA", None)
        if data_directory:
            args.data = os.path.join(data_directory, "FakeImageNet")

    if args.fixed_batch:
        args.synthetic_data = True

    if args.synthetic_data:
        args.data = None

    if args.data:
        folder = os.path.join(args.data, "train")

        if args.loader == "dali":
            return dali(folder, args.batch_size, args.num_workers)
        
        return pytorch(folder, args.batch_size, args.num_workers)

    return synthetic(
        model=model,
        batch_size=args.batch_size,
        fixed_batch=args.fixed_batch
    )