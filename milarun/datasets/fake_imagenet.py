import argparse
import os
import sys
import multiprocessing
from tqdm import tqdm
from torchvision.datasets import FakeData
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def write(args):
    image_size, count, offset, outdir = args
    dataset = FakeData(
        size=count,
        image_size=image_size,
        num_classes=1000,
        random_offset=offset
    )

    for i, (image, y) in tqdm(enumerate(dataset), total=count):

        class_val = int(y.item())
        image_name = f'{i}.jpeg'

        path = os.path.join(outdir, str(class_val))
        os.makedirs(path, exist_ok=True)

        image_path = os.path.join(path, image_name)
        image.save(image_path)

        if i > count:
            break


def generate(image_size, n, outdir):
    p_count = min(multiprocessing.cpu_count(), 4)
    count = n // p_count
    offset_list = [(image_size, count, i, outdir) for i in range(0, n, count)]
    pool = multiprocessing.Pool(p_count)
    pool.map(write, offset_list)


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])


class FakeImagenet:
    def __init__(self, path, transform):
        self.path = os.path.join(path, "ImageNet")
        self.size = 512
        self.n_train = 100000 // 100
        self.n_val = 5000 // 100
        self.n_test = 5000 // 100
        self.transform = transform

    def avail(self, download=True):
        if download:
            self.download()
        self.train = datasets.ImageFolder(
            os.path.join(self.path, "train"),
            self.transform
        )
        self.val = datasets.ImageFolder(
            os.path.join(self.path, "val"),
            self.transform
        )
        self.test = datasets.ImageFolder(
            os.path.join(self.path, "test"),
            self.transform
        )

    def download(self):
        if os.path.exists(self.path):
            return
        image_size = (3, self.size, self.size)
        generate(image_size, self.n_train, os.path.join(self.path, "train"))
        generate(image_size, self.n_val, os.path.join(self.path, "val"))
        generate(image_size, self.n_test, os.path.join(self.path, "test"))


def fake_imagenet(path, transform=data_transforms):
    return FakeImagenet(path, transform)
