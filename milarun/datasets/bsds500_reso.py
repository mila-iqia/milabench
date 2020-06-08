import torch
import os
import subprocess
from os import listdir
from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


class DatasetFromFolder(torch.utils.data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [os.path.join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = input.copy()
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)


class BSDS500Reso:
    def __init__(self, path, input_transform, target_transform):
        self.path = os.path.join(path, "bsds500")
        self.input_transform = input_transform
        self.target_transform = target_transform

    def avail(self, download=True):
        if download:
            self.download()
        train_path = os.path.join(self.path, "BSR/BSDS500/data/images/train")
        self.train = DatasetFromFolder(train_path, self.input_transform, self.target_transform)

    def download(self):
        if os.path.exists(self.path):
            return
        os.makedirs(self.path, exist_ok=True)
        subprocess.run(
            f"""
            cd {self.path}
            wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz
            tar -xzf BSR_bsds500.tgz
            rm BSR_bsds500.tgz
            """,
            shell=True
        )


def bsds500_reso(path, input_transform=None, target_transform=None):
    return BSDS500Reso(path, input_transform, target_transform)
