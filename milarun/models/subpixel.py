import argparse
from math import log10

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from coleo import Argument, default
from milarun.lib import coleo_main, init_torch, iteration_wrapper, dataloop


# def is_image_file(filename):
#     return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


# def load_img(filepath):
#     img = Image.open(filepath).convert('YCbCr')
#     y, _, _ = img.split()
#     return y


# class DatasetFromFolder(torch.utils.data.Dataset):
#     def __init__(self, image_dir, input_transform=None, target_transform=None):
#         super(DatasetFromFolder, self).__init__()
#         self.image_filenames = [os.path.join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
#         self.input_transform = input_transform
#         self.target_transform = target_transform

#     def __getitem__(self, index):
#         input = load_img(self.image_filenames[index])
#         target = input.copy()
#         if self.input_transform:
#             input = self.input_transform(input)
#         if self.target_transform:
#             target = self.target_transform(target)

#         return input, target

#     def __len__(self):
#         return len(self.image_filenames)


class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.Resize(crop_size // upscale_factor),
        transforms.ToTensor(),
    ])


def target_transform(crop_size):
    return transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
    ])


def get_dataset(exp, dataset_name, upscale_factor):
    crop_size = calculate_valid_crop_size(256, upscale_factor)
    return exp.get_dataset(
        dataset_name,
        input_transform=input_transform(crop_size, upscale_factor),
        target_transform=target_transform(crop_size)
    )


@coleo_main
def main(exp):

    # Dataset to use
    dataset: Argument

    # super resolution upscale factor
    upscale_factor: Argument & int = default(2)

    # # testing batch size (default: 10)
    # test_batch_size: Argument & int = default(10)

    # Learning rate (default: 0.1)
    lr: Argument & float = default(0.1)

    # Batch size (default: 64)
    batch_size: Argument & int = default(64)

    torch_settings = init_torch()
    device = torch_settings.device

    print('===> Loading datasets')
    # dataset_instance = exp.resolve_dataset("milabench.presets:bsds500")
    # folder = dataset_instance["environment"]["root"]
    sets = get_dataset(exp, dataset, upscale_factor)
    train_set = sets.train
    # train_set = get_dataset(os.path.join(folder, "bsds500/BSR/BSDS500/data/images/train"), upscale_factor)
    # test_set = get_dataset(os.path.join(folder, "bsds500/BSR/BSDS500/data/images/test"), upscale_factor)

    training_data_loader = DataLoader(
        dataset=train_set,
        num_workers=torch_settings.workers,
        batch_size=batch_size,
        shuffle=True
    )
    # testing_data_loader = DataLoader(
    #     dataset=test_set,
    #     num_workers=torch_settings.workers,
    #     batch_size=test_batch_size,
    #     shuffle=False
    # )

    print('===> Building model')
    model = Net(upscale_factor=upscale_factor).to(device)
    model.train()
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    wrapper = iteration_wrapper(exp, sync=torch_settings.sync)
    for it, (input, target) in dataloop(training_data_loader, wrapper=wrapper):
        it.set_count(batch_size)

        input = input.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        loss = criterion(model(input), target)
        it.log(loss=loss.item())
        loss.backward()
        optimizer.step()


    # def test():
    #     avg_psnr = 0
    #     with torch.no_grad():
    #         for batch in testing_data_loader:
    #             input, target = batch[0].to(device), batch[1].to(device)

    #             prediction = model(input)
    #             mse = criterion(prediction, target)
    #             psnr = 10 * log10(1 / mse.item())
    #             avg_psnr += psnr

    #     return avg_psnr / len(testing_data_loader)
