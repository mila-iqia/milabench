import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import time

from coleo import Argument, default, auto_cli
from milarun.lib import coleo_main, init_torch, iteration_wrapper, dataloop


def _set_cudnn_benchmark():
    cudnn.benchmark = True


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, ngpu, nz, nc, ngf):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, ngpu, nz, nc, ndf):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


@coleo_main
def main(exp):

    # dataset to use
    dataset: Argument & str

    # Number of examples per batch
    batch_size: Argument & int = default(64)

    # the height / width of the input image to network
    image_size: Argument & int = default(64)

    # size of the latent z vector
    nz: Argument & int = default(100)
    ngf: Argument & int = default(64)
    ndf: Argument & int = default(64)

    # learning rate, default is 0.0002
    lr: Argument & float = default(0.0002)

    # beta1 for adam. default=0.5
    beta1: Argument & float = default(0.5)

    # Number of gpus to use
    ngpu: Argument & int = default(None)

    # path to netG (to continue training)
    saved_netG: Argument = default(None)

    # path to netD (to continue training)
    saved_netD: Argument = default(None)

    torch_settings = init_torch()
    device = torch_settings.device

    if torch_settings.cuda and ngpu is None:
        ngpu_value = torch.cuda.device_count()
    else:
        ngpu_value = 1

    real_batch_size = batch_size * ngpu_value
    _set_cudnn_benchmark()

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = exp.get_dataset(dataset, transform).train
    nc = 3

    assert train_dataset
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(torch_settings.workers)
    )

    netG = Generator(ngpu_value, nz, nc, ngf).to(device)
    netG.apply(weights_init)
    if saved_netG:
        netG.load_state_dict(torch.load(saved_netG))

    netD = Discriminator(ngpu_value, nz, nc, ndf).to(device)
    netD.apply(weights_init)
    if saved_netD:
        netD.load_state_dict(torch.load(saved_netD))

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    compute_time = 0
    compute_count = 0

    wrapper = iteration_wrapper(exp, sync=torch_settings.sync)

    for it, data in dataloop(dataloader, wrapper=wrapper):
        it.set_count(real_batch_size)

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        data = [d.to(device) for d in data]

        batch_compute_start = time.time()
        netD.zero_grad()
        real_cpu = data[0].to(device)
        bsize = real_cpu.size(0)
        label = torch.full((bsize,), real_label, device=device)

        output = netD(real_cpu)
        errD_real = criterion(output, label)
        # exp.log_metric('errD_real', errD_real.item())
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(bsize, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        # exp.log_metric('errG_fake', errD_fake.item())
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        # exp.log_metric('errG', errG.item())
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        it.log(
            errD_real=errD_real.item(),
            errD_fake=errD_fake.item(),
            errG=errG.item(),
        )
