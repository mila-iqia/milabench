import torch
import torchvision
import random
import time
import argparse
import os
import sys
import math
import torch.nn as nn
import json

from .fp16util import network_to_half, get_param_copy

from coleo import Argument, default
from milarun.lib import coleo_main, init_torch, iteration_wrapper, dataloop


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def get_network(net):
    classification_models = torchvision.models.__dict__
    try:
        segmentation_models = torchvision.models.segmentation.__dict__
    except AttributeError:
        segmentation_models = {}

    if net in classification_models:
        return classification_models[net]().cuda()

    if net in segmentation_models:
        return segmentation_models[net]().cuda()

    print("ERROR: not a supported model.")
    sys.exit(1)


def forwardbackward(inp, optimizer, network, target):
    optimizer.zero_grad()
    out = network(inp)
    # WIP: googlenet, deeplabv3_*, fcn_* missing log_softmax for this to work
    loss = torch.nn.functional.cross_entropy(out, target)
    loss.backward()
    optimizer.step()


def rendezvous(distributed_parameters):
    print("Initializing process group...")
    torch.distributed.init_process_group(
        backend=distributed_parameters['dist_backend'],
        init_method=distributed_parameters['dist_url'],
        rank=distributed_parameters['rank'],
        world_size=distributed_parameters['world_size']
    )
    print("Rendezvous complete. Created process group...")


@coleo_main
def main(exp):
    models = [
        'alexnet', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn',
        'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'shufflenet',
        'shufflenet_v2_x05', 'shufflenet_v2_x10', 'shufflenet_v2_x15', 'SqueezeNet',
        'SqueezeNet1.1', 'densenet121', 'densenet169', 'densenet201', 'densenet161',
        'inception', 'inception_v3', 'resnext50', 'resnext101', 'mobilenet_v2', 'googlenet',
        'deeplabv3_resnet50', 'deeplabv3_resnet101', 'fcn_resnet50', 'fcn_resnet101'
    ]

    # Network to run.
    network: Argument & str

    # Batch size (will be split among devices used by this invocation)
    batch_size: Argument & int = default(64)

    # FP16 mixed precision benchmarking
    fp16: Argument & int = default(0)

    # Use torch.nn.DataParallel api to run single process on multiple devices. Use only one of --dataparallel or --distributed_dataparallel
    dataparallel: Argument & bool = default(False)

    # Use torch.nn.parallel.DistributedDataParallel api to run on multiple processes/nodes. The multiple processes need to be launched manually, this script will only launch ONE process per invocation. Use only one of --dataparallel or --distributed_dataparallel
    distributed_dataparallel: Argument & bool = default(False)

    # Comma-separated list (no spaces) to specify which HIP devices (0-indexed) to run dataparallel or distributedDataParallel api on. Might need to use HIP_VISIBLE_DEVICES to limit visiblity of devices to different processes.
    device_ids: Argument & str = default(None)

    # Rank of this process. Required for --distributed_dataparallel
    rank: Argument & int = default(None)

    # Total number of ranks/processes. Required for --distributed_dataparallel
    world_size: Argument & int = default(None)

    # Backend used for distributed training. Can be one of 'nccl' or 'gloo'. Required for --distributed_dataparallel
    dist_backend: Argument & str = default(None)

    # url used for rendezvous of processes in distributed training. Needs to contain IP and open port of master rank0 eg. 'tcp://172.23.2.1:54321'. Required for --distributed_dataparallel
    dist_url: Argument & str = default(None)

    torch_settings = init_torch()

    if device_ids:
        device_ids_values = [int(x) for x in device_ids.split(",")]
    else:
        device_ids_values = None

    distributed_parameters = dict()
    distributed_parameters['rank'] = rank
    distributed_parameters['world_size'] = world_size
    distributed_parameters['dist_backend'] = dist_backend
    distributed_parameters['dist_url'] = dist_url

    # Some arguments are required for distributed_dataparallel
    if distributed_dataparallel:
        assert rank is not None and \
               world_size is not None and \
               dist_backend is not None and \
               dist_url is not None, "rank, world-size, dist-backend and dist-url are required arguments for distributed_dataparallel"

    wrapper = iteration_wrapper(exp, sync=torch_settings.sync)

    run_benchmarking(exp, wrapper, network, batch_size, fp16, dataparallel,
                     distributed_dataparallel, device_ids_values, distributed_parameters)


def run_benchmarking(exp, wrapper, net, batch_size, run_fp16, dataparallel,
                     distributed_dataparallel, device_ids=None,
                     distributed_parameters=None):
    if device_ids:
        torch.cuda.set_device("cuda:%d" % device_ids[0])
    else:
        torch.cuda.set_device("cuda:0")

    network = get_network(net)
    if run_fp16:
        network = network_to_half(network)

    if dataparallel:
        network = torch.nn.DataParallel(network, device_ids=device_ids)
        num_devices = len(device_ids) if device_ids is not None else torch.cuda.device_count()

    elif distributed_dataparallel:
        rendezvous(distributed_parameters)
        network = torch.nn.parallel.DistributedDataParallel(network, device_ids=device_ids)
        num_devices = len(device_ids) if device_ids is not None else torch.cuda.device_count()

    else:
        num_devices = 1

    if net == "inception_v3":
        inp = torch.randn(batch_size, 3, 299, 299, device="cuda")
    else:
        inp = torch.randn(batch_size, 3, 224, 224, device="cuda")

    if run_fp16:
        inp = inp.half()

    target = torch.randint(0, 1, size=(batch_size,), device='cuda')

    param_copy = network.parameters()
    if run_fp16:
        param_copy = get_param_copy(network)

    optimizer = torch.optim.SGD(param_copy, lr=0.01, momentum=0.9)

    rank = distributed_parameters.get('rank', -1)

    ## benchmark.
    print("INFO: running the benchmark..")
    tm = time.time()
    while not wrapper.done():
        with wrapper(count=batch_size) as it:
            forwardbackward(inp, optimizer, network, target)
            if rank <= 0:
                it.log(ndev=num_devices)

    torch.cuda.synchronize()
