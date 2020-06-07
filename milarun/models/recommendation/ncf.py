import os
import heapq
import math
import time
from functools import partial
from datetime import datetime
from collections import OrderedDict
from argparse import ArgumentParser

import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch import multiprocessing as mp

from . import utils
from .neumf import NeuMF

from coleo import Argument, default, auto_cli
from milarun.lib import coleo_main, init_torch, iteration_wrapper, dataloop


def predict(model, users, items, batch_size=1024, use_cuda=True):
    batches = [(users[i:i + batch_size], items[i:i + batch_size])
               for i in range(0, len(users), batch_size)]
    preds = []
    for user, item in batches:
        def proc(x):
            x = np.array(x)
            x = torch.from_numpy(x)
            if use_cuda:
                x = x.cuda(non_blocking=True)
            return torch.autograd.Variable(x)
        outp = model(proc(user), proc(item), sigmoid=True)
        outp = outp.data.cpu().numpy()
        preds += list(outp.flatten())
    return preds


def _calculate_hit(ranked, test_item):
    return int(test_item in ranked)


def _calculate_ndcg(ranked, test_item):
    for i, item in enumerate(ranked):
        if item == test_item:
            return math.log(2) / math.log(i + 2)
    return 0.


def eval_one(rating, items, model, K, use_cuda=True):
    user = rating[0]
    test_item = rating[1]
    items.append(test_item)
    users = [user] * len(items)
    predictions = predict(model, users, items, use_cuda=use_cuda)

    map_item_score = {item: pred for item, pred in zip(items, predictions)}
    ranked = heapq.nlargest(K, map_item_score, key=map_item_score.get)

    hit = _calculate_hit(ranked, test_item)
    ndcg = _calculate_ndcg(ranked, test_item)
    return hit, ndcg, len(predictions)


@coleo_main
def main(exp):
    # dataset to use
    dataset: Argument & str

    # batch size
    batch_size: Argument & int = default(128)

    # number of predictive factors
    # [alias: -f]
    factors: Argument & int = default(8)

    # size of hidden layers for MLP
    layers: Argument = default("64,32,16,8")

    # number of negative examples per interaction
    # [alias: -n]
    negative_samples: Argument & int = default(4)

    # learning rate for optimizer
    # [alias: -l]
    learning_rate: Argument & float = default(0.001)

    # rank for test examples to be considered a hit
    # [alias: -k]
    topk: Argument & int = default(10)

    layer_sizes = [int(x) for x in layers.split(",")]

    torch_settings = init_torch()
    device = torch_settings.device

    # Load Data
    # ------------------------------------------------------------------------------------------------------------------
    print('Loading data')
    with exp.time('loading_data'):
        t1 = time.time()

        train_dataset = exp.get_dataset(dataset, nb_neg=negative_samples).train

        # mlperf_log.ncf_print(key=# mlperf_log.INPUT_BATCH_SIZE, value=batch_size)
        # mlperf_log.ncf_print(key=# mlperf_log.INPUT_ORDER)  # set shuffle=True in DataLoader
        train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=torch_settings.workers,
            pin_memory=True
        )

        nb_users, nb_items = train_dataset.nb_users, train_dataset.nb_items

        print('Load data done [%.1f s]. #user=%d, #item=%d, #train=%d'
              % (time.time()-t1, nb_users, nb_items, train_dataset.mat.nnz))
    # ------------------------------------------------------------------------------------------------------------------

    # Create model
    model = NeuMF(nb_users, nb_items,
                  mf_dim=factors, mf_reg=0.,
                  mlp_layer_sizes=layer_sizes,
                  mlp_layer_regs=[0. for i in layer_sizes]).to(device)
    print(model)
    print("{} parameters".format(utils.count_parameters(model)))

    # Save model text description
    run_dir = exp.results_directory()
    with open(os.path.join(run_dir, 'model.txt'), 'w') as file:
        file.write(str(model))

    # Add optimizer and loss to graph
    # mlperf_log.ncf_print(key=# mlperf_log.OPT_LR, value=learning_rate)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8

    optimizer = torch.optim.Adam(
        model.parameters(),
        betas=(beta1, beta2),
        lr=learning_rate, eps=epsilon)

    # mlperf_log.ncf_print(key=# mlperf_log.MODEL_HP_LOSS_FN, value=# mlperf_log.BCE)
    criterion = nn.BCEWithLogitsLoss().to(device)

    model.train()

    wrapper = iteration_wrapper(exp, sync=None)

    for it, (user, item, label) in dataloop(train_dataloader, wrapper=wrapper):
        it.set_count(batch_size)

        user = torch.autograd.Variable(user, requires_grad=False).to(device)
        item = torch.autograd.Variable(item, requires_grad=False).to(device)
        label = torch.autograd.Variable(label, requires_grad=False).to(device)

        outputs = model(user, item)
        loss = criterion(outputs, label)
        it.log(loss=loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
