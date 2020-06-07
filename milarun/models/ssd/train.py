import os
from argparse import ArgumentParser
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
import numpy as np
from types import SimpleNamespace as NS

from .base_model import Loss
from .ssd300 import SSD300
from .utils import DefaultBoxes, Encoder, SSDTransformer

from coleo import Argument, default
from milarun.lib import coleo_main, init_torch, iteration_wrapper, dataloop


def show_memusage(device=0):
    import gpustat
    gpu_stats = gpustat.GPUStatCollection.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    print("{}/{}".format(item["memory.used"], item["memory.total"]))


def dboxes300_coco():
    figsize = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    # mlperf_log.ssd_print(key=# mlperf_log.FEATURE_SIZES, value=feat_size)

    steps = [8, 16, 32, 64, 100, 300]
    # mlperf_log.ssd_print(key=# mlperf_log.STEPS, value=steps)

    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [21, 45, 99, 153, 207, 261, 315]
    # mlperf_log.ssd_print(key=# mlperf_log.SCALES, value=scales)

    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    # mlperf_log.ssd_print(key=# mlperf_log.ASPECT_RATIOS, value=aspect_ratios)

    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    # mlperf_log.ssd_print(key=# mlperf_log.NUM_DEFAULTS, value=len(dboxes.default_boxes))
    return dboxes


def coco_eval(model, coco, cocoGt, encoder, inv_map, threshold,
              iteration, use_cuda=True):
    from pycocotools.cocoeval import COCOeval
    print("")
    model.eval()
    if use_cuda:
        model.cuda()
    ret = []

    overlap_threshold = 0.50
    nms_max_detections = 200

    start = time.time()
    for idx, image_id in enumerate(coco.img_keys):
        img, (htot, wtot), _, _ = coco[idx]

        with torch.no_grad():
            print("Parsing image: {}/{}".format(idx+1, len(coco)), end="\r")
            inp = img.unsqueeze(0)
            if use_cuda:
                inp = inp.cuda()
            ploc, plabel = model(inp)

            try:
                result = encoder.decode_batch(ploc, plabel,
                                              overlap_threshold,
                                              nms_max_detections)[0]

            except:
                #raise
                print("")
                print("No object detected in idx: {}".format(idx))
                continue

            loc, label, prob = [r.cpu().numpy() for r in result]
            for loc_, label_, prob_ in zip(loc, label, prob):
                ret.append([image_id, loc_[0]*wtot, \
                                      loc_[1]*htot,
                                      (loc_[2] - loc_[0])*wtot,
                                      (loc_[3] - loc_[1])*htot,
                                      prob_,
                                      inv_map[label_]])
    print("")
    print("Predicting Ended, total time: {:.2f} s".format(time.time()-start))

    cocoDt = cocoGt.loadRes(np.array(ret))

    E = COCOeval(cocoGt, cocoDt, iouType='bbox')
    E.evaluate()
    E.accumulate()
    E.summarize()
    print("Current AP: {:.5f} AP goal: {:.5f}".format(E.stats[0], threshold))

    # put your model back into training mode
    model.train()

    current_accuracy = E.stats[0]
    return current_accuracy>= threshold #Average Precision  (AP) @[ IoU=050:0.95 | area=   all | maxDets=100 ]


def train300_mlperf_coco(exp, args):

    torch.backends.cudnn.benchmark = True

    device = args.torch_settings.device

    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)

    input_size = 300
    train_trans = SSDTransformer(dboxes, (input_size, input_size), val=False)
    val_trans = SSDTransformer(dboxes, (input_size, input_size), val=True)
    # mlperf_log.ssd_print(key=# mlperf_log.INPUT_SIZE, value=input_size)

    # val_annotate = os.path.join(args.data, "annotations/instances_val2017.json")
    # val_coco_root = os.path.join(args.data, "val2017")
    # train_annotate = os.path.join(args.data, "annotations/instances_train2017.json")
    # train_coco_root = os.path.join(args.data, "train2017")

    # cocoGt = COCO(annotation_file=val_annotate)
    # val_coco = COCODetection(val_coco_root, val_annotate, val_trans)
    # train_coco = COCODetection(train_coco_root, train_annotate, train_trans)
    coco_dataset = exp.get_dataset(
        args.dataset,
        train_transform=train_trans,
        val_transform=val_trans
    )
    cocoGt = coco_dataset.coco
    val_coco = coco_dataset.val
    train_coco = coco_dataset.train

    #print("Number of labels: {}".format(train_coco.labelnum))
    train_dataloader = DataLoader(train_coco, batch_size=args.batch_size, shuffle=True, num_workers=4)
    # set shuffle=True in DataLoader
    # mlperf_log.ssd_print(key=# mlperf_log.INPUT_SHARD, value=None)
    # mlperf_log.ssd_print(key=# mlperf_log.INPUT_ORDER)
    # mlperf_log.ssd_print(key=# mlperf_log.INPUT_BATCH_SIZE, value=args.batch_size)

    ssd300 = SSD300(train_coco.labelnum)
    if args.checkpoint is not None:
        print("loading model checkpoint", args.checkpoint)
        od = torch.load(args.checkpoint)
        ssd300.load_state_dict(od["model"])

    ssd300.train()
    ssd300 = ssd300.to(device)
    loss_func = Loss(dboxes).to(device)

    current_lr = 1e-3
    current_momentum = 0.9
    current_weight_decay = 5e-4

    optim = torch.optim.SGD(
        ssd300.parameters(),
        lr=current_lr,
        momentum=current_momentum,
        weight_decay=current_weight_decay
    )

    # mlperf_log.ssd_print(key=# mlperf_log.OPT_NAME, value="SGD")
    # mlperf_log.ssd_print(key=# mlperf_log.OPT_LR, value=current_lr)
    # mlperf_log.ssd_print(key=# mlperf_log.OPT_MOMENTUM, value=current_momentum)
    # mlperf_log.ssd_print(key=# mlperf_log.OPT_WEIGHT_DECAY,  value=current_weight_decay)

    avg_loss = 0.0
    inv_map = {v:k for k,v in val_coco.label_map.items()}

    # mlperf_log.ssd_print(key=# mlperf_log.TRAIN_LOOP)

    train_loss = 0
    for it, (img, img_size, bbox, label) in dataloop(train_dataloader, wrapper=args.wrapper):
        it.set_count(args.batch_size)

        img = Variable(img.to(device), requires_grad=True)

        ploc, plabel = ssd300(img)

        trans_bbox = bbox.transpose(1,2).contiguous()

        trans_bbox = trans_bbox.to(device)
        label = label.to(device)

        gloc = Variable(trans_bbox, requires_grad=False)
        glabel = Variable(label, requires_grad=False)

        loss = loss_func(ploc, plabel, gloc, glabel)

        if not np.isinf(loss.item()):
            avg_loss = 0.999 * avg_loss + 0.001 * loss.item()

        it.log(loss=loss.item())

        optim.zero_grad()
        loss.backward()
        optim.step()


@coleo_main
def main(exp):

    # dataset to use
    dataset: Argument

    # batch size
    batch_size: Argument & int = default(32)

    # path to model checkpoint file
    checkpoint: Argument = default(None)

    torch_settings = init_torch()
    wrapper = iteration_wrapper(exp, sync=torch_settings.sync)

    args = NS(
        dataset=dataset,
        checkpoint=checkpoint,
        batch_size=batch_size,
        torch_settings=torch_settings,
        wrapper=wrapper,
    )
    train300_mlperf_coco(exp, args)
