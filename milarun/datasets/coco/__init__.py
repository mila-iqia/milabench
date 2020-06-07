import os

from .coco import COCO
from .detect import COCODetection


class Coco:

    def __init__(self, path, train_transform, val_transform):
        self.path = os.path.join(path, "coco")
        self.train_transform = train_transform
        self.val_transform = val_transform

    def avail(self, download=True):
        if download:
            self.download()
        val_annotate = os.path.join(self.path, "annotations/instances_val2017.json")
        val_coco_root = os.path.join(self.path, "val2017")
        train_annotate = os.path.join(self.path, "annotations/instances_train2017.json")
        train_coco_root = os.path.join(self.path, "train2017")
        self.coco = COCO(annotation_file=val_annotate)
        self.train = COCODetection(train_coco_root, train_annotate, self.train_transform)
        self.val = COCODetection(val_coco_root, val_annotate, self.val_transform)

    def download(self):
        if os.path.exists(self.path):
            return
        raise Exception("TODO")


def coco(path, train_transform, val_transform):
    return Coco(path, train_transform, val_transform)
