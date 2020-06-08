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
        os.makedirs(self.path, exist_ok=True)
        subprocess.run(
            f"""
            cd {self.path}

            wget http://images.cocodataset.org/zips/train2017.zip
            unzip -u train2017.zip
            rm train2017.zip

            wget http://images.cocodataset.org/zips/val2017.zip
            unzip -u val2017.zip
            rm val2017.zip

            wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
            unzip -u annotations_trainval2017.zip
            rm annotations_trainval2017.zip
            """,
            shell=True
        )


def coco(path, train_transform=None, val_transform=None):
    return Coco(path, train_transform, val_transform)
