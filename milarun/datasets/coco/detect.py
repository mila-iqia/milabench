import torch
import torch.utils.data as data
import bz2
import json
import time
import os
from PIL import Image

# Implement a datareader for COCO dataset
class COCODetection(data.Dataset):
    def __init__(self, img_folder, annotate_file, transform=None):
        self.img_folder = img_folder
        self.annotate_file = annotate_file

        # Start processing annotation
        with open(annotate_file) as fin:
            self.data = json.load(fin)

        self.images = {}

        self.label_map = {}
        self.label_info = {}
        # print("Parsing COCO data...")
        start_time = time.time()
        # 0 stand for the background
        cnt = 0
        self.label_info[cnt] = "background"
        for cat in self.data["categories"]:
            cnt += 1
            self.label_map[cat["id"]] = cnt
            self.label_info[cnt] = cat["name"]

        # build inference for images
        for img in self.data["images"]:
            img_id = img["id"]
            img_name = img["file_name"]
            img_size = (img["height"], img["width"])
            # print(img_name)
            if img_id in self.images: raise Exception("duplicated image record")
            self.images[img_id] = (img_name, img_size, [])

        # read bboxes
        for bboxes in self.data["annotations"]:
            img_id = bboxes["image_id"]
            category_id = bboxes["category_id"]
            bbox = bboxes["bbox"]
            bbox_label = self.label_map[bboxes["category_id"]]
            self.images[img_id][2].append((bbox, bbox_label))

        for k, v in list(self.images.items()):
            if len(v[2]) == 0:
                # print("empty image: {}".format(k))
                self.images.pop(k)

        self.img_keys = list(self.images.keys())
        self.transform = transform
        # print("End parsing COCO data, total time {}".format(time.time()-start_time))

    @property
    def labelnum(self):
        return len(self.label_info)

    @staticmethod
    def load(pklfile):
        # print("Loading from {}".format(pklfile))
        with bz2.open(pklfile, "rb") as fin:
            ret = pickle.load(fin)
        return ret

    def save(self, pklfile):
        # print("Saving to {}".format(pklfile))
        with bz2.open(pklfile, "wb") as fout:
            pickle.dump(self, fout)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_id = self.img_keys[idx]
        img_data = self.images[img_id]
        fn = img_data[0]
        img_path = os.path.join(self.img_folder, fn)
        img = Image.open(img_path).convert("RGB")

        htot, wtot = img_data[1]
        bbox_sizes = []
        bbox_labels = []

        # for (xc, yc, w, h), bbox_label in img_data[2]:
        for (l, t, w, h), bbox_label in img_data[2]:
            r = l + w
            b = t + h
            # l, t, r, b = xc - 0.5*w, yc - 0.5*h, xc + 0.5*w, yc + 0.5*h
            bbox_size = (l / wtot, t / htot, r / wtot, b / htot)
            bbox_sizes.append(bbox_size)
            bbox_labels.append(bbox_label)

        bbox_sizes = torch.tensor(bbox_sizes)
        bbox_labels = torch.tensor(bbox_labels)

        if self.transform != None:
            img, (htot, wtot), bbox_sizes, bbox_labels = \
                self.transform(img, (htot, wtot), bbox_sizes, bbox_labels)
        else:
            pass

        return img, (htot, wtot), bbox_sizes, bbox_labels

    # Implement a datareader for VOC dataset
