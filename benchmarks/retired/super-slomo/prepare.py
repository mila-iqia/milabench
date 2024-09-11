#!/usr/bin/env python

import torchvision
from benchmate.datagen import generate_fakeimagenet

if __name__ == "__main__":
    # This will download the weights for vgg16
    generate_fakeimagenet()
    torchvision.models.vgg16(pretrained=True)
