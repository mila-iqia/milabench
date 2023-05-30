#!/usr/bin/env python

import torchvision


if __name__ == "__main__":
    # This will download the weights for vgg16
    torchvision.models.vgg16(pretrained=True)
