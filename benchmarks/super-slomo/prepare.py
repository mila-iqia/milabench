#!/usr/bin/env python

import torchvision



def download_celebA():
    # celebA use Google drive, and google drive wants to tell us that 
    # they cant scan for virus so the download fails
    # torchvision 0.17.1 might solve this issue though but we dont have it
    pass


if __name__ == "__main__":
    # This will download the weights for vgg16
    torchvision.models.vgg16(pretrained=True)
