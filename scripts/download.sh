#!/bin/bash

set -ex

milarun dataset milarun.datasets:mnist $@
milarun dataset milarun.datasets:fake_imagenet $@
milarun dataset milarun.datasets:bsds500_reso $@
milarun dataset milarun.datasets:wmt16 $@
milarun dataset milarun.datasets:wiki2 $@
milarun dataset milarun.datasets:coco $@
milarun dataset milarun.datasets:ml20m $@
