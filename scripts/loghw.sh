#!/bin/bash

OUTDIR=$1

if [ -z $OUTDIR ]
then
    echo "loghw.sh: error: missing required argument OUTDIR"
    exit
fi

mkdir -p $OUTDIR
echo "# Logging hardware information in: $OUTDIR"

set -x

lshw > $OUTDIR/lshw.log
nvidia-smi -q -f $OUTDIR/nvidia.log
hwinfo > $OUTDIR/hwinfo.log
hwinfo --short > $OUTDIR/hwinfo-short.log
lsblk > $OUTDIR/lsblk.log
inxi -v7 > $OUTDIR/inxi.log

poetry run pip list > $OUTDIR/pip-list.log
