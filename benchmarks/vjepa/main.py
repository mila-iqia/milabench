# This is the script run by milabench run (by default)

# It is possible to use a script from a GitHub repo if it is cloned using
# clone_subtree in the benchfile.py, in which case this file can simply
# be deleted.

import time
import random
from argparse import Namespace
import torchcompat.core as accelerator
from benchmate.observer import BenchObserver
import os
import argparse
import multiprocessing as mp
import sys
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'jepa'))
from jepa.app.main import process_main

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--fname', type=str,
        help='name of config file to load',
        default='./config/vith16.yaml')
    parser.add_argument(
        '--devices', type=str, nargs='+', default=['cuda:0'],
        help='which devices to use on local machine')
    return parser

def main():
    observer = BenchObserver(batch_size_fn=lambda batch: 1)
    criterion = observer.criterion(criterion)
    device = accelerator.fetch_device(0) # <= This is your cuda device

    # Change this to use torch run
    args = arg_parser()
    num_gpus = len(args.devices)
    mp.set_start_method('spawn')
    for rank in range(num_gpus):
        mp.Process(
            target=process_main,
            args=(rank, args.fname, num_gpus, args.devices)
        ).start()
    


if __name__ == "__main__":
    main()
