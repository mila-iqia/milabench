#!/usr/bin/env python

from bench.__main__ import parser
from bench.models import models

if __name__ == "__main__":
    args = parser().parse_args()
    print(f"Preparing {args.model}")
    make_config = models[args.model]
    make_config(args)

    # bert dataset
    # t5 dataset
    # reformer dataset
    # whisper dataset