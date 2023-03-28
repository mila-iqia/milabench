#!/usr/bin/env python

from main import parser
from models import models

if __name__ == "__main__":
    args = parser().parse_args()
    print(f"Preparing {args.model}")
    make_config = models[args.model]
    make_config()
