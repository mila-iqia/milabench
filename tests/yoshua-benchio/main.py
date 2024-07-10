#!/usr/bin/env python
import argparse
import time

import voir
from giving import give


def main():
    parser = argparse.ArgumentParser(description="Test benchmark")
    parser.add_argument(
        "--start",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--end",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--sleep",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--bad",
        action="store_true"
    )

    args = parser.parse_args()

    if args.sleep is not None:
        time.sleep(args.sleep)

    data = [[[i]] for i in range(args.start, args.end)]

    if args.bad:
        raise RuntimeError()

    for [[x]] in voir.iterate("train", data, True):
        give(loss=1 / x)


if __name__ == "__main__":
    main()
