import argparse

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

    args = parser.parse_args()

    data = [[[i]] for i in range(args.start, args.end)]

    for [[x]] in voir.iterate("train", data, True):
        give(loss=1 / x)


if __name__ == "__main__":
    main()
