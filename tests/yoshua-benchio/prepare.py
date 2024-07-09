#!/usr/bin/env python


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test benchmark")
    parser.add_argument(
        "--bad",
        action="store_true"
    )

    args = parser.parse_args()

    if args.bad:
        raise RuntimeError()
