import sys

import baselines


def main():
    args = sys.argv
    print(args)
    baselines.run.main(args)


if __name__ == "__main__":
    main()
