# This is the script run by milabench run (by default)

# It is possible to use a script from a GitHub repo if it is cloned using
# clone_subtree in the benchfile.py, in which case this file can simply
# be deleted.

import argparse


def main():
    parser = argparse.ArgumentParser(description="PureJaxRL")
    parser.add_argument(
        "-b",
        "--benchmark",
        type=str,
        choices=["dqn", "ppo"],
        default="dqn",
        help="Benchmark to run",
    )
    args = parser.parse_args()
    if args.benchmark == "dqn":
        from benchmarks.purejaxrl.dqn import main

        main()
    elif args.benchmark == "ppo":
        from benchmarks.purejaxrl.ppo import main

        main()
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")


if __name__ == "__main__":
    main()
