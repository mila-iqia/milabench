# This is the script run by milabench run (by default)

# It is possible to use a script from a GitHub repo if it is cloned using
# clone_subtree in the benchfile.py, in which case this file can simply
# be deleted.

import argparse
import argklass


from dqn import add_dqn_command, main as dqn_main
from ppo import add_ppo_command, main as ppo_main


def main():
    parser = argklass.ArgumentParser(description="PureJaxRL")
    subparser = parser.add_subparsers(title="Benchmark", dest="benchmark")

    add_dqn_command(subparser)
    add_ppo_command(subparser)

    bench = {
        "dqn": dqn_main,
        "ppo": ppo_main
    }

    args = parser.parse_args()

    if benchmark := bench.get(args.benchmark):
        benchmark(args)

    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")


if __name__ == "__main__":
    main()
