"""Print instructions to get access to gated models."""

from collections import defaultdict
from dataclasses import dataclass

from argklass.command import Command
from argklass.arguments import group

from ...common import CommonArguments, _get_multipack


def _print_gated_info(args):
    """Print info about gated model requirements.

    Accepts any object with CommonArguments fields (parsed args namespace).
    Also called from install.py.
    """
    benchmarks = _get_multipack(args, return_config=True)
    urls = defaultdict(list)

    for bench, config in benchmarks.items():
        tags = config.get("tags", [])

        if "gated" in tags and "url" in config:
            urls[config["url"]].append((bench, config))

    if len(urls) > 0:
        print("#. Setup huggingface access: benchmark use gated models or datasets")
        print("   You need to request permission to huggingface")
        print()
        print("   1. Request access to gated models")
        print()

        for url, benches in urls.items():
            names = " ".join([k for k, _ in benches])
            print(f"      - `{names} <{url}>`_")

        print()
        print(
            "   2. Create a new `read token <https://huggingface.co/settings/tokens/new?tokenType=read>`_"
            " to download the models"
        )
        print()
        print(
            "   3. Add the token to your environment"
            " ``export MILABENCH_HF_TOKEN={your_token}``"
        )
        print()
        print("Now you are ready to execute `milabench prepare`")


class Gated(Command):
    """Print instructions to get access to gated models."""

    name = "gated"

    # fmt: off
    @dataclass
    class Arguments:
        """Print instructions to get access to gated models."""
        shared : CommonArguments = group(CommonArguments)
    # fmt: on

    @staticmethod
    def execute(args):
        _print_gated_info(args)


COMMANDS = Gated
