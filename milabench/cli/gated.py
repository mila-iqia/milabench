

from collections import defaultdict
from milabench.common import arguments, _get_multipack


def cli_gated(args=None):
    """Print instruction to get access to gated models"""

    if args is None:
        args = arguments()

    benchmarks = _get_multipack(args, return_config=True)
    urls = defaultdict(list)

    for bench, config in benchmarks.items():
        tags = config.get("tags", [])

        if "gated" in tags and 'url' in config:
            urls[config["url"]].append((bench, config))


    if len(urls) > 0:
        #
        #   This match the documentation in milabench/docs/usage.rst
        #
        print("#. Setup huggingface access: benchmark use gated models or datasets")
        print("   You need to request permission to huggingface")
        print()
        print("   1. Request access to gated models")
        print()

        for url, benches in urls.items():
            names = ' '.join([k for k, _ in benches])
            print(f"      - `{names} <{url}>`_")

        print()
        print("   2. Create a new `read token <https://huggingface.co/settings/tokens/new?tokenType=read>`_ to download the models")
        print()
        print("   3. Add the token to your environment ``export MILABENCH_HF_TOKEN={your_token}``")
        print()
        print("Now you are ready to execute `milabench prepare`")
