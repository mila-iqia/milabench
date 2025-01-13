


from milabench.common import arguments, _get_multipack


def cli_gated():
    args = arguments()

    benchmarks = _get_multipack(args, return_config=True)
    gated_bench = []

    for bench, config in benchmarks.items():
        tags = config.get("tags", [])

        if "gated" in tags and 'url' in config:
            gated_bench.append((bench, config))

    if len(gated_bench) > 0:
        print("benchmark use gated models or datasets")
        print("You need to request permission to huggingface")
        print()
        for bench, config in gated_bench:
            print(f"{bench}")
            print(f"    url: {config.get('url')}")

        print()
        print("Create a new token")
        print("    - https://huggingface.co/settings/tokens/new?tokenType=read")
        print("")
        print("Add your token to your environment")
        print("    export MILABENCH_HF_TOKEN={your_token}")
        print("")
        print("Now you are ready to execute `milabench prepare`")
