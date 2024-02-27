from dataclasses import dataclass

from coleo import Option, tooled

from milabench.config import _config_layers, merge


# fmt: off
@dataclass
class Arguments:
    config  : str
# fmt: on


@tooled
def arguments():
    # The name of the benchmark to develop
    config: str

    return Arguments(config)


@tooled
def cli_resolve(args=None):
    """Generate a configuration"""

    if args is None:
        args = arguments()

    overrides = {}
    configs = [args.config, overrides]

    config = {}
    for layer in _config_layers(configs):
        config = merge(config, layer)

    wip_config = {}
    parents = []

    #
    #   Only keep enabled benchmarks
    #
    for benchname, benchconfig in config.items():
        is_enabled = benchconfig.get("enabled", False)
        parent = benchconfig.get("inherits", None)

        if parent:
            parents.append(parent)

        if is_enabled:
            wip_config[benchname] = benchconfig

    #
    #   Keep the parents as well
    #
    parents = set(parents)
    for parent in parents:
        wip_config[parent] = config[parent]

    #
    # Remove resolved fields
    #
    resolved = ["dirs", "config_file", "config_base"]
    for benchname, benchconfig in wip_config.items():
        for field in resolved:
            benchconfig.pop(field, None)

    #
    # Finished
    #

    import yaml

    print(yaml.dump(wip_config))


if __name__ == "__main__":
    args = Arguments("/workspaces/milabench/config/standard.yaml")

    cli_resolve(args)
