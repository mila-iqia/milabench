

from milabench.system import _global_options, as_environment_variable, SystemConfig


from dataclasses import asdict


def cli_env():
    _ = SystemConfig()

    # import yaml
    # print(yaml.dump(asdict(_)))

    for k, option in _global_options.items():
        env_name = as_environment_variable(k)
        value = option["value"]
        default = option["default"]

        if value is None or value == default:
            print("# ", end="")
        
        print(f"export {env_name}={value}")


if __name__ == "__main__":
    cli_env()
