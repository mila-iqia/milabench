import yaml

from .fs import XPath
from .merge import self_merge


def parse_config(config_file, base=None):
    config_file = XPath(config_file).absolute()
    config_base = config_file.parent
    with open(config_file) as cf:
        config = yaml.safe_load(cf)
    config.setdefault("defaults", {})
    config["defaults"]["config_base"] = str(config_base)
    config["defaults"]["config_file"] = str(config_file)

    if base is not None:
        config["defaults"]["dirs"]["base"] = base

    config = self_merge(config)

    for name, defn in config["benchmarks"].items():
        defn.setdefault("name", name)
        defn.setdefault("group", name)
        defn["tag"] = [defn["name"]]

        pack = XPath(defn["definition"]).expanduser()
        if not pack.is_absolute():
            pack = (XPath(defn["config_base"]) / pack).resolve()
            defn["definition"] = str(pack)

    return config
