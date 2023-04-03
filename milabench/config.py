import yaml

from .fs import XPath
from .merge import self_merge


def load_meta(config_base, metafile):
    with open(config_base / metafile) as cf:
        config = yaml.safe_load(cf)
        
    return config


def insert_meta(meta, defn):
    name = defn["name"]
    metadata = meta.get(name, dict())
    for k, v in metadata.items():
        if k not in defn:
            defn[k] = v


def parse_config(config_file, base=None):
    config_file = XPath(config_file).absolute()
    config_base = config_file.parent
    with open(config_file) as cf:
        config = yaml.safe_load(cf)
        
    config.setdefault("defaults", {})
    config["defaults"]["config_base"] = str(config_base)
    config["defaults"]["config_file"] = str(config_file)
    config["defaults"].setdefault("dirs", {})

    if base is not None:
        config["defaults"]["dirs"]["base"] = base

    config = self_merge(config)
    metafile = config['defaults'].get('benchmeta')
    meta = load_meta(config_base, metafile)

    for name, defn in config["benchmarks"].items():
        defn.setdefault("name", name)
        defn.setdefault("group", name)
        
        name = defn["name"]
        defn["tag"] = [name]
        
        insert_meta(meta, defn)
    
        pack = XPath(defn["definition"]).expanduser()
        if not pack.is_absolute():
            pack = (XPath(defn["config_base"]) / pack).resolve()
            defn["definition"] = str(pack)

    return config
