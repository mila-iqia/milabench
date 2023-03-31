import os

from .fs import XPath
from .merge import self_merge


from omegaconf import OmegaConf



def parse_config(config_file, base=None):
    config_file = XPath(config_file).absolute()
    config_base = config_file.parent
    config = OmegaConf.load(config_file)
    
    base_path = config.get('from')
    
    if base_path:
        # Handles relative path
        if base_path.startswith('.'):
            base_path = os.path.join(config_base, base_path)
        
        base_config = OmegaConf.load(base_path)
        config = OmegaConf.merge(base_config, config)

    # Save the base 
    config.setdefault("defaults", {})
    config["defaults"]["config_base"] = str(config_base)
    config["defaults"]["config_file"] = str(config_file)
    config["defaults"].setdefault("dirs", {})

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
