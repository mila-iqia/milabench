from dataclasses import dataclass
import os
import os
import yaml

import numpy as np


ROOT = os.path.dirname(__file__)

default_scaling_config = os.path.join(ROOT, '..', 'config', 'scaling.yaml')


@dataclass
class AutoScalerOptions:
    size: int = os.getenv("MILABENCH_BATCH_SIZE")
    default: bool = True
    autoscale: bool = False


metric_prefixes = {
    "T": (12, 4),
    "G": (9, 3),
    "M": (6, 2),
    "k": (3, 1),
    "h": (2, None),
    "da": (1, None),
    "d": (-1, None),
    "c": (-2, None),
    "m": (-3, None),
    "u": (-6, None),
    "n": (-9, None),
    "p": (-12, None),
}


def to_octet(value: str) -> float:
    for p, (vm, vb) in metric_prefixes.items():
        if f"{p}iB" in value or f"{p}io" in value:
            return float(value[:len(p) + 1]) * 1024 ** vb
        
        if f"{p}B" in value or f"{p}o" in value:
            return float(value[:len(p) + 1]) * 10 ** vm
        
    raise float(value[:-1])


class AutoScaler:
    """Automatically scale the batch size to match GPU spec"""
    def __init__(self, options=AutoScalerOptions(), scaling_config=None):
        self.options = options
        
        if scaling_config is None:
            scaling_config = default_scaling_config
            
        with open(scaling_config, 'r') as sconf:
            self.scaling_config = yaml.safe_load(sconf)
    
    def auto_size(self, benchmark, capacity):
        
        if isinstance(capacity, str):
            capacity = to_octet(capacity)
        
        config = self.scaling_config[benchmark]
        
        mem = [to_octet(v) for v in config["model"].values()]
        size = [float(v) for v in config["model"].keys()]
        
        return np.interp(capacity, mem, size)
    
    def size(self, benchmark, capacity):
        
        if self.options.autoscale:
            
            if self.options.default:
                config = self.scaling_config[benchmark]
                return config["default"]
            
            return self.auto_size(self, benchmark, capacity)
            
        return None
    
    def scale(self, benchmark, capacity, argv):
        """Find the batch size and override it with a new value"""
        
        newsize = self.size(benchmark, capacity)
        
        if newsize is None:
            return argv
        
        # <param> <value>
        argv = list(argv)
        config = self.scaling_config[benchmark]
        
        for i, arg in enumerate(argv):
            if arg == config["arg"]:
                break
        else:
            # add the new argument
            return argv + [config["arg"], str(newsize)]

        argv[i + 1] = str(newsize)
        return argv
        
    

if __name__ == "__main__":
    print(AutoScaler().size("llama", "18Go"))
