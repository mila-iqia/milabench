import contextvars
import os
import socket
from copy import deepcopy

import psutil
from copy import deepcopy
import yaml
from omegaconf import OmegaConf

from .fs import XPath
from .merge import merge


config_global = contextvars.ContextVar("config", default=None)


def relative_to(pth, cwd):
    pth = XPath(pth).expanduser()
    if not pth.is_absolute():
        pth = (XPath(cwd) / pth).resolve()
    return pth


def _config_layers(config_files):
    for config_file in config_files:
        if isinstance(config_file, dict):
            yield config_file
        else:
            config_file = XPath(config_file).absolute()
            config_base = config_file.parent
            with open(config_file) as cf:
                config = yaml.safe_load(cf)
                includes = config.pop("include", [])
                if isinstance(includes, str):
                    includes = [includes]
                yield from _config_layers(
                    relative_to(incl, config_base) for incl in includes
                )
                for v in config.values():
                    assert isinstance(v, dict)
                    v.setdefault("config_base", str(config_base))
                    v.setdefault("config_file", str(config_file))
                    v.setdefault("dirs", {})
                yield config


def resolve_inheritance(bench_config, all_configs):
    while inherit := bench_config.pop("inherits", None):
        parent = all_configs[inherit]
        tags = {*parent.get("tags", []), *bench_config.get("tags", [])}
        bench_config = merge(parent, bench_config)
        bench_config["tags"] = sorted(tags)

    if "*" in all_configs:
        bench_config = merge(bench_config, all_configs["*"])

    return bench_config


def finalize_config(name, bench_config):
    bench_config["name"] = name
    if "definition" in bench_config:
        pack = XPath(bench_config["definition"]).expanduser()
        if not pack.is_absolute():
            pack = (XPath(bench_config["config_base"]) / pack).resolve()
            bench_config["definition"] = str(pack)

    bench_config["tag"] = [bench_config["name"]]

    bench_config = OmegaConf.to_object(OmegaConf.create(bench_config))
    return bench_config


def combine_args(args, kwargs):
    if len(args) == 0:
        yield kwargs
    else:
        key, values = args.popitem()
        for value in values:
            kwargs[key] = value
            yield from combine_args(deepcopy(args), kwargs)


def expand_matrix(name, bench_config):
    if "matrix" not in bench_config:
        return [(name, bench_config)]

    arguments = deepcopy(bench_config["matrix"])
    template = bench_config["job"]

    newbenches = []

    for matrix_args in combine_args(arguments, dict()):
        newbench = deepcopy(template)
        name = newbench.pop("name").format(**matrix_args)

        for karg, varg in template["argv"].items():
            try:
                varg = varg.format(**matrix_args)
            except:
                pass
            newbench["argv"][karg] = varg

        newbenches.append((name, newbench))

    return newbenches


def build_matrix_bench(all_configs):
    expanded_config = {}

    for name, bench_config in all_configs.items():
        for k, v in expand_matrix(name, bench_config):

            if k in expanded_config:
                raise ValueError("Bench name is not unique")

            expanded_config[k] = v

    return expanded_config


def build_config(*config_files):
    all_configs = {}
    for layer in _config_layers(config_files):
        all_configs = merge(all_configs, layer)

    all_configs = build_matrix_bench(all_configs)

    for name, bench_config in all_configs.items():
        all_configs[name] = resolve_inheritance(bench_config, all_configs)

    for name, bench_config in all_configs.items():
        all_configs[name] = finalize_config(name, bench_config)

    config_global.set(all_configs)
    return all_configs
