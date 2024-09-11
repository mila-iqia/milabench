from copy import deepcopy
import os
import subprocess
import sys
import warnings

from coleo import Option, tooled
from omegaconf import OmegaConf
import yaml

from milabench.fs import XPath
from milabench.utils import blabla

from .. import ROOT_FOLDER
from ..common import get_multipack

_SETUP = "setup"
_TEARDOWN = "teardown"
_LIST = "list"
_ACTIONS = (_SETUP, _TEARDOWN, _LIST)


def _flatten_cli_args(**kwargs):
    return sum(
        (
            (f"--{str(k).replace('_', '-')}", *([str(v)] if v is not None else []))
            for k, v in kwargs.items()
        ), ()
    )


def _or_sudo(cmd:str):
    return f"( {cmd} || sudo {cmd} )"


def _get_common_dir(first_dir:XPath, second_dir:XPath):
    f_parents, s_parents = (
        list(reversed((first_dir / "_").parents)),
        list(reversed((second_dir / "_").parents))
    )
    f_parents, s_parents = (
        f_parents[:min(len(f_parents), len(s_parents))],
        s_parents[:min(len(f_parents), len(s_parents))]
    )
    while f_parents != s_parents:
        f_parents = f_parents[:-1]
        s_parents = s_parents[:-1]
    if f_parents[-1] == XPath("/"):
        # no common dir
        return None
    else:
        return f_parents[-1]


def manage_cloud(pack, run_on, action="setup"):
    assert run_on in pack.config["system"]["cloud_profiles"], f"{run_on} cloud profile not found in {list(pack.config['system']['cloud_profiles'].keys())}"

    key_map = {
        "hostname":(lambda v: ("ip",v)),
        "private_ip":(lambda v: ("internal_ip",v)),
        "username":(lambda v: ("user",v)),
        "ssh_key_file":(lambda v: ("key",v)),
        # "env":(lambda v: ("env",[".", v, ";", "conda", "activate", "milabench", "&&"])),
    }
    plan_params = deepcopy(pack.config["system"]["cloud_profiles"][run_on])
    run_on, *profile = run_on.split("__")
    profile = profile[0] if profile else ""
    default_state_prefix = profile or run_on
    default_state_id = "_".join((pack.config["hash"][:6], blabla()))

    local_base = pack.dirs.base.absolute()
    local_data_dir = _get_common_dir(ROOT_FOLDER.parent, local_base.parent)
    if local_data_dir is None:
        local_data_dir = local_base.parent
    remote_data_dir = XPath("/data") / local_data_dir.name

    nodes = iter(enumerate(pack.config["system"]["nodes"]))
    for i, n in nodes:
        if n["ip"] != "1.1.1.1":
            continue

        plan_params["state_prefix"] = plan_params.get("state_prefix", default_state_prefix)
        plan_params["state_id"] = plan_params.get("state_id", default_state_id)
        plan_params["cluster_size"] = max(len(pack.config["system"]["nodes"]), i + 1)
        plan_params["keep_alive"] = None

        import milabench.scripts.covalent as cv

        subprocess.run(
            [
                sys.executable,
                "-m", cv.__name__,
                "serve", "start"
            ]
            , stdout=sys.stderr
            , check=True
        )

        cmd = [
            sys.executable,
            "-m", cv.__name__,
            run_on,
            f"--{action}",
            *_flatten_cli_args(**plan_params)
        ]
        if action == _SETUP:
            cmd += [
                "--",
                "bash", "-c",
                _or_sudo(f"mkdir -p '{local_data_dir.parent}'") +
                " && " + _or_sudo(f"chmod a+rwX '{local_data_dir.parent}'") +
                f" && mkdir -p '{remote_data_dir}'"
                f" && ln -sfT '{remote_data_dir}' '{local_data_dir}'"
            ]
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        stdout_chunks = []
        while True:
            line = p.stdout.readline()
            if not line:
                break
            line_str = line.decode("utf-8").strip()
            stdout_chunks.append(line_str)
            print(line_str, file=sys.stderr)

            if not line_str:
                continue
            try:
                k, v = line_str.split("::>")
            except ValueError:
                continue
            try:
                k, v = key_map[k](v)
            except KeyError:
                warnings.warn(f"Ignoring invalid key received: {k}:{v}")
                continue
            if k == "ip" and n[k] != "1.1.1.1":
                i, n = next(nodes)
            n[k] = v

        _, stderr = p.communicate()
        stderr = stderr.decode("utf-8").strip()
        print(stderr, file=sys.stderr)

        if p.returncode != 0:
            stdout = os.linesep.join(stdout_chunks)
            raise subprocess.CalledProcessError(
                p.returncode,
                cmd,
                stdout,
                stderr
            )

    return pack.config["system"]


@tooled
def _setup():
    """Setup a cloud infrastructure"""

    # Setup cloud on target infra
    run_on: Option & str

    mp = get_multipack()
    setup_pack = mp.setup_pack()
    system_config = manage_cloud(setup_pack, run_on, action=_SETUP)
    del system_config["arch"]

    print(f"# hash::>{setup_pack.config['hash']}")
    print(yaml.dump({"system": system_config}))


@tooled
def _teardown():
    """Teardown a cloud infrastructure"""

    # Teardown cloud instance on target infra
    run_on: Option & str

    # Teardown all cloud instances
    all: Option & bool = False

    overrides = {}
    if all:
        overrides = {
            "*": OmegaConf.to_object(OmegaConf.from_dotlist([
                f"system.cloud_profiles.{run_on}.state_id='*'",
            ]))
        }

    mp = get_multipack(overrides=overrides)
    setup_pack = mp.setup_pack()
    manage_cloud(setup_pack, run_on, action=_TEARDOWN)


@tooled
def cli_cloud():
    """Manage cloud instances."""

    # Setup a cloud infrastructure
    setup: Option & bool = False
    # Teardown a cloud infrastructure
    teardown: Option & bool = False

    assert any((setup, teardown)) and not all((setup, teardown))

    if setup:
        _setup()
    elif teardown:
        _teardown()
