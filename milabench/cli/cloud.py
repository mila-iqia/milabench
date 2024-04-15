from copy import deepcopy
import os
import socket
import subprocess
import sys

from coleo import Option, tooled
from omegaconf import OmegaConf
import yaml

from milabench.fs import XPath
from milabench.utils import blabla

from ..common import get_multipack

_SETUP = "setup"
_TEARDOWN = "teardown"
_LIST = "list"
_ACTIONS = (_SETUP, _TEARDOWN, _LIST)


def _flatten_cli_args(**kwargs):
    return sum(
        (
            (f"--{str(k).replace('_', '-')}", *([str(v)] if str(v) else []))
            for k, v in kwargs.items()
        ), ()
    )


def _or_sudo(cmd:str):
    return f"( {cmd} || sudo {cmd} )"


def manage_cloud(pack, run_on, action="setup"):
    assert run_on in pack.config["system"]["cloud_profiles"], f"{run_on} cloud profile not found in {list(pack.config['system']['cloud_profiles'].keys())}"

    key_map = {
        "hostname":(lambda v: ("ip",v)),
        "username":(lambda v: ("user",v)),
        "ssh_key_file":(lambda v: ("key",v)),
        "env":(lambda v: ("env",[".", v, ";", "conda", "activate", "milabench", "&&"])),
    }
    plan_params = deepcopy(pack.config["system"]["cloud_profiles"][run_on])
    run_on, *profile = run_on.split("__")
    profile = profile[0] if profile else ""
    default_state_prefix = profile or run_on
    default_state_id = "_".join((pack.config["hash"][:6], blabla()))

    remote_base = XPath("/data") / pack.dirs.base.name
    local_base = pack.dirs.base.absolute().parent

    nodes = iter(enumerate(pack.config["system"]["nodes"]))
    for i, n in nodes:
        if n["ip"] != "1.1.1.1":
            continue

        plan_params["state_prefix"] = plan_params.get("state_prefix", default_state_prefix)
        plan_params["state_id"] = plan_params.get("state_id", default_state_id)
        plan_params["cluster_size"] = max(len(pack.config["system"]["nodes"]), i + 1)

        import milabench.cli.covalent as cv

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
                _or_sudo(f"mkdir -p '{local_base.parent}'") +
                " && " + _or_sudo(f"chmod a+rwX '{local_base.parent}'") +
                f" && mkdir -p '{remote_base}'"
                f" && ln -sfT '{remote_base}' '{local_base}'"
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
                k, v = key_map[k](v)
                if k == "ip" and n[k] != "1.1.1.1":
                    i, n = next(nodes)
                n[k] = v
            except ValueError:
                pass

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
                f"system.cloud_profiles.{run_on}.state_prefix='*'",
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
