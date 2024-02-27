from copy import deepcopy
import os
import subprocess
import sys

from coleo import Option, tooled
import yaml

# import milabench as mb
from ..common import get_multipack


_SETUP = "setup"
_TEARDOWN = "teardown"
_LIST = "list"
_ACTIONS = (_SETUP, _TEARDOWN, _LIST)


def manage_cloud(pack, packs, run_on, action="setup"):
    assert run_on in pack.config["system"]["cloud_profiles"]

    key_map = {
        "hostname":(lambda v: ("ip",v)),
        "username":(lambda v: ("user",v)),
        "ssh_key_file":(lambda v: ("key",v)),
        "env":(lambda v: ("env",[".", v, ";", "conda", "activate", "milabench", "&&"])),
    }
    plan_params = deepcopy(pack.config["system"]["cloud_profiles"][run_on])

    nodes = iter(enumerate(pack.config["system"]["nodes"]))

    state_prefix = []
    for p in packs.values():
        state_prefix.append(p.config["name"])
        state_prefix.append(p.config["install_variant"])

    while True:
        try:
            i, n = next(nodes)
            if n["ip"] != "1.1.1.1":
                continue
        except StopIteration:
            break

        plan_params["state_prefix"] = plan_params.get("state_prefix", None) or "-".join([str(i), *state_prefix])
        plan_params["state_id"] = plan_params.get("state_id", None) or pack.config["hash"]

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
            *[
                f"--{k.replace('_', '-')}={v}"
                for k, v in plan_params.items()
            ],
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
    system_config = manage_cloud(setup_pack, mp.packs, run_on, action=_SETUP)

    print(f"# hash::>{setup_pack.config['hash']}")
    print(yaml.dump({"system": system_config}))


@tooled
def _teardown():
    """Teardown a cloud infrastructure"""

    # Setup cloud on target infra
    run_on: Option & str

    mp = get_multipack()
    setup_pack = mp.setup_pack()
    manage_cloud(setup_pack, mp.packs, run_on, action=_TEARDOWN)


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
