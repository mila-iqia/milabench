import json
import os
import subprocess
import traceback
from datetime import datetime

import cpuinfo
from voir.instruments.gpu import get_gpu_info

import milabench.scripts.torchversion as torchversion

from ._version import __commit__, __date__, __tag__
from .scripts.vcs import retrieve_git_versions
from .utils import error_guard

def _get_gpu_info():
    try:
        return get_gpu_info()
    except Exception:
        traceback.print_exc()
        return {}


@error_guard({})
def fetch_torch_version(pack):
    cwd = pack.dirs.code
    exec_env = pack.full_env(dict())

    result = subprocess.run(
        [str(x) for x in ["python", torchversion.__file__]],
        env=exec_env,
        cwd=cwd,
        capture_output=True,
    )

    return json.loads(result.stdout)


def machine_metadata(pack=None):
    """Retrieve machine metadata"""

    uname = os.uname()
    gpus = _get_gpu_info()
    cpu = cpuinfo.get_cpu_info()

    if pack is None:
        torchv = torchversion.get_pytorch_version()
    else:
        torchv = fetch_torch_version(pack)

    return {
        "cpu": {
            "count": os.cpu_count(),
            "brand": cpu.pop("brand_raw", "<unknown>"),
        },
        "os": {
            "sysname": uname.sysname,
            "nodename": uname.nodename,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
        },
        "accelerators": gpus,
        "date": datetime.utcnow().timestamp(),
        "milabench": retrieve_git_versions(
            __tag__,
            __commit__,
            __date__,
        ),
        "pytorch": torchv,
    }
