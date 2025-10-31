import json
import os
import subprocess
import traceback
import datetime
import functools

import cpuinfo
import voir.instruments.gpu as gpu



from ._version import __commit__, __date__, __tag__
from .utils import error_guard


def _get_gpu_info():
    try:
        smi = gpu.select_backend()

        return {
            "arch": smi.arch,
            "gpus": smi.get_gpus_info(),
            "system": smi.system_info(),
        }
    except Exception:
        traceback.print_exc()
        return {}


@error_guard({})
def fetch_torch_version(pack):
    import milabench.scripts.torchversion as torchversion

    cwd = pack.dirs.code
    exec_env = pack.full_env(dict())
    
    result = subprocess.run(
        [str(x) for x in ["python", torchversion.__file__]],
        env=exec_env,
        cwd=cwd,
        capture_output=True,
    )

    return json.loads(result.stdout)


@functools.cache
@error_guard({})
def machine_metadata(pack=None):
    """Retrieve machine metadata"""
    from .scripts.vcs import retrieve_git_versions

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
        "date": datetime.datetime.utcnow().timestamp(),
        "milabench": retrieve_git_versions(
            __tag__,
            __commit__,
            __date__,
        ),
        "pytorch": torchv,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(machine_metadata(), indent=2))
