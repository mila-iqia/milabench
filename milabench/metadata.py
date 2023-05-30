import os
from datetime import datetime
import subprocess
import cpuinfo
import warnings

from voir.instruments.gpu import get_gpu_info

from ._version import __commit__, __tag__, __date__


def _exec(cmd, default):
    try:
        return subprocess.check_output(cmd.split(" "), encoding="utf-8").strip()
    except subprocess.CalledProcessError:
        warnings.warn("out of tree; milabench could not retrieve version info")
        return default


def machine_metadata():
    """Retrieve machine metadata"""

    uname = os.uname()
    gpus = get_gpu_info()
    cpu = cpuinfo.get_cpu_info()

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
        "date": datetime.utcnow(),
        "milabench": {
            "tag": _exec("git describe --tags", __tag__),
            "commit": _exec("git rev-parse HEAD", __commit__),
            "date": _exec("git show -s --format=%ci", __date__),
        },
    }
