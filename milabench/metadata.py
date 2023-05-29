import os
from datetime import datetime
import subprocess
import cpuinfo

from voir.instruments.gpu import get_gpu_info


def _exec(cmd, default):
    try:
        return subprocess.check_output(cmd.split(" "), encoding="utf-8").strip()
    except:
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
            "tag": _exec("git describe --tags", "<tag>"),
            "commit": _exec("git rev-parse HEAD", "<commit>"),
            "date": _exec("git show -s --format=%ci", "<date>"),
        },
    }
