import os
from datetime import datetime
import cpuinfo

from voir.instruments.gpu import get_gpu_info

from ._version import __commit__, __tag__, __date__
from .vcs import retrieve_git_versions


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
        "date": datetime.utcnow().timestamp(),
        "milabench": retrieve_git_versions(
            __tag__,
            __commit__,
            __date__,
        ),
    }
