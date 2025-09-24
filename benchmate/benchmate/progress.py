"""THIS FILE CANNOT IMPORT ANYTHING FROM benchmate"""

import os

import tqdm as _tqdm


MIN_INTERVAL = 30
MAX_INTERVAL = 60


def _patched_tqdm(*args, **kwargs):
    disable = kwargs.get("disable", False) or int(os.getenv("MILABENCH_NO_PROGRESS", 0))

    mininterval = max(MIN_INTERVAL, kwargs.get("mininterval", MIN_INTERVAL))
    maxinterval = max(MAX_INTERVAL, kwargs.get("maxinterval", MAX_INTERVAL))

    return _tqdm._original_tqdm(*args, mininterval=mininterval, maxinterval=maxinterval, disable=disable, **kwargs)


if not hasattr(_tqdm, "_original_tqdm"):
    _tqdm._original_tqdm = _tqdm.tqdm
    _tqdm.tqdm = _patched_tqdm

    # also patch tqdm.auto
    try:
        import tqdm.auto as _tqdm_auto
        _tqdm_auto._original_tqdm = _tqdm_auto.tqdm
        _tqdm_auto.tqdm = _patched_tqdm

    except ImportError:
        pass


tqdm = _patched_tqdm
