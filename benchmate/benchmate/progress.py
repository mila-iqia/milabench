"""THIS FILE CANNOT IMPORT ANYTHING FROM benchmate"""

import os

import tqdm as _tqdm


MIN_INTERVAL = 30
MAX_INTERVAL = 60


class PatchedTQDM(_tqdm.tqdm):
    def __init__(self, *args, **kwargs):
        disable = kwargs.pop("disable", False) or int(os.getenv("MILABENCH_NO_PROGRESS", 0))

        mininterval = max(MIN_INTERVAL, kwargs.pop("mininterval", MIN_INTERVAL))
        maxinterval = max(MAX_INTERVAL, kwargs.pop("maxinterval", MAX_INTERVAL))

        super().__init__(*args, mininterval=mininterval, maxinterval=maxinterval, disable=disable, **kwargs)


if not hasattr(_tqdm, "_original_tqdm"):
    _tqdm._original_tqdm = _tqdm.tqdm
    _tqdm.tqdm = PatchedTQDM

    # also patch tqdm.auto
    try:
        import tqdm.auto as _tqdm_auto
        _tqdm_auto._original_tqdm = _tqdm_auto.tqdm
        _tqdm_auto.tqdm = PatchedTQDM

    except ImportError:
        pass


tqdm = PatchedTQDM
