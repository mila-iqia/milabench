from coleo import tooled

from .prepare import cli_prepare
from .run import cli_run

@tooled
def cli_prepare_run(args=None):
    """Prepare a benchmark: download datasets, weights etc."""
    
    rc = cli_prepare()
    
    if rc == 0:
        rc = cli_run()

    return rc
