from milabench.testing import replay_run, replay_zipfile
from milabench.utils import multilogger


def publish_archived_run(backend, folder, print_msg=False, stop_on_exception=False):
    """Publish an archived run to a database"""

    with multilogger(backend, stop_on_exception=stop_on_exception) as log:
        for msg in replay_run(folder):
            if print_msg:
                print(msg)
            log(msg)


def publish_zipped_run(backend, folder, **options):
    """Publish an archived run to a database"""

    with multilogger(backend, **options) as log:
        replay_zipfile(folder, backend)
