from milabench.testing import replay_run, replay_zipfile
from milabench.utils import multilogger


def publish_archived_run(backend, folder):
    """Publish an archived run to a database"""

    with multilogger(backend) as log:
        for msg in replay_run(folder):
            print(msg)
            log(msg)



def publish_zipped_run(backend, folder, **options):
    """Publish an archived run to a database"""

    with multilogger(backend, **options) as log:
        replay_zipfile(folder, backend)
