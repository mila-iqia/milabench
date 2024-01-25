from milabench.testing import replay_run
from milabench.utils import multilogger


def publish_archived_run(backend, folder):
    """Publish an archived run to a database"""

    with multilogger(backend) as log:
        for msg in replay_run(folder):
            print(msg)
            log(msg)
