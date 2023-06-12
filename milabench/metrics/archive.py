from milabench.utils import multilogger
from milabench.testing import replay_run


def publish_archived_run(backend, folder):
    """Publish an archived run to a database"""

    with multilogger(backend) as log:
        for msg in replay_run(folder):
            print(msg)
            log(msg)
