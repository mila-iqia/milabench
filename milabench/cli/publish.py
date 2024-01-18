import json
from dataclasses import dataclass

from coleo import Option, tooled


# fmt: off
@dataclass
class Arguments:
    uri: str
    folder: str
    meta: str = None
# fmt: on


@tooled
def arguments():
    # URI to the database
    #   ex:
    #       - postgresql://user:password@hostname:27017/database
    #       - sqlite:///sqlite.db
    uri: str

    # Run folder to save
    folder: str

    # Json string of file to append to the meta dictionary
    meta: Option & str = None

    return Arguments(uri, folder, meta)


@tooled
def cli_publish(args = arguments()):
    """Publish an archived run to a database"""

    from ..metrics.archive import publish_archived_run
    from ..metrics.sqlalchemy import SQLAlchemy

    if args.meta is not None:
        with open(args.meta, "r") as file:
            args.meta = json.load(file)

    backend = SQLAlchemy(args.uri, meta_override=args.meta)
    publish_archived_run(backend, args.folder)