"""Publish an archived run to a database."""

import json
import sys
from dataclasses import dataclass
from typing import Optional

from argklass.command import Command

from .._publish_utils import reverse_proxy


class Publish(Command):
    """Publish an archived run to a database."""

    name = "publish"

    # fmt: off
    @dataclass
    class Arguments:
        """Publish an archived run to a database."""
        uri     : str           = None  # URI to the database
        folder  : str           = None  # Run folder to save
        meta    : Optional[str] = None  # JSON file to append to meta dictionary
        testing : bool          = True  # Enable reverse proxy for testing
    # fmt: on

    @staticmethod
    def execute(args):
        from ...metrics.archive import publish_archived_run
        from ...metrics.sqlalchemy import SQLAlchemy

        if args.meta is not None:
            with open(args.meta, "r") as file:
                args.meta = json.load(file)

        with reverse_proxy(args.uri, enabled=args.testing) as uri:
            backend = SQLAlchemy(uri, meta_override=args.meta)
            publish_archived_run(backend, args.folder)

        sys.exit(0)


COMMANDS = Publish
