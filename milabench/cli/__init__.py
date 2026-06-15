"""milabench CLI entry point using argklass for command discovery."""

import argparse
import os
import sys

from argklass.argformat import HelpAction, HelpActionException
from argklass.command import ParentCommand
from argklass.plugin import discover_module_commands_no_cache

from benchmate.progress import timed_flush

timed_flush()


def discover_commands():
    import milabench.cli
    return discover_module_commands_no_cache(
        milabench.cli,
        None,
    ).found_commands


def build_parser(commands):
    parser = argparse.ArgumentParser(
        prog="milabench",
        add_help=False,
        description="Benchmarking suite for machine learning algorithms",
    )
    parser.add_argument(
        "-h", "--help", action=HelpAction, help="show this help message and exit"
    )

    subparsers = parser.add_subparsers(dest="command")

    ParentCommand.dispatch = dict()
    for k, command in commands.items():
        command.arguments(subparsers)

    return parser


def main(argv=None):
    sys.path.insert(0, os.path.abspath(os.curdir))

    if argv is None:
        argv = sys.argv[1:]
    argv = [str(x) for x in argv]

    commands = discover_commands()

    try:
        parser = build_parser(commands)
        parsed_args = parser.parse_args(argv)
    except HelpActionException:
        return 0

    cmd_name = parsed_args.command
    if cmd_name is None:
        parser.print_usage()
        return 1

    command = commands.get(cmd_name)

    if command is None:
        print(f"Action `{cmd_name}` not implemented")
        return 1

    try:
        returncode = command.execute(parsed_args)
        return returncode if returncode is not None else 0
    except KeyboardInterrupt:
        return 130
