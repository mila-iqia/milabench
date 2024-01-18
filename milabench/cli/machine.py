from ..metadata import machine_metadata


def cli_machine():
    """Display machine metadata.
    Used to generate metadata json to back populate archived run

    """
    from bson.json_util import dumps as to_json

    print(to_json(machine_metadata(), indent=2))