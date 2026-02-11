
from dataclasses import dataclass
import os
import pathlib
import argparse

from coleo import Option, tooled

benchmark = (pathlib.Path(__file__).parent / '..' / '..' / "benchmarks").resolve()
template =  benchmark / "_templates"


multigpu = "\n".join([
    "method: njobs", 
    "    n: 1\n\n",
])

multinode = "\n".join([
    "  num_machines: 2",
    "  requires_capabilities:",
    "    - \"len(nodes) >= ${..num_machines}\"\n\n",
])

placeholder_repo = "https://github.com/Delaunay/extern_example.git"


# fmt: off
@dataclass
class Arguments:
    name       : str
    template   : str = "simple"
    repo_url   : str = None
    multi_gpu  : bool = False
    multi_node : bool = False
# fmt: on


@tooled
def arguments():
    # Name of the benchmark
    name: Option & str

    # Name of the template to use (simple, voir, stdout)
    template: Option & str = "simple"

    # Repo URL to clone
    repo_url: Option & str = None

    # is benchmark multi gpu
    multi_gpu: Option & bool = False

    # is the benchmark is multi node
    multi_node: Option & bool = False

    return Arguments(name, template, repo_url, multi_gpu, multi_node)


@tooled
def cli_new(args=None):
    """Create a new benchmark from the template"""

    if args is None:
        args = arguments()

    if args.repo_url is not None:
        args.template = "voir"

    package_name =  args.name.capitalize()

    template_dir = template / args.template
    destination = benchmark / args.name
    os.makedirs(destination, exist_ok=True)

    for file in os.listdir(template_dir):
        if file in ("base",):
            continue

        source = template_dir / file
        dest = destination / file

        with open(source, "r") as fp:
            content = fp.read()

        placeholders = [
            ("Template", package_name),
            ("template", args.name),
            (placeholder_repo, args.repo_url),
        ]

        if args.multi_gpu or args.multi_node:
            placeholders.append(("method: per_gpu\n", multigpu))
            
            if not args.multi_node:
                placeholders.append("multigpu")
            else:
                placeholders.append("multinode")

        if args.repo_url:
            placeholders.append(("https://github.com/Delaunay/extern_example.git",  args.repo_url))

        if args.multi_node:
            placeholders((None, multinode))

        for placeholder, value in placeholders:
            if value is not None:
                if placeholder is not None:
                    content = content.replace(placeholder, value)
                else:
                    content += value
        
        with open(dest, "w") as fp:
            fp.write(content)

        st = os.stat(source)
        os.chown(dest, st.st_uid, st.st_gid)
        os.chmod(dest, st.st_mode)


if __name__ == "__main__":
    cli_new()
