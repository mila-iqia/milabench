import sys
from dataclasses import dataclass
import sys

import yaml
from coleo import Option, tooled

from ..common import get_multipack
from ..sizer import resolve_argv
from ..system import build_system_config, multirun, apply_system, SizerOptions, option, as_environment_variable


# fmt: off
@dataclass
class Arguments:
    base      : str = None
    system    : str = None
    config    : str = None
    select    : str = None
    exclude   : str = None
# fmt: on


@tooled
def arguments():
    base: Option & str = None

    system: Option & str = None

    config: Option & str = None

    select: Option & str = None

    exclude: Option & str = None

    return Arguments(base, system, config, select, exclude)


@tooled
def cli_multirun(args=None):
    """Generate environment variable overrides for a multirun.
    Populates a template file replacing `# PLACEHOLDER` with the environment variables

    This is used to schedule matrix runs on slurm clusters.
    """
    if args is None:
        args = arguments()

    # Load the configuration and system
    mp = get_multipack()

    template_filename = "/home/d/delaunay/links/scratch/shared/milabench.bash"

    with open(template_filename, "r") as fp:
        template = fp.read()

    files = []

    success = 0
    for name, conf in multirun():
        run_name = name or args.run_name

        env = [
            f"export MILABENCH_RUN_NAME={run_name}"
        ]
        for k, v in conf.items():
            conf_key = as_environment_variable(k)
            env.append(f"export {conf_key}=\"{v}\"")

        env = "\n".join(env)

        fname = f"{name}.bash"

        with open(fname, "w") as fp:
            specialized = template.replace("# PLACEHOLDER", env)
            fp.write(specialized)

        files.append(f"sbatch {fname}")

    print("\n".join(files))

    return success


def tree_run():
    from .schedule import sbatch 

    tree = [
        [
            "bs_m0.5.bash",
            "bs_m2.bash",
            "bs_a-8.bash",
            "bs_a0.bash",
            "bs_a8.bash",
            "bs_a16.bash",
        ],
        [
            "c32Go_m8_w8.bash",
            "c32Go_m8_w16.bash",
            "c64Go_m8_w8.bash",
            "c64Go_m8_w16.bash",
            "cAll_m8_w8.bash",
            "All_m8_w16.bash",
        ],
        [
            "auto.bash"
        ]
    ]

    dependencies = []
    for step in steps:
        base_cmd = []
        if dependencies:
            base_cmd.append("--dependency=" + ":".join(dependencies))
        dependencies = []
        
        for run in step:            
            cmd = base_cmd + [run]

            code, jobid = sbatch(cmd, sync=False)
            
            dependencies.append(str(jobid))


if __name__ == "__main__":
    tree_run()


# sbatch --dependency=afterok:1154:1155:1156:1157:1158:1159 

# sbatch c32Go_m8_w8.bash
# sbatch c32Go_m8_w16.bash
# sbatch c64Go_m8_w8.bash
# sbatch c64Go_m8_w16.bash
# sbatch cAll_m8_w8.bash
# sbatch cAll_m8_w16.bash