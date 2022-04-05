""" IDEA: Setup the environment variables that aren't being set by `mila code`.
"""

import os
import subprocess
import shlex


def main():
    user = os.environ["USER"]
    out = subprocess.run(
        shlex.split(f"squeue -u {user} -h -o '%i'"), stdout=subprocess.PIPE
    )
    job_id = str(out.stdout, "utf-8")
    slurm_tmp_dir = f"/Tmp/slurm.{job_id}.0"
    os.environ["SLURM_TMPDIR"] = slurm_tmp_dir


if __name__ == "__main__":
    # Set the environment variables that aren't being set by `mila code`.
    main()
