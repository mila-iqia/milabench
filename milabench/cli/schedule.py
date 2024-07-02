from dataclasses import dataclass, field
import os
import re
import subprocess

import importlib_resources
import requests
import yaml
from coleo import Option, tooled


# fmt: off
@dataclass
class Arguments:
    sync: bool = False
    dry : bool = False
    args: list = field(default_factory=list)
    profile: str = None
# fmt: on


@tooled
def arguments():
    # tail -f on the slurm job
    sync: Option & bool = False

    # Print the command and return without running it
    dry: Option & bool = False

    # sbatch run profile
    profile: Option & str = None

    # script arguments
    # [remainder]
    args: Option = []

    return Arguments(sync, dry, args, profile)


def get_sbatch_profiles(profile, default):
    ROOT = os.path.dirname(__file__)
    default_scaling_config = os.path.join(ROOT, "..", "..", "config", "slurm.yaml")

    with open(default_scaling_config, "r") as fp:
        sbatch_profiles = yaml.safe_load(fp)

    args = sbatch_profiles.get(profile)

    if args is None:
        args = sbatch_profiles.get(default)
    
    return args 


@tooled
def cli_schedule(args=None):
    """Launch a slurm job to run milabench"""
    # milabench schedule --sync -- --select resnet50
    if args is None:
        args = arguments()

    sbatch_args = get_sbatch_profiles(args.profile, "single-node-small")

    launch_milabench(args.args, sbatch_args=sbatch_args, dry=args.dry, sync=args.sync)


def popen(cmd, callback=None):
    def println(line):
        print(line, end="")

    if callback is None:
        callback = println

    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=False,
    ) as process:

        def readoutput():
            process.stdout.flush()
            for line in process.stdout.readlines():
                if callback:
                    callback(line)

        try:
            while process.poll() is None:
                readoutput()

            readoutput()
            return 0

        except KeyboardInterrupt:
            print("Stopping due to user interrupt")
            process.kill()
            return -1


def sbatch(args, sync=False, tags=None, **kwargs):
    jobid_regex = re.compile(r"Submitted batch job (?P<jobid>[0-9]*)")
    jobid = None

    def readline(line):
        nonlocal jobid

        if match := jobid_regex.match(line):
            data = match.groupdict()
            jobid = data["jobid"]

        print(line, end="")

    code = popen(["sbatch"] + args, readline)

    if jobid is not None and sync:
        try:
            subprocess.run(["touch", f"slurm-{jobid}.out"])
            subprocess.run(["tail", "-f", f"slurm-{jobid}.out"])
        except KeyboardInterrupt:
            pass

    return code, jobid


def shell(cmd):
    return subprocess.check_output(
        cmd.split(" "), stderr=subprocess.STDOUT, text=True
    ).strip()


class SlurmBatchOptions:
    pass


@dataclass
class SetupOptions:
    branch: str = "master"
    origin: str = "https://github.com/mila-iqia/milabench.git"
    config: str = "milabench/config/standard.yaml"
    env: str = "./env"
    python: str = "3.10"
    fun: str = "run"

    def deduce_remote(self, current_branch):
        prefix = "refs/heads/"

        # Fetch all remotes
        remotes = shell("git remote").splitlines()
        possible_remotes = []

        # Find remotes that have our branch
        for remote in remotes:
            branches = shell(f"git ls-remote --heads {remote}").splitlines()

            for branch in branches:
                _, name = branch.split("\t")
                name = name[len(prefix) :]

                if current_branch == name:
                    possible_remotes.append(remote)

        if len(possible_remotes) == 1:
            return possible_remotes[0]

        raise RuntimeError(f"Multiple suitable remotes found {possible_remotes}")

    def deduce_from_repository(self, remote=None):
        self.branch = shell("git rev-parse --abbrev-ref HEAD")

        if remote is None:
            remote = self.deduce_remote(self.branch)

        self.origin = shell(f"git remote get-url {remote}")

    def arguments(self):
        return [
            "-b",
            self.branch,
            "-o",
            self.origin,
            "-c",
            self.config,
            "-e",
            self.env,
            "-p",
            self.python,
            "-f",
            self.fun
        ]


def launch_milabench(args, sbatch_args=None, dry: bool = False, sync: bool = False):
    sbatch_script = (
        os.path.abspath(importlib_resources.files(__name__) / ".." / "scripts" / "milabench_run.bash")
    )
    sbatch_script = str(sbatch_script)

    script_args = SetupOptions()
    script_args.deduce_from_repository()
    script_args = script_args.arguments()

    cmd = sbatch_args + [sbatch_script] + script_args + args
    print("sbatch " + " ".join(cmd))

    if dry:
        code = 0
    else:
        code, _ = sbatch(cmd, sync=sync, tags=None)

    return code


def get_remote_owner(remote):
    sshremote = re.compile(
        r"git@[A-Za-z]*\.[A-Za-z]*:(?P<owner>[A-Za-z\-.0-9]*)\/([A-Za-z]*).([A-Za-z]*)"
    )
    httpsremote = re.compile(
        r"https:\/\/[A-Za-z]*\.[A-Za-z]*\/(?P<owner>[A-Za-z\-.0-9]*)\/([A-Za-z]*).([A-Za-z]*)"
    )

    patterns = [sshremote, httpsremote]

    for pat in patterns:
        if match := pat.match(remote):
            results = match.groupdict()
            return results["owner"]

    return None


def post_comment_on_pr(remote, branch, comment, access_token=None):
    owner = get_remote_owner(remote)
    assert owner is not None, "Remote owner not found"

    if access_token is None:
        access_token = os.getenv("MILABENCH_GITHUB_PAT")

    url = "https://api.github.com/repos/mila-iqia/milabench/pulls"

    response = requests.get(url, params={"head": f"{owner}:{branch}"})

    if response.status_code != 200:
        raise RuntimeError(response)

    pull_requests = response.json()

    if not pull_requests:
        raise RuntimeError("No matching pull requests found.")

    assert len(pull_requests) == 1, "Multiple PR found"

    pr = pull_requests[0]
    post_url = pr["_links"]["comments"]["href"]

    data = {
        "body": comment,
    }

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/vnd.github.v3+json",
    }

    response = requests.post(post_url, json=data, headers=headers)

    if response.status_code != 201:
        raise RuntimeError(response, response.json())
