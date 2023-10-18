
from dataclasses import dataclass
import re
import importlib_resources
import subprocess


def popen(cmd, callback=None):
    def println(line):
        print(line, end="")

    if callback is None:
        callback=println
    
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=False,
    ) as process:
        def readoutput():
            process.stdout.flush()
            line = process.stdout.readline()

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

    code = popen(['sbatch'] + args, readline)

    if jobid is not None and sync:
        try:
            subprocess.run(["touch", f"slurm-{jobid}.out"])
            subprocess.run(["tail", "-f", f"slurm-{jobid}.out"])
        except KeyboardInterrupt:
            pass
        
    return code, jobid


def shell(cmd):
       return subprocess.check_output(
            cmd.split(" "), 
            stderr=subprocess.STDOUT, 
            text=True
        ).strip()


class SlurmBatchOptions:
    pass


@dataclass
class SetupOptions:
    branch: str = "master"
    origin: str = "https://github.com/mila-iqia/milabench.git"
    config: str = "milabench/config/standard.yaml"
    env: str = "./env"
    python: str = "3.9"
    
    def deduce_remote(self, branch):
        prefix = "refs/heads/"
        
        # Fetch all remotes
        remotes = shell("get remote").splitlines()
        possible_remotes = []
        
        # Find remotes that have our branch
        for remote in remotes:
            branches = shell(f"git ls-remote --heads {remote}").splitlines()
            
            for ref, name in branches.split('\t'):
                name = name[len(prefix):]
                
                if branch == name:
                    possible_remotes.append(remote)
        
        if len(possible_remotes) == 1:
            return possible_remotes[0]
        
        raise RuntimeError(f"Multiple suitable remotes found {possible_remotes}")
        
    def deduce_from_repository(self, remote="origin"):
        self.branch = shell("git rev-parse --abbrev-ref HEAD")
        
        if remote is None:
            remote = self.deduce_remote(self.branch)
        
        self.origin = shell(f"git remote get-url {remote}")
    
    def arguments(self):
        return [
            "-b", self.branch,
            "-o", self.origin,
            "-c", self.config,
            "-e", self.env,
            "-p", self.python,
        ]


def launch_milabench(sbatch_args=None, dry: bool = False, sync: bool = False):
    sbatch_script = importlib_resources.files(__name__) / "scripts" / "milabench.bash"
    sbatch_script = str(sbatch_script)

    if sbatch_args is None:
        sbatch_args = [
            "--gpus-per-task=1",
            "--cpus-per-task=4",
            "--time=01:00:00",
            "--ntasks-per-node=1",
            "--mem=32G"
        ]
        
    script_args = SetupOptions()
    script_args.deduce_from_repository()
    script_args = script_args.arguments()

    cmd = ["sbatch"] + sbatch_args + [sbatch_script] + script_args
    
    if dry:
        print(' '.join(cmd))
        code = 0
    else:
        code, _ = sbatch(cmd, sync=sync, tags=None)
    
    return code