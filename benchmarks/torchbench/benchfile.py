import contextlib
import os
import subprocess

from milabench.pack import Package

BRANCH = "ff7114655294aa3ba57127a260dbd1ef5190f610"


@contextlib.contextmanager
def tmp_path(path):
    curr_path = os.environ["PATH"]
    os.environ["PATH"] = path

    yield

    os.environ["PATH"] = curr_path


class TorchBenchmarkPack(Package):
    def install(self):
        code = self.dirs.code
        env_path = os.environ["PATH"]
        # Download and extract git-lfs if it is missing
        try:
            subprocess.check_call(["which", "git-lfs"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            subprocess.check_call([code / "download_extract_git-lfs.sh"], cwd=code, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._nox_session.env["PATH"] += f":{code / 'bin'}"
            env_path += f":{code / 'bin'}"
        with tmp_path(env_path):
            code.clone_subtree("https://github.com/pytorch/benchmark", BRANCH)
        # Add missing .git dirs even if they are empty to keep the git project
        # integrity. In particular, this is required to have a functional git-lfs
        os.makedirs(code / ".git/branches", exist_ok=True)
        os.makedirs(code / ".git/refs", exist_ok=True)
        self.pip_install("-r", code / "requirements-bench.txt")
        if os.path.exists(code / f"requirements-{self.config['model']}.txt"):
            self.pip_install("-r", code / f"requirements-{self.config['model']}.txt")
        self.python("install.py", "--models", self.config["model"])

    def run(self, args, voirargs, env):
        args.insert(0, self.config["model"])
        super().run
        return self.launch("run.py", args=args, voirargs=voirargs, env=env)


__pack__ = TorchBenchmarkPack
