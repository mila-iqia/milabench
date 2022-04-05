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
        data = self.dirs.data
        env_path = os.environ["PATH"]
        try:
            subprocess.check_call(["which", "git-lfs"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            subprocess.check_call([code / "download_extract_git-lfs.sh"], cwd=data, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._nox_session.env["PATH"] += f":{data / 'bin'}"
            env_path += f":{data / 'bin'}"
        with tmp_path(env_path):
            code.clone_subtree("https://github.com/pytorch/benchmark", BRANCH)
        os.makedirs(code / ".git/branches", exist_ok=True)
        os.makedirs(code / ".git/refs", exist_ok=True)
        self.pip_install("-r", code / "requirements-bench.txt")
        self.python("install.py", "--models", self.config["model"])

    def run(self, args, voirargs, env):
        args.insert(0, self.config["model"])
        super().run
        return self.launch("run.py", args=args, voirargs=voirargs, env=env)


__pack__ = TorchBenchmarkPack
