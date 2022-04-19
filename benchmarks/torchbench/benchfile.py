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
        headless = os.environ.get("HEADLESS", False)
        env_path = os.environ["PATH"]
        self._nox_session.env["PATH"] += f":{code / 'bin'}"
        env_path += f":{code / 'bin'}"
        if not self.code_mark_file.exists():
            # Download and extract git-lfs if it is missing
            with tmp_path(env_path):
                try:
                    subprocess.check_call(
                        ["which", "git-lfs"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except subprocess.CalledProcessError as e:
                    subprocess.check_call(
                        [code / "download_extract_git-lfs.sh"],
                        cwd=code,
                    )
                code.clone_subtree("https://github.com/pytorch/benchmark", BRANCH)
            # Add missing .git dirs even if they are empty to keep the git project
            # integrity. In particular, this is required to have a functional git-lfs
            os.makedirs(code / ".git/branches", exist_ok=True)
            os.makedirs(code / ".git/refs", exist_ok=True)
        self.pip_install("-r", code / "requirements-bench.txt")
        model_name = self.config["model"]
        if headless and (code / f"requirements-{model_name}-headless.txt").exists():
            self.pip_install("-r", code / f"requirements-{model_name}-headless.txt")
        elif (code / f"requirements-{model_name}.txt").exists():
            self.pip_install("-r", code / f"requirements-{model_name}.txt")
        if (code / f"noinstallpy-{model_name}.txt").exists():
            # We don't want to use the model's install.py, but we still want to
            # run the global install.py, so we pass resnet18 which has an empty
            # install.py.
            self.python("install.py", "--models", "resnet18")
        else:
            self.python("install.py", "--models", self.config["model"])

    def run(self, args, voirargs, env):
        args.insert(0, self.config["model"])
        super().run
        return self.launch("run.py", args=args, voirargs=voirargs, env=env)


__pack__ = TorchBenchmarkPack
