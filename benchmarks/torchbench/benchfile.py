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
    requirements_file = "requirements-bench.txt"

    def install(self):
        code = self.dirs.code
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

            # Patch to avoid a race condition in the stargan test
            os.makedirs(code / "stargan/logs", exist_ok=True)
            os.makedirs(code / "stargan/models", exist_ok=True)
            os.makedirs(code / "stargan/samples", exist_ok=True)
            os.makedirs(code / "stargan/results", exist_ok=True)
            (code / "torchbenchmark/models/Super_SloMo/slomo_model.py").sub(
                "ind = indices",
                "ind = indices.detach().cpu().numpy()"
            )

        super().install()

        model_name = self.config["model"]

        with open(code / f"pre-post-{model_name}.out", "w") as _out, open(code / f"pre-post-{model_name}.err", "w") as _err:
            self.execute("python3", "-m", "pip", "freeze", stdout=_out, stderr=_err)

        if (code / "reqs" / model_name / "requirements-post.txt").exists():
            self.pip_install("-r", code / "reqs" / model_name / "requirements-post.txt")

        with open(code / f"pre-install-{model_name}.out", "w") as _out, open(code / f"pre-install-{model_name}.err", "w") as _err:
            self.execute("python3", "-m", "pip", "freeze", stdout=_out, stderr=_err)

        if (code / f"installpy-{model_name}.txt").exists():
            self.python("install.py", "--models", self.config["model"])

        with open(code / f"post-install-{model_name}.out", "w") as _out, open(code / f"post-install-{model_name}.err", "w") as _err:
            self.execute("python3", "-m", "pip", "freeze", stdout=_out, stderr=_err)

    def run(self, args, voirargs, env):
        args.insert(0, self.config["model"])
        super().run
        return self.launch("run.py", args=args, voirargs=voirargs, env=env)


__pack__ = TorchBenchmarkPack
