import contextlib
import os
import subprocess
from tempfile import TemporaryDirectory

from milabench.fs import XPath
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

    def _clone_tb(self, to):
        to = XPath(to)
        env_path = os.environ["PATH"]
        env_path += f":{to / 'bin'}"
        with tmp_path(env_path):
            # Download and extract git-lfs if it is missing
            try:
                subprocess.check_call(
                    ["which", "git-lfs"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except subprocess.CalledProcessError:
                subprocess.check_call(
                    [self.pack_path / "download_extract_git-lfs.sh"],
                    cwd=to,
                )
            to.clone_subtree("https://github.com/pytorch/benchmark", BRANCH)

    def install(self):
        code = self.dirs.code
        self._nox_session.env["PATH"] += f":{code / 'bin'}"
        if not self.code_mark_file.exists():
            self._clone_tb(code)
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

        # Log the requirements installed from the first install phase
        # super().install(). requirements-bench.txt being derived from
        # reqs/requirements-bench.in, most of the requirements should be
        # installed
        with open(code / f"pre-post-{model_name}.out", "w") as _out, open(code / f"pre-post-{model_name}.err", "w") as _err:
            self.execute("python3", "-m", "pip", "freeze", stdout=_out, stderr=_err)

        # Some benches requires some packages to already be installed as they
        # use them in their setup.py. Very few packages should be installed here
        # as every non-weird requirements should be in requirements-bench.txt so
        # those that can should be moved to the model's requirements.in (e.g.
        # reqs/MODEL/requirements.in)
        if (code / "reqs" / model_name / "requirements-post.txt").exists():
            self.pip_install("-r", code / "reqs" / model_name / "requirements-post.txt")

        # Log the requirements installed from the second install phase and make
        # sure to move any requirements that can to the model's requirements.in
        with open(code / f"pre-install-{model_name}.out", "w") as _out, open(code / f"pre-install-{model_name}.err", "w") as _err:
            self.execute("python3", "-m", "pip", "freeze", stdout=_out, stderr=_err)

        if (code / f"installpy-{model_name}.txt").exists():
            self.python("install.py", "--models", self.config["model"])

        # Log the requirements installed from the third install phase and make
        # sure to move any requirements that can to the model's requirements.in
        with open(code / f"post-install-{model_name}.out", "w") as _out, open(code / f"post-install-{model_name}.err", "w") as _err:
            self.execute("python3", "-m", "pip", "freeze", stdout=_out, stderr=_err)

    def pin(self, *pip_compile_args):
        with TemporaryDirectory(dir=self.pack_path) as pin_dir:
            pin_dir = XPath(pin_dir)
            reqs_dir = pin_dir / "reqs"
            self._clone_tb(pin_dir)

            (self.pack_path / "reqs").copy(pin_dir / "reqs")
            (self.pack_path / "benchtest.yaml").copy(pin_dir)
            (self.pack_path / self.requirements_file).copy(pin_dir)

            self.pip_install("pip-tools")
            self.execute(reqs_dir / "pip-compile.sh", "--reqs", reqs_dir.stem,
                "--tb-root", ".", "--", "--resolver", "backtracking",
                "--output-file", self.requirements_file, *pip_compile_args,
                cwd=pin_dir)

            (pin_dir / self.requirements_file).copy(self.pack_path)

    def run(self, args, voirargs, env):
        args.insert(0, self.config["model"])
        super().run
        return self.launch("run.py", args=args, voirargs=voirargs, env=env)


__pack__ = TorchBenchmarkPack
