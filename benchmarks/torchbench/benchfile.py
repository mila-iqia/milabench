import contextlib
import os
import subprocess
from tempfile import TemporaryDirectory

from milabench.fs import XPath
from milabench.pack import Package
from milabench.config import parse_config

BRANCH = "ff7114655294aa3ba57127a260dbd1ef5190f610"


@contextlib.contextmanager
def tmp_path(path):
    curr_path = os.environ["PATH"]
    try:
        os.environ["PATH"] = path
        yield
    finally:
        os.environ["PATH"] = curr_path


class TorchBenchmarkPack(Package):
    requirements_file = "requirements-bench.txt"

    def make_env(self):
        """Return a dict of environment variables to use for prepare/run.

        By default, ``make_env()`` will return:

        .. code-block:: python

            {
                "MILABENCH_DIR_CODE": self.dirs.code,
                "MILABENCH_DIR_DATA": self.dirs.data,
                "MILABENCH_DIR_VENV": self.dirs.venv,
                "MILABENCH_DIR_RUNS": self.dirs.runs,
                "MILABENCH_CONFIG": json.dumps(self.config),
                "TORCH_HOME": self.dirs.data / "torch"
            }
        """
        return {
            **super().make_env(),
            "TORCH_HOME": self.dirs.data / "torch"
        }

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
            (code / "run.py").sub(
                r"run_one_step\(test, model_flops=model_flops\)",
                "run_one_step(test, model_flops=model_flops, num_iter=1_000_000)",
            )

        super().install()

        model_name = self.config["model"]

        freeze_files = {stage: code / f"{stage}-{model_name}"
                        for stage in ("pre-post", "pre-install", "post-install")}

        # Log the requirements installed from the first install phase
        # super().install(). requirements-bench.txt being derived from
        # reqs/requirements-bench.in, most of the requirements should be
        # installed
        with open(f"{freeze_files['pre-post']}.out", "w") as _out, \
             open(f"{freeze_files['pre-post']}.err", "w") as _err:
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
        with open(f"{freeze_files['pre-install']}.out", "w") as _out, \
             open(f"{freeze_files['pre-install']}.err", "w") as _err:
            self.execute("python3", "-m", "pip", "freeze", stdout=_out, stderr=_err)

        if (code / f"installpy-{model_name}.txt").exists():
            self.python("install.py", "--models", self.config["model"])

        # Log the requirements installed from the third install phase and make
        # sure to move any requirements that can to the model's requirements.in
        with open(f"{freeze_files['post-install']}.out", "w") as _out, \
             open(f"{freeze_files['post-install']}.err", "w") as _err:
            self.execute("python3", "-m", "pip", "freeze", stdout=_out, stderr=_err)

    def pin(self, *pip_compile_args, constraints:list=tuple()):
        with TemporaryDirectory(dir=self.pack_path) as pin_dir:
            pin_dir = XPath(pin_dir)
            req_file = XPath(self.requirements_file)
            self._clone_tb(pin_dir)

            (self.pack_path / "reqs").copy(pin_dir / "reqs")
            super().pin(*pip_compile_args, requirements_file=req_file,
                        input_files=(req_file.with_suffix('.in'),), constraints=constraints,
                        cwd=pin_dir)
            (pin_dir / "reqs").merge_into(self.pack_path / "reqs")

    def exec_pip_compile(self, requirements_file:XPath, input_files:list,
                         *pip_compile_args, cwd:XPath):
        config = parse_config(self.config["config_file"])
        models = set()
        for name, defn in config["benchmarks"].items():
            group = defn.get("group", name)
            if group == self.config["group"]:
                models.add(defn["model"])
        models_args = []
        for m in models:
            models_args.extend(("-m", m))
        self.execute(cwd / "reqs/pip-compile.sh", "--reqs", "reqs", "--tb-root", ".",
                     "--config", (self.pack_path / "benchtest.yaml").absolute(),
                     *models_args, "--", "--resolver", "backtracking",
                     "--output-file", requirements_file, *pip_compile_args,
                     *input_files, cwd=cwd)

    def run(self, args, voirargs, env):
        args.insert(0, self.config["model"])
        env = {**self.make_env(), **(env if env else {})}
        return self.launch("run.py", args=args, voirargs=voirargs, env=env)

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


__pack__ = TorchBenchmarkPack
