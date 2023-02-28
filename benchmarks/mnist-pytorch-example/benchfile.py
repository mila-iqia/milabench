from tempfile import TemporaryDirectory

from milabench.fs import XPath
from milabench.pack import Package

BRANCH = "3970e068c7f18d2d54db2afee6ddd81ef3f93c24"


class MNISTPack(Package):
    main_script = "main.py"
    prepare_script = "prepare.py"
    requirements_file = "requirements-bench.txt"

    def _clone_mnist(self, to):
        XPath(to).clone_subtree("https://github.com/pytorch/examples", BRANCH, "mnist")

    def install(self):
        code = self.dirs.code
        self._clone_mnist(code)
        main = code / "main.py"
        main.sub("../data", self.dirs.data)
        super().install()

    def pin(self, *pip_compile_args):
        with TemporaryDirectory(dir=self.pack_path) as pin_dir:
            pin_dir = XPath(pin_dir)
            self._clone_mnist(pin_dir)

            super().pin(*(*pip_compile_args, "requirements.txt"), cwd=pin_dir)


__pack__ = MNISTPack
