from milabench.pack import Package

BRANCH = "3970e068c7f18d2d54db2afee6ddd81ef3f93c24"


class MNISTPack(Package):
    main_script = "main.py"
    prepare_script = "prepare.py"

    def install(self):
        code = self.dirs.code
        code.clone_subtree("https://github.com/pytorch/examples", BRANCH, "mnist")
        main = code / "main.py"
        main.sub("../data", self.dirs.data)
        self.pip_install("-r", code / "requirements.txt")


__pack__ = MNISTPack
