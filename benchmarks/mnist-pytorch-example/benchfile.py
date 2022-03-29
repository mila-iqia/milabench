from milabench.pack import Package


BRANCH = "3970e068c7f18d2d54db2afee6ddd81ef3f93c24"


class MNISTPack(Package):

    def setup(self):
        code = self.dirs.code
        code.clone_subtree("https://github.com/pytorch/examples", BRANCH, "mnist")
        main = code / "mnist" / "main.py"
        main.sub("../data", str(self.dirs.data))
        self.install("-r", code / "mnist" / "requirements.txt")

    def prepare(self):
        # return self.launch_script("prepare.py", args=[self.dirs.data], voir=False)
        self.run("python", self.dirs.code / "prepare.py", self.dirs.data)

    def launch(self, args, voirargs, env):
        return self.launch_script("mnist/main.py", args=args, voirargs=voirargs, env=env)


__pack__ = MNISTPack
