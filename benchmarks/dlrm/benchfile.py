from milabench.fs import XPath
from milabench.pack import Package

BRANCH = "69d22b99ec02ff868dbc1170e39686935f9d1274"


class DLRMBenchmarkPack(Package):
    base_requirements = "requirements.in"
    main_script = "dlrm/dlrm_s_pytorch.py"

    async def install(self):
        code:XPath = self.dirs.code
        dlrm:XPath = code / "dlrm"
        if not dlrm.exists():
            dlrm.clone_subtree("https://github.com/facebookresearch/dlrm", BRANCH)
            reqs:XPath = code / "requirements.in"
            reqs.write_text("\n".join(((dlrm / "requirements.txt").read_text(),
                                       (code / "_requirements.in").read_text())))

        await super().install()


__pack__ = DLRMBenchmarkPack
