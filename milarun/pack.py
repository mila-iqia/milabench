from argparse import Namespace as NS
from sys import version_info as pyv

from giving import give
from nox.sessions import Session, SessionRunner

from .fs import XPath
from .runner import Runner, make_runner
from .utils import extract_instruments


class Package:
    def __init__(self, config):
        self.config = config
        self.pack_path = XPath(config["definition"])

        self.dirs = NS(**{name: XPath(d) for name, d in config["dirs"].items()})

        reuse = True
        self._nox_runner = SessionRunner(
            name=self.dirs.venv.name,
            signatures=[],
            func=NS(
                python=f"{pyv.major}.{pyv.minor}.{pyv.micro}",
                venv_backend="virtualenv",
                venv_params=[],
                reuse_venv=reuse,
            ),
            global_config=NS(
                posargs=[],
                envdir=self.dirs.venv.parent,
                force_venv_backend=False,
                default_venv_backend=None,
                reuse_existing_virtualenvs=reuse,
                no_install=False,
                error_on_external_run=False,
                install_only=False,
            ),
            manifest=None,
        )
        self._nox_session = Session(self._nox_runner)
        self._nox_runner._create_venv()

    def do_install(self):
        self.install(".")
        self.setup()
        self.pack_path.copy(self.dirs.code / "__bench__.py")

    def do_main(self, *args):
        return self.run_function(self.main, args)

    def setup(self):
        pass

    def install(self, *args, **kwargs):
        args = [str(x) for x in args]
        return self._nox_session.install(*args, **kwargs)

    def conda_install(self, *args, **kwargs):
        args = [str(x) for x in args]
        return self._nox_session.conda_install(*args, **kwargs)

    def run(self, *args, **kwargs):
        args = [str(x) for x in args]
        return self._nox_session.run(*args, **kwargs, external=True)

    def bridge(self, runner, gv):
        pass

    def complete_instruments(self, instruments):
        instruments = extract_instruments({"instruments": instruments})
        instruments += extract_instruments(self.config)
        return instruments

    def run_script(self, script, args=(), instruments={}, field="__main__"):
        instruments = self.complete_instruments(instruments)

        if not XPath(script).is_absolute():
            script = str(self.dirs.code / script)

        runner = make_runner(
            script=script,
            field=field,
            args=args,
            instruments=[self.bridge, *instruments],
        )
        return runner()

    def run_function(self, fn, args=(), instruments={}):
        instruments = self.complete_instruments(instruments)

        runner = Runner(fn, instruments=instruments)
        return runner(*args)
