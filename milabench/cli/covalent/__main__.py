import argparse
import asyncio
import json
import os
import pathlib
import subprocess
import sys
import tempfile


def _load_venv(venv:pathlib.Path) -> dict:
    activate = venv / "bin/activate"
    if not activate.exists():
        raise FileNotFoundError(str(activate))
    env = subprocess.run(
        f". '{activate}' && python3 -c 'import os ; import json ; print(json.dumps(dict(os.environ)))'",
        shell=True,
        capture_output=True
    ).stdout
    return json.loads(env)


def serve(*argv):
    return subprocess.run([
        "covalent",
        *argv
    ]).returncode


def _get_executor_kwargs(args):
    return {
        **{k:v for k,v in vars(args).items() if k not in ("setup", "teardown")},
        **{"action":k for k,v in vars(args).items() if k in ("setup", "teardown") and v},
    }


def executor(executor_cls, args, *argv):
    import covalent as ct

    executor:ct.executor.BaseExecutor = executor_cls(
        **_get_executor_kwargs(args),
    )

    def _popen(cmd, *args, _env=None, **kwargs):
        _env = _env if _env is not None else {}

        for envvar in _env.keys():
            envvar_val = _env[envvar]

            if not envvar_val:
                continue

            envvar_val = pathlib.Path(envvar_val).expanduser()
            if str(envvar_val) != _env[envvar]:
                _env[envvar] = str(envvar_val)

        if "MILABENCH_CONFIG_CONTENT" in _env:
            _config_dir = pathlib.Path(_env["MILABENCH_CONFIG"]).parent
            with tempfile.NamedTemporaryFile("wt", dir=str(_config_dir), suffix=".yaml", delete=False) as _f:
                _f.write(_env["MILABENCH_CONFIG_CONTENT"])
                _env["MILABENCH_CONFIG"] = _f.name

        try:
            cmd = (str(pathlib.Path(cmd[0]).expanduser()), *cmd[1:])
        except IndexError:
            pass

        cwd = kwargs.pop("cwd", None)
        if cwd is not None:
            cwd = str(pathlib.Path(cwd).expanduser())
            kwargs["cwd"] = cwd

        _env = {**os.environ.copy(), **kwargs.pop("env", {}), **_env}

        kwargs = {
            **kwargs,
            "env": _env,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
        }
        p = subprocess.Popen(cmd, *args, **kwargs)

        stdout_chunks = []
        while True:
            line = p.stdout.readline()
            if not line:
                break
            line_str = line.decode("utf-8").strip()
            stdout_chunks.append(line_str)
            print(line_str)

        _, stderr = p.communicate()
        stderr = stderr.decode("utf-8").strip()
        stdout = os.linesep.join(stdout_chunks)

        if p.returncode != 0:
            raise subprocess.CalledProcessError(
                p.returncode,
                (cmd, args, kwargs),
                stdout,
                stderr
            )
        return p.returncode, stdout, stderr

    @ct.lattice
    def lattice(argv=(), deps_bash = None):
        return ct.electron(
            _popen,
            executor=executor,
            deps_bash=deps_bash,
        )(
            argv,
        )

    return_code = 0
    try:
        dispatch_id = None
        result = None
        deps_bash = None

        if not argv and args.setup:
            conda_prefix = "eval \"$(conda shell.bash hook)\""
            conda_activate = "conda activate milabench"
            deps_bash = []
            for _cmd in (
                f"{conda_activate} || conda create -n milabench -y",
                f"{conda_activate}"
                f" && conda install python={sys.version_info.major}.{sys.version_info.minor} virtualenv pip -y"
                f" || >&2 echo First attempt to install python in milabench env failed",
                f"{conda_activate}"
                f" && conda install python={sys.version_info.major}.{sys.version_info.minor} virtualenv pip -y"
                f" || conda remove -n milabench --all -y",
            ):
                deps_bash.append(f"{conda_prefix} && ({_cmd})")
            deps_bash = ct.DepsBash(deps_bash)
            argv = ["conda", "env", "list"]

        if argv:
            dispatch_id = ct.dispatch(lattice, disable_run=False)(argv, deps_bash=deps_bash)
            result = ct.get_result(dispatch_id=dispatch_id, wait=True)
            return_code, stdout, _ = result.result if result.result is not None else (1, "", "")

        if return_code == 0 and args.setup:
            assert any([l for l in stdout.split("\n") if l.startswith("milabench ")])
            _executor:ct.executor.BaseExecutor = executor_cls(
                **{
                    **_get_executor_kwargs(args),
                    **{"action": "teardown"},
                }
            )
            asyncio.run(_executor.setup({}))

            assert _executor.hostname
            print(f"hostname::>{_executor.hostname}")
            print(f"username::>{_executor.username}")
            print(f"ssh_key_file::>{_executor.ssh_key_file}")
            print(f"env::>{_executor.env}")
    finally:
        result = ct.get_result(dispatch_id=dispatch_id, wait=False) if dispatch_id else None
        results_dir = result.results_dir if result else ""
        if args.teardown:
            try:
                _executor:ct.executor.BaseExecutor = executor_cls(
                    **{
                        **_get_executor_kwargs(args),
                        **{"action": "teardown"},
                    }
                )
                asyncio.run(_executor.setup({}))
                asyncio.run(
                    _executor.teardown(
                        {"dispatch_id": dispatch_id, "node_id": 0, "results_dir": results_dir}
                    )
                )
            except FileNotFoundError:
                pass

    return return_code


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    try:
        import covalent as ct
    except (KeyError, ImportError):
        module = pathlib.Path(__file__).resolve().parent
        cache_dir = pathlib.Path(f"/tmp/milabench/{module.name}_venv")
        python3 = str(cache_dir / "bin/python3")
        check_module = "import covalent"
        try:
            subprocess.run([python3, "-c", check_module], check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            cache_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run([sys.executable, "-m", "virtualenv", str(cache_dir)], check=True)
            subprocess.run([python3, "-m", "pip", "install", "-U", "pip"], check=True)
            subprocess.run([
                python3,
                "-m",
                "pip",
                "install",
                "-r",
                str(module / "requirements.txt")
            ], stdout=sys.stderr, check=True)
            subprocess.run([python3, "-c", check_module], check=True)
        return subprocess.call(
            [python3, __file__, *argv],
            env=_load_venv(cache_dir)
        )

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparser = subparsers.add_parser("serve")
    subparser.add_argument(f"argv", nargs=argparse.REMAINDER)
    for p in ("azure","ec2"):
        try:
            config = ct.get_config(f"executors.{p}")
        except KeyError:
            continue
        subparser = subparsers.add_parser(p)
        subparser.add_argument(f"--setup", action="store_true")
        subparser.add_argument(f"--teardown", action="store_true")
        for param, default in config.items():
            if param == "action":
                continue
            subparser.add_argument(f"--{param.replace('_', '-')}", default=default)

    try:
        cv_argv, argv = argv[:argv.index("--")], argv[argv.index("--")+1:]
    except ValueError:
        cv_argv, argv = argv, []

    args = parser.parse_args(cv_argv)

    if cv_argv[0] == "serve":
        assert not argv
        return serve(*args.argv)
    elif cv_argv[0] == "azure":
        executor_cls = ct.executor.AzureExecutor
    elif cv_argv[0] == "ec2":
        executor_cls = ct.executor.EC2Executor
    else:
        raise

    return executor(executor_cls, args, *argv)


if __name__ == "__main__":
    sys.exit(main())
