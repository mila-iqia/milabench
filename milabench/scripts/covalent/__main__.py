import argparse
import ast
import os
import pathlib
import subprocess
import sys
import tempfile
from time import sleep
import uuid


def _arg_pop(args:argparse.Namespace, key:str):
    value = args.__getattribute__(key)
    args.__delattr__(key)
    return value

ARGS_DEFAULT_SETUP = {
    "slurm": {
        "state_prefix": {},
        "state_id": {},
        "cluster_size": {},
        "keep_alive": {"action": "store_true"},
    }
}

ARGS_MAP = {
    "slurm": {
        "state_prefix": lambda args, k:_arg_pop(args, k),
        "state_id": lambda args, k:args.options.setdefault("job-name", _arg_pop(args, k)),
        "cluster_size": lambda args, k:args.options.setdefault("nodes", _arg_pop(args, k)),
        "keep_alive": lambda args, k:_arg_pop(args, k),
    }
}

_SETUP = {}

_TEARDOWN = {}

_CONNECTION_ATTRIBUTES = {
    "hostname": None,
    "username": None,
    "ssh_key_file": None,
    "private_ip": None,
    "env": None,
    "python_path": None,
    "slurm_job_id": None
}


def serve(*argv):
    return subprocess.run([
        "covalent",
        *argv
    ]).returncode


def _get_executor_kwargs(args):
    return {
        **{k:v for k,v in vars(args).items() if k not in ("setup", "teardown")},
    }


def _wait_for_any(*dispatch_ids):
    import covalent as ct

    dispatch_ids = set(dispatch_ids)
    while True:
        for dispatch_id in set(dispatch_ids):
            status = ct.get_result(
                dispatch_id=dispatch_id,
                wait=False,
                status_only=True
            )["status"]
            if status in [ct.status.COMPLETED]:
                yield dispatch_id
                dispatch_ids.remove(dispatch_id)
            elif status in [ct.status.FAILED, ct.status.CANCELLED]:
                raise RuntimeError(f"Job {dispatch_id} failed")
        sleep(5)


def _format(lines:list, **template_kv):
    for l in lines:
        for k, v in template_kv.items():
            if "{{" + k + "}}" in l:
                yield l[:l.find("{{")] + v + l[l.find("}}")+2:]
                break
        else:
            yield l


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


def _setup_terraform(executor:"ct.executor.BaseExecutor"):
    import covalent as ct

    result = ct.dispatch_sync(
        ct.lattice(executor.get_connection_attributes)
    )().result

    assert result and result[0]

    all_connection_attributes, _ = result
    master_host:str = next(iter(all_connection_attributes))

    if len(all_connection_attributes) > 1:
        # Add master node to known host to avoid unknown host error The
        # authenticity of host '[hostname] ([IP address])' can't be established.
        new_host = subprocess.run(
            ["ssh-keyscan", master_host],
            stdout=subprocess.PIPE,
            check=True
        ).stdout.decode("utf8")
        known_hosts = pathlib.Path("~/.ssh/known_hosts").expanduser()
        with known_hosts.open("at") as _f:
            _f.write(new_host)

        # Add ssh file to master node to allow connections to worker nodes
        ssh_key_file = all_connection_attributes[master_host]["ssh_key_file"]
        fn = pathlib.Path(ssh_key_file)
        result = ct.dispatch_sync(
            ct.lattice(executor.cp_to_remote)
        )(f".ssh/{fn.name.split('.')[0]}", str(fn))

        assert result.status == ct.status.COMPLETED

    return all_connection_attributes


def _teardown_terraform(executor:"ct.executor.BaseExecutor"):
    result = executor.stop_cloud_instance().result
    assert result is not None


def _slurm_executor(executor:"ct.executor.SlurmExecutor", job_uuid:uuid.UUID):
    import covalent as ct

    _executor = ct.executor.SlurmExecutor()
    _executor.from_dict(executor.to_dict())

    executor = _executor
    executor.conda_env = executor.conda_env or "covalent"
    bashrc_path = f"""''
{pathlib.Path(__file__).with_name("covalent_bashrc.sh").read_text()}
"""
    bashrc_path = "\n".join(
        _format(
            bashrc_path.splitlines(),
            conda_env=executor.conda_env,
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
            covalent_version=ct.__version__,
        )
    )
    executor.bashrc_path = executor.bashrc_path or "{bashrc_path}"
    if "{bashrc_path}" in executor.bashrc_path:
        executor.bashrc_path = executor.bashrc_path.format(bashrc_path=bashrc_path)
    executor.remote_workdir = executor.remote_workdir or "cov-{job_uuid}-workdir"
    executor.remote_workdir = executor.remote_workdir.format(job_uuid=job_uuid)
    executor.options["job-name"] = (
        executor.options.get("job-name", None) or f"cov-{job_uuid}"
    )

    return executor


def _setup_slurm(executor:"ct.executor.SlurmExecutor"):
    import covalent as ct

    job_uuid = uuid.uuid4()
    job_file = f"covalent_job_{job_uuid}"

    _executor = ct.executor.SlurmExecutor()
    _executor.from_dict(executor.to_dict())

    executor = _slurm_executor(executor, job_uuid)

    job_connection_executor = ct.executor.SlurmExecutor()
    job_connection_executor.from_dict(executor.to_dict())
    # Store job connection attributes
    job_connection_executor.prerun_commands = f"""
# print connection attributes
printenv |
    grep -E ".*SLURM.*NODENAME|.*SLURM.*JOB_ID" |
    sort -u >>"{job_file}" &&
srun printenv |
    grep -E ".*SLURM.*NODENAME|.*SLURM.*JOB_ID" |
    sort -u >>"{job_file}" &&
echo "USERNAME=$USER" >>"{job_file}" &&
echo "{job_uuid}" >>"{job_file}"
""".splitlines()

    query_executor = ct.executor.SlurmExecutor()
    query_executor.from_dict(executor.to_dict())
    query_executor.options = {
        "nodes": 1,
        "cpus-per-task": 1,
        "mem": 1000,
        "job-name": executor.options["job-name"],
    }

    @ct.electron()
    def _empty():
        pass

    @ct.electron()
    def _keep_alive():
        while True:
            sleep(60)

    @ct.electron()
    def _query_connection_attributes(milabench_bashrc:str=""):
        _job_file = pathlib.Path(job_file).expanduser()
        _job_file.touch()
        content = _job_file.read_text().splitlines()
        while (not content or content[-1].strip() != f"{job_uuid}"):
            sleep(5)
            content = _job_file.read_text().splitlines()

        nodes = []
        connection_attributes = _CONNECTION_ATTRIBUTES.copy()

        milabench_bashrc = "\n".join(
            _format(
                milabench_bashrc.splitlines(),
                milabench_env="cov-slurm-milabench",
                python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
            )
        )
        if milabench_bashrc:
            milabench_bashrc_file = _job_file.with_name("milabench_bashrc.sh").resolve()
            milabench_bashrc_file.write_text(milabench_bashrc)
            connection_attributes["env"] = str(milabench_bashrc_file)

        for l in _job_file.read_text().splitlines():
            try:
                key, value = l.strip().split("=")
            except ValueError:
                # end flag
                break
            if "NODENAME" in key and value not in nodes:
                nodes.append(value)
            elif "USERNAME" in key:
                connection_attributes["username"] = value
            elif "JOB_ID" in key:
                connection_attributes["slurm_job_id"] = value

        return {
            hostname: {
                **connection_attributes,
                **{
                    "hostname": hostname,
                    "private_ip": hostname,
                },
            }
            for hostname in nodes
        }

    try:
        # setup covalent for jobs
        next(_wait_for_any(ct.dispatch(ct.lattice(_empty, executor=query_executor))()))
        # setup nodes and retrieve connection attributes 
        job_dispatch_id = ct.dispatch(
            ct.lattice(
                lambda:_keep_alive(),
                executor=job_connection_executor
            ),
            disable_run=False
        )()
        query_dispatch_id = ct.dispatch(
            ct.lattice(
                _query_connection_attributes,
                executor=query_executor
            ),
            disable_run=False
        )(
            milabench_bashrc=pathlib.Path(__file__).with_name("milabench_bashrc.sh").read_text()
        )
        next(_wait_for_any(job_dispatch_id, query_dispatch_id))
        all_connection_attributes = ct.get_result(
            dispatch_id=query_dispatch_id,
            wait=False
        ).result

        assert all_connection_attributes

    except:
        _teardown_slurm(query_executor)
        raise

    return all_connection_attributes


def _teardown_slurm(executor:"ct.executor.SlurmExecutor"):
    import covalent as ct

    @ct.electron()
    def _empty():
        pass

    assert executor.options["job-name"], "Jobs to teardown must have an explicit name"

    _exec = _slurm_executor(executor, "DELETE")
    _exec.options = {
        "nodes": 1,
        "cpus-per-task": 1,
        "mem": 1000,
        "job-name": executor.options["job-name"],
    }
    _exec.prerun_commands = f"""
# cancel jobs
scancel --jobname="{_exec.options['job-name']}"
""".splitlines()
    ct.dispatch_sync(ct.lattice(_empty, executor=_exec))()


def executor(executor_cls, args:argparse.Namespace, *argv):
    import covalent as ct

    _SETUP[ct.executor.AzureExecutor] = _setup_terraform
    _SETUP[ct.executor.EC2Executor] = _setup_terraform
    _SETUP[ct.executor.SlurmExecutor] = _setup_slurm
    _TEARDOWN[ct.executor.AzureExecutor] = _teardown_terraform
    _TEARDOWN[ct.executor.EC2Executor] = _teardown_terraform
    _TEARDOWN[ct.executor.SlurmExecutor] = _teardown_slurm

    executor:ct.executor.BaseExecutor = executor_cls(
        **_get_executor_kwargs(args),
    )
    return_code = 0
    try:
        if args.setup:
            for hostname, connection_attributes in _SETUP[executor_cls](executor).items():
                print(f"hostname::>{hostname}")
                for attribute,value in connection_attributes.items():
                    if attribute == "hostname" or value is None:
                        continue
                    print(f"{attribute}::>{value}")

        if argv:
            result = ct.dispatch_sync(
                ct.lattice(executor.list_running_instances)
            )().result

            assert result

            dispatch_ids = set()
            for connection_attributes in result.get(
                (executor.state_prefix, executor.state_id),
                {"env": None}
            ).values():
                kwargs = {
                    **_get_executor_kwargs(args),
                    **connection_attributes
                }
                del kwargs["env"]
                del kwargs["private_ip"]

                _executor:ct.executor.BaseExecutor = executor_cls(**kwargs)

                dispatch_ids.add(
                    ct.dispatch(
                        ct.lattice(
                            lambda:ct.electron(_popen, executor=_executor)(argv)
                        ),
                        disable_run=False
                    )()
                )

            for dispatch_id in dispatch_ids:
                result = ct.get_result(dispatch_id=dispatch_id, wait=True).result

                _return_code, _, _ = result if result is not None else (1, "", "")
                return_code = return_code or _return_code

    finally:
        if args.teardown:
            _TEARDOWN[executor_cls](executor)

    return return_code


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    try:
        import covalent as ct
    except (KeyError, ImportError):
        from ..utils import run_in_module_venv
        check_if_module = "import covalent"
        return run_in_module_venv(__file__, check_if_module, argv)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparser = subparsers.add_parser("serve")
    subparser.add_argument(f"argv", nargs=argparse.REMAINDER)
    for p in ("azure", "ec2", "slurm"):
        try:
            config = ct.get_config(f"executors.{p}")
        except KeyError:
            continue
        subparser = subparsers.add_parser(p)
        subparser.add_argument(f"--setup", action="store_true")
        subparser.add_argument(f"--teardown", action="store_true")
        for param, default in config.items():
            if param.startswith("_"):
                continue
            add_argument_kwargs = {}
            if isinstance(default, bool):
                add_argument_kwargs["action"] = "store_false" if default else "store_true"
            elif any(isinstance(default, t) for t in [dict, list]):
                add_argument_kwargs["type"] = ast.literal_eval
                add_argument_kwargs["default"] = str(default)
            else:
                add_argument_kwargs["default"] = default
            subparser.add_argument(f"--{param.replace('_', '-')}", **add_argument_kwargs)

        for param, add_argument_kwargs in ARGS_DEFAULT_SETUP.get(p, {}).items():
            if param in config:
                raise ValueError(
                    f"Found existing argument {param} in both {config} and"
                    f" {ARGS_DEFAULT_SETUP}"
                )
            subparser.add_argument(f"--{param.replace('_', '-')}", **add_argument_kwargs)

    try:
        cv_argv, argv = argv[:argv.index("--")], argv[argv.index("--")+1:]
    except ValueError:
        cv_argv, argv = argv, []

    args = parser.parse_args(cv_argv)

    for arg, _map in ARGS_MAP.get(cv_argv[0], {}).items():
        _map(args, arg)

    if cv_argv[0] == "serve":
        assert not argv
        return serve(*args.argv)
    elif cv_argv[0] == "azure":
        executor_cls = ct.executor.AzureExecutor
    elif cv_argv[0] == "ec2":
        executor_cls = ct.executor.EC2Executor
    elif cv_argv[0] == "slurm":
        executor_cls = ct.executor.SlurmExecutor
    else:
        raise

    return executor(executor_cls, args, *argv)


if __name__ == "__main__":
    sys.exit(main())
