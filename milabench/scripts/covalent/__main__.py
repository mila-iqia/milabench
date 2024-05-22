import argparse
import subprocess
import sys


def serve(*argv):
    return subprocess.run([
        "covalent",
        *argv
    ]).returncode


def _get_executor_kwargs(args):
    return {
        **{k:v for k,v in vars(args).items() if k not in ("setup", "teardown")},
    }


def executor(executor_cls, args):
    import covalent as ct

    return_code = 0
    try:
        executor:ct.executor.BaseExecutor = executor_cls(
            **_get_executor_kwargs(args),
        )

        if args.setup:
            dispatch_id = ct.dispatch(
                ct.lattice(executor.get_connection_attributes), disable_run=False
            )()

            result = ct.get_result(dispatch_id=dispatch_id, wait=True).result

            assert result and result[0]

            all_connection_attributes, _ = result
            for hostname, connection_attributes in all_connection_attributes.items():
                print(f"hostname::>{hostname}")
                for attribute,value in connection_attributes.items():
                    if attribute == "hostname":
                        continue
                    print(f"{attribute}::>{value}")
    finally:
        if args.teardown:
            executor.stop_cloud_instance({})

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
    for p in ("azure","ec2"):
        try:
            config = ct.get_config(f"executors.{p}")
        except KeyError:
            continue
        subparser = subparsers.add_parser(p)
        subparser.add_argument(f"--setup", action="store_true")
        subparser.add_argument(f"--teardown", action="store_true")
        for param, default in config.items():
            add_argument_kwargs = {}
            if isinstance(default, bool):
                add_argument_kwargs["action"] = "store_false" if default else "store_true"
            else:
                add_argument_kwargs["default"] = default
            subparser.add_argument(f"--{param.replace('_', '-')}", **add_argument_kwargs)

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

    return executor(executor_cls, args)


if __name__ == "__main__":
    sys.exit(main())
