from copy import deepcopy
import io
import json
import os
import re
import runpy
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime

from coleo import Option, default, tooled
import git
from omegaconf import OmegaConf
from voir.instruments.gpu import deduce_backend, select_backend
from milabench import ROOT_FOLDER

from milabench.alt_async import proceed
from milabench.utils import available_layers, blabla, multilogger

from .config import build_config
from .fs import XPath
from .log import TerminalFormatter
from .merge import merge
from .multi import MultiPackage
from .report import make_report
from .summary import aggregate, make_summary
from .system import build_system_config, option


def get_pack(defn):
    pack = XPath(defn["definition"])
    pack_glb = runpy.run_path(str(pack / "benchfile.py"))
    pack_cls = pack_glb["__pack__"]
    pack_obj = pack_cls(defn)
    return pack_obj


def dlist():
    return field(default_factory=list)


# fmt: off
@dataclass
class CommonArguments:
    config          : str       = None      # Configuration file
    system          : str       = None      # System Configuration file
    base            : str       = None      # Base path for code, venvs, data and runs
    use_current_env : bool      = False     # Whether to use the current environment
    select          : str       = ""        # Packs to select
    exclude         : str       = ""        # Packs to exclude
    override        : list[str] = dlist()   # Override configuration values
    capabilities    : str       = ""        # Define capabilities
# fmt : on

@tooled
def arguments():
    # Configuration file
    config: Option & str = None

    # System Configuration file
    system: Option & str = None

    # Base path for code, venvs, data and runs
    base: Option & str = None

    # Whether to use the current environment
    use_current_env: Option & bool = False

    # Packs to select
    select: Option & str = default("")

    # Packs to exclude
    exclude: Option & str = default("")

    # Override configuration values
    # [action: append]
    override: Option = []

    # Define capabilities
    capabilities: Option = ""

    return CommonArguments(
        config,
        system,
        base,
        use_current_env,
        select,
        exclude,
        override,
        capabilities,
    )

@tooled
def get_multipack(args = None, run_name=None, overrides={}):
    if args is None:
        args = arguments()

    override = [
        o if re.match(pattern=r"[.\w]+=", string=o) else f"={o}" for o in args.override
    ]

    override.extend(
        [f"*.capabilities.{entry}" for entry in args.capabilities.split(",") if entry]
    )

    if override:
        override_obj = OmegaConf.to_object(OmegaConf.from_dotlist(override))
        if "" in override_obj:
            override_obj = merge(override_obj, override_obj.pop(""))
        overrides = merge(overrides, override_obj)

    return _get_multipack(
        args=args,
        run_name=run_name,
        overrides=overrides,
    )


def selection_keys(defn):
    if "group" not in defn:
        raise Exception("Invalid benchmark:", defn["name"])
    sel = {
        "*",
        defn["name"],
        defn["group"],
        defn["install_group"],
        *defn.get("tags", []),
    }
    return sel


def get_base_defaults(base, arch="none", run_name="none"):
    try:
        user = os.getlogin()
    except OSError:
        user = "root"
    return {
        "_defaults": {
            "system": {
                "arch": arch,
                "sshkey": None,
                "nodes": [
                    {
                        "name": "local",
                        "ip": "127.0.0.1",
                        "sshport": 22,
                        "user": user,
                        "main": True,
                    }
                ],
            },
            "dirs": {
                "base": base,
                "venv": option("dirs.venv", str, default="${dirs.base}/venv/${install_group}"),
                "data": option("dirs.data", str, default="${dirs.base}/data"),
                "runs": option("dirs.runs", str, default="${dirs.base}/runs"),
                "extra": option("dirs.extra", str, default="${dirs.base}/extra/${group}"),
                "cache": option("dirs.cache", str, default="${dirs.base}/cache"),
            },
            "group": "${name}",
            "install_group": "${group}",
            "install_variant": "${system.arch}",
            "run_name": run_name,
            "enabled": True,
            "capabilities": {
                "nodes": 1,
            },
        }
    }


def deduce_arch():
    """Deduce the arch for installation and preparation purposes"""
    arch_guess = os.environ.get("MILABENCH_GPU_ARCH", None)

    if arch_guess is not None:
        return arch_guess

    return deduce_backend()


def init_arch(arch=None):
    """Initialize the monitor for the given arch"""
    arch = arch or deduce_arch()
    return select_backend(arch)


def is_selected(defn, args):
    if defn["name"] == "*":
        return False
    keys = selection_keys(defn)
    return (
        defn["enabled"]
        and not defn["name"].startswith("_")
        and defn.get("definition", None)
        and (not args.select or (keys & args.select))
        and (not args.exclude or not (keys & args.exclude))
    )

def _get_multipack(
    args: CommonArguments = None,
    run_name=None,
    overrides={},
    return_config=False,
):
    if args is None:
        args = arguments()

    if args.config is None:
        args.config = os.environ.get("MILABENCH_CONFIG", None)

    if args.config is None:
        sys.exit("Error: CONFIG argument not provided and no $MILABENCH_CONFIG")

    if args.system is None:
        args.system = os.environ.get("MILABENCH_SYSTEM", None)

    if args.system is None:
        if XPath(f"{args.config}.system").exists():
            args.system = f"{args.config}.system"

    if args.select:
        args.select = set(args.select.split(","))

    if args.exclude:
        args.exclude = set(args.exclude.split(","))

    if args.base is None:
        args.base = os.environ.get("MILABENCH_BASE", None)

    if not return_config and not args.base:
        sys.exit("Error: Neither --base nor $MILABENCH_BASE are set.")

    args.base = args.base and os.path.abspath(os.path.expanduser(args.base))

    if args.use_current_env:
        venv = os.environ.get("CONDA_PREFIX", None)
        if venv is None:
            venv = os.environ.get("VIRTUAL_ENV", None)
        if venv is None:
            print("Could not find virtual environment", file=sys.stderr)
            sys.exit(1)
        overrides = merge(overrides, {"*": {"dirs": {"venv": venv}}})

    if run_name is None:
        run_name = blabla() + ".{time}"

    now = str(datetime.today()).replace(" ", "_")
    run_name = run_name.format(time=now)

    arch = deduce_arch()
    base_defaults = get_base_defaults(
        base=args.base,
        arch=arch,
        run_name=run_name
    )
    system_config = build_system_config(
        args.system,
        defaults={"system": base_defaults["_defaults"]["system"]},
        gpu=True
    )
    overrides = merge({"*": system_config}, overrides)

    config = build_config(base_defaults, args.config, overrides)

    selected_config = {name: defn for name, defn in config.items() if is_selected(defn, args)}
    if return_config:
        return selected_config
    else:
        return MultiPackage(
            {name: get_pack(deepcopy(defn)) for name, defn in selected_config.items()}
        )


def _parse_report(pth, open=lambda x: x.open()):
    with open(pth) as f:
        lines = f.readlines()
        data = []
        good_lines = 0
        bad_lines = 0

        for line in lines:
            try:
                data.append(json.loads(line))
                good_lines += 1
            except Exception:
                import traceback

                print(f"Could not parse line inside {pth}\n\t- {line}")
                traceback.print_exc()
                bad_lines += 1

    if good_lines == 0 and bad_lines == 0:
        print(f"Empty file {pth}")

    if good_lines == 0 and bad_lines > 0:
        print(f"Unknow format for file {pth}")

    return data


def _read_reports(*runs):
    all_data = {}
    for folder in runs:
        for parent, _, filenames in os.walk(folder):
            for file in filenames:
                if not file.endswith(".data"):
                    continue
                pth = XPath(parent) / file
                all_data[str(pth)] = _parse_report(pth)

    return all_data


def _filter_reports(**reports):
    _reports = {}

    for k, report in reports.items():
        config = next(iter(e for e in report if e["event"] == "config"), None)
        if config is None:
            continue

        if config["data"]["name"] != "remote":
            _reports[k] = report

    return _reports


def _push_reports(reports_repo, runs):
    _SVG_COLORS = {
        "pass": "blue",
        "partial": "yellow",
        "failure": "red",
    }
    import milabench.scripts.badges as badges

    try:
        reports_repo = git.Repo(str(reports_repo))
    except (git.exc.InvalidGitRepositoryError, git.exc.NoSuchPathError):
        _repo = git.Repo(ROOT_FOLDER)
        repo_url = next(iter(_r.url for _r in _repo.remotes if _r.name == "origin"), None)
        reports_repo = git.Repo.clone_from(repo_url, str(reports_repo), branch="reports")
        config_reader = _repo.config_reader()
        config_writer = reports_repo.config_writer()
        for section in config_reader.sections():
            if not section.startswith("credential"):
                continue
            for option in config_reader.options(section):
                if not option.strip("_") == option:
                    continue
                for value in config_reader.get_values(section, option):
                    config_writer.add_value(section, option, value)
                config_writer.write()

    device_reports = {}
    for run in runs:
        reports = _read_reports(run)
        reports = list(_filter_reports(**reports).values())

        if not reports:
            continue

        meta = [e["data"] for _r in reports for e in _r if e["event"] == "meta"]

        for gpu in (_ for _meta in meta for _ in _meta["accelerators"]["gpus"].values()):
            device = gpu["product"].replace(" ", "_")
            break
        else:
            for _meta in meta:
                device = _meta["cpu"]["brand"].replace(" ", "_")
                break

        build = meta[0]["milabench"]["tag"]
        reports_dir = XPath(reports_repo.working_tree_dir) / build

        run = XPath(run)
        try:
            run.copy(reports_dir / device / run.name)
        except FileExistsError:
            pass

        for _f in (reports_dir / device / run.name).glob("*.stderr"):
            _f.unlink()

        device_reports.setdefault((device, build), set())
        device_reports[(device, build)].update(
            (reports_dir / device).glob("*/")
        )

    for (device, build), reports in device_reports.items():
        reports_dir = XPath(reports_repo.working_tree_dir) / build
        reports = _read_reports(*reports)
        reports = _filter_reports(**reports)
        summary = make_summary(reports)

        successes = [s["successes"] for s in summary.values()]
        failures = [s["failures"] for s in summary.values()]

        if sum(successes) == 0:
            text = "failure"
        elif any(failures):
            text = "partial"
        else:
            text = "pass"

        result = subprocess.run(
            [
                sys.executable,
                "-m", badges.__name__,
                "--left-text", device,
                "--right-text", text,
                "--right-color", _SVG_COLORS[text],
            ],
            capture_output=True,
            check=True
        )
        if result.returncode == 0:
            (reports_dir / device / "badge.svg").write_text(result.stdout.decode("utf8"))

        with open(str(reports_dir / device / "README.md"), "wt") as _f:
            _f.write("```\n")
            make_report(summary, stream=_f)
            _f.write("```\n")

        for cmd, _kwargs in (
            (["git", "pull"], {"check": True}),
            (["git", "add", build], {"check": True}),
            (["git", "commit", "-m", build], {"check": False}),
            (["git", "push"], {"check": True})
        ):
            subprocess.run(
                cmd,
                cwd=reports_repo.working_tree_dir,
                **_kwargs
            )


def _error_report(reports):
    out = {}
    for r, data in reports.items():
        try:
            agg = aggregate(data)
            (success,) = agg["data"]["success"]
            if not success:
                out[r] = [line for line in data if "#stdout" in line or "#stderr" in line]
        except:
            pass
    
    return out


def run_with_loggers(coro, loggers, mp=None):
    retcode = 0
    loggers = [logger for logger in loggers if logger is not None]

    try:
        with multilogger(*loggers) as log:
            for entry in proceed(coro):
                log(entry)

        retcode = log.result()

    except Exception:
        traceback.print_exc()
        retcode = -1

    finally:
        if mp:
            logdirs = {pack.logdir for pack in mp.packs.values() if pack.logdir}
            for logdir in logdirs:
                print(f"[DONE] Reports directory: {logdir}")

        return retcode


def run_sync(coro, terminal=True):
    return run_with_loggers(coro, [TerminalFormatter()] if terminal else [])


def validation_names(layers):
    if layers is None:
        layers = ""

    layers = layers.split(",")
    all_layers = available_layers()

    if "all" in layers:
        return all_layers

    results = set(["error", "ensure_rate", "version"])
    for layer in layers:
        if layer in all_layers:
            results.add(layer)
        elif layer != "":
            print(f"Layer {layer} does not exist")

    return results


def _short_make_report(runs, config):
    reports = None

    if runs:
        reports = _read_reports(*runs)
        summary = make_summary(reports)

    if config:
        config = _get_multipack(CommonArguments(config), return_config=True)

    stream = io.StringIO()

    make_report(
        summary,
        weights=config,
        stream=stream,
        sources=runs,
        errdata=reports and _error_report(reports),
    )

    return stream.getvalue()
