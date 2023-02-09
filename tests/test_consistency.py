import os
from pathlib import Path
import subprocess


tests_folder = Path(os.path.dirname(__file__))
config_folder = tests_folder / ".." / "config"
test_config_folder = tests_folder / "config"

assert config_folder.exists()
assert test_config_folder.exists()


def compute_diff(basefile, otherfile):
    """Note that python difflib is too basic for this and generates too big diffs"""
    # no need to check the return code because it is not 0 when there is a diff
    output = subprocess.run(
        ["diff", "-Z", basefile, otherfile], encoding="utf-8", capture_output=True
    )
    return output.stdout


def check_diff(line, allowed_diffs):
    if len(line) > 2 and line[0] in (">", "<"):

        for diff in allowed_diffs:
            if diff in line:
                return True

        assert False, f"{line} does not contain {allowed_diffs}"


def check_consistency(configurations, allowed_diffs):
    """Makes sure the diff between configuration files are expected"""
    base = config_folder / configurations[0]
    rest = configurations[1:]

    assert base.exists()

    for config in rest:
        other = config_folder / config
        assert other.exists()
        diff = compute_diff(base, other)

        # We print here since we expect developers will want
        # to see the actual diff when fixing this test
        print()
        print(f"Diff:")
        print(f" - {base}")
        print(f" - {other}")
        print(">>>> START")
        print(diff)
        print("<<<< END")

        for line in diff.splitlines():
            check_diff(line, allowed_diffs)


def test_ci_configuration_consistency():
    """Makes sure ci-configurations are not too far from each other.
    Only pip args/constraints should change.
    """

    configurations = ["ci-cuda.yaml", "ci-rocm.yaml"]

    authorized_diffs = [
        "pip:",
        "args:",  # Only pip use args, the bench use argv
        "--extra-index-url",
        "https://download.pytorch",
    ]

    check_consistency(configurations, authorized_diffs)


def test_configuration_consistency_standard():
    """Makes sure ci-configuration is not too far from the reality.
    Only batch-size tweaks should be necessary.
    """

    configurations_pair = [
        ("ci-cuda.yaml", "standard-cuda.yaml"),
        ("ci-rocm.yaml", "standard-rocm.yaml"),
    ]

    authorized_diffs = ["--batch-size", "argv", "--bs"]

    for configurations in configurations_pair:
        check_consistency(configurations, authorized_diffs)
