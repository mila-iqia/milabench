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
    line = line.strip()
    
    if len(line) > 2 and line[0] in (">", "<"):
        line = line[1:].strip()
        if line.startswith("#"):
            return True

        for diff in allowed_diffs:
            if diff in line:
                return True

        return False
    else:
        return True


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

        difflines = diff.splitlines()

    badlines = []
    for line in difflines:
        if not check_diff(line, allowed_diffs):
            badlines.append(line)

    for line in badlines:
        print("DISALLOWED LINE IN DIFF:", line)

    assert not badlines


def test_configuration_consistency_standard():
    """Makes sure ci-configuration is not too far from the reality.
    Only batch-size tweaks should be necessary.
    """

    configurations_pair = [
        ("ci.yaml", "standard.yaml"),
    ]

    authorized_diffs = [
        "--train_batch_size",
        "--batch-size", 
        "--batch_size",
        "argv", 
        "--bs",
        "- no-rocm",
        "stop:",
        "skip:",
        'model_name: "facebook/bart-base"',
        'model_name: "facebook/opt-2.7b"',
        "max_train_steps: 5",
        "max_train_steps: 30",
    ]

    for configurations in configurations_pair:
        check_consistency(configurations, authorized_diffs)
