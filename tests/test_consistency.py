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
        if line[1:].strip().startswith("#"):
            return True

        for diff in allowed_diffs:
            if diff in line:
                return True

        return False
    
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
        
        badlines = []
        for line in diff.splitlines():
            
            if check_diff(line, allowed_diffs):
                ok = '[  OK]'
            else:
                badlines.append(line)
                ok = '[FAIL]'
                
            print(f'{ok}{line}')
            
        if badlines:
            assert False, "Unexpected diff, check {badlines}"


def test_configuration_consistency_standard():
    """Makes sure ci-configuration is not too far from the reality.
    Only batch-size tweaks should be necessary.
    """

    configurations_pair = [
        ("ci.yaml", "standard.yaml"),
    ]

    authorized_diffs = [
        "--batch-size", 
        "--batch_size",
        "argv", 
        "--bs", 
        "- no-rocm", 
        "stop: 10",
        "stop: 60",
        "skip: 5",
        "skip: 1",
        'model_name: "facebook/bart-base"',
        'model_name: "facebook/opt-2.7b"',
        "max_train_steps: 5",
        "max_train_steps: 30",
    ]

    for configurations in configurations_pair:
        check_consistency(configurations, authorized_diffs)
