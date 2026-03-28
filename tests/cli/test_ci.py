import yaml
import pytest

from milabench.cli.ci import get_benchmark_groups, format_groups_for_ci


def _write_config(tmp_path, benchmarks):
    """Write a YAML config file with the given benchmark definitions."""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml.dump(benchmarks, default_flow_style=False))
    return config_file


@pytest.fixture
def multi_bench_config(tmp_path):
    """Config with several benchmarks across different definitions and tags."""
    benchmarks = {
        "_defaults": {
            "enabled": True,
        },
        "bench_a": {
            "inherits": "_defaults",
            "definition": "benchmarks/group_alpha",
            "tags": ["monogpu"],
        },
        "bench_b": {
            "inherits": "_defaults",
            "definition": "benchmarks/group_alpha",
            "tags": ["monogpu"],
        },
        "bench_c": {
            "inherits": "_defaults",
            "definition": "benchmarks/group_beta",
            "tags": ["multigpu"],
        },
        "bench_d": {
            "inherits": "_defaults",
            "definition": "benchmarks/group_gamma",
            "tags": ["multinode"],
        },
        "bench_disabled": {
            "inherits": "_defaults",
            "definition": "benchmarks/group_alpha",
            "tags": ["monogpu"],
            "enabled": False,
        },
        "bench_gated": {
            "inherits": "_defaults",
            "definition": "benchmarks/group_beta",
            "tags": ["monogpu", "gated"],
        },
    }
    return _write_config(tmp_path, benchmarks)


def test_groups_by_definition(multi_bench_config):
    """Benchmarks sharing a definition are grouped together."""
    groups = get_benchmark_groups(multi_bench_config)

    alpha = [v for k, v in groups.items() if k.endswith("group_alpha")]
    assert len(alpha) == 1
    assert alpha[0] == ["bench_a", "bench_b"]


def test_disabled_benchmarks_excluded(multi_bench_config):
    """Benchmarks with enabled=false do not appear in any group."""
    groups = get_benchmark_groups(multi_bench_config)
    all_names = [name for names in groups.values() for name in names]
    assert "bench_disabled" not in all_names


def test_exclude_tags_multinode(multi_bench_config):
    """Excluding 'multinode' drops benchmarks tagged multinode."""
    groups = get_benchmark_groups(multi_bench_config, exclude_tags={"multinode"})
    all_names = [name for names in groups.values() for name in names]
    assert "bench_d" not in all_names
    assert "bench_a" in all_names


def test_exclude_tags_gated(multi_bench_config):
    """Excluding 'gated' drops only gated benchmarks."""
    groups = get_benchmark_groups(multi_bench_config, exclude_tags={"gated"})
    all_names = [name for names in groups.values() for name in names]
    assert "bench_gated" not in all_names
    assert "bench_c" in all_names


def test_exclude_multiple_tags(multi_bench_config):
    """Excluding multiple tags filters all matching benchmarks."""
    groups = get_benchmark_groups(
        multi_bench_config, exclude_tags={"multinode", "gated"}
    )
    all_names = [name for names in groups.values() for name in names]
    assert "bench_d" not in all_names
    assert "bench_gated" not in all_names
    assert "bench_a" in all_names
    assert "bench_c" in all_names


def test_no_exclude_tags(multi_bench_config):
    """Without exclude_tags all enabled benchmarks appear."""
    groups = get_benchmark_groups(multi_bench_config)
    all_names = sorted(name for names in groups.values() for name in names)
    assert all_names == ["bench_a", "bench_b", "bench_c", "bench_d", "bench_gated"]


def test_format_groups_for_ci(multi_bench_config):
    """format_groups_for_ci produces sorted {name, select} entries."""
    groups = get_benchmark_groups(multi_bench_config, exclude_tags={"multinode", "gated"})
    result = format_groups_for_ci(groups)

    assert isinstance(result, list)
    ci_names = [entry["name"] for entry in result]
    assert ci_names == sorted(ci_names), "CI matrix entries should be sorted by name"

    for entry in result:
        assert "name" in entry
        assert "select" in entry
        bench_names = entry["select"].split(",")
        assert bench_names == sorted(bench_names), "select list should be sorted"


def test_format_uses_folder_basename(multi_bench_config):
    """The 'name' field in CI output is the definition folder basename."""
    groups = get_benchmark_groups(multi_bench_config, exclude_tags={"multinode", "gated"})
    result = format_groups_for_ci(groups)

    names = {entry["name"] for entry in result}
    assert "group_alpha" in names
    assert "group_beta" in names


def test_empty_config(tmp_path):
    """A config with only private entries yields no groups."""
    config_file = _write_config(tmp_path, {
        "_defaults": {"enabled": True},
        "_internal": {"definition": "benchmarks/x", "tags": ["monogpu"]},
    })
    groups = get_benchmark_groups(config_file)
    assert groups == {}


def test_no_definition_skipped(tmp_path):
    """Benchmarks without a 'definition' key are skipped."""
    config_file = _write_config(tmp_path, {
        "bench_no_def": {
            "tags": ["monogpu"],
            "enabled": True,
        },
    })
    groups = get_benchmark_groups(config_file)
    assert groups == {}


def test_include_resolution(tmp_path):
    """Configs using 'include' correctly pull in benchmarks from included files."""
    base_file = tmp_path / "base.yaml"
    base_file.write_text(yaml.dump({
        "_defaults": {"enabled": True},
        "base_bench": {
            "inherits": "_defaults",
            "definition": "benchmarks/base_group",
            "tags": ["monogpu"],
        },
    }))

    main_file = tmp_path / "main.yaml"
    main_file.write_text(yaml.dump({
        "include": ["base.yaml"],
        "extra_bench": {
            "inherits": "_defaults",
            "definition": "benchmarks/extra_group",
            "tags": ["multigpu"],
        },
    }))

    groups = get_benchmark_groups(main_file)
    all_names = [name for names in groups.values() for name in names]
    assert "base_bench" in all_names
    assert "extra_bench" in all_names


def test_inheritance_merges_tags(tmp_path):
    """Tags from parent and child are merged through inheritance."""
    config_file = _write_config(tmp_path, {
        "_base": {
            "definition": "benchmarks/x",
            "tags": ["monogpu"],
        },
        "child_bench": {
            "inherits": "_base",
            "tags": ["special"],
        },
    })
    groups = get_benchmark_groups(config_file, exclude_tags={"special"})
    all_names = [name for names in groups.values() for name in names]
    assert "child_bench" not in all_names


def test_all_disabled(tmp_path):
    """When every benchmark is disabled, no groups are returned."""
    config_file = _write_config(tmp_path, {
        "bench_x": {
            "definition": "benchmarks/x",
            "tags": ["monogpu"],
            "enabled": False,
        },
        "bench_y": {
            "definition": "benchmarks/y",
            "tags": ["multigpu"],
            "enabled": False,
        },
    })
    groups = get_benchmark_groups(config_file)
    assert groups == {}
