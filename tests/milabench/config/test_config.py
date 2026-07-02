import yaml
import pytest
from copy import deepcopy

from milabench.config import (
    build_config,
    _config_layers,
    _filter_config,
    resolve_inheritance,
    finalize_config,
    combine_args,
    expand_matrix,
    build_matrix_bench,
    relative_to,
    set_run_count,
    get_run_count,
    get_bench_count,
    get_config_global,
    get_base_folder,
    get_run_folder,
    config_global,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_yaml(path, data):
    path.write_text(yaml.dump(data, default_flow_style=False))
    return path


# ---------------------------------------------------------------------------
# Tests for set_run_count / get_run_count / get_bench_count (lines 17-31)
# ---------------------------------------------------------------------------


def test_set_and_get_run_count():
    set_run_count(10, 5)
    assert get_run_count() == 10


def test_get_bench_count():
    set_run_count(7, 3)
    assert get_bench_count() == 3


# ---------------------------------------------------------------------------
# Tests for get_base_folder / get_run_folder (lines 34-49)
# ---------------------------------------------------------------------------


def test_get_base_folder(tmp_path, monkeypatch):
    from milabench.system import system_global

    system_global.set({"base": str(tmp_path), "run_name": "run1"})
    try:
        result = get_base_folder()
        assert str(result) == str(tmp_path)
    finally:
        system_global.set(None)


def test_get_run_folder(tmp_path, monkeypatch):
    from milabench.system import system_global

    system_global.set({"base": str(tmp_path), "run_name": "my_run"})
    try:
        result = get_run_folder()
        assert str(result) == str(tmp_path / "runs" / "my_run")
    finally:
        system_global.set(None)


# ---------------------------------------------------------------------------
# Tests for relative_to (line 51-55)
# ---------------------------------------------------------------------------


def test_relative_to_absolute_path(tmp_path):
    p = relative_to(str(tmp_path / "foo"), "/ignored")
    assert str(p) == str(tmp_path / "foo")


def test_relative_to_relative_path(tmp_path):
    p = relative_to("sub/file.yaml", str(tmp_path))
    assert str(p) == str((tmp_path / "sub" / "file.yaml").resolve())


# ---------------------------------------------------------------------------
# Tests for _config_layers (lines 58-78)
# ---------------------------------------------------------------------------


def test_config_layers_from_file(tmp_path):
    cfg = {
        "bench_a": {"tags": ["monogpu"], "definition": "benchmarks/a"},
    }
    f = write_yaml(tmp_path / "config.yaml", cfg)
    layers = list(_config_layers([f]))
    assert len(layers) == 1
    assert "bench_a" in layers[0]
    assert "config_base" in layers[0]["bench_a"]
    assert "config_file" in layers[0]["bench_a"]
    assert "dirs" in layers[0]["bench_a"]


def test_config_layers_from_dict():
    cfg = {"bench_x": {"tags": ["monogpu"]}}
    layers = list(_config_layers([cfg]))
    assert layers == [cfg]


def test_config_layers_include_string(tmp_path):
    """include as a single string (not a list) is handled (line 68-69)."""
    base_cfg = {"base_bench": {"tags": ["multigpu"], "definition": "benchmarks/b"}}
    write_yaml(tmp_path / "base.yaml", base_cfg)

    main_cfg = {
        "include": "base.yaml",
        "main_bench": {"tags": ["monogpu"], "definition": "benchmarks/m"},
    }
    write_yaml(tmp_path / "main.yaml", main_cfg)

    layers = list(_config_layers([tmp_path / "main.yaml"]))
    assert len(layers) == 2
    names = [list(l.keys())[0] for l in layers]
    assert "base_bench" in names
    assert "main_bench" in names


def test_config_layers_include_list(tmp_path):
    inc1 = {"inc_a": {"tags": ["monogpu"], "definition": "benchmarks/ia"}}
    inc2 = {"inc_b": {"tags": ["multigpu"], "definition": "benchmarks/ib"}}
    write_yaml(tmp_path / "inc1.yaml", inc1)
    write_yaml(tmp_path / "inc2.yaml", inc2)

    main_cfg = {
        "include": ["inc1.yaml", "inc2.yaml"],
        "top_bench": {"tags": ["multinode"], "definition": "benchmarks/t"},
    }
    write_yaml(tmp_path / "main.yaml", main_cfg)

    layers = list(_config_layers([tmp_path / "main.yaml"]))
    all_keys = set()
    for l in layers:
        all_keys.update(l.keys())
    assert {"inc_a", "inc_b", "top_bench"} <= all_keys


def test_config_layers_nested_include(tmp_path):
    """Nested includes are recursively resolved."""
    deep = {"deep_bench": {"tags": ["monogpu"], "definition": "benchmarks/d"}}
    write_yaml(tmp_path / "deep.yaml", deep)

    mid_cfg = {
        "include": ["deep.yaml"],
        "mid_bench": {"tags": ["multigpu"], "definition": "benchmarks/mid"},
    }
    write_yaml(tmp_path / "mid.yaml", mid_cfg)

    top_cfg = {
        "include": ["mid.yaml"],
        "top_bench": {"tags": ["multinode"], "definition": "benchmarks/top"},
    }
    write_yaml(tmp_path / "top.yaml", top_cfg)

    layers = list(_config_layers([tmp_path / "top.yaml"]))
    all_keys = set()
    for l in layers:
        all_keys.update(l.keys())
    assert {"deep_bench", "mid_bench", "top_bench"} <= all_keys


# ---------------------------------------------------------------------------
# Tests for resolve_inheritance (lines 81-91)
# ---------------------------------------------------------------------------


def test_resolve_inheritance_basic():
    all_configs = {
        "_base": {"definition": "benchmarks/x", "tags": ["monogpu"], "enabled": True},
        "child": {"inherits": "_base", "tags": ["special"]},
    }
    result = resolve_inheritance(all_configs["child"], all_configs)
    assert result["definition"] == "benchmarks/x"
    assert "monogpu" in result["tags"]
    assert "special" in result["tags"]
    assert result["enabled"] is True


def test_resolve_inheritance_chain():
    """Multi-level inheritance."""
    all_configs = {
        "_grandparent": {"tags": ["monogpu"], "a": 1},
        "_parent": {"inherits": "_grandparent", "tags": ["fast"], "b": 2},
        "child": {"inherits": "_parent", "tags": ["special"], "c": 3},
    }
    result = resolve_inheritance(all_configs["child"], all_configs)
    assert result["a"] == 1
    assert result["b"] == 2
    assert result["c"] == 3
    assert sorted(result["tags"]) == ["fast", "monogpu", "special"]


def test_resolve_inheritance_star_config():
    """The '*' entry is merged into every bench."""
    all_configs = {
        "*": {"global_key": "hello"},
        "bench": {"tags": ["monogpu"]},
    }
    result = resolve_inheritance(all_configs["bench"], all_configs)
    assert result["global_key"] == "hello"
    assert "monogpu" in result["tags"]


def test_resolve_inheritance_no_parent():
    """No inheritance, no '*' — config is returned as-is."""
    all_configs = {"bench": {"tags": ["monogpu"], "x": 1}}
    result = resolve_inheritance(all_configs["bench"], all_configs)
    assert result == {"tags": ["monogpu"], "x": 1}


# ---------------------------------------------------------------------------
# Tests for finalize_config (lines 94-114)
# ---------------------------------------------------------------------------


def test_finalize_config_sets_name_and_tag():
    cfg = {"tags": ["monogpu"], "config_base": "/tmp"}
    result = finalize_config("my_bench", cfg)
    assert result["name"] == "my_bench"
    assert result["tag"] == ["my_bench"]


def test_finalize_config_warns_no_monitor_tag(capsys):
    """Prints warning when no monitor tag is present (line 111)."""
    cfg = {"tags": ["custom"], "config_base": "/tmp"}
    finalize_config("bench_no_monitor", cfg)
    captured = capsys.readouterr()
    assert "should have exactly one monitor tag" in captured.out


def test_finalize_config_warns_multiple_monitor_tags(capsys):
    """Prints warning when multiple monitor tags are present."""
    cfg = {"tags": ["monogpu", "multigpu"], "config_base": "/tmp"}
    finalize_config("bench_multi", cfg)
    captured = capsys.readouterr()
    assert "should have exactly one monitor tag" in captured.out


def test_finalize_config_private_name_no_warning(capsys):
    """Private names (starting with _) skip the monitor tag check."""
    cfg = {"tags": [], "config_base": "/tmp"}
    finalize_config("_internal", cfg)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_finalize_config_star_name_no_warning(capsys):
    """The '*' name skips the monitor tag check."""
    cfg = {"tags": [], "config_base": "/tmp"}
    finalize_config("*", cfg)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_finalize_config_resolves_relative_definition(tmp_path):
    defn_dir = tmp_path / "benchmarks" / "mybench"
    defn_dir.mkdir(parents=True)
    cfg = {
        "tags": ["monogpu"],
        "definition": "benchmarks/mybench",
        "config_base": str(tmp_path),
    }
    result = finalize_config("my_bench", cfg)
    assert str(defn_dir.resolve()) in result["definition"]


# ---------------------------------------------------------------------------
# Tests for combine_args (lines 117-129)
# ---------------------------------------------------------------------------


def test_combine_args_empty():
    results = list(combine_args({}, {}))
    assert results == [{}]


def test_combine_args_single_key_list():
    args = {"size": [1, 2, 3]}
    results = [deepcopy(r) for r in combine_args(args, {})]
    assert len(results) == 3
    sizes = [r["size"] for r in results]
    assert set(sizes) == {1, 2, 3}


def test_combine_args_multiple_keys():
    args = {"a": [1, 2], "b": ["x", "y"]}
    results = list(combine_args(args, {}))
    assert len(results) == 4


def test_combine_args_non_iterable_value():
    """When a value is not iterable, the bare except catches and uses it directly (line 127-129)."""
    args = {"scalar": 42}
    results = list(combine_args(args, {}))
    assert len(results) == 1
    assert results[0]["scalar"] == 42


# ---------------------------------------------------------------------------
# Tests for expand_matrix (lines 131-170)
# ---------------------------------------------------------------------------


def test_expand_matrix_no_matrix():
    """No 'matrix' key returns original bench unchanged."""
    cfg = {"tags": ["monogpu"], "definition": "benchmarks/x"}
    result = expand_matrix("bench_a", cfg)
    assert result == [("bench_a", cfg)]


def test_expand_matrix_basic():
    """Expands a matrix with a name template and argv substitution (lines 135-170)."""
    cfg = {
        "matrix": {"size": [128, 256]},
        "job": {
            "name": "bench_{size}",
            "tags": ["monogpu"],
            "definition": "benchmarks/x",
            "argv": {"--batch-size": "{size}"},
        },
    }
    result = expand_matrix("template", cfg)
    assert len(result) == 2
    names = [name for name, _ in result]
    assert "bench_128" in names
    assert "bench_256" in names
    for name, bench in result:
        assert bench["argv"]["--batch-size"] in ("128", "256")
        assert bench["matrix"]["size"] in (128, 256)


def test_expand_matrix_client_server_argv():
    """Matrix expansion substitutes into client/server argv paths."""
    cfg = {
        "matrix": {"model": ["small", "large"]},
        "job": {
            "name": "infer_{model}",
            "tags": ["monogpu"],
            "definition": "benchmarks/x",
            "client": {"argv": {"--model": "{model}"}},
            "server": {"argv": {"--checkpoint": "{model}"}},
        },
    }
    result = expand_matrix("template", cfg)
    assert len(result) == 2
    names = {name for name, _ in result}
    assert names == {"infer_small", "infer_large"}
    for name, bench in result:
        expected_model = name.replace("infer_", "")
        assert bench["client"]["argv"]["--model"] == expected_model
        assert bench["server"]["argv"]["--checkpoint"] == expected_model


def test_expand_matrix_non_format_value():
    """Argv values that can't be formatted are kept as-is (bare except line 163-164)."""
    cfg = {
        "matrix": {"n": [1]},
        "job": {
            "name": "bench_{n}",
            "tags": ["monogpu"],
            "definition": "benchmarks/x",
            "argv": {"--flag": 999},
        },
    }
    result = expand_matrix("template", cfg)
    assert len(result) == 1
    _, bench = result[0]
    assert bench["argv"]["--flag"] == 999


# ---------------------------------------------------------------------------
# Tests for build_matrix_bench (lines 173-183)
# ---------------------------------------------------------------------------


def test_build_matrix_bench_duplicate_name_raises():
    """Duplicate bench names from matrix expansion raise ValueError (line 178-179)."""
    configs = {
        "a": {
            "matrix": {"x": [1]},
            "job": {"name": "dup", "tags": ["monogpu"]},
        },
        "b": {
            "matrix": {"x": [1]},
            "job": {"name": "dup", "tags": ["multigpu"]},
        },
    }
    with pytest.raises(ValueError, match="not unique"):
        build_matrix_bench(configs)


def test_build_matrix_bench_no_matrix():
    configs = {
        "simple": {"tags": ["monogpu"], "definition": "benchmarks/x"},
    }
    result = build_matrix_bench(configs)
    assert "simple" in result


# ---------------------------------------------------------------------------
# Tests for build_config (full integration, lines 186-201)
# ---------------------------------------------------------------------------


def test_build_config_single_file(tmp_path):
    cfg = {
        "_defaults": {"enabled": True},
        "bench_a": {
            "inherits": "_defaults",
            "tags": ["monogpu"],
            "definition": "benchmarks/a",
        },
    }
    f = write_yaml(tmp_path / "config.yaml", cfg)
    result = build_config(f)
    assert "bench_a" in result
    assert "_defaults" not in result
    assert result["bench_a"]["name"] == "bench_a"
    assert result["bench_a"]["enabled"] is True


def test_build_config_excludes_private_and_star(tmp_path):
    cfg = {
        "_private": {"definition": "benchmarks/p", "tags": ["monogpu"]},
        "*": {"global_val": 1},
        "public": {"tags": ["monogpu"], "definition": "benchmarks/pub"},
    }
    f = write_yaml(tmp_path / "config.yaml", cfg)
    result = build_config(f)
    assert "public" in result
    assert "_private" not in result
    assert "*" not in result


def test_build_config_with_includes(tmp_path):
    base = {"base_bench": {"tags": ["multigpu"], "definition": "benchmarks/base"}}
    write_yaml(tmp_path / "base.yaml", base)

    main = {
        "include": ["base.yaml"],
        "main_bench": {"tags": ["monogpu"], "definition": "benchmarks/main"},
    }
    write_yaml(tmp_path / "main.yaml", main)

    result = build_config(tmp_path / "main.yaml")
    assert "base_bench" in result
    assert "main_bench" in result


def test_build_config_inheritance_merges_tags(tmp_path):
    cfg = {
        "_base": {"tags": ["monogpu"], "definition": "benchmarks/x"},
        "child": {"inherits": "_base", "tags": ["extra"]},
    }
    f = write_yaml(tmp_path / "config.yaml", cfg)
    result = build_config(f)
    assert "child" in result
    assert "extra" in sorted(result["child"]["tags"])
    assert "monogpu" in sorted(result["child"]["tags"])


def test_build_config_multiple_files(tmp_path):
    cfg1 = {"bench1": {"tags": ["monogpu"], "definition": "benchmarks/b1"}}
    cfg2 = {"bench2": {"tags": ["multigpu"], "definition": "benchmarks/b2"}}
    f1 = write_yaml(tmp_path / "a.yaml", cfg1)
    f2 = write_yaml(tmp_path / "b.yaml", cfg2)
    result = build_config(f1, f2)
    assert "bench1" in result
    assert "bench2" in result


def test_build_config_star_merges_globally(tmp_path):
    cfg = {
        "*": {"extra": "global_value"},
        "bench": {"tags": ["monogpu"], "definition": "benchmarks/x"},
    }
    f = write_yaml(tmp_path / "config.yaml", cfg)
    result = build_config(f)
    assert result["bench"]["extra"] == "global_value"


def test_build_config_with_matrix(tmp_path):
    defn = tmp_path / "benchmarks" / "x"
    defn.mkdir(parents=True)
    cfg = {
        "template": {
            "matrix": {"n": [1, 2]},
            "job": {
                "name": "bench_{n}",
                "tags": ["monogpu"],
                "definition": str(defn),
                "argv": {"--n": "{n}"},
            },
        },
    }
    f = write_yaml(tmp_path / "config.yaml", cfg)
    result = build_config(f)
    assert "bench_1" in result
    assert "bench_2" in result
    assert "template" not in result


def test_build_config_empty_file(tmp_path):
    """Empty YAML file (None content) raises an error at iteration."""
    f = tmp_path / "empty.yaml"
    f.write_text("")
    with pytest.raises((TypeError, AttributeError)):
        build_config(f)


def test_build_config_sets_global(tmp_path):
    """build_config sets the config_global context var."""
    cfg = {"bench": {"tags": ["monogpu"], "definition": "benchmarks/x"}}
    f = write_yaml(tmp_path / "config.yaml", cfg)
    build_config(f)
    raw = config_global.get()
    assert raw is not None
    assert "bench" in raw


def test_get_config_global_after_build(tmp_path):
    """get_config_global returns filtered config (line 23)."""
    cfg = {
        "_private": {"tags": ["monogpu"], "definition": "benchmarks/p"},
        "public": {"tags": ["monogpu"], "definition": "benchmarks/pub"},
    }
    f = write_yaml(tmp_path / "config.yaml", cfg)
    build_config(f)
    result = get_config_global()
    assert "public" in result
    assert "_private" not in result


# ---------------------------------------------------------------------------
# Tests for _filter_config (lines 204-207)
# ---------------------------------------------------------------------------


def test_filter_config_removes_private():
    cfg = {
        "_hidden": {"name": "_hidden"},
        "*": {"name": "*"},
        "visible": {"name": "visible"},
    }
    result = _filter_config(cfg)
    assert "visible" in result
    assert "_hidden" not in result
    assert "*" not in result


def test_filter_config_custom_filter():
    cfg = {
        "keep_me": {"name": "keep_me", "x": 1},
        "drop_me": {"name": "drop_me", "x": 2},
    }
    result = _filter_config(cfg, filter=lambda d: d["x"] == 1)
    assert "keep_me" in result
    assert "drop_me" not in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_config_layers_file_not_found():
    """Missing config file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        list(_config_layers(["/nonexistent/path/config.yaml"]))


def test_config_layers_include_file_not_found(tmp_path):
    """Missing include file raises FileNotFoundError."""
    cfg = {
        "include": ["missing.yaml"],
        "bench": {"tags": ["monogpu"], "definition": "benchmarks/x"},
    }
    write_yaml(tmp_path / "main.yaml", cfg)
    with pytest.raises(FileNotFoundError):
        list(_config_layers([tmp_path / "main.yaml"]))


def test_resolve_inheritance_missing_parent_raises():
    """Reference to nonexistent parent raises KeyError."""
    all_configs = {"child": {"inherits": "nonexistent"}}
    with pytest.raises(KeyError):
        resolve_inheritance(all_configs["child"], all_configs)


def test_build_config_dict_input():
    """_config_layers accepts dicts directly — verify through build_config-like flow."""
    cfg = {"bench_d": {"tags": ["monogpu"], "definition": "benchmarks/d"}}
    layers = list(_config_layers([cfg]))
    assert layers == [cfg]
