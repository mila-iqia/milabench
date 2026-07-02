"""Tests for milabench.merge module."""

import pytest
import yaml

from milabench.merge import DELETE, cleanup, merge, _tweak


# ============================================================
# cleanup tests
# ============================================================


class TestCleanup:
    def test_cleanup_primitive_values(self):
        assert cleanup(42) == 42
        assert cleanup("hello") == "hello"
        assert cleanup(3.14) == 3.14
        assert cleanup(None) is None
        assert cleanup(True) is True

    def test_cleanup_dict_removes_delete_values(self):
        d = {"a": 1, "b": DELETE, "c": 3}
        result = cleanup(d)
        assert result == {"a": 1, "c": 3}

    def test_cleanup_dict_empty(self):
        assert cleanup({}) == {}

    def test_cleanup_dict_all_delete(self):
        d = {"a": DELETE, "b": DELETE}
        assert cleanup(d) == {}

    def test_cleanup_dict_nested_delete(self):
        d = {"a": {"x": 1, "y": DELETE}, "b": 2}
        result = cleanup(d)
        assert result == {"a": {"x": 1}, "b": 2}

    def test_cleanup_dict_preserves_type(self):
        from collections import OrderedDict

        d = OrderedDict([("a", 1), ("b", DELETE), ("c", 3)])
        result = cleanup(d)
        assert isinstance(result, OrderedDict)
        assert result == OrderedDict([("a", 1), ("c", 3)])

    def test_cleanup_list(self):
        assert cleanup([1, 2, 3]) == [1, 2, 3]

    def test_cleanup_list_empty(self):
        assert cleanup([]) == []

    def test_cleanup_list_with_nested_dicts(self):
        lst = [{"a": 1, "b": DELETE}, {"c": 3}]
        result = cleanup(lst)
        assert result == [{"a": 1}, {"c": 3}]

    def test_cleanup_tuple(self):
        result = cleanup((1, 2, 3))
        assert result == (1, 2, 3)
        assert isinstance(result, tuple)

    def test_cleanup_set(self):
        result = cleanup({1, 2, 3})
        assert result == {1, 2, 3}
        assert isinstance(result, set)

    def test_cleanup_frozenset(self):
        result = cleanup(frozenset([1, 2, 3]))
        assert result == frozenset([1, 2, 3])
        assert isinstance(result, frozenset)

    def test_cleanup_nested_list_in_dict(self):
        d = {"a": [1, 2], "b": DELETE}
        result = cleanup(d)
        assert result == {"a": [1, 2]}


# ============================================================
# merge tests - dict + dict
# ============================================================


class TestMergeDict:
    def test_merge_simple_dicts(self):
        d1 = {"a": 1, "b": 2}
        d2 = {"b": 3, "c": 4}
        result = merge(d1, d2)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_merge_empty_dicts(self):
        assert merge({}, {}) == {}

    def test_merge_first_empty(self):
        result = merge({}, {"a": 1})
        assert result == {"a": 1}

    def test_merge_second_empty(self):
        result = merge({"a": 1}, {})
        assert result == {"a": 1}

    def test_merge_nested_dicts(self):
        d1 = {"a": {"x": 1, "y": 2}, "b": 3}
        d2 = {"a": {"y": 99, "z": 100}}
        result = merge(d1, d2)
        assert result == {"a": {"x": 1, "y": 99, "z": 100}, "b": 3}

    def test_merge_deeply_nested(self):
        d1 = {"a": {"b": {"c": {"d": 1}}}}
        d2 = {"a": {"b": {"c": {"e": 2}}}}
        result = merge(d1, d2)
        assert result == {"a": {"b": {"c": {"d": 1, "e": 2}}}}

    def test_merge_delete_key(self):
        """Line 49: DELETE sentinel removes key from merged result."""
        d1 = {"a": 1, "b": 2, "c": 3}
        d2 = {"b": DELETE}
        result = merge(d1, d2)
        assert result == {"a": 1, "c": 3}

    def test_merge_delete_nested_key(self):
        d1 = {"a": {"x": 1, "y": 2}, "b": 3}
        d2 = {"a": {"y": DELETE}}
        result = merge(d1, d2)
        assert result == {"a": {"x": 1}, "b": 3}

    def test_merge_delete_all_keys(self):
        d1 = {"a": 1, "b": 2}
        d2 = {"a": DELETE, "b": DELETE}
        result = merge(d1, d2)
        assert result == {}

    def test_merge_new_key_with_delete_values(self):
        """New key from d2 with nested DELETE is cleaned up."""
        d1 = {"a": 1}
        d2 = {"b": {"x": 1, "y": DELETE}}
        result = merge(d1, d2)
        assert result == {"a": 1, "b": {"x": 1}}

    def test_merge_preserves_dict_type(self):
        from collections import OrderedDict

        d1 = OrderedDict([("a", 1)])
        d2 = {"b": 2}
        result = merge(d1, d2)
        assert isinstance(result, OrderedDict)

    def test_merge_overwrite_scalar_with_dict(self):
        d1 = {"a": 1}
        d2 = {"a": {"nested": True}}
        result = merge(d1, d2)
        assert result == {"a": {"nested": True}}

    def test_merge_dict_value_with_scalar_raises(self):
        """Merging a dict with a non-iterable scalar as d2 raises TypeError
        because the dict overload tries 'k in d2'."""
        d1 = {"a": {"nested": True}}
        d2 = {"a": 42}
        with pytest.raises(TypeError):
            merge(d1, d2)


# ============================================================
# merge tests - list + list
# ============================================================


class TestMergeList:
    def test_merge_lists_replaces(self):
        """List merge replaces l1 with l2."""
        result = merge([1, 2, 3], [4, 5])
        assert result == [4, 5]

    def test_merge_lists_empty_second(self):
        result = merge([1, 2, 3], [])
        assert result == []

    def test_merge_lists_empty_first(self):
        result = merge([], [4, 5])
        assert result == [4, 5]

    def test_merge_lists_both_empty(self):
        result = merge([], [])
        assert result == []


# ============================================================
# merge tests - list + dict (append mode)
# ============================================================


class TestMergeListDict:
    def test_merge_list_dict_append(self):
        """Lines 67-68: dict with 'append' key appends to list."""
        result = merge([1, 2], {"append": [3, 4]})
        assert result == [1, 2, 3, 4]

    def test_merge_list_dict_append_empty(self):
        result = merge([1, 2], {"append": []})
        assert result == [1, 2]

    def test_merge_list_dict_append_to_empty_list(self):
        result = merge([], {"append": [1, 2]})
        assert result == [1, 2]

    def test_merge_list_dict_no_append_key_raises(self):
        """Lines 69-70: dict without 'append' raises TypeError."""
        with pytest.raises(TypeError, match="Cannot merge list and dict"):
            merge([1, 2], {"other": [3, 4]})

    def test_merge_list_dict_no_append_key_empty_dict_raises(self):
        with pytest.raises(TypeError, match="Cannot merge list and dict"):
            merge([1, 2], {})


# ============================================================
# merge tests - object fallback
# ============================================================


class TestMergeObject:
    def test_merge_scalar_override(self):
        """Line 78: fallback returns cleanup(b) when a has no __merge__."""
        assert merge(1, 2) == 2
        assert merge("old", "new") == "new"
        assert merge(True, False) is False

    def test_merge_object_with_merge_method(self):
        """Line 75-76: object with __merge__ method uses it."""

        class Mergeable:
            def __init__(self, val):
                self.val = val

            def __merge__(self, other):
                return Mergeable(self.val + other.val)

        a = Mergeable(10)
        b = Mergeable(5)
        result = merge(a, b)
        assert result.val == 15

    def test_merge_object_without_merge_method(self):
        """Line 78: fallback cleans and returns b."""

        class Plain:
            pass

        a = Plain()
        b = "replacement"
        assert merge(a, b) == "replacement"

    def test_merge_none_values(self):
        assert merge(None, 42) == 42
        assert merge(None, None) is None

    def test_merge_scalar_with_dict_cleanup(self):
        """Fallback cleanup(b) applies when b is a dict with DELETE."""
        result = merge(42, {"keep": 1, "remove": DELETE})
        assert result == {"keep": 1}


# ============================================================
# _tweak tests
# ============================================================


class TestTweak:
    def test_tweak_no_dots(self):
        """Lines 87-98: simple dict with no dots passes through."""
        d = {"a": 1, "b": 2}
        result = _tweak(d)
        assert result == {"a": 1, "b": 2}

    def test_tweak_single_dot(self):
        """Dot notation expands to nested dict."""
        d = {"a.b": 1}
        result = _tweak(d)
        assert result == {"a": {"b": 1}}

    def test_tweak_multiple_dots(self):
        d = {"a.b.c": 1}
        result = _tweak(d)
        assert result == {"a": {"b": {"c": 1}}}

    def test_tweak_mixed_dot_and_plain(self):
        d = {"a.b": 1, "c": 2}
        result = _tweak(d)
        assert result == {"a": {"b": 1}, "c": 2}

    def test_tweak_merges_overlapping_dot_keys(self):
        d = {"a.x": 1, "a.y": 2}
        result = _tweak(d)
        assert result == {"a": {"x": 1, "y": 2}}

    def test_tweak_triple_chevron_inlines_subdict(self):
        """Lines 94-95: '<<<' key inlines its value's items."""
        d = {"<<<": {"x": 10, "y": 20}, "z": 30}
        result = _tweak(d)
        assert result == {"x": 10, "y": 20, "z": 30}

    def test_tweak_triple_chevron_with_dots(self):
        d = {"<<<": {"a.b": 1}, "c": 2}
        result = _tweak(d)
        assert result == {"a": {"b": 1}, "c": 2}

    def test_tweak_triple_chevron_override(self):
        """'<<<' values merged with later keys."""
        d = {"<<<": {"a": 1}, "a": 2}
        result = _tweak(d)
        assert result == {"a": 2}

    def test_tweak_empty_dict(self):
        with pytest.raises(TypeError):
            _tweak({})

    def test_tweak_single_key(self):
        assert _tweak({"key": "value"}) == {"key": "value"}


# ============================================================
# YAML integration tests
# ============================================================


class TestYAMLIntegration:
    def test_yaml_expand_dot_tag(self):
        """Line 106: !expand-dot tag triggers tweak."""
        doc = "!expand-dot\na.b: 1\nc: 2\n"
        result = yaml.safe_load(doc)
        assert result == {"a": {"b": 1}, "c": 2}

    def test_yaml_delete_tag(self):
        """Line 107: !delete tag produces DELETE sentinel."""
        doc = "key: !delete\n"
        result = yaml.safe_load(doc)
        assert result == {"key": DELETE}

    def test_yaml_expand_dot_nested(self):
        doc = "!expand-dot\na.b.c: 42\n"
        result = yaml.safe_load(doc)
        assert result == {"a": {"b": {"c": 42}}}

    def test_yaml_delete_in_merge(self):
        """DELETE from YAML can be used in merge."""
        doc = "key: !delete\n"
        parsed = yaml.safe_load(doc)
        d1 = {"key": "value", "other": 1}
        result = merge(d1, parsed)
        assert result == {"other": 1}


# ============================================================
# DELETE sentinel tests
# ============================================================


class TestDeleteSentinel:
    def test_delete_is_named(self):
        assert repr(DELETE) == "DELETE"

    def test_delete_identity(self):
        assert DELETE is DELETE


# ============================================================
# Edge cases and complex scenarios
# ============================================================


class TestComplexScenarios:
    def test_merge_chain(self):
        """Multiple sequential merges."""
        from functools import reduce

        dicts = [
            {"a": 1, "b": 2},
            {"b": 3, "c": 4},
            {"a": DELETE, "d": 5},
        ]
        result = reduce(merge, dicts)
        assert result == {"b": 3, "c": 4, "d": 5}

    def test_merge_large_dicts(self):
        d1 = {f"key_{i}": i for i in range(100)}
        d2 = {f"key_{i}": i * 10 for i in range(50, 150)}
        result = merge(d1, d2)
        assert len(result) == 150
        assert result["key_0"] == 0
        assert result["key_50"] == 500
        assert result["key_149"] == 1490

    def test_merge_dict_with_list_value_replaced(self):
        d1 = {"items": [1, 2, 3]}
        d2 = {"items": [4, 5]}
        result = merge(d1, d2)
        assert result == {"items": [4, 5]}

    def test_merge_dict_with_list_append_via_nested(self):
        d1 = {"items": [1, 2, 3]}
        d2 = {"items": {"append": [4, 5]}}
        result = merge(d1, d2)
        assert result == {"items": [1, 2, 3, 4, 5]}

    def test_cleanup_deeply_nested_structure(self):
        d = {
            "level1": {
                "level2": {
                    "keep": "yes",
                    "remove": DELETE,
                    "level3": {"also_remove": DELETE, "also_keep": True},
                }
            }
        }
        result = cleanup(d)
        expected = {"level1": {"level2": {"keep": "yes", "level3": {"also_keep": True}}}}
        assert result == expected

    def test_tweak_with_reduce_merge(self):
        """_tweak uses reduce(merge, parts) internally."""
        d = {"server.host": "localhost", "server.port": 8080, "debug": True}
        result = _tweak(d)
        assert result == {"server": {"host": "localhost", "port": 8080}, "debug": True}
