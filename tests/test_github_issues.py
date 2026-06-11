import pytest
from unittest.mock import patch, MagicMock
from collections import defaultdict
from dataclasses import dataclass, field

from milabench.validation.github_issues import GitHubIssueLinker, MAX_URL_LENGTH, REPO_URL
from milabench.validation.error import ParsedTraceback


@pytest.fixture
def linker():
    return GitHubIssueLinker(token=None)


class TestBuildIssueTitle:
    def test_basic(self, linker):
        assert linker.build_issue_title("geo_gnn") == "[geo_gnn] benchmark failure"

    def test_with_special_chars(self, linker):
        assert linker.build_issue_title("timm-tf32") == "[timm-tf32] benchmark failure"


class TestSmartTruncateTrace:
    def test_short_trace_unchanged(self, linker):
        tb = ParsedTraceback([
            "Traceback (most recent call last):",
            '  File "main.py", line 10, in <module>',
            "    x = 1 / 0",
            "ZeroDivisionError: division by zero",
        ])
        result = linker._smart_truncate_trace(tb, 5000)
        assert "ZeroDivisionError" in result
        assert "[... truncated ...]" not in result

    def test_long_trace_truncated(self, linker):
        lines = ["Traceback (most recent call last):"]
        for i in range(50):
            lines.append(f'  File "module_{i}.py", line {i}, in func_{i}')
            lines.append(f"    some_call_{i}()")
        lines.append("RuntimeError: CUDA out of memory")

        tb = ParsedTraceback(lines)
        result = linker._smart_truncate_trace(tb, 200)
        assert "[... truncated ...]" in result
        assert len(result) <= 200

    def test_very_short_trace(self, linker):
        tb = ParsedTraceback(["Error", "details", "RuntimeError: boom"])
        result = linker._smart_truncate_trace(tb, 5000)
        assert "RuntimeError: boom" in result


class TestBuildNewIssueUrl:
    def test_basic_url(self, linker):
        url = linker.build_new_issue_url("title", "body")
        assert url.startswith(f"{REPO_URL}/issues/new?")
        assert "title=title" in url
        assert "body=body" in url
        assert "labels=bug" in url

    def test_url_within_limit(self, linker):
        long_body = "x" * 10000
        url = linker.build_new_issue_url("title", long_body)
        assert len(url) <= MAX_URL_LENGTH

    def test_truncation_marker_present(self, linker):
        long_body = "x" * 10000
        url = linker.build_new_issue_url("title", long_body)
        assert "truncated" in url

    def test_custom_labels(self, linker):
        url = linker.build_new_issue_url("title", "body", labels=["bug", "benchmark"])
        assert "labels=bug%2Cbenchmark" in url or "labels=bug,benchmark" in url


class TestBuildSearchUrl:
    def test_basic(self, linker):
        url = linker.build_search_url("geo_gnn")
        assert "/issues?q=" in url
        assert "geo_gnn" in url
        assert "mila-iqia/milabench" in url


class TestBuildIssueBody:
    def test_single_exception(self, linker):
        tb = ParsedTraceback([
            "Traceback (most recent call last):",
            '  File "main.py", line 10, in <module>',
            "RuntimeError: CUDA out of memory",
        ])
        exceptions = {"RuntimeError: CUDA out of memory": {"geo_gnn.0": [tb]}}
        body = linker.build_issue_body("geo_gnn", exceptions)

        assert "**Benchmark:** `geo_gnn`" in body
        assert "RuntimeError: CUDA out of memory" in body
        assert "(x1)" in body
        assert "## Exceptions" in body

    def test_multiple_exceptions(self, linker):
        tb1 = ParsedTraceback(["Traceback:", "RuntimeError: OOM"])
        tb2 = ParsedTraceback(["Traceback:", "ImportError: no module"])
        exceptions = {
            "RuntimeError: OOM": {"geo_gnn.0": [tb1, tb1]},
            "ImportError: no module": {"geo_gnn.1": [tb2]},
        }
        body = linker.build_issue_body("geo_gnn", exceptions)

        assert "(x2)" in body
        assert "(x1)" in body
        assert "RuntimeError: OOM" in body
        assert "ImportError: no module" in body


class TestSearchExistingIssues:
    def test_returns_none_on_failure(self, linker):
        with patch("requests.get", side_effect=Exception("network error")):
            result = linker.search_existing_issues("geo_gnn")
            assert result is None

    def test_returns_none_on_non_200(self, linker):
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        with patch("requests.get", return_value=mock_resp):
            result = linker.search_existing_issues("geo_gnn")
            assert result is None

    def test_returns_results_on_success(self, linker):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "items": [
                {
                    "number": 412,
                    "title": "[geo_gnn] benchmark failure",
                    "html_url": "https://github.com/mila-iqia/milabench/issues/412",
                    "state": "open",
                }
            ]
        }
        with patch("requests.get", return_value=mock_resp):
            result = linker.search_existing_issues("geo_gnn")
            assert len(result) == 1
            assert result[0] == (
                412,
                "[geo_gnn] benchmark failure",
                "https://github.com/mila-iqia/milabench/issues/412",
                "open",
            )

    def test_uses_token_in_header(self):
        linker = GitHubIssueLinker(token="test-token")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"items": []}

        with patch("requests.get", return_value=mock_resp) as mock_get:
            linker.search_existing_issues("timm")
            headers = mock_get.call_args[1]["headers"]
            assert headers["Authorization"] == "token test-token"


class TestGenerateLinks:
    def _make_grouped_errors(self, bench_name, failures=1):
        @dataclass
        class GroupedError:
            packs: dict = field(default_factory=lambda: defaultdict(list))
            early_stopped: int = 0
            total: int = 1
            exceptions: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
            failures: int = 0
            success: int = 0

        group = GroupedError(failures=failures)
        tb = ParsedTraceback(["Traceback:", "RuntimeError: test error"])
        group.exceptions["RuntimeError: test error"][f"{bench_name}.0"] = [tb]
        return {bench_name: group}

    def test_no_links_for_success(self, linker):
        groups = self._make_grouped_errors("geo_gnn", failures=0)
        links = linker.generate_links(groups)
        assert links == []

    def test_generates_link_for_failure(self, linker):
        groups = self._make_grouped_errors("geo_gnn", failures=1)
        with patch.object(linker, "search_existing_issues", return_value=None):
            links = linker.generate_links(groups)

        assert len(links) == 1
        assert links[0]["bench_name"] == "geo_gnn"
        assert "new_issue_url" in links[0]
        assert "search_url" in links[0]

    def test_existing_issues_populated(self, linker):
        groups = self._make_grouped_errors("timm", failures=1)
        existing = [(100, "[timm] benchmark failure", "https://github.com/mila-iqia/milabench/issues/100", "open")]
        with patch.object(linker, "search_existing_issues", return_value=existing):
            links = linker.generate_links(groups)

        assert links[0]["existing"] == existing


class TestRenderReport:
    def test_render_with_existing_issue(self, linker):
        from milabench.validation.validation import Summary

        @dataclass
        class GroupedError:
            packs: dict = field(default_factory=lambda: defaultdict(list))
            early_stopped: int = 0
            total: int = 1
            exceptions: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
            failures: int = 1
            success: int = 0

        group = GroupedError()
        tb = ParsedTraceback(["Traceback:", "RuntimeError: boom"])
        group.exceptions["RuntimeError: boom"]["geo_gnn.0"] = [tb]
        groups = {"geo_gnn": group}

        existing = [(42, "[geo_gnn] benchmark failure", "https://github.com/mila-iqia/milabench/issues/42", "open")]

        summary = Summary()
        with patch.object(linker, "search_existing_issues", return_value=existing):
            linker.render_report(summary, groups)

        output = []
        summary._show(summary.root.body, 0, output)
        text = "\n".join(output)

        assert "GitHub Issues" in text
        assert "geo_gnn" in text
        assert "#42" in text

    def test_render_with_no_existing_issue(self, linker):
        from milabench.validation.validation import Summary

        @dataclass
        class GroupedError:
            packs: dict = field(default_factory=lambda: defaultdict(list))
            early_stopped: int = 0
            total: int = 1
            exceptions: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
            failures: int = 1
            success: int = 0

        group = GroupedError()
        tb = ParsedTraceback(["Traceback:", "ImportError: no module"])
        group.exceptions["ImportError: no module"]["timm.0"] = [tb]
        groups = {"timm": group}

        summary = Summary()
        with patch.object(linker, "search_existing_issues", return_value=None):
            linker.render_report(summary, groups)

        output = []
        summary._show(summary.root.body, 0, output)
        text = "\n".join(output)

        assert "Search:" in text
        assert "New issue:" in text
        assert "/issues/new?" in text
