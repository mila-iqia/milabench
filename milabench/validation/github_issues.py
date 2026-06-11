"""Generate pre-filled GitHub issue URLs for benchmark failures."""

import os
from urllib.parse import quote, urlencode

REPO_URL = "https://github.com/mila-iqia/milabench"
MAX_URL_LENGTH = 7500


class GitHubIssueLinker:
    def __init__(self, repo_url=None, token=None):
        self.repo_url = (repo_url or REPO_URL).rstrip("/")
        self.token = token or os.environ.get("MILABENCH_GITHUB_PAT")

    def build_issue_title(self, bench_name):
        return f"[{bench_name}] benchmark failure"

    def build_issue_body(self, bench_name, grouped_exceptions):
        """Build a markdown issue body aggregating all exceptions for one benchmark.

        grouped_exceptions: dict of {raised_exception_str: {pack_key: [ParsedTraceback, ...]}}
        """
        parts = []
        parts.append(f"**Benchmark:** `{bench_name}`\n")

        n_groups = max(len(grouped_exceptions), 1)
        per_exception_budget = 6000 // n_groups

        parts.append("## Exceptions\n")

        for raised, packs in grouped_exceptions.items():
            total_count = sum(len(traces) for traces in packs.values())
            parts.append(f"### {raised} (x{total_count})\n")

            _, traces = next(iter(packs.items()))
            trace_text = self._smart_truncate_trace(traces[0], per_exception_budget)

            parts.append("<details>")
            parts.append("<summary>Traceback</summary>\n")
            parts.append("```")
            parts.append(trace_text)
            parts.append("```\n")
            parts.append("</details>\n")

        parts.append("## Environment")
        parts.append(f"- milabench version: `{self._get_version()}`")

        return "\n".join(parts)

    def _smart_truncate_trace(self, traceback, max_chars):
        """Keep first frame, last N frames, and exception line. Truncate middle."""
        lines = traceback.lines
        text = "\n".join(lines)

        if len(text) <= max_chars:
            return text

        if len(lines) <= 4:
            return text[:max_chars]

        exception_line = lines[-1]
        first_lines = lines[:2]
        last_lines = lines[-3:]

        truncation_note = "\n[... truncated ...]\n"
        overhead = len("\n".join(first_lines)) + len(truncation_note) + len(exception_line) + 2

        available = max_chars - overhead
        last_text = "\n".join(last_lines)

        if len(last_text) > available:
            last_text = last_text[:available]

        return "\n".join(first_lines) + truncation_note + last_text

    def build_new_issue_url(self, title, body, labels=None):
        """Build a /issues/new URL, truncating body if needed to stay under limit."""
        if labels is None:
            labels = ["bug"]

        params = {"title": title, "body": body}
        if labels:
            params["labels"] = ",".join(labels)

        url = f"{self.repo_url}/issues/new?{urlencode(params, quote_via=quote)}"

        if len(url) > MAX_URL_LENGTH:
            truncation_msg = "\n\n[... body truncated due to URL length limit ...]"
            available_body = self._fit_body_to_url(title, labels, truncation_msg)
            if available_body is not None:
                body = body[:available_body] + truncation_msg
                params["body"] = body
                url = f"{self.repo_url}/issues/new?{urlencode(params, quote_via=quote)}"
            else:
                params = {"title": title}
                if labels:
                    params["labels"] = ",".join(labels)
                url = f"{self.repo_url}/issues/new?{urlencode(params, quote_via=quote)}"

        return url

    def _fit_body_to_url(self, title, labels, suffix):
        """Compute how many raw body chars fit before the URL exceeds the limit."""
        params = {"title": title, "body": suffix}
        if labels:
            params["labels"] = ",".join(labels)
        base_url = f"{self.repo_url}/issues/new?{urlencode(params, quote_via=quote)}"
        remaining = MAX_URL_LENGTH - len(base_url)
        if remaining <= 0:
            return None
        # URL encoding can expand chars (space -> %20 = 3x), estimate conservatively
        return remaining // 3

    def build_search_url(self, bench_name):
        """Generate a URL that searches existing issues for this benchmark."""
        query = f'repo:mila-iqia/milabench is:issue "[{bench_name}]" in:title'
        return f"{self.repo_url}/issues?q={quote(query)}"

    def search_existing_issues(self, bench_name):
        """Search GitHub for existing issues matching this benchmark.

        Returns list of (number, title, html_url, state) or None on failure.
        """
        try:
            import requests
        except ImportError:
            return None

        query = f'repo:mila-iqia/milabench is:issue "[{bench_name}]" in:title'
        url = "https://api.github.com/search/issues"
        headers = {"Accept": "application/vnd.github.v3+json"}

        if self.token:
            headers["Authorization"] = f"token {self.token}"

        try:
            response = requests.get(
                url, params={"q": query}, headers=headers, timeout=10
            )
            if response.status_code != 200:
                return None

            data = response.json()
            results = []
            for item in data.get("items", []):
                results.append((
                    item["number"],
                    item["title"],
                    item["html_url"],
                    item["state"],
                ))
            return results

        except Exception:
            return None

    def generate_links(self, grouped_errors):
        """Generate issue links for all failed benchmarks.

        grouped_errors: dict from group_errors() -- {bench_name: GroupedError}
        Returns list of dicts with keys: bench_name, errors_summary, existing, search_url, new_issue_url
        """
        results = []

        for bench_name, group in grouped_errors.items():
            if group.failures == 0:
                continue

            errors_summary = ", ".join(
                f"{sum(len(t) for t in packs.values())} x {raised}"
                for raised, packs in group.exceptions.items()
            )

            existing = self.search_existing_issues(bench_name)

            title = self.build_issue_title(bench_name)
            body = self.build_issue_body(bench_name, group.exceptions)
            new_issue_url = self.build_new_issue_url(title, body)
            search_url = self.build_search_url(bench_name)

            results.append({
                "bench_name": bench_name,
                "errors_summary": errors_summary,
                "existing": existing,
                "search_url": search_url,
                "new_issue_url": new_issue_url,
            })

        return results

    def render_report(self, summary, grouped_errors):
        """Render the GitHub Issues section into the summary."""
        links = self.generate_links(grouped_errors)

        if not links:
            return

        with summary.section("GitHub Issues"):
            for entry in links:
                with summary.section(entry["bench_name"]):
                    summary.add(f"* Errors: {entry['errors_summary']}")

                    existing = entry["existing"]
                    if existing:
                        for number, title, url, state in existing:
                            summary.add(f"* Existing: #{number} ({state}) - {title}")
                            summary.add(f"    {url}")
                    else:
                        summary.add(f"* Search:    {entry['search_url']}")
                        summary.add(f"* New issue: {entry['new_issue_url']}")

    def _get_version(self):
        try:
            from milabench._version import __tag__
            return __tag__
        except Exception:
            return "unknown"
