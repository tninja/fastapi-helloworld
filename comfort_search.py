import asyncio
import os
import re
import shlex
from pathlib import Path
from typing import Any, Protocol

try:
    import mcp.types as mcp_types
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
except ImportError:
    mcp_types = None
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None


EMPTY_SEARCH_FINDINGS = (
    "News findings:\n"
    "- None\n"
    "Reddit/public discussion findings:\n"
    "- None\n"
    "Named entities:\n"
    "- None"
)

DISCUSSION_MARKERS = (
    "reddit",
    "forum",
    "discussion",
    "commentary",
    "comments",
    "users discuss",
)


class SearchProvider(Protocol):
    def search(self, query: Any) -> str:
        ...


def resolve_server_args(template: str) -> list[str]:
    return shlex.split(template)


class DuckDuckGoSearchProvider:
    def __init__(
        self,
        server_cmd: str,
        server_args: tuple[str, ...],
        server_dir: Path | str | None,
        max_results: int,
        *,
        mcp_types_module: Any = mcp_types,
        client_session_cls: Any = ClientSession,
        stdio_server_parameters_cls: Any = StdioServerParameters,
        stdio_client_fn: Any = stdio_client,
    ) -> None:
        self.server_cmd = server_cmd
        self.server_args = server_args
        self.server_dir = server_dir
        self.max_results = max_results
        self._mcp_types = mcp_types_module
        self._client_session_cls = client_session_cls
        self._stdio_server_parameters_cls = stdio_server_parameters_cls
        self._stdio_client_fn = stdio_client_fn

    @classmethod
    def from_env(cls) -> "DuckDuckGoSearchProvider":
        server_dir = os.getenv("DDG_MCP_DIR")
        return cls(
            server_cmd=os.getenv("DDG_MCP_CMD", "uvx"),
            server_args=tuple(
                resolve_server_args(os.getenv("DDG_MCP_ARGS", "duckduckgo-mcp-server"))
            ),
            server_dir=Path(server_dir) if server_dir else None,
            max_results=int(os.getenv("DDG_MCP_MAX_RESULTS", "10")),
        )

    def search(self, query: Any) -> str:
        return asyncio.run(self._search_async(self._coerce_query_text(query)))

    def _coerce_query_text(self, query: Any) -> str:
        if hasattr(query, "situation"):
            return str(query.situation).strip()
        return str(query).strip()

    def _require_mcp_dependencies(self) -> None:
        if any(
            dependency is None
            for dependency in (
                self._mcp_types,
                self._client_session_cls,
                self._stdio_server_parameters_cls,
                self._stdio_client_fn,
            )
        ):
            raise RuntimeError("DuckDuckGo MCP dependencies are not available.")

    def _extract_text_parts(self, result: Any) -> list[str]:
        return [
            item.text.strip()
            for item in result.content
            if isinstance(item, self._mcp_types.TextContent) and item.text.strip()
        ]

    async def _search_async(self, query_text: str) -> str:
        self._require_mcp_dependencies()

        server_params = self._stdio_server_parameters_cls(
            command=self.server_cmd,
            args=list(self.server_args),
            cwd=str(self.server_dir) if self.server_dir else None,
        )

        async with self._stdio_client_fn(server_params) as (read_stream, write_stream):
            async with self._client_session_cls(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    "search",
                    {"query": query_text, "max_results": self.max_results},
                )

                if result.isError:
                    message = "\n".join(self._extract_text_parts(result)) or "Unknown MCP error"
                    raise RuntimeError(f"MCP server reported an error: {message}")

                return "\n".join(self._extract_text_parts(result))


class SearchFindingsFormatter:
    def __init__(
        self,
        *,
        empty_findings: str = EMPTY_SEARCH_FINDINGS,
        discussion_markers: tuple[str, ...] = DISCUSSION_MARKERS,
    ) -> None:
        self._empty_findings = empty_findings
        self._discussion_markers = discussion_markers

    def format(self, search_context: str) -> str:
        entries = self._parse_search_entries(search_context)
        if not entries:
            return self._empty_findings

        news_findings: list[str] = []
        discussion_findings: list[str] = []
        named_entities: list[str] = []

        for entry in entries:
            title = entry["title"]
            summary = entry["summary"]
            url = entry["url"]
            finding = self._compose_finding(title, summary)
            if self._is_discussion_entry(title, summary, url):
                discussion_findings.append(finding)
            else:
                news_findings.append(finding)
            named_entities.extend(self._extract_named_entities(f"{title} {summary} {url}"))

        named_entities = self._dedupe_keep_order(named_entities)[:8]
        return "\n".join(
            [
                "News findings:",
                *self._format_section(news_findings),
                "Reddit/public discussion findings:",
                *self._format_section(discussion_findings),
                "Named entities:",
                *self._format_section(named_entities),
            ]
        )

    def _parse_search_entries(self, search_context: str) -> list[dict[str, str]]:
        entries: list[dict[str, str]] = []
        current: dict[str, str] | None = None

        for raw_line in search_context.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            if re.match(r"^\d+\.\s+", line):
                if current:
                    entries.append(current)
                current = {"title": re.sub(r"^\d+\.\s+", "", line), "summary": "", "url": ""}
                continue

            if current is None:
                current = {"title": line, "summary": "", "url": ""}
                continue

            if line.startswith("URL:"):
                current["url"] = line[len("URL:") :].strip()
            elif line.startswith("Summary:"):
                current["summary"] = line[len("Summary:") :].strip()
            elif current["summary"]:
                current["summary"] = f"{current['summary']} {line}".strip()
            else:
                current["summary"] = line

        if current:
            entries.append(current)

        return entries[:5]

    def _compose_finding(self, title: str, summary: str) -> str:
        if title and summary:
            return f"{title}. {summary}"
        return title or summary or "None"

    def _is_discussion_entry(self, title: str, summary: str, url: str) -> bool:
        haystack = f"{title} {summary} {url}".lower()
        return any(marker in haystack for marker in self._discussion_markers)

    def _extract_named_entities(self, text: str) -> list[str]:
        matches = re.findall(r"\b[A-Z][A-Za-z0-9&.-]{1,}\b", text)
        return [
            match
            for match in matches
            if match.lower() not in {"url", "summary", "news", "public"}
        ]

    def _dedupe_keep_order(self, items: list[str]) -> list[str]:
        seen = set()
        deduped: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    def _format_section(self, items: list[str]) -> list[str]:
        if not items:
            return ["- None"]
        return [f"- {item}" for item in items[:3]]
