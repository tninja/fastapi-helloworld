from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Protocol, Sequence, Tuple

import mcp.types as types
import openai
from dateutil import tz
from dotenv import load_dotenv

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

load_dotenv()


@dataclass(frozen=True)
class Article:
    title: str
    source: str | None = None
    author: str | None = None
    published_at: str | None = None
    description: str | None = None
    url: str | None = None
    content: str | None = None


@dataclass(frozen=True)
class FetchResult:
    raw_feed: str
    extra_notes: Tuple[str, ...]
    articles: Tuple[Article, ...]


@dataclass(frozen=True)
class GoodNewsConfig:
    query: str
    timezone: str
    output_dir: Path
    model: str
    ddg_server_cmd: str
    ddg_server_dir: Path | None
    ddg_server_args: Tuple[str, ...]
    ddg_max_results: int
    fetch_server_cmd: str
    fetch_server_dir: Path | None
    fetch_server_args: Tuple[str, ...]
    fetch_max_chars: int
    fetch_article_limit: int


@dataclass(frozen=True)
class GoodNewsDigest:
    report: str
    summary: str
    articles: Tuple[Article, ...]
    raw_feed: str
    extra_notes: Tuple[str, ...]
    output_path: Path
    written: bool


class NewsFetcher(Protocol):
    def fetch(self) -> FetchResult:  # pragma: no cover - interface definition
        ...


class NewsSummarizer(Protocol):
    def summarize(
        self,
        articles: Sequence[Article],
        extra_notes: Sequence[str],
        generated_at: datetime,
    ) -> str:  # pragma: no cover - interface definition
        ...


def ensure_env(var_name: str, friendly_name: str) -> str:
    value = os.getenv(var_name)
    if not value or value.startswith("YOUR_"):
        raise RuntimeError(
            f"Missing {friendly_name}. Set the {var_name} environment variable or .env entry."
        )
    return value


def resolve_news_server_args(template: str, server_dir: Path | None) -> Tuple[str, ...]:
    directory = str(server_dir) if server_dir else ""
    formatted = template.format(dir=directory)
    return tuple(shlex.split(formatted))


def parse_articles(raw_feed: str) -> Tuple[Article, ...]:
    if not raw_feed.strip():
        return tuple()

    blocks = [block for block in re.split(r"Article \d+:\s*", raw_feed) if block.strip()]

    parsed: list[Article] = []
    for block in blocks:
        title_match = re.search(r"^Title:\s*(.+)$", block, re.MULTILINE)
        url_match = re.search(r"^URL:\s*(.+)$", block, re.MULTILINE)
        source_match = re.search(r"^Source:\s*(.+)$", block, re.MULTILINE)
        author_match = re.search(r"^Author:\s*(.+)$", block, re.MULTILINE)
        published_match = re.search(r"^Published:\s*(.+)$", block, re.MULTILINE)
        summary_match = re.search(r"^Summary:\s*(.+)$", block, re.MULTILINE)
        if not summary_match:
            summary_match = re.search(r"^Description:\s*(.+)$", block, re.MULTILINE)
        content_match = re.search(r"^Full Text Excerpt:\s*(.+)$", block, re.MULTILINE | re.DOTALL)

        if not title_match and not url_match:
            continue

        parsed.append(
            Article(
                title=title_match.group(1).strip() if title_match else "未命名",
                source=source_match.group(1).strip() if source_match else None,
                author=author_match.group(1).strip() if author_match else None,
                published_at=published_match.group(1).strip() if published_match else None,
                url=url_match.group(1).strip() if url_match else None,
                description=summary_match.group(1).strip() if summary_match else None,
                content=content_match.group(1).strip() if content_match else None,
            )
        )

    return tuple(parsed)


def build_report(
    summary_body: str,
    articles: Sequence[Article],
    extra_notes: Sequence[str],
    generated_at: datetime,
) -> str:
    date_str = generated_at.strftime("%Y-%m-%d")
    lines = [f"# 好消息速递 · {date_str}", "", summary_body.strip(), ""]

    lines.append("## 信息来源")
    if articles:
        for index, article in enumerate(articles, 1):
            suffix = []
            if article.source:
                suffix.append(article.source)
            if article.published_at:
                suffix.append(f"({article.published_at})")
            suffix_text = " ".join(suffix)
            if article.url:
                lines.append(f"{index}. [{article.title}]({article.url}) {suffix_text}".rstrip())
            else:
                lines.append(f"{index}. {article.title} {suffix_text}".rstrip())
    else:
        lines.append("暂无可引用的文章链接。")

    if extra_notes:
        lines.append("")
        lines.extend(extra_notes)

    lines.append("")
    lines.append(f"_Generated on: {generated_at.strftime('%Y-%m-%d %H:%M %Z')}_")
    lines.append("")

    return "\n".join(lines)


class GoodNewsFetcher:
    def __init__(self, config: GoodNewsConfig) -> None:
        self._config = config

    def _parse_ddg_results(self, search_report: str) -> list[Article]:
        articles: list[Article] = []
        current_title: str | None = None
        current_url: str | None = None
        current_summary: str | None = None

        for line in search_report.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            match = re.match(r"^(\d+)\.\s*(.+)$", stripped)
            if match:
                if current_title or current_url or current_summary:
                    articles.append(
                        Article(
                            title=current_title or "未命名",
                            url=current_url,
                            description=current_summary,
                        )
                    )
                current_title = match.group(2).strip()
                current_url = None
                current_summary = None
                continue

            if stripped.startswith("URL:"):
                current_url = stripped[len("URL:") :].strip()
                continue

            if stripped.startswith("Summary:"):
                current_summary = stripped[len("Summary:") :].strip()
                continue

        if current_title or current_url or current_summary:
            articles.append(
                Article(
                    title=current_title or "未命名",
                    url=current_url,
                    description=current_summary,
                )
            )

        return articles

    def _compose_feed(
        self, search_report: str, articles: Sequence[Article]
    ) -> str:
        if not articles:
            return search_report

        sections: list[str] = [search_report]
        for idx, article in enumerate(articles, 1):
            body_lines = [f"Article {idx}"]
            if article.title:
                body_lines.append(f"Title: {article.title}")
            if article.url:
                body_lines.append(f"URL: {article.url}")
            if article.description:
                body_lines.append(f"Summary: {article.description}")
            if article.content:
                body_lines.append("Full Text Excerpt:")
                body_lines.append(article.content)
            sections.append("\n".join(body_lines))

        return "\n\n".join(sections)

    async def _fetch_async(self) -> FetchResult:
        server_params = StdioServerParameters(
            command=self._config.ddg_server_cmd,
            args=list(self._config.ddg_server_args),
            cwd=str(self._config.ddg_server_dir) if self._config.ddg_server_dir else None,
        )

        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                result = await session.call_tool(
                    "search",
                    {
                        "query": self._config.query,
                        "max_results": self._config.ddg_max_results,
                    },
                )

                if result.isError:
                    message = (
                        "\n".join(
                            item.text
                            for item in result.content
                            if isinstance(item, types.TextContent)
                        )
                        or "Unknown MCP error"
                    )
                    raise RuntimeError(f"MCP server reported an error: {message}")

                search_report_parts: list[str] = []
                for item in result.content:
                    if isinstance(item, types.TextContent) and item.text.strip():
                        search_report_parts.append(item.text.strip())

                search_report = "\n".join(search_report_parts)
                articles = self._parse_ddg_results(search_report)
                if not articles:
                    raise RuntimeError("DuckDuckGo search returned no parseable results.")

                try:
                    enriched_articles = await self._enrich_articles_with_content(articles)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    enriched_articles = articles

                augmented_feed = self._compose_feed(search_report, enriched_articles)
                return FetchResult(augmented_feed, tuple(), tuple(enriched_articles))

    async def _enrich_articles_with_content(
        self, articles: Sequence[Article]
    ) -> Tuple[Article, ...]:
        if not articles:
            return tuple()

        urls: list[str] = []
        for article in articles:
            if article.url and article.url not in urls:
                urls.append(article.url)

        if not urls:
            return tuple(articles)

        if self._config.fetch_article_limit <= 0:
            return tuple(articles)

        if not self._config.fetch_server_cmd or not self._config.fetch_server_args:
            return tuple(articles)

        urls = urls[: self._config.fetch_article_limit]

        server_params = StdioServerParameters(
            command=self._config.fetch_server_cmd,
            args=list(self._config.fetch_server_args),
            cwd=str(self._config.fetch_server_dir) if self._config.fetch_server_dir else None,
        )

        contents: dict[str, str] = {}
        try:
            async with stdio_client(server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    for url in urls:
                        try:
                            result = await session.call_tool(
                                "fetch",
                                {"url": url, "max_length": self._config.fetch_max_chars},
                            )
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            continue

                        if result.isError:
                            continue

                        collected = [
                            item.text
                            for item in result.content
                            if isinstance(item, types.TextContent) and item.text.strip()
                        ]
                        if collected:
                            contents[url] = "\n".join(collected)
        except asyncio.CancelledError:
            raise
        except Exception:
            return tuple(articles)

        enriched: list[Article] = []
        for article in articles:
            if article.url and article.url in contents:
                enriched.append(replace(article, content=contents[article.url]))
            else:
                enriched.append(article)

        return tuple(enriched)

    def fetch(self) -> FetchResult:
        return asyncio.run(self._fetch_async())


class GoodNewsSummarizer:
    def __init__(
        self, config: GoodNewsConfig, client: openai.OpenAI | None = None
    ) -> None:
        self._config = config
        self._client = client

    def summarize(
        self,
        articles: Sequence[Article],
        extra_notes: Sequence[str],
        generated_at: datetime,
    ) -> str:
        if not articles:
            return "## 精选亮点\n- 无法找到今日的正面新闻。\n\n## 正面影响\n- 期待明天会传来更好的消息。\n\n## 鼓励寄语\n继续保持希望，新的祝福就在路上。"

        client = self._client or openai.OpenAI(api_key=ensure_env("OPENAI_API_KEY", "OpenAI API key"))
        trailing = "\n".join(extra_notes) if extra_notes else "None"
        date_label = generated_at.strftime("%Y-%m-%d")

        article_sections: list[str] = []
        for idx, article in enumerate(articles, 1):
            content_excerpt = article.content or "(no content fetched)"
            if content_excerpt and len(content_excerpt) > self._config.fetch_max_chars:
                content_excerpt = content_excerpt[: self._config.fetch_max_chars] + "..."
            summary_text = article.description or "(no summary provided)"
            article_sections.append(
                (
                    "Article {idx}\nTitle: {title}\nSource: {source}\nAuthor: {author}\nPublished: {published}\nURL: {url}\nSummary: {summary}\nFull Text Excerpt:\n{content}"
                ).format(
                    idx=idx,
                    title=article.title,
                    source=article.source or "(unknown source)",
                    author=article.author or "(unknown author)",
                    published=article.published_at or "(unknown time)",
                    url=article.url or "(no URL)",
                    summary=summary_text,
                    content=content_excerpt,
                )
            )

        context_block = "\n\n---\n\n".join(article_sections)

        user_prompt = f"""
今天日期：{date_label}

以下提供了若干正面新闻的原文内容。请严格依据 Full Text Excerpt（若存在）撰写总结，如若全文不可用，可参考 Summary 或其他元数据。

要求：
- 全文使用简体中文撰写。
- 输出结构必须包含：
  ## 精选亮点
  - 针对每篇文章说明积极亮点，若有 URL 且可引用请使用 Markdown 链接。
  ## 正面影响
  - 至少 3 条要点，基于正文中的具体细节，解释这些新闻如何带来积极影响。
  ## 鼓励寄语
  - 以温暖的一句话鼓励读者。
- 请引用正文中的关键信息，不要虚构内容。若信息不足，请明确指出。
- 文字不少于 400 字。

文章内容：
{context_block}

额外说明：{trailing}
"""

        response = client.chat.completions.create(
            model=self._config.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an upbeat journalist who only writes truthful, uplifting reports in Simplified Chinese. Always ground your writing in the provided article content.",
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.45,
            max_tokens=1400,
        )

        return response.choices[0].message.content.strip()


class GoodNewsService:
    def __init__(
        self,
        config: GoodNewsConfig,
        fetcher: NewsFetcher | None = None,
        summarizer: NewsSummarizer | None = None,
    ) -> None:
        self._config = config
        self._fetcher = fetcher or GoodNewsFetcher(config)
        self._summarizer = summarizer or GoodNewsSummarizer(config)

    def _current_time(self) -> datetime:
        target_tz = tz.gettz(self._config.timezone) or tz.UTC
        now_utc = datetime.utcnow().replace(tzinfo=tz.UTC)
        return now_utc.astimezone(target_tz)

    def generate(
        self,
        now: datetime | None = None,
        write: bool = True,
    ) -> GoodNewsDigest:
        generated_at = now or self._current_time()
        fetch_result = self._fetcher.fetch()

        if not fetch_result.raw_feed:
            raise RuntimeError("No news data received from the MCP server.")

        articles = fetch_result.articles or parse_articles(fetch_result.raw_feed)
        if not articles:
            raise RuntimeError("No articles parsed from the MCP server output.")

        summary_body = self._summarizer.summarize(articles, fetch_result.extra_notes, generated_at)
        report = build_report(summary_body, articles, fetch_result.extra_notes, generated_at)

        output_dir = self._config.output_dir
        output_path = output_dir / f"{generated_at.strftime('%Y-%m-%d')}.md"

        written = False
        if write:
            output_dir.mkdir(parents=True, exist_ok=True)
            if output_path.exists():
                written = False
            else:
                output_path.write_text(report, encoding="utf-8")
                written = True

        return GoodNewsDigest(
            report=report,
            summary=summary_body,
            articles=tuple(articles),
            raw_feed=fetch_result.raw_feed,
            extra_notes=fetch_result.extra_notes,
            output_path=output_path,
            written=written,
        )


def load_config() -> GoodNewsConfig:
    query = os.getenv("GOOD_NEWS_QUERY", "warm heart positive news about good people, good behavior, please provide detail individual story")
    timezone = os.getenv("GOOD_NEWS_TIMEZONE", "America/Los_Angeles")
    output_dir = Path(os.getenv("GOOD_NEWS_OUTPUT_DIR", os.path.join("daily", "good-news")))
    model = os.getenv("GOOD_NEWS_MODEL", "gpt-4o-mini")

    ddg_server_cmd = os.getenv("DDG_MCP_CMD", "uvx")
    ddg_server_template = os.getenv("DDG_MCP_ARGS", "duckduckgo-mcp-server")
    ddg_server_dir_env = os.getenv("DDG_MCP_DIR")
    ddg_server_dir = Path(ddg_server_dir_env) if ddg_server_dir_env else None
    ddg_server_args = resolve_news_server_args(ddg_server_template, ddg_server_dir)
    ddg_max_results = int(os.getenv("DDG_MCP_MAX_RESULTS", "10"))

    fetch_server_cmd = os.getenv("FETCH_MCP_CMD", "uvx")
    fetch_server_template = os.getenv("FETCH_MCP_ARGS", "mcp-server-fetch")
    fetch_server_dir_env = os.getenv("FETCH_MCP_DIR")
    fetch_server_dir = Path(fetch_server_dir_env) if fetch_server_dir_env else None
    fetch_server_args = resolve_news_server_args(fetch_server_template, fetch_server_dir)
    fetch_max_chars = int(os.getenv("FETCH_MCP_MAX_LENGTH", "4000"))
    fetch_article_limit = int(os.getenv("FETCH_MCP_ARTICLE_LIMIT", "3"))

    return GoodNewsConfig(
        query=query,
        timezone=timezone,
        output_dir=output_dir,
        model=model,
        ddg_server_cmd=ddg_server_cmd,
        ddg_server_dir=ddg_server_dir,
        ddg_server_args=ddg_server_args,
        ddg_max_results=ddg_max_results,
        fetch_server_cmd=fetch_server_cmd,
        fetch_server_dir=fetch_server_dir,
        fetch_server_args=fetch_server_args,
        fetch_max_chars=fetch_max_chars,
        fetch_article_limit=fetch_article_limit,
    )


def main() -> None:
    try:
        config = load_config()
        openai_client = openai.OpenAI(api_key=ensure_env("OPENAI_API_KEY", "OpenAI API key"))
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return

    service = GoodNewsService(
        config,
        fetcher=GoodNewsFetcher(config),
        summarizer=GoodNewsSummarizer(config, client=openai_client),
    )

    try:
        digest = service.generate()
    except Exception as exc:
        print(f"Failed to create good news digest: {exc}")
        return

    if digest.written:
        print(f"Saved good news digest to {digest.output_path}")
    else:
        print(f"Digest already exists at {digest.output_path}. Skipped writing.")
        print(digest.report)


if __name__ == "__main__":
    main()
