import os
import openai
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from good_news import (
    Article,
    GoodNewsConfig,
    GoodNewsFetcher,
    GoodNewsService,
    GoodNewsSummarizer,
    build_report,
    ensure_env,
    parse_articles,
    resolve_news_server_args,
    load_config,
)


class EnsureEnvTest(unittest.TestCase):
    def test_returns_existing_value(self) -> None:
        os.environ["TEST_ENV_VALUE"] = "present"
        try:
            result = ensure_env("TEST_ENV_VALUE", "Test Value")
        finally:
            del os.environ["TEST_ENV_VALUE"]

        self.assertEqual(result, "present")

    def test_raises_for_missing_value(self) -> None:
        os.environ.pop("TEST_ENV_MISSING", None)
        with self.assertRaisesRegex(RuntimeError, "Missing Test Value"):
            ensure_env("TEST_ENV_MISSING", "Test Value")


class ResolveNewsServerArgsTest(unittest.TestCase):
    def test_handles_spaces_and_quotes(self) -> None:
        template = '--directory "{dir}" run "src/news_api_mcp/server.py" --flag'
        args = resolve_news_server_args(template, Path("/tmp/mcp path"))
        self.assertEqual(
            args,
            (
                "--directory",
                "/tmp/mcp path",
                "run",
                "src/news_api_mcp/server.py",
                "--flag",
            ),
        )


class ParseArticlesTest(unittest.TestCase):
    def test_parse_articles_extracts_fields(self) -> None:
        raw_feed = (
            "Article 1:\n"
            "Title: Joyful Discovery\n"
            "Source: Inspiring Times\n"
            "Author: Jane Doe\n"
            "Published: 2024-05-01 08:00 UTC\n"
            "Description: Scientists report a breakthrough.\n"
            "URL: https://example.com/story\n"
            "---\n"
            "Article 2:\n"
            "Title: Community Triumph\n"
            "Source: Hope Daily\n"
            "Description: Volunteers rebuilt a playground.\n"
            "URL: https://example.com/another\n"
            "---\n"
        )

        articles = parse_articles(raw_feed)

        self.assertEqual(len(articles), 2)
        self.assertEqual(articles[0].title, "Joyful Discovery")
        self.assertEqual(articles[0].source, "Inspiring Times")
        self.assertEqual(articles[0].author, "Jane Doe")
        self.assertEqual(articles[1].title, "Community Triumph")
        self.assertEqual(articles[1].url, "https://example.com/another")

    def test_parse_articles_returns_empty_for_blank_feed(self) -> None:
        self.assertEqual(parse_articles(""), tuple())


class BuildReportTest(unittest.TestCase):
    def test_build_report_contains_sources_and_notes(self) -> None:
        summary_body = "## 精选亮点\n- 点亮世界"
        articles = (
            Article(
                title="Joyful Discovery",
                source="Inspiring Times",
                published_at="2024-05-01 08:00 UTC",
                url="https://example.com/story",
            ),
        )
        notes = ("... and 2 more articles",)
        generated_at = datetime(2024, 5, 2, 9, 0, tzinfo=timezone.utc)

        report = build_report(summary_body, articles, notes, generated_at)

        self.assertIn("# 好消息速递", report)
        self.assertIn("## 信息来源", report)
        self.assertIn("Joyful Discovery", report)
        self.assertIn("... and 2 more articles", report)
        self.assertIn("_Generated on:", report)


class GoodNewsSummarizerTest(unittest.TestCase):
    def _config(self, tmp_dir: Path) -> GoodNewsConfig:
        base = load_config()
        return replace(base, output_dir=tmp_dir)

    def test_returns_fallback_when_no_articles(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config = self._config(Path(tmp_dir))
            summarizer = GoodNewsSummarizer(config)
            result = summarizer.summarize(tuple(), tuple(), datetime.now(timezone.utc))

        self.assertIn("无法找到今日的正面新闻", result)

    @unittest.skipUnless(os.getenv("OPENAI_API_KEY"), "Requires OPENAI_API_KEY")
    def test_summarize_with_real_openai(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config = self._config(Path(tmp_dir))
            summarizer = GoodNewsSummarizer(config)
            try:
                result = summarizer.summarize(
                    [
                        Article(
                            title="Community Garden Blossoms",
                            source="Local News",
                            url="https://example.com/garden",
                        )
                    ],
                    tuple(),
                    datetime.now(timezone.utc),
                )
            except openai.APIConnectionError as exc:
                self.skipTest(f"OpenAI API unreachable: {exc}")
            except openai.OpenAIError as exc:
                self.skipTest(f"OpenAI API error: {exc}")

        self.assertIn("## 精选亮点", result)
        self.assertIn("## 正面影响", result)
        self.assertIn("## 鼓励寄语", result)


class GoodNewsServiceTest(unittest.TestCase):
    def _config(self, output_dir: Path) -> GoodNewsConfig:
        base = load_config()
        return replace(base, output_dir=output_dir)

    @unittest.skipUnless(
        os.getenv("OPENAI_API_KEY") and os.getenv("NEWS_API_KEY"),
        "Requires OPENAI_API_KEY and NEWS_API_KEY",
    )
    def test_generate_creates_digest_without_writing_when_disabled(self) -> None:
        artifact_root = Path("/tmp/test_artifacts")
        artifact_root.mkdir(parents=True, exist_ok=True)
        unique_dir = artifact_root / datetime.now(timezone.utc).strftime("good_news_preview_%Y%m%d%H%M%S%f")
        unique_dir.mkdir(parents=True, exist_ok=True)

        config = self._config(unique_dir)
        openai_client = openai.OpenAI(api_key=ensure_env("OPENAI_API_KEY", "OpenAI API key"))
        service = GoodNewsService(
            config,
            fetcher=GoodNewsFetcher(config),
            summarizer=GoodNewsSummarizer(config, client=openai_client),
        )
        try:
            digest = service.generate(
                now=datetime.now(timezone.utc),
                write=False,
            )
        except (openai.OpenAIError, RuntimeError, OSError) as exc:
            self.skipTest(f"Integration dependencies unavailable: {exc}")

        fetch_log = unique_dir / "fetch_result.txt"
        fetch_log.write_text(
            "Raw feed:\n"
            + digest.raw_feed
            + "\n\nExtra notes:\n"
            + ("\n".join(digest.extra_notes) if digest.extra_notes else "(none)"),
            encoding="utf-8",
        )

        articles_log = unique_dir / "articles.txt"
        articles_lines = [
            (
                "Title: {title}\nSource: {source}\nAuthor: {author}\nPublished: {published}\nURL: {url}\nContent length: {length}\nContent: \n{content}"
            ).format(
                title=article.title,
                source=article.source,
                author=article.author,
                published=article.published_at,
                url=article.url,
                length=len(article.content or ""),
                content=(article.content or "(no content fetched)"),
            )
            for article in digest.articles
        ]
        articles_log.write_text("\n\n---\n\n".join(articles_lines), encoding="utf-8")

        preview_path = unique_dir / "digest_preview.md"
        preview_path.write_text(digest.report, encoding="utf-8")

        summary_path = unique_dir / "summary_body.md"
        summary_path.write_text(digest.summary, encoding="utf-8")

        self.assertFalse(digest.written)
        self.assertTrue(fetch_log.exists())
        self.assertTrue(articles_log.exists())
        fetch_contents = fetch_log.read_text(encoding="utf-8")
        self.assertIn("Content:", fetch_contents)
        self.assertTrue(preview_path.exists())
        self.assertTrue(summary_path.exists())
        self.assertTrue(preview_path.read_text(encoding="utf-8").strip())
        self.assertTrue(digest.output_path.name.endswith(".md"))

    @unittest.skipUnless(
        os.getenv("OPENAI_API_KEY") and os.getenv("NEWS_API_KEY"),
        "Requires OPENAI_API_KEY and NEWS_API_KEY",
    )
    def test_generate_writes_digest_when_enabled(self) -> None:
        artifact_root = Path("/tmp/test_artifacts")
        artifact_root.mkdir(parents=True, exist_ok=True)
        unique_dir = artifact_root / datetime.now(timezone.utc).strftime("good_news_%Y%m%d%H%M%S%f")
        unique_dir.mkdir(parents=True, exist_ok=True)

        config = self._config(unique_dir)
        service = GoodNewsService(config)
        try:
            digest = service.generate(
                now=datetime.now(timezone.utc),
                write=True,
            )
        except (openai.OpenAIError, RuntimeError, OSError) as exc:
            self.skipTest(f"Integration dependencies unavailable: {exc}")

        self.assertTrue(digest.written)
        self.assertTrue(digest.output_path.exists())
        contents = digest.output_path.read_text(encoding="utf-8")

        fetch_log = unique_dir / "fetch_result.txt"
        fetch_log.write_text(
            "Raw feed:\n"
            + digest.raw_feed
            + "\n\nExtra notes:\n"
            + ("\n".join(digest.extra_notes) if digest.extra_notes else "(none)"),
            encoding="utf-8",
        )

        articles_log = unique_dir / "articles.txt"
        articles_lines = [
            (
                "Title: {title}\nSource: {source}\nAuthor: {author}\nPublished: {published}\nURL: {url}\nContent length: {length}\nContent: \n{content}"
            ).format(
                title=article.title,
                source=article.source,
                author=article.author,
                published=article.published_at,
                url=article.url,
                length=len(article.content or ""),
                content=(article.content or "(no content fetched)"),
            )
            for article in digest.articles
        ]
        articles_log.write_text("\n\n---\n\n".join(articles_lines), encoding="utf-8")

        summary_path = unique_dir / "summary_body.md"
        summary_path.write_text(digest.summary, encoding="utf-8")

        self.assertTrue(contents.strip())
        self.assertTrue(digest.output_path.name.endswith(".md"))
        self.assertTrue(fetch_log.exists())
        self.assertTrue(articles_log.exists())
        self.assertIn("Content:", fetch_log.read_text(encoding="utf-8"))
        self.assertTrue(summary_path.exists())


if __name__ == "__main__":
    unittest.main()
