import unittest
from pathlib import Path
from unittest.mock import patch

from comfort_search import DuckDuckGoSearchProvider, SearchFindingsFormatter


class _FakeTextContent:
    def __init__(self, text: str):
        self.text = text


class _FakeStdioServerParameters:
    def __init__(self, command, args, cwd=None):
        self.command = command
        self.args = args
        self.cwd = cwd


class _FakeAsyncContextManager:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeClientSession:
    result_content = []
    last_call = None

    def __init__(self, read_stream, write_stream):
        self.read_stream = read_stream
        self.write_stream = write_stream

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def initialize(self):
        return None

    async def call_tool(self, name, arguments):
        type(self).last_call = {"name": name, "arguments": arguments}
        return type(
            "Result",
            (),
            {"isError": False, "content": type(self).result_content},
        )()


def _fake_stdio_client(server_params):
    _fake_stdio_client.last_server_params = server_params
    return _FakeAsyncContextManager(("read-stream", "write-stream"))


_fake_stdio_client.last_server_params = None


class _FakeMCPTypes:
    TextContent = _FakeTextContent


class DuckDuckGoSearchProviderTest(unittest.TestCase):
    def test_search_returns_raw_search_results_from_mcp(self):
        _FakeClientSession.result_content = [
            _FakeTextContent(
                "\n".join(
                    [
                        "1. Reuters reports layoffs continue",
                        "URL: https://example.com/reuters",
                        "Summary: Hiring slows across multiple firms.",
                    ]
                )
            )
        ]
        provider = DuckDuckGoSearchProvider(
            server_cmd="uvx",
            server_args=("duckduckgo-mcp-server",),
            server_dir=Path("/tmp"),
            max_results=2,
            mcp_types_module=_FakeMCPTypes,
            client_session_cls=_FakeClientSession,
            stdio_server_parameters_cls=_FakeStdioServerParameters,
            stdio_client_fn=_fake_stdio_client,
        )

        result = provider.search("I feel anxious about work.")

        self.assertEqual(_FakeClientSession.last_call["name"], "search")
        self.assertEqual(
            _FakeClientSession.last_call["arguments"],
            {"query": "I feel anxious about work.", "max_results": 2},
        )
        self.assertEqual(_fake_stdio_client.last_server_params.command, "uvx")
        self.assertEqual(_fake_stdio_client.last_server_params.args, ["duckduckgo-mcp-server"])
        self.assertEqual(_fake_stdio_client.last_server_params.cwd, "/tmp")
        self.assertIn("Reuters reports layoffs continue", result)
        self.assertIn("Summary: Hiring slows across multiple firms.", result)


class SearchFindingsFormatterTest(unittest.TestCase):
    def test_format_search_findings_groups_news_discussion_and_entities(self):
        formatter = SearchFindingsFormatter()

        formatted = formatter.format(
            "\n".join(
                [
                    "1. Meta cuts more teams while increasing AI investment",
                    "URL: https://example.com/meta",
                    "Summary: News reports describe AI-driven restructuring and employee uncertainty.",
                    "2. Reddit users discuss frozen hiring and longer job searches",
                    "URL: https://reddit.com/example",
                    "Summary: Public discussion centers on fear, delayed callbacks, and emotional exhaustion.",
                ]
            )
        )

        self.assertIn("News findings:", formatted)
        self.assertIn("Reddit/public discussion findings:", formatted)
        self.assertIn("Named entities:", formatted)
        self.assertIn("AI-driven restructuring", formatted)
        self.assertIn("longer job searches", formatted)
        self.assertIn("Meta", formatted)
        self.assertIn("Reddit", formatted)


if __name__ == "__main__":
    unittest.main()
