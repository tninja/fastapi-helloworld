import unittest
import os
import json
from bible_comfort_service import (
    BibleComfortService,
    BibleComfortQuery,
    BibleComfortResponse,
)


@unittest.skipUnless(os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY is not set, skipping BibleComfortService integration test.")
class TestBibleComfortServiceIntegration(unittest.TestCase):
    def test_service_real_api_call_zh(self):
        service = BibleComfortService()
        query = BibleComfortQuery(
            language="zh",
            situation="近期情绪低落，难以入睡",
            max_passages=1,
        )
        result = service.get_comfort(query)
        try:
            response_obj = BibleComfortResponse(**result)
        except Exception as e:
            self.fail(f"Service response did not match ComfortResponse: {e}")
        print(response_obj)
        self.assertGreaterEqual(len(response_obj.passages), 1)
        self.assertTrue(len(response_obj.devotional) > 10)
        self.assertTrue(len(response_obj.prayer) > 10)

    def test_service_real_api_call_zh_2(self):
        service = BibleComfortService()
        query = BibleComfortQuery(
            language="zh",
            situation="最近有很多公司宣称因为AI而layoff员工, 比如meta, block, amazon. 这让人对工作感到不确定和焦虑",
            max_passages=3,
        )
        result = service.get_comfort(query)
        try:
            response_obj = BibleComfortResponse(**result)
        except Exception as e:
            self.fail(f"Service response did not match ComfortResponse: {e}")
        print(response_obj.passages)
        self.assertGreaterEqual(len(response_obj.passages), 1)
        print(response_obj.devotional)
        self.assertTrue(len(response_obj.devotional) > 10)
        print(response_obj.prayer)
        self.assertTrue(len(response_obj.prayer) > 10)

    def test_service_real_api_call_en(self):
        service = BibleComfortService()
        query = BibleComfortQuery(
            language="en",
            situation="Struggling with uncertainty at work",
            max_passages=1,
        )
        result = service.get_comfort(query)
        try:
            response_obj = BibleComfortResponse(**result)
        except Exception as e:
            self.fail(f"Service response did not match ComfortResponse: {e}")
        self.assertEqual(len(response_obj.passages), 1)
        self.assertTrue(len(response_obj.devotional.split()) > 10)
        self.assertTrue(len(response_obj.prayer.split()) > 5)


class _FakeSearchProvider:
    def __init__(self, context: str):
        self.context = context
        self.queries = []

    def search(self, query: BibleComfortQuery) -> str:
        self.queries.append(query)
        return self.context


class _FakeCompletions:
    def __init__(self, response_content: str):
        self.response_content = response_content
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return type(
            "Response",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {
                            "message": type(
                                "Message", (), {"content": self.response_content}
                            )()
                        },
                    )()
                ]
            },
        )()


class _FakeResponses:
    def __init__(self, output_text: str):
        self.output_text = output_text
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return type("Response", (), {"output_text": self.output_text})()


class _FakeOpenAIClient:
    def __init__(self, response_content: str, research_output_text: str = ""):
        self.completions = _FakeCompletions(response_content)
        self.chat = type(
            "Chat",
            (),
            {"completions": self.completions},
        )()
        self.responses = _FakeResponses(research_output_text)


class TestBibleComfortServiceSearchContext(unittest.TestCase):
    def test_get_comfort_skips_web_search_by_default(self):
        response_payload = json.dumps(
            {
                "passages": [],
                "devotional": "Comfort",
                "prayer": "Prayer",
                "disclaimer": "Disclaimer",
            }
        )
        fake_client = _FakeOpenAIClient(
            response_payload,
            research_output_text="News findings:\n- This should not be used.",
        )
        search_provider = _FakeSearchProvider(
            "News findings:\n- Provider result that should not be used."
        )
        service = BibleComfortService(
            openai_client=fake_client,
            search_provider=search_provider,
        )

        service.get_comfort(
            BibleComfortQuery(
                language="en",
                situation="I feel anxious about work.",
            )
        )

        prompt = fake_client.completions.last_kwargs["messages"][1]["content"]
        self.assertEqual(search_provider.queries, [])
        self.assertNotIn("OpenAI web search findings:", prompt)

    def test_get_comfort_calls_web_search_when_enabled(self):
        response_payload = json.dumps(
            {
                "passages": [],
                "devotional": "Comfort",
                "prayer": "Prayer",
                "disclaimer": "Disclaimer",
            }
        )
        fake_client = _FakeOpenAIClient(
            response_payload,
            research_output_text="News findings:\n- Reuters reports layoffs continue.",
        )
        service = BibleComfortService(openai_client=fake_client)

        service.get_comfort(
            BibleComfortQuery(
                language="en",
                situation="I feel anxious about work.",
                enable_web_search=True,
            )
        )

        prompt = fake_client.completions.last_kwargs["messages"][1]["content"]
        self.assertEqual(fake_client.responses.last_kwargs["tools"], [{"type": "web_search"}])
        self.assertIn("OpenAI web search findings:", prompt)

    def test_build_web_search_prompt_focuses_on_news_reddit_and_public_discussion(self):
        service = BibleComfortService(openai_client=_FakeOpenAIClient("{}"))

        prompt = service._build_web_search_prompt(
            BibleComfortQuery(
                language="zh",
                situation="最近有很多公司宣称因为AI而layoff员工, 比如meta, block, amazon. 这让人对工作感到不确定和焦虑",
                enable_web_search=True,
            )
        )

        self.assertIn("news", prompt.lower())
        self.assertIn("reddit", prompt.lower())
        self.assertIn("public discussion", prompt.lower())
        self.assertIn("meta", prompt.lower())
        self.assertIn("amazon", prompt.lower())

    def test_build_messages_mentions_web_search_news_and_reddit_context(self):
        service = BibleComfortService(openai_client=_FakeOpenAIClient("{}"))

        messages = service.build_messages(
            BibleComfortQuery(
                language="en",
                situation="I feel overwhelmed by layoffs in the news.",
            ),
            search_context="Recent web search findings about layoffs and Reddit discussions about job insecurity.",
        )

        system_prompt = messages[0]["content"]
        self.assertIn("web search", system_prompt)
        self.assertIn("news", system_prompt)
        self.assertIn("Reddit", system_prompt)
        self.assertIn("more accurate and relevant comfort", system_prompt)

    def test_build_messages_requires_explicit_external_context_in_devotional_and_prayer(self):
        service = BibleComfortService(openai_client=_FakeOpenAIClient("{}"))

        messages = service.build_messages(
            BibleComfortQuery(
                language="en",
                situation="I feel anxious about AI layoffs.",
            ),
            search_context="Meta and Amazon layoffs in news coverage and Reddit job insecurity discussions.",
        )

        system_prompt = messages[0]["content"]
        user_prompt = messages[1]["content"]
        self.assertIn("explicitly mention", system_prompt)
        self.assertIn("company names", system_prompt)
        self.assertIn("platform names", system_prompt)
        self.assertIn("devotional", system_prompt)
        self.assertIn("prayer", system_prompt)
        self.assertIn("If the web search findings include company names", user_prompt)
        self.assertIn("devotional", user_prompt)
        self.assertIn("prayer", user_prompt)

    def test_get_comfort_uses_openai_web_search_and_includes_findings_in_prompt(self):
        search_provider = _FakeSearchProvider(
            "1. Sleep hygiene from trusted Christian counseling resources."
        )
        response_payload = json.dumps(
            {
                "passages": [],
                "devotional": "Comfort",
                "prayer": "Prayer",
                "disclaimer": "Disclaimer",
            }
        )
        fake_client = _FakeOpenAIClient(
            response_payload,
            research_output_text=(
                "News findings:\n"
                "- Reuters reports that several firms are slowing hiring after AI-led restructuring.\n"
                "Reddit/public discussion findings:\n"
                "- Reddit users describe delayed callbacks and longer job searches.\n"
                "Named entities:\n"
                "- Reuters\n- Reddit\n- Meta"
            ),
        )
        service = BibleComfortService(
            openai_client=fake_client,
        )

        service.get_comfort(
            BibleComfortQuery(
                language="en",
                situation="I feel anxious at night and cannot sleep.",
                guidance="Focus on biblical comfort for insomnia.",
                enable_web_search=True,
            )
        )

        prompt = fake_client.completions.last_kwargs["messages"][1]["content"]
        self.assertEqual(fake_client.responses.last_kwargs["tools"], [{"type": "web_search"}])
        self.assertIn("OpenAI web search findings:", prompt)
        self.assertIn("Reuters reports that several firms are slowing hiring", prompt)
        self.assertIn("Reddit users describe delayed callbacks", prompt)

    def test_get_comfort_formats_web_search_findings_for_devotional_and_prayer(self):
        search_provider = _FakeSearchProvider(
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
        response_payload = json.dumps(
            {
                "passages": [],
                "devotional": "Comfort",
                "prayer": "Prayer",
                "disclaimer": "Disclaimer",
            }
        )
        fake_client = _FakeOpenAIClient(
            response_payload,
            research_output_text=(
                "News findings:\n"
                "- Meta cuts more teams while increasing AI investment. News reports describe AI-driven restructuring and employee uncertainty.\n"
                "Reddit/public discussion findings:\n"
                "- Reddit users discuss frozen hiring and longer job searches. Public discussion centers on fear, delayed callbacks, and emotional exhaustion.\n"
                "Named entities:\n"
                "- Meta\n- Reddit"
            ),
        )
        service = BibleComfortService(
            openai_client=fake_client,
        )

        service.get_comfort(
            BibleComfortQuery(
                language="en",
                situation="I feel anxious about layoffs at work.",
                enable_web_search=True,
            )
        )

        prompt = fake_client.completions.last_kwargs["messages"][1]["content"]
        self.assertIn("OpenAI web search findings:", prompt)
        self.assertIn("News findings:", prompt)
        self.assertIn("Reddit/public discussion findings:", prompt)
        self.assertIn("Named entities:", prompt)
        self.assertIn("use at least one new external detail", prompt)
        self.assertIn("AI-driven restructuring", prompt)
        self.assertIn("longer job searches", prompt)

    def test_get_comfort_omits_web_search_section_when_search_returns_nothing(self):
        response_payload = json.dumps(
            {
                "passages": [],
                "devotional": "Comfort",
                "prayer": "Prayer",
                "disclaimer": "Disclaimer",
            }
        )
        fake_client = _FakeOpenAIClient(response_payload, research_output_text="")
        service = BibleComfortService(
            openai_client=fake_client,
        )

        service.get_comfort(
            BibleComfortQuery(
                language="en",
                situation="I feel overwhelmed.",
            )
        )

        prompt = fake_client.completions.last_kwargs["messages"][1]["content"]
        self.assertNotIn("OpenAI web search findings:", prompt)


if __name__ == '__main__':
    unittest.main()
