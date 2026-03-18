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


class _FakeOpenAIClient:
    def __init__(self, response_content: str):
        self.completions = _FakeCompletions(response_content)
        self.chat = type(
            "Chat",
            (),
            {"completions": self.completions},
        )()


class TestBibleComfortServiceSearchContext(unittest.TestCase):
    def test_build_messages_mentions_duckduckgo_news_and_reddit_context(self):
        service = BibleComfortService(openai_client=_FakeOpenAIClient("{}"))

        messages = service.build_messages(
            BibleComfortQuery(
                language="en",
                situation="I feel overwhelmed by layoffs in the news.",
            ),
            search_context="Recent layoffs and Reddit discussions about job insecurity.",
        )

        system_prompt = messages[0]["content"]
        self.assertIn("DuckDuckGo", system_prompt)
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
        self.assertIn("If the DuckDuckGo context includes company names", user_prompt)
        self.assertIn("devotional", user_prompt)
        self.assertIn("prayer", user_prompt)

    def test_get_comfort_includes_duckduckgo_context_in_prompt(self):
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
        fake_client = _FakeOpenAIClient(response_payload)
        service = BibleComfortService(
            openai_client=fake_client,
            search_provider=search_provider,
        )

        service.get_comfort(
            BibleComfortQuery(
                language="en",
                situation="I feel anxious at night and cannot sleep.",
                guidance="Focus on biblical comfort for insomnia.",
            )
        )

        prompt = fake_client.completions.last_kwargs["messages"][1]["content"]
        self.assertIn("DuckDuckGo research context:", prompt)
        self.assertIn("Sleep hygiene from trusted Christian counseling resources.", prompt)
        self.assertEqual(len(search_provider.queries), 1)

    def test_get_comfort_omits_duckduckgo_section_when_search_returns_nothing(self):
        search_provider = _FakeSearchProvider("")
        response_payload = json.dumps(
            {
                "passages": [],
                "devotional": "Comfort",
                "prayer": "Prayer",
                "disclaimer": "Disclaimer",
            }
        )
        fake_client = _FakeOpenAIClient(response_payload)
        service = BibleComfortService(
            openai_client=fake_client,
            search_provider=search_provider,
        )

        service.get_comfort(
            BibleComfortQuery(
                language="en",
                situation="I feel overwhelmed.",
            )
        )

        prompt = fake_client.completions.last_kwargs["messages"][1]["content"]
        self.assertNotIn("DuckDuckGo research context:", prompt)


if __name__ == '__main__':
    unittest.main()
