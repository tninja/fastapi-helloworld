import json
import os
import unittest
from unittest.mock import patch
import philosophy_comfort_service as philosophy_comfort_service_module

from philosophy_comfort_service import (
    PhilosophyComfortQuery,
    PhilosophyComfortResponse,
    PhilosophyComfortService,
)
from test_comfort_support import (
    FakeDuckDuckGoSearchProvider,
    FakeOpenAIClient,
)


@unittest.skipUnless(os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY is not set, skipping PhilosophyComfortService integration test.")
class TestPhilosophyComfortServiceIntegration(unittest.TestCase):
    def test_service_real_api_call_zh(self):
        service = PhilosophyComfortService()
        query = PhilosophyComfortQuery(
            language="zh",
            situation="最近因为工作不稳定而感到焦虑",
        )
        result = service.get_comfort(query)
        try:
            response_obj = PhilosophyComfortResponse(**result)
        except Exception as e:
            self.fail(f"Service response did not match PhilosophyComfortResponse: {e}")
        self.assertTrue(len(response_obj.reflection) > 10)
        self.assertTrue(len(response_obj.exercise) > 10)


class TestPhilosophyComfortServiceSearchContext(unittest.TestCase):
    def test_get_comfort_skips_web_search_by_default(self):
        response_payload = json.dumps(
            {
                "reflection": "Reflection",
                "exercise": "Exercise",
                "disclaimer": "Disclaimer",
            }
        )
        fake_client = FakeOpenAIClient(
            response_payload,
            research_output_text="News findings:\n- This should not be used.",
        )
        service = PhilosophyComfortService(openai_client=fake_client)

        service.get_comfort(
            PhilosophyComfortQuery(
                language="en",
                situation="I feel anxious about work.",
            )
        )

        prompt = fake_client.completions.last_kwargs["messages"][1]["content"]
        self.assertIsNone(fake_client.responses.last_kwargs)
        self.assertNotIn("Web search findings:", prompt)

    def test_get_comfort_uses_duckduckgo_search_by_default_when_enabled(self):
        response_payload = json.dumps(
            {
                "reflection": "Reflection",
                "exercise": "Exercise",
                "disclaimer": "Disclaimer",
            }
        )
        fake_client = FakeOpenAIClient(
            response_payload,
            research_output_text=(
                "News findings:\n"
                "- Reuters reports more companies are slowing hiring.\n"
                "Reddit/public discussion findings:\n"
                "- Reddit users describe longer job searches and delayed callbacks.\n"
                "Named entities:\n"
                "- Reuters\n- Reddit"
            ),
        )
        fake_provider = FakeDuckDuckGoSearchProvider(
            "\n".join(
                [
                    "1. Reuters reports more companies are slowing hiring",
                    "URL: https://example.com/reuters",
                    "Summary: News coverage describes AI-driven restructuring and job insecurity.",
                    "2. Reddit users describe longer job searches and delayed callbacks",
                    "URL: https://reddit.com/example",
                    "Summary: Public discussion centers on uncertainty, exhaustion, and fear.",
                ]
            )
        )

        with (
            patch.object(
                philosophy_comfort_service_module.DuckDuckGoSearchProvider,
                "from_env",
                return_value=fake_provider,
            ),
        ):
            service = PhilosophyComfortService(openai_client=fake_client)

            service.get_comfort(
                PhilosophyComfortQuery(
                    language="en",
                    situation="I feel anxious about work.",
                    enable_web_search=True,
                )
            )

        prompt = fake_client.completions.last_kwargs["messages"][1]["content"]
        self.assertIsNone(fake_client.responses.last_kwargs)
        self.assertEqual(len(fake_provider.queries), 1)
        self.assertEqual(fake_provider.queries[0].situation, "I feel anxious about work.")
        self.assertIn("Web search findings:", prompt)
        self.assertIn("News findings:", prompt)
        self.assertIn("Reddit/public discussion findings:", prompt)
        self.assertIn("Named entities:", prompt)
        self.assertIn("Reuters reports more companies are slowing hiring", prompt)
        self.assertIn("longer job searches and delayed callbacks", prompt)

    def test_build_web_search_prompt_focuses_on_news_reddit_and_public_discussion(self):
        service = PhilosophyComfortService(openai_client=FakeOpenAIClient("{}"))

        prompt = service._build_web_search_prompt(
            PhilosophyComfortQuery(
                language="zh",
                situation="最近因为AI和裁员新闻而感到焦虑",
                enable_web_search=True,
            )
        )

        self.assertIn("news", prompt.lower())
        self.assertIn("reddit", prompt.lower())
        self.assertIn("public discussion", prompt.lower())

    def test_build_messages_mentions_web_search_context(self):
        service = PhilosophyComfortService(openai_client=FakeOpenAIClient("{}"))

        messages = service.build_messages(
            PhilosophyComfortQuery(
                language="en",
                situation="I feel overwhelmed by layoffs in the news.",
            ),
            search_context="OpenAI web search findings about layoffs and Reddit discussions about job insecurity.",
        )

        system_prompt = messages[0]["content"]
        user_prompt = messages[1]["content"]
        self.assertIn("web search", system_prompt)
        self.assertIn("news", system_prompt)
        self.assertIn("Reddit", system_prompt)
        self.assertIn("Web search findings:", user_prompt)

    def test_get_comfort_uses_openai_web_search_when_explicitly_requested(self):
        response_payload = json.dumps(
            {
                "reflection": "Reflection",
                "exercise": "Exercise",
                "disclaimer": "Disclaimer",
            }
        )
        fake_client = FakeOpenAIClient(
            response_payload,
            research_output_text=(
                "News findings:\n"
                "- Reuters reports more companies are slowing hiring.\n"
                "Reddit/public discussion findings:\n"
                "- Reddit users describe longer job searches and delayed callbacks.\n"
                "Named entities:\n"
                "- Reuters\n- Reddit"
            ),
        )
        service = PhilosophyComfortService(openai_client=fake_client)

        result = service._search_with_openai_web(
            fake_client,
            PhilosophyComfortQuery(
                language="en",
                situation="I feel anxious about work.",
                enable_web_search=True,
            ),
        )

        self.assertEqual(fake_client.responses.last_kwargs["tools"], [{"type": "web_search"}])
        self.assertIn("Reuters reports more companies are slowing hiring", result)
        self.assertIn("Reddit users describe longer job searches", result)


if __name__ == "__main__":
    unittest.main()
