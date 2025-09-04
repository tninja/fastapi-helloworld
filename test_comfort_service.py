import unittest
import os
from comfort_service import (
    ComfortService,
    ComfortQuery,
    ComfortResponse,
)


@unittest.skipUnless(os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY is not set, skipping ComfortService integration test.")
class TestComfortServiceIntegration(unittest.TestCase):
    def test_service_real_api_call_zh(self):
        service = ComfortService()
        query = ComfortQuery(
            language="zh",
            situation="近期情绪低落，难以入睡",
            max_passages=1,
        )
        result = service.get_comfort(query)
        try:
            response_obj = ComfortResponse(**result)
        except Exception as e:
            self.fail(f"Service response did not match ComfortResponse: {e}")
        self.assertGreaterEqual(len(response_obj.passages), 1)
        self.assertTrue(len(response_obj.devotional) > 10)
        self.assertTrue(len(response_obj.prayer) > 10)

    def test_service_real_api_call_en(self):
        service = ComfortService()
        query = ComfortQuery(
            language="en",
            situation="Struggling with uncertainty at work",
            max_passages=1,
        )
        result = service.get_comfort(query)
        try:
            response_obj = ComfortResponse(**result)
        except Exception as e:
            self.fail(f"Service response did not match ComfortResponse: {e}")
        self.assertEqual(len(response_obj.passages), 1)
        self.assertTrue(len(response_obj.devotional.split()) > 10)
        self.assertTrue(len(response_obj.prayer.split()) > 5)


if __name__ == '__main__':
    unittest.main()

