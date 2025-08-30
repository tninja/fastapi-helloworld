import unittest
import os
from app import get_comfort_from_openai, ComfortQuery, ComfortResponse

# This is an integration test that calls the actual OpenAI API.
# It will be skipped unless the OPENAI_API_KEY environment variable is set.
# To run it, execute: OPENAI_API_KEY="your_key_here" python -m unittest test_app.py
@unittest.skipUnless(os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY is not set, skipping integration test.")
class TestOpenAIFunctionIntegration(unittest.TestCase):

    def test_real_api_call_zh(self):
        """Tests a real API call with a Chinese query."""
        query = ComfortQuery(
            language="zh",
            situation="工作压力很大，感到很累",
            max_passages=2
        )

        # Call the function that interacts with the OpenAI API
        result = get_comfort_from_openai(query)

        # Validate the structure and types of the response using the Pydantic model
        try:
            response_obj = ComfortResponse(**result)
        except Exception as e:
            self.fail(f"Response from API did not match ComfortResponse model. Error: {e}")

        # Basic assertions for the content
        self.assertGreater(len(response_obj.passages), 0, "Should return at least one passage")
        self.assertLessEqual(len(response_obj.passages), query.max_passages, "Should not return more passages than requested")
        self.assertIsInstance(response_obj.passages[0].ref, str)
        self.assertTrue(len(response_obj.devotional) > 10, "Devotional should have meaningful content")
        self.assertTrue(len(response_obj.prayer) > 10, "Prayer should have meaningful content")
        self.assertTrue(len(response_obj.disclaimer) > 10, "Disclaimer should not be empty")
        print("\n[Integration Test - ZH] Passed. Devotional snippet:", response_obj.devotional[:50])

    def test_real_api_call_en(self):
        """Tests a real API call with an English query."""
        query = ComfortQuery(
            language="en",
            situation="Feeling anxious about the future",
            max_passages=1
        )

        # Call the function that interacts with the OpenAI API
        result = get_comfort_from_openai(query)

        # Validate the structure
        try:
            response_obj = ComfortResponse(**result)
        except Exception as e:
            self.fail(f"Response from API did not match ComfortResponse model. Error: {e}")

        # Basic assertions
        self.assertEqual(len(response_obj.passages), 1)
        self.assertIsInstance(response_obj.passages[0].ref, str)
        self.assertTrue(len(response_obj.devotional.split()) > 10, "Devotional should have meaningful content")
        self.assertTrue(len(response_obj.prayer.split()) > 5, "Prayer should have meaningful content")
        self.assertTrue(len(response_obj.disclaimer) > 10, "Disclaimer should not be empty")
        print("\n[Integration Test - EN] Passed. Devotional snippet:", " ".join(response_obj.devotional.split()[:10]))

if __name__ == '__main__':
    unittest.main()