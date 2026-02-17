import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import generate_devotional as devotional


class GenerateDevotionalPromptTest(unittest.TestCase):
    def _capture_user_prompt(self, theme: str) -> str:
        with patch("generate_devotional.openai.OpenAI") as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
            mock_client.chat.completions.create.return_value = mock_response

            devotional.generate_devotional_with_ai(
                theme,
                ["James 1:5"],
                datetime(2026, 2, 17),
            )

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            messages = call_kwargs["messages"]
            return next(msg["content"] for msg in messages if msg["role"] == "user")

    @patch("generate_devotional.openai.OpenAI")
    def test_prompt_includes_theme_specific_prayer_points_for_time_management(self, _):
        user_prompt = self._capture_user_prompt("Time Management and Wisdom")
        self.assertIn(
            "wisdom to manage time and to make wise judgments in the age of artificial intelligence",
            user_prompt,
        )
        self.assertNotIn(
            "my younger son to become obedient and dependable",
            user_prompt,
        )
        self.assertNotIn(
            "personal and family health and happiness, and that my talents may be used by God",
            user_prompt,
        )

    @patch("generate_devotional.openai.OpenAI")
    def test_prompt_includes_theme_specific_prayer_points_for_children(self, _):
        user_prompt = self._capture_user_prompt("Children's Education/Prayer for Children")
        self.assertIn("my younger son to become obedient and dependable", user_prompt)
        self.assertNotIn(
            "wisdom to manage time and to make wise judgments in the age of artificial intelligence",
            user_prompt,
        )

    def test_theme_config_is_single_source_of_truth(self):
        self.assertTrue(hasattr(devotional, "THEME_CONFIG"))
        config = devotional.THEME_CONFIG
        self.assertIn("Family Responsibility and Care", config)
        self.assertIn("Children's Education/Prayer for Children", config)
        self.assertIn("Time Management and Wisdom", config)
        for theme in (
            "Family Responsibility and Care",
            "Children's Education/Prayer for Children",
            "Time Management and Wisdom",
        ):
            self.assertIn("weight", config[theme])
            self.assertIn("scriptures", config[theme])
            self.assertIn("prayer_focus", config[theme])
            self.assertIsInstance(config[theme]["scriptures"], list)

    def test_theme_config_includes_related_scripture_for_new_prayer_focuses(self):
        config = devotional.THEME_CONFIG
        self.assertIn("1 Peter 4:10", config["Family Responsibility and Care"]["scriptures"])
        self.assertIn("Ephesians 6:1-4", config["Children's Education/Prayer for Children"]["scriptures"])
        self.assertIn("Philippians 1:9-10", config["Time Management and Wisdom"]["scriptures"])


if __name__ == "__main__":
    unittest.main()
