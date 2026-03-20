import unittest
from pathlib import Path


class TestBibleComfortIndexGuidanceControl(unittest.TestCase):
    def test_guidance_uses_combobox_with_localized_common_candidates(self):
        html = Path("index.html").read_text(encoding="utf-8")

        self.assertIn('<input id="guide" list="guideOptions" />', html)
        self.assertIn('<datalist id="guideOptions"></datalist>', html)
        self.assertNotIn('<select id="guide">', html)
        self.assertIn(
            '请列举一位与当前处境最相似的圣经正面人物，并详细说明他或她有哪些相似经历，以及神如何带领他或她。',
            html,
        )
        self.assertIn(
            "Identify one Bible figure whose hopeful experience most closely matches the situation, and explain in detail how their journey parallels mine and how God guided them.",
            html,
        )

    def test_bible_service_marks_guidance_todo_as_done(self):
        source = Path("bible_comfort_service.py").read_text(encoding="utf-8")
        self.assertIn('# DONE: Update "Optional Guidance" from input box to combo box', source)


if __name__ == "__main__":
    unittest.main()
