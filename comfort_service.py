import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from openai import OpenAI


def _init_openai_client() -> Optional[OpenAI]:
    """Initialize OpenAI client if API key is available; otherwise return None.
    Avoids raising during module import so unit tests can run without network/keys.
    """
    try:
        return OpenAI()
    except Exception:
        return None


class ComfortQuery(BaseModel):
    language: str = "zh"
    situation: str
    faith_background: Optional[str] = "christian"
    max_passages: int = 3
    guidance: Optional[str] = ""


class Passage(BaseModel):
    ref: str
    short_quote: str
    reason: str
    full_passage_text: str


class ComfortResponse(BaseModel):
    passages: List[Passage]
    devotional: str
    prayer: str
    disclaimer: str


SYSTEM_PROMPT = """You are a gentle Christian pastoral counselor and Bible study helper.
You MUST respond STRICTLY in the user's requested language (zh for Chinese, en for English).
DO NOT mix languages. All output, including Bible references, must be in the selected language.
Propose Bible passages (book chapter:verse), fitting the user's situation.
For quotes: provide at most a very short paraphrase (<= 20 words/chars) or leave empty.
Write a longer pastoral devotional (300-500 zh characters / 300-400 English words) and a prayer (4-8 sentences).
Begin the devotional with 1-2 empathetic sentences acknowledging the user's feelings and situation before offering guidance.
Use validating, non-judgmental, warm language; avoid clichés and platitudes.
Avoid doctrinal disputes, be comforting and practical.
Return STRICT JSON only, matching the schema the user supplies.
If unsure about an exact verse, choose one you are confident in.
Do NOT include long verbatim quotes from copyrighted translations.
"""


USER_PROMPT_TMPL = """User language: {language}
Faith background: {faith_background}
Situation detail: {situation}
Additional guidance: {guidance}

Return JSON with fields:
- passages: array of at most {max_passages} objects with fields:
  - ref (string, e.g., "Psalm 46:1-3" or localized equivalent)
  - short_quote (string, <= 20 words/chars; a paraphrase or public-domain-short snippet; MAY be empty)
  - reason (string, 1-2 sentences why this fits)
  - full_passage_text (string, the full text of the passage from a public domain version. Use WEB (World English Bible) if user language is English, use CUV (Chinese Union Version) if user language is Chinese)
- devotional: a 300-500 {lang_unit} pastoral reflection applying these passages to the user's situation. Begin with 1-2 empathetic sentences acknowledging the user's feelings and context before offering guidance.
- prayer: 4-8 sentences prayer.
- disclaimer: one sentence kindly asking the user to verify in their preferred translation.

Use the requested language for everything.
"""


class ComfortService:
    """Service responsible for building prompts and calling the OpenAI API."""

    def __init__(self, openai_client: Optional[OpenAI] = None) -> None:
        self.client: Optional[OpenAI] = openai_client or _init_openai_client()

    def build_messages(self, q: ComfortQuery) -> List[Dict[str, str]]:
        lang_unit = "characters" if q.language.startswith("zh") else "words"
        uprompt = USER_PROMPT_TMPL.format(
            language=q.language,
            faith_background=q.faith_background or "christian",
            situation=q.situation,
            guidance=q.guidance or "None",
            max_passages=max(1, min(q.max_passages, 10)),
            lang_unit=lang_unit,
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": uprompt},
        ]

    def get_comfort(self, q: ComfortQuery, *, openai_client: Optional[OpenAI] = None) -> Dict[str, Any]:
        """
        Builds the prompt, calls the OpenAI API, and processes the response.
        Returns a dict that matches ComfortResponse schema.
        """
        messages = self.build_messages(q)

        try:
            oc = openai_client or self.client
            if oc is None:
                raise RuntimeError("OpenAI client not configured")

            resp = oc.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            if not content:
                raise ValueError("LLM returned empty content")

            data = json.loads(content)

            # Constraint: Trim passages count and short quote length to avoid copyright/length issues
            max_passages = max(1, min(q.max_passages, 10))
            passages = (data.get("passages") or [])[:max_passages]
            for p in passages:
                sq = (p.get("short_quote") or "").strip()
                if q.language.startswith("zh"):
                    if len(sq) > 40:
                        p["short_quote"] = ""
                else:
                    if len(sq.split()) > 20:
                        p["short_quote"] = ""
            data["passages"] = passages

            # Ensure other required fields have default values if missing from LLM response
            data.setdefault("devotional", "")
            data.setdefault("prayer", "")

            if not data.get("disclaimer"):
                data["disclaimer"] = (
                    "请在你常用的圣经译本中核对经文原文与上下文；以上解读仅作灵修参考。"
                    if q.language.startswith("zh")
                    else "Please verify these references in your preferred Bible translation; the reflection is for devotional support."
                )

            return data

        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {e}") from e
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {e}") from e

