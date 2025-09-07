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


class BibleComfortQuery(BaseModel):
    language: str = "zh"
    situation: str
    faith_background: Optional[str] = "christian"
    max_passages: int = 3
    guidance: Optional[str] = ""


class BiblePassage(BaseModel):
    ref: str
    short_quote: str
    reason: str
    full_passage_text: str


class BibleComfortResponse(BaseModel):
    passages: List[BiblePassage]
    devotional: str
    prayer: str
    disclaimer: str


SYSTEM_PROMPT = """You are a Christian pastoral counselor and Bible study assistant serving a Christian audience.
You MUST respond STRICTLY in the user's requested language (zh for Chinese, en for English).
DO NOT mix languages. All output, including Bible references, must be in the selected language.

Tone and style (professional, Scripture-centered):
- Use respectful, clear, pastoral language with a measured, professional tone.
- Ground counsel in Scripture and historic Christian understanding; avoid interfaith syncretism.
- Lead with empathy without over-emphasizing therapeutic language; avoid clichés and platitudes.
- Be concise and structured; avoid debates and speculative theology.

Verse accuracy requirements:
- Ensure every reference (book name, chapter:verse range) is accurate and commonly recognized in the requested language.
- Use localized book names (Chinese for zh, English for en).
- Provide full_passage_text verbatim from a public-domain translation: WEB for English, CUV for Chinese.
- Do NOT paraphrase in full_passage_text; ensure it matches the cited reference precisely.
- If unsure about a verse, choose a different passage you are confident is correct. Never invent verses or numbers.

Content requirements:
- Propose Bible passages (book chapter:verse) that fit the user's situation.
- For short_quote: give at most a very brief paraphrase (<= 20 words/chars) or leave empty.
- Write a pastoral devotional (300–500 zh characters / 300–500 English words) with a professional tone for Christians.
- Begin the devotional with 1–2 empathetic sentences, then provide Scripture-based reflection and one concise practical application or reflection question.
- Write a reverent, concise prayer (4–8 sentences).
- Avoid heavy emphasis on therapeutic techniques; keep the focus on biblical encouragement and practical wisdom.

Output rules:
- Return STRICT JSON only, matching the schema the user supplies.
- Use only the requested language for all content and references.
- Do NOT include long verbatim quotes from copyrighted translations (only WEB/CUV for full_passage_text).
"""


USER_PROMPT_TMPL = """User language: {language}
Faith background: {faith_background}
Situation detail: {situation}
Additional guidance: {guidance}

Return JSON with fields:
- passages: array of at most {max_passages} objects with fields:
  - ref (string, e.g., "Psalm 46:1-3" or localized equivalent)
  - short_quote (string, <= 20 words/chars; a minimal paraphrase; MAY be empty)
  - reason (string, 1–2 sentences explaining why this passage fits)
  - full_passage_text (string, VERBATIM full text from a public-domain version. Use WEB for English; use CUV for Chinese. Ensure the verses exactly match the cited reference.)
- devotional: a 300–500 {lang_unit} pastoral reflection for Christians, professional in tone; begin with 1–2 empathetic sentences, then provide Scripture-based reflection with one concise application or reflection question.
- prayer: 4–8 sentences, reverent and concise.
- disclaimer: one sentence kindly asking the user to verify in their preferred translation.

Use the requested language for everything.
"""


class BibleComfortService:
    """Service responsible for building prompts and calling the OpenAI API."""

    def __init__(self, openai_client: Optional[OpenAI] = None) -> None:
        self.client: Optional[OpenAI] = openai_client or _init_openai_client()

    def build_messages(self, q: BibleComfortQuery) -> List[Dict[str, str]]:
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

    def get_comfort(self, q: BibleComfortQuery, *, openai_client: Optional[OpenAI] = None) -> Dict[str, Any]:
        """
        Builds the prompt, calls the OpenAI API, and processes the response.
        Returns a dict that matches BibleComfortResponse schema.
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

# Backward-compatible aliases (to avoid breaking external imports)
ComfortQuery = BibleComfortQuery
Passage = BiblePassage
ComfortResponse = BibleComfortResponse
