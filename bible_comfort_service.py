import json
import os
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from openai import OpenAI
from comfort_search import DuckDuckGoSearchProvider, SearchFindingsFormatter, SearchProvider


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
    enable_web_search: bool = False


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
- Write a pastoral devotional (500–700 zh characters / 500–700 English words) with a professional tone for Christians.
- Begin the devotional with 1–2 empathetic sentences, then provide Scripture-based reflection or healing.
- Write a reverent, concise prayer (4–8 sentences).
- Avoid heavy emphasis on therapeutic techniques; keep the focus on biblical encouragement and practical wisdom.
- When web search findings are provided, use them to gather relevant current context such as news, Reddit posts, public discussions, and situational background.
- Use those web search findings to provide more accurate and relevant comfort, including better-matched passages, devotional reflection, and prayer.
- In the devotional, explicitly mention at least one concrete external detail from web search findings when available.
- In the prayer, explicitly mention at least one concrete external detail from web search findings when available.
- If web search findings include company names, platform names, source types, or public discussion themes, prefer to mention those exact details rather than generic wording.
- Use web search findings wisely inside devotional and prayer part to connect with the user's current reality, but do not let them overshadow Scripture as the primary authority.

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
- devotional: a 300–500 {lang_unit} pastoral reflection for Christians, professional in tone; begin with 1–2 empathetic sentences, then provide Scripture-based reflection / healing.
- prayer: 4–8 sentences, reverent and concise.
- disclaimer: one sentence kindly asking the user to verify in their preferred translation.

If web search findings are included below, use them to understand the user's current reality more precisely, including relevant news, Reddit posts, and public discussion themes. Use them to improve relevance, but keep Scripture as the primary authority and do not invent facts.
If the web search findings include company names, platform names, source types, or public discussion themes, explicitly mention at least one or two of those details in the devotional and at least one in the prayer when natural to do so.

Use the requested language for everything.
"""
class BibleComfortService:
    """Service responsible for building prompts and calling the OpenAI API."""

    def __init__(
        self,
        openai_client: Optional[OpenAI] = None,
        search_provider: Optional[SearchProvider] = None,
    ) -> None:
        self.client: Optional[OpenAI] = openai_client or _init_openai_client()
        self.search_provider = search_provider or DuckDuckGoSearchProvider.from_env()
        self._search_findings_formatter = SearchFindingsFormatter()

    def build_messages(
        self,
        q: BibleComfortQuery,
        search_context: str = "",
    ) -> List[Dict[str, str]]:
        lang_unit = "characters" if q.language.startswith("zh") else "words"
        uprompt = USER_PROMPT_TMPL.format(
            language=q.language,
            faith_background=q.faith_background or "christian",
            situation=q.situation,
            guidance=q.guidance or "None",
            max_passages=max(1, min(q.max_passages, 10)),
            lang_unit=lang_unit,
        )
        ## DONE: I want to have a checkbox in the UI to enable web search context. If not checked (by default), we should not call the search provider at all and can skip appending search context to the prompt. If enabled, the default waiting time is around 40 sec
        uprompt = self._append_search_context(uprompt, search_context)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": uprompt},
        ]

    def _append_search_context(self, prompt: str, search_context: str) -> str:
        if not search_context.strip():
            return prompt

        return (
            f"{prompt}\n"
            "Web search findings:\n"
            f"{search_context.strip()}\n\n"
            "When web search findings contain concrete details beyond the user's own wording, "
            "use at least one new external detail in the devotional and at least one in the prayer when relevant.\n"
            "Use these findings only as supporting background. "
            "Prioritize biblically faithful, situation-relevant passages, devotional guidance, and prayer."
        )

    def _build_web_search_prompt(self, q: BibleComfortQuery) -> str:
        return (
            "Search the web for recent and concrete context related to the user's situation. "
            "Focus on news, Reddit posts, and public discussion. "
            "Prefer details beyond the user's own wording, including named companies, source types, and specific concerns people are discussing.\n\n"
            f"User language: {q.language}\n"
            f"Faith background: {q.faith_background or 'christian'}\n"
            f"Situation detail: {q.situation}\n"
            f"Additional guidance: {q.guidance or 'None'}\n\n"
            "Return concise plain text with exactly these sections:\n"
            "News findings:\n"
            "- ...\n"
            "Reddit/public discussion findings:\n"
            "- ...\n"
            "Named entities:\n"
            "- ...\n"
        )

    def _search_with_openai_web(self, oc: OpenAI, q: BibleComfortQuery) -> str:
        responses_api = getattr(oc, "responses", None)
        if responses_api is None:
            return ""

        result = responses_api.create(
            model=os.getenv("BIBLE_COMFORT_SEARCH_MODEL", "gpt-5"),
            tools=[{"type": "web_search"}],
            input=self._build_web_search_prompt(q),
        )
        return (getattr(result, "output_text", "") or "").strip()

    def _format_search_findings(self, search_context: str) -> str:
        return self._search_findings_formatter.format(search_context)

    def _should_use_web_search(self, q: BibleComfortQuery) -> bool:
        return q.enable_web_search

    def _get_search_context(self, q: BibleComfortQuery, oc: OpenAI) -> str:
        if not self._should_use_web_search(q):
            return ""
        if self.search_provider is not None:
            search_context = self.search_provider.search(q)
            if isinstance(self.search_provider, DuckDuckGoSearchProvider):
                return self._format_search_findings(search_context)
            return search_context
        return self._search_with_openai_web(oc, q)

    def get_comfort(self, q: BibleComfortQuery, *, openai_client: Optional[OpenAI] = None) -> Dict[str, Any]:
        """
        Builds the prompt, calls the OpenAI API, and processes the response.
        Returns a dict that matches BibleComfortResponse schema.
        """
        try:
            oc = openai_client or self.client
            if oc is None:
                raise RuntimeError("OpenAI client not configured")

            messages = self.build_messages(q, search_context=self._get_search_context(q, oc))

            resp = oc.chat.completions.create(
                model="gpt-5-mini",
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
