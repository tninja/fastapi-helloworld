import asyncio
import json
import os
import shlex
from typing import List, Optional, Dict, Any, Protocol
from pydantic import BaseModel
from openai import OpenAI

try:
    import mcp.types as mcp_types
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
except ImportError:
    mcp_types = None
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None


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


class SearchProvider(Protocol):
    def search(self, query: "BibleComfortQuery") -> str:
        ...


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
- When DuckDuckGo research context is provided, use it to gather relevant current context such as news, Reddit posts, public discussions, and situational background.
- Use that DuckDuckGo context to provide more accurate and relevant comfort, including better-matched passages, devotional reflection, and prayer.
- In the devotional, explicitly mention at least one concrete external detail from DuckDuckGo context when available.
- In the prayer, explicitly mention at least one concrete external detail from DuckDuckGo context when available.
- If DuckDuckGo context includes company names, platform names, source types, or public discussion themes, prefer to mention those exact details rather than generic wording.
- Use DuckDuckGo context wisely inside devotional and prayer part to connect with the user's current reality, but do not let it overshadow Scripture as the primary authority.

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

If DuckDuckGo research context is included below, use it to understand the user's current reality more precisely, including relevant news, Reddit posts, and public discussion themes. Use it to improve relevance, but keep Scripture as the primary authority and do not invent facts.
If the DuckDuckGo context includes company names, platform names, source types, or public discussion themes, explicitly mention at least one or two of those details in the devotional and at least one in the prayer when natural to do so.

Use the requested language for everything.
"""


def _resolve_server_args(template: str) -> List[str]:
    return shlex.split(template)


class DuckDuckGoMCPSearchProvider:
    def __init__(
        self,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        max_results: Optional[int] = None,
    ) -> None:
        self.command = command or os.getenv("DDG_MCP_CMD", "uvx")
        self.args = args or _resolve_server_args(
            os.getenv("DDG_MCP_ARGS", "duckduckgo-mcp-server")
        )
        self.max_results = max_results or int(os.getenv("DDG_MCP_MAX_RESULTS", "5"))

    def _build_query_text(self, query: BibleComfortQuery) -> str:
        parts = [
            query.situation.strip(),
            (query.guidance or "").strip(),
            f"faith background: {query.faith_background or 'christian'}",
            "Find relevant current public discussion, including news, Reddit posts, forum discussions, and other public commentary about this situation.",
            "Prioritize recent real-world context that helps explain what people are experiencing, fearing, or discussing.",
            "Return context that can help a Christian pastoral response choose more relevant passages, devotional reflection, and prayer.",
        ]
        return " ".join(part for part in parts if part)

    async def _search_async(self, query_text: str) -> str:
        if (
            not query_text
            or mcp_types is None
            or ClientSession is None
            or StdioServerParameters is None
            or stdio_client is None
        ):
            return ""

        server_params = StdioServerParameters(
            command=self.command,
            args=list(self.args),
        )

        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    "search",
                    {
                        "query": query_text,
                        "max_results": self.max_results,
                    },
                )

                if result.isError:
                    return ""

                text_parts = [
                    item.text.strip()
                    for item in result.content
                    if isinstance(item, mcp_types.TextContent) and item.text.strip()
                ]
                return "\n".join(text_parts)

    def search(self, query: BibleComfortQuery) -> str:
        query_text = self._build_query_text(query)
        try:
            return asyncio.run(self._search_async(query_text))
        except Exception:
            return ""


class BibleComfortService:
    """Service responsible for building prompts and calling the OpenAI API."""

    def __init__(
        self,
        openai_client: Optional[OpenAI] = None,
        search_provider: Optional[SearchProvider] = None,
    ) -> None:
        self.client: Optional[OpenAI] = openai_client or _init_openai_client()
        self.search_provider = search_provider or DuckDuckGoMCPSearchProvider()

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
            "DuckDuckGo research context:\n"
            f"{search_context.strip()}\n\n"
            "Use this research context only as supporting background. "
            "Prioritize biblically faithful, situation-relevant passages, devotional guidance, and prayer."
        )

    def _get_search_context(self, q: BibleComfortQuery) -> str:
        if self.search_provider is None:
            return ""
        return self.search_provider.search(q)

    def get_comfort(self, q: BibleComfortQuery, *, openai_client: Optional[OpenAI] = None) -> Dict[str, Any]:
        """
        Builds the prompt, calls the OpenAI API, and processes the response.
        Returns a dict that matches BibleComfortResponse schema.
        """
        try:
            oc = openai_client or self.client
            if oc is None:
                raise RuntimeError("OpenAI client not configured")

            messages = self.build_messages(q, search_context=self._get_search_context(q))

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
