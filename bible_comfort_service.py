import asyncio
import json
import os
import re
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
        return self._build_search_queries(query)[0]

    def _build_search_queries(self, query: BibleComfortQuery) -> List[str]:
        entities = self._extract_query_entities(query)
        entity_phrase = " ".join(entities[:4]) or "AI layoffs job insecurity"
        guidance = (query.guidance or "").strip()
        guidance_suffix = f" {guidance}" if guidance else ""

        return [
            f"{entity_phrase} news layoffs AI job market{guidance_suffix}".strip(),
            f"{entity_phrase} reddit discussion layoffs hiring freeze{guidance_suffix}".strip(),
            f"{entity_phrase} public discussion forum job insecurity recession fears{guidance_suffix}".strip(),
        ]

    def _extract_query_entities(self, query: BibleComfortQuery) -> List[str]:
        source_text = f"{query.situation} {query.guidance or ''}"
        matches = re.findall(r"[a-zA-Z][a-zA-Z0-9&.-]{1,}", source_text)
        stop_words = {
            "the",
            "and",
            "for",
            "with",
            "this",
            "that",
            "have",
            "many",
            "because",
            "about",
            "there",
            "recently",
            "ai",
            "layoff",
            "layoffs",
            "employee",
            "employees",
            "work",
            "job",
        }
        cleaned = [match.lower() for match in matches if match.lower() not in stop_words]
        return self._dedupe_keep_order(cleaned)

    async def _search_async(self, query_texts: List[str]) -> str:
        if (
            not query_texts
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
                collected_reports: List[str] = []
                for query_text in query_texts:
                    result = await session.call_tool(
                        "search",
                        {
                            "query": query_text,
                            "max_results": self.max_results,
                        },
                    )

                    if result.isError:
                        continue

                    text_parts = [
                        item.text.strip()
                        for item in result.content
                        if isinstance(item, mcp_types.TextContent) and item.text.strip()
                    ]
                    report = "\n".join(text_parts).strip()
                    if report and not self._looks_like_no_results(report):
                        collected_reports.append(report)

                return "\n\n".join(collected_reports)

    def _looks_like_no_results(self, report: str) -> bool:
        normalized = report.lower()
        return "no results were found" in normalized or "query returned no matches" in normalized

    def search(self, query: BibleComfortQuery) -> str:
        query_texts = self._build_search_queries(query)
        try:
            return asyncio.run(self._search_async(query_texts))
        except Exception:
            return ""

    def _dedupe_keep_order(self, items: List[str]) -> List[str]:
        seen = set()
        deduped: List[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped


class BibleComfortService:
    """Service responsible for building prompts and calling the OpenAI API."""

    def __init__(
        self,
        openai_client: Optional[OpenAI] = None,
        search_provider: Optional[SearchProvider] = None,
    ) -> None:
        self.client: Optional[OpenAI] = openai_client or _init_openai_client()
        self.search_provider = search_provider

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
            "OpenAI web search findings:\n"
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
        entries = self._parse_search_entries(search_context)
        if not entries:
            return "News findings:\n- None\nReddit/public discussion findings:\n- None\nNamed entities:\n- None"

        news_findings: List[str] = []
        discussion_findings: List[str] = []
        named_entities: List[str] = []

        for entry in entries:
            title = entry["title"]
            summary = entry["summary"]
            url = entry["url"]
            finding = self._compose_finding(title, summary)
            if self._is_discussion_entry(title, summary, url):
                discussion_findings.append(finding)
            else:
                news_findings.append(finding)
            named_entities.extend(self._extract_named_entities(f"{title} {summary} {url}"))

        named_entities = self._dedupe_keep_order(named_entities)[:8]
        return "\n".join(
            [
                "News findings:",
                *self._format_section(news_findings),
                "Reddit/public discussion findings:",
                *self._format_section(discussion_findings),
                "Named entities:",
                *self._format_section(named_entities),
            ]
        )

    def _parse_search_entries(self, search_context: str) -> List[Dict[str, str]]:
        entries: List[Dict[str, str]] = []
        current: Optional[Dict[str, str]] = None

        for raw_line in search_context.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            if re.match(r"^\d+\.\s+", line):
                if current:
                    entries.append(current)
                current = {"title": re.sub(r"^\d+\.\s+", "", line), "summary": "", "url": ""}
                continue

            if current is None:
                current = {"title": line, "summary": "", "url": ""}
                continue

            if line.startswith("URL:"):
                current["url"] = line[len("URL:") :].strip()
            elif line.startswith("Summary:"):
                current["summary"] = line[len("Summary:") :].strip()
            elif current["summary"]:
                current["summary"] = f"{current['summary']} {line}".strip()
            else:
                current["summary"] = line

        if current:
            entries.append(current)

        return entries[:5]

    def _compose_finding(self, title: str, summary: str) -> str:
        if title and summary:
            return f"{title}. {summary}"
        return title or summary or "None"

    def _is_discussion_entry(self, title: str, summary: str, url: str) -> bool:
        haystack = f"{title} {summary} {url}".lower()
        discussion_markers = [
            "reddit",
            "forum",
            "discussion",
            "commentary",
            "comments",
            "users discuss",
        ]
        return any(marker in haystack for marker in discussion_markers)

    def _extract_named_entities(self, text: str) -> List[str]:
        matches = re.findall(r"\b[A-Z][A-Za-z0-9&.-]{1,}\b", text)
        return [match for match in matches if match.lower() not in {"url", "summary", "news", "public"}]

    def _dedupe_keep_order(self, items: List[str]) -> List[str]:
        seen = set()
        deduped: List[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    def _format_section(self, items: List[str]) -> List[str]:
        if not items:
            return ["- None"]
        return [f"- {item}" for item in items[:3]]

    def _get_search_context(self, q: BibleComfortQuery, oc: OpenAI) -> str:
        if self.search_provider is not None:
            return self.search_provider.search(q)
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
