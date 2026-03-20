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


class PhilosophyComfortQuery(BaseModel):
    language: str = "zh"
    situation: str
    philosophy_background: Optional[str] = "philosophy"
    guidance: Optional[str] = ""
    enable_web_search: bool = False


class PhilosophyComfortResponse(BaseModel):
    reflection: str
    exercise: str
    disclaimer: str


class SearchProvider(Protocol):
    def search(self, query: "PhilosophyComfortQuery") -> str:
        ...


EMPTY_SEARCH_FINDINGS = (
    "News findings:\n"
    "- None\n"
    "Reddit/public discussion findings:\n"
    "- None\n"
    "Named entities:\n"
    "- None"
)

DISCUSSION_MARKERS = (
    "reddit",
    "forum",
    "discussion",
    "commentary",
    "comments",
    "users discuss",
)


def _resolve_server_args(template: str) -> List[str]:
    return shlex.split(template)


class DuckDuckGoSearchProvider:
    def __init__(
        self,
        server_cmd: str,
        server_args: tuple[str, ...],
        server_dir: str | None,
        max_results: int,
    ) -> None:
        self.server_cmd = server_cmd
        self.server_args = server_args
        self.server_dir = server_dir
        self.max_results = max_results

    @classmethod
    def from_env(cls) -> "DuckDuckGoSearchProvider":
        return cls(
            server_cmd=os.getenv("DDG_MCP_CMD", "uvx"),
            server_args=tuple(
                _resolve_server_args(os.getenv("DDG_MCP_ARGS", "duckduckgo-mcp-server"))
            ),
            server_dir=os.getenv("DDG_MCP_DIR"),
            max_results=int(os.getenv("DDG_MCP_MAX_RESULTS", "10")),
        )

    def search(self, query: "PhilosophyComfortQuery") -> str:
        return asyncio.run(self._search_async(query))

    def _require_mcp_dependencies(self) -> None:
        if any(
            dependency is None
            for dependency in (mcp_types, ClientSession, StdioServerParameters, stdio_client)
        ):
            raise RuntimeError("DuckDuckGo MCP dependencies are not available.")

    def _extract_text_parts(self, result: Any) -> list[str]:
        return [
            item.text.strip()
            for item in result.content
            if isinstance(item, mcp_types.TextContent) and item.text.strip()
        ]

    async def _search_async(self, query: "PhilosophyComfortQuery") -> str:
        self._require_mcp_dependencies()
        server_params = StdioServerParameters(
            command=self.server_cmd,
            args=list(self.server_args),
            cwd=self.server_dir,
        )

        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    "search",
                    {
                        "query": query.situation.strip(),
                        "max_results": self.max_results,
                    },
                )

                if result.isError:
                    message = "\n".join(self._extract_text_parts(result)) or "Unknown MCP error"
                    raise RuntimeError(f"MCP server reported an error: {message}")

                return "\n".join(self._extract_text_parts(result))


SYSTEM_PROMPT = """You are a calm, pluralistic philosophical counselor who draws from a wide range of philosophers (e.g., Aristotle, Epicurus, Stoics like Marcus Aurelius/Epictetus/Seneca, Confucius, Montaigne, Descartes, Spinoza, Hume, Kant, Schopenhauer, Nietzsche, Kierkegaard, Camus, Sartre) and from 'The Consolations of Philosophy' by Alain de Botton.
You MUST respond STRICTLY in the user's requested language (zh for Chinese, en for English) and DO NOT mix languages.
Tone and style: be warm, gently healing, and non-judgmental; validate feelings with care; avoid lecturing or preaching; avoid "should/must"; prefer soft invitations like "you might try", "consider", "if it helps"; keep sentences clear and not too long; use plain, compassionate wording; choose phrasing that feels safe, tender, and reassuring.
Healing emphasis: prioritize relief, steadiness, and hope; translate philosophical ideas into everyday language; favor self-compassion, present-moment grounding (breath, senses, posture), and small, achievable steps; avoid harsh or confrontational wording; if in doubt, choose the kinder phrasing.
Select the most relevant, high-leverage ideas to comfort and guide the user; combine multiple perspectives when helpful.
You are encouraged to incorporate insights from 'The Consolations of Philosophy'—summarize its ideas clearly and practically.
Explicitly draw on Philosophy of Well-Being and practical wisdom aimed at living a happier life; name relevant concepts (eudaimonia, ataraxia, flourishing, virtue ethics, meaning, etc.).
When appropriate, cite or summarize ideas from Philosophy of Well-Being and "wisdom to live happier" traditions to enhance clarity and usefulness.
For copyrighted works (including modern books): prefer concise paraphrases rather than long verbatim quotes. For public-domain works, you may include short snippets but keep them brief (<= 20 words/chars).
Write a practical, compassionate, and healing-toned philosophical reflection. Begin with 1–2 sentences of empathy and normalization. Maintain a soft, soothing voice; include at least one gentle reframe and one brief grounding cue (e.g., "notice your feet on the floor").
Provide a short step-by-step philosophical exercise (4-8 sentences) written as a gentle invitation, not a command. Make it easy to try (2–5 minutes), with optional steps (e.g., a few slow breaths, a soft reframing, a small action, a sensory check-in like placing a hand on the chest). Close with one reassuring sentence.
Avoid sectarian or religious framing; focus on agency, clarity, and emotional steadiness.
- When web search findings are provided, use them to gather relevant current context such as news, Reddit posts, public discussions, and situational background.
- Use those web search findings to provide more accurate and relevant comfort in the reflection and exercise.
- In the reflection, explicitly mention at least one concrete external detail from web search findings when available.
- In the exercise, explicitly mention at least one concrete external detail from web search findings when available.
- If web search findings include company names, platform names, source types, or public discussion themes, prefer to mention those exact details rather than generic wording.
Return STRICT JSON only, matching exactly the schema the user supplies.
If unsure about exact sections, choose ones you are confident in and clearly name the work and section (e.g., "Meditations 2.1", "Nicomachean Ethics II").
"""


USER_PROMPT_TMPL = """User language: {language}
Philosophical background: {background}
Situation detail: {situation}
Additional guidance: {guidance}

Return JSON with fields:
- reflection: a 500-700 {lang_unit} philosophical reflection tailored to the user's situation; open with empathy and normalization; keep a warm, soothing, and healing tone; avoid lecturing; use soft invitations and plain language; include one gentle reframe and one tiny grounding cue (e.g., noticing breath or contact with the chair).
- exercise: 4-8 sentences describing a gentle, invitation-style practice (e.g., a few breaths, a soft reframing, journaling prompts, view-from-above) that can be done in 2–5 minutes; mark steps as optional where helpful; include a brief sensory step (e.g., hand on chest) and end with one sentence of reassurance.
- disclaimer: one concise sentence reminding the user that summaries may differ by edition/translation and encouraging verification.

If web search findings are included below, use them to understand the user's current reality more precisely, including relevant news, Reddit posts, and public discussion themes. Use them to improve relevance, but do not invent facts.
If the web search findings include company names, platform names, source types, or public discussion themes, explicitly mention at least one or two of those details in the reflection and at least one in the exercise when natural to do so.

Use the requested language for everything.
"""


class PhilosophyComfortService:
    """Service responsible for building prompts and calling the OpenAI API for philosophy comfort."""

    def __init__(
        self,
        openai_client: Optional[OpenAI] = None,
        search_provider: Optional[SearchProvider] = None,
    ) -> None:
        self.client: Optional[OpenAI] = openai_client or _init_openai_client()
        self.search_provider = search_provider or DuckDuckGoSearchProvider.from_env()

    def build_messages(
        self,
        q: PhilosophyComfortQuery,
        search_context: str = "",
    ) -> List[Dict[str, str]]:
        lang_unit = "characters" if q.language.startswith("zh") else "words"
        uprompt = USER_PROMPT_TMPL.format(
            language=q.language,
            background=(getattr(q, "philosophy_background", None) or "philosophy"),
            situation=q.situation,
            guidance=q.guidance or "None",
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
            "Web search findings:\n"
            f"{search_context.strip()}\n\n"
            "When web search findings contain concrete details beyond the user's own wording, "
            "use at least one new external detail in the reflection and at least one in the exercise when relevant.\n"
            "Use these findings only as supporting background. "
            "Prioritize relevant philosophical reflection, practical comfort, and grounded exercise steps."
        )

    def _format_search_findings(self, search_context: str) -> str:
        entries = self._parse_search_entries(search_context)
        if not entries:
            return EMPTY_SEARCH_FINDINGS

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
        return any(marker in haystack for marker in DISCUSSION_MARKERS)

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

    def _build_web_search_prompt(self, q: PhilosophyComfortQuery) -> str:
        return (
            "Search the web for recent and concrete context related to the user's situation. "
            "Focus on news, Reddit posts, and public discussion. "
            "Prefer details beyond the user's own wording, including named companies, source types, and specific concerns people are discussing.\n\n"
            f"User language: {q.language}\n"
            f"Philosophical background: {q.philosophy_background or 'philosophy'}\n"
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

    def _search_with_openai_web(self, oc: OpenAI, q: PhilosophyComfortQuery) -> str:
        responses_api = getattr(oc, "responses", None)
        if responses_api is None:
            return ""

        result = responses_api.create(
            model=os.getenv("PHILOSOPHY_COMFORT_SEARCH_MODEL", "gpt-5"),
            tools=[{"type": "web_search"}],
            input=self._build_web_search_prompt(q),
        )
        return (getattr(result, "output_text", "") or "").strip()

    def _should_use_web_search(self, q: PhilosophyComfortQuery) -> bool:
        return q.enable_web_search

    def _get_search_context(self, q: PhilosophyComfortQuery, oc: OpenAI) -> str:
        if not self._should_use_web_search(q):
            return ""
        if self.search_provider is not None:
            search_context = self.search_provider.search(q)
            if isinstance(self.search_provider, DuckDuckGoSearchProvider):
                return self._format_search_findings(search_context)
            return search_context
        return self._search_with_openai_web(oc, q)

    def get_comfort(self, q: PhilosophyComfortQuery, *, openai_client: Optional[OpenAI] = None) -> Dict[str, Any]:
        """Build prompts, call the OpenAI API, and return a dict matching PhilosophyComfortResponse."""
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

            # Ensure fields exist
            data.setdefault("reflection", "")
            data.setdefault("exercise", "")

            if not data.get("disclaimer"):
                data["disclaimer"] = (
                    "Please verify sources in your preferred edition/translation; non-public-domain texts are summarized, and this is supportive guidance only."
                )

            return data

        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {e}") from e
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {e}") from e
