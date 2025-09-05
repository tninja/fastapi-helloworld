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


class PhilosophyComfortQuery(BaseModel):
    language: str = "zh"
    situation: str
    philosophy_background: Optional[str] = "philosophy"
    guidance: Optional[str] = ""


class PhilosophyComfortResponse(BaseModel):
    reflection: str
    exercise: str
    disclaimer: str


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

Use the requested language for everything.
"""


class PhilosophyComfortService:
    """Service responsible for building prompts and calling the OpenAI API for philosophy comfort."""

    def __init__(self, openai_client: Optional[OpenAI] = None) -> None:
        self.client: Optional[OpenAI] = openai_client or _init_openai_client()

    def build_messages(self, q: PhilosophyComfortQuery) -> List[Dict[str, str]]:
        lang_unit = "characters" if q.language.startswith("zh") else "words"
        uprompt = USER_PROMPT_TMPL.format(
            language=q.language,
            background=(getattr(q, "philosophy_background", None) or "philosophy"),
            situation=q.situation,
            guidance=q.guidance or "None",
            lang_unit=lang_unit,
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": uprompt},
        ]

    def get_comfort(self, q: PhilosophyComfortQuery, *, openai_client: Optional[OpenAI] = None) -> Dict[str, Any]:
        """Build prompts, call the OpenAI API, and return a dict matching PhilosophyComfortResponse."""
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


# Backward-compatible aliases (symmetry with bible service)
ComfortQuery = PhilosophyComfortQuery
ComfortResponse = PhilosophyComfortResponse
