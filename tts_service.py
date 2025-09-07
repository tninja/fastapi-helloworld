from typing import Optional, Tuple
from pydantic import BaseModel
from openai import OpenAI
from tempfile import NamedTemporaryFile
import os


def _init_openai_client() -> Optional[OpenAI]:
    """Initialize OpenAI client if API key is available; otherwise return None.
    Avoids raising during module import so unit tests can run without network/keys.
    """
    try:
        return OpenAI()
    except Exception:
        return None


class TTSRequest(BaseModel):
    text: str
    language: Optional[str] = "zh"
    voice: Optional[str] = None  # if None, pick by language
    format: Optional[str] = "mp3"  # mp3 or wav


class TTSService:
    """Service for Text-to-Speech generation via OpenAI."""

    def __init__(self, openai_client: Optional[OpenAI] = None) -> None:
        self.client: Optional[OpenAI] = openai_client or _init_openai_client()

    def select_voice(self, language: str, override: Optional[str] = None) -> str:
        """Pick a reasonable default voice by language, unless override provided."""
        if override:
            return override
        # OpenAI voices like 'alloy', 'verse', 'onyx', 'nova' are multi-lingual.
        # Use 'alloy' as a safe default for both zh/en.
        # 哪种更适合做 Devotional / Prayer？
        # 这部分有点主观，但根据社区反馈和我的理解：
        # fable：声音柔和、叙事感强，适合讲故事或做灵修分享。
        # alloy：较自然、稳重，男女声中性，适合祷告、沉静的场景。
        # shimmer：稍微明亮、轻快一些，适合鼓励、安慰式的语境。
        # 如果你想要一种“温柔、安静、带陪伴感”的声音，我建议先试 fable 或 alloy，效果通常比较贴近祷告与灵修氛围。
        return "fabel" # "alloy"

    def generate_audio(
            self,
            text: str,
            language: str = "zh",
            voice: Optional[str] = None,
            fmt: str = "mp3",
            *,
            openai_client: Optional[OpenAI] = None,
    ) -> Tuple[str, str]:
        """
        Generate TTS audio to a temporary file using OpenAI TTS.

        Returns a tuple (temp_file_path, media_type).
        The caller is responsible for deleting the temp file after streaming/usage.
        """
        if not text or not text.strip():
            raise ValueError("Missing text for TTS")

        # Basic caps to avoid extremely long synthesis
        max_chars = 6000
        text = text.strip()
        if len(text) > max_chars:
            text = text[:max_chars]

        chosen_voice = self.select_voice(language or "zh", voice)
        fmt = (fmt or "mp3").lower()
        if fmt not in {"mp3", "wav"}:
            fmt = "mp3"

        media_type = "audio/mpeg" if fmt == "mp3" else "audio/wav"

        oc = openai_client or self.client
        if oc is None:
            raise RuntimeError("OpenAI client not configured for TTS")

        with NamedTemporaryFile(delete=False, suffix=f".{fmt}") as tmp:
            temp_path = tmp.name

        # Stream audio to file via SDK
        with oc.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice=chosen_voice,
                input=text,
        ) as response:
            response.stream_to_file(temp_path)

        return temp_path, media_type
