import os, json
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from tempfile import NamedTemporaryFile
from io import BytesIO
from typing import Tuple
from comfort_service import ComfortService, ComfortQuery, ComfortResponse

def _init_openai_client() -> Optional[OpenAI]:
    """Initialize OpenAI client if API key is available; otherwise return None.
    Avoids raising during module import so unit tests can run without network/keys.
    """
    try:
        return OpenAI()
    except Exception:
        return None

# Lazily-initialized client; may be None in test environments without API key
client: Optional[OpenAI] = _init_openai_client()

# Instantiate service for comfort generation (separate client inside service)
comfort_service = ComfortService()

app = FastAPI(title="Comfort API (OpenAI SDK)")

# Change to your GitHub Pages domain (user page and/or project page)
ALLOWED_ORIGINS = [
    "https://tninja.github.io",
    "https://tninja.github.io/fastapi-helloworld"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TTSRequest(BaseModel):
    text: str
    language: Optional[str] = "zh"
    voice: Optional[str] = None  # if None, pick by language
    format: Optional[str] = "mp3"  # mp3 or wav

 

@app.post("/comfort", response_model=ComfortResponse)
def comfort(q: ComfortQuery):
    # Basic validation
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="Server missing OPENAI_API_KEY")

    try:
        # Inline the previous wrapper body by delegating directly to the service
        result = comfort_service.get_comfort(q)
        return result
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


def select_voice(language: str, override: Optional[str] = None) -> str:
    """Pick a reasonable default voice by language, unless override provided."""
    if override:
        return override
    # OpenAI voices like 'alloy', 'verse', 'onyx', 'nova' are multi-lingual.
    # Use 'alloy' as a safe default for both zh/en.
    return "alloy"


def generate_tts_audio(
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

    chosen_voice = select_voice(language or "zh", voice)
    fmt = (fmt or "mp3").lower()
    if fmt not in {"mp3", "wav"}:
        fmt = "mp3"

    media_type = "audio/mpeg" if fmt == "mp3" else "audio/wav"

    oc = openai_client or client

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


@app.post("/tts")
def tts(req: TTSRequest):
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="Server missing OPENAI_API_KEY")

    try:
        temp_path, media_type = generate_tts_audio(
            text=req.text,
            language=req.language or "zh",
            voice=req.voice,
            fmt=req.format or "mp3",
            openai_client=client,
        )

        def file_iterator(path: str, chunk_size: int = 8192):
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
            try:
                os.remove(path)
            except Exception:
                pass
        return StreamingResponse(file_iterator(temp_path), media_type=media_type)

    except Exception as e:
        # Clean up temp file if allocation happened but streaming failed
        try:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
        raise HTTPException(status_code=502, detail=f"TTS failed: {e}")
