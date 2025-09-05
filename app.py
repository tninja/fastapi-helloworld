import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from bible_comfort_service import BibleComfortService, ComfortQuery, ComfortResponse
from philosophy_comfort_service import (
    PhilosophyComfortService,
    PhilosophyComfortQuery,
    PhilosophyComfortResponse,
)
from tts_service import TTSRequest, TTSService

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

# Instantiate services
comfort_service = BibleComfortService()
philosophy_service = PhilosophyComfortService()
tts_service = TTSService()

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

 

@app.post("/bible-comfort", response_model=ComfortResponse)
def bible_comfort(q: ComfortQuery):
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


@app.post("/philosophy-comfort", response_model=PhilosophyComfortResponse)
def philosophy_comfort(q: PhilosophyComfortQuery):
    # Basic validation
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="Server missing OPENAI_API_KEY")

    try:
        result = philosophy_service.get_comfort(q)
        return result
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.post("/tts")
def tts(req: TTSRequest):
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="Server missing OPENAI_API_KEY")

    try:
        temp_path, media_type = tts_service.generate_audio(
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
