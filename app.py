"""
FastAPI app that calls the Gemini model (Google Generative AI / Gemini).

Usage:
- Install dependencies:
    pip install fastapi uvicorn google-generativeai

- Configure credentials (one of these):
    - Set GOOGLE_API_KEY with an API key, or
    - Set GOOGLE_APPLICATION_CREDENTIALS pointing to a service account JSON (ADC).

- Optional environment variable:
    - GEMINI_MODEL (defaults to "gemini-1.3")

Run:
    export GOOGLE_API_KEY="ya29.A0..."              # or set ADC
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

from typing import Optional
import os
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

# Try to import the Google generative AI helper library.
try:
    import google.generativeai as genai  # type: ignore
    HAS_GENAI = True
except Exception:
    genai = None  # type: ignore
    HAS_GENAI = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Gemini Q&A (FastAPI)")

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.3")


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 256
    model: Optional[str] = None


class GenerateResponse(BaseModel):
    model: str
    text: str


def _configure_genai_if_needed():
    """
    Configure the genai client if an API key is provided.
    If no API key is provided, the library will typically fall back to Application Default Credentials.
    """
    if not HAS_GENAI:
        return

    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        try:
            # configure is available in many client versions
            genai.configure(api_key=api_key)  # type: ignore
            logger.info("Configured google.generativeai with GOOGLE_API_KEY.")
        except Exception:
            # Some versions might not expose a configure function; ignore and rely on ADC if available.
            logger.debug("genai.configure not available; relying on ADC or library defaults.")


def extract_text_from_response(resp) -> str:
    """
    Try to extract human-readable text from a variety of possible response shapes.
    The google.generativeai library has gone through iterations; this helper is defensive.
    """
    if resp is None:
        return ""

    # common attribute in simple wrappers
    if hasattr(resp, "text") and resp.text:
        return resp.text

    # some return an 'output' field (string or list)
    out = getattr(resp, "output", None)
    if out:
        if isinstance(out, str):
            return out
        if isinstance(out, list) and len(out) > 0:
            # items may be dicts
            first = out[0]
            if isinstance(first, dict):
                return first.get("content") or str(first)
            return str(first)

    # choices/candidates style responses
    if hasattr(resp, "candidates") and len(getattr(resp, "candidates")) > 0:
        c0 = resp.candidates[0]
        return getattr(c0, "output_text", getattr(c0, "text", str(c0)))

    if hasattr(resp, "choices") and len(getattr(resp, "choices")) > 0:
        c0 = resp.choices[0]
        # chat-like choice shape
        if isinstance(getattr(c0, "message", None), dict):
            return c0.message.get("content", str(c0))
        return str(c0)

    # As a final fallback stringify
    return str(resp)


def call_gemini_sync(prompt: str, model: str, max_tokens: int) -> str:
    """
    Synchronous call to Gemini (blocking). We'll run this in a threadpool from the async endpoint.
    Uses google.generativeai when available; otherwise raises a RuntimeError explaining missing dependency.
    """
    if not HAS_GENAI:
        raise RuntimeError(
            "google-generativeai is not installed. Install it with: pip install google-generativeai"
        )

    _configure_genai_if_needed()
    model_name = model or DEFAULT_MODEL

    # Try the simple text generation API if present
    try:
        # Many versions expose a generate_text or similar. This is the preferred simple call.
        resp = genai.generate_text(model=model_name, prompt=prompt, max_output_tokens=max_tokens)  # type: ignore
        return extract_text_from_response(resp)
    except Exception as e:
        logger.debug("generate_text call failed or not available: %s", e)

    # Try chat-like interface
    try:
        chat_resp = genai.chat.create(model=model_name, messages=[{"role": "user", "content": prompt}])  # type: ignore
        return extract_text_from_response(chat_resp)
    except Exception as e:
        logger.debug("chat.create call failed or not available: %s", e)

    # If none of the stable calls worked, raise a helpful error with guidance
    raise RuntimeError(
        "Could not call Gemini via google.generativeai. "
        "Check that google-generativeai is installed and up to date, and that credentials are configured."
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """
    Generate text from the Gemini model.

    Example request body:
    {
      "prompt": "Write a short poem about the moon.",
      "max_tokens": 100,
      "model": "gemini-1.3"
    }
    """
    model_name = req.model or DEFAULT_MODEL
    try:
        # run the blocking network call in a threadpool so the FastAPI event loop is not blocked
        text = await run_in_threadpool(call_gemini_sync, req.prompt, model_name, int(req.max_tokens or 256))
    except Exception as e:
        logger.exception("Gemini call failed")
        raise HTTPException(status_code=500, detail=f"Gemini call failed: {e}")

    return GenerateResponse(model=model_name, text=text)


@app.get("/health")
async def health():
    """
    Health check endpoint. Returns whether the genai client appears to be installed and the default model.
    """
    return {"status": "ok", "has_genai": HAS_GENAI, "default_model": DEFAULT_MODEL}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
