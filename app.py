from typing import Optional
import os

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "local-stub")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 256
    model: Optional[str] = None

class GenerateResponse(BaseModel):
    model: str
    text: str

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Return a very small stubbed response for testing."""
    model_name = req.model or DEFAULT_MODEL
    # very simple behavior: echo the prompt and include token hint
    text = f"Echo: {req.prompt} (max_tokens={int(req.max_tokens or 256)})"
    return GenerateResponse(model=model_name, text=text)


@app.get("/")
async def health():
    return {"status": "ok", "default_model": DEFAULT_MODEL}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
