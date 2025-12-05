import os
from typing import Generator, Optional

try:
    from groq import Groq
except Exception:  # pragma: no cover
    Groq = None  # type: ignore

# Read env once
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def systems_supported() -> bool:
    """
    Best-effort capability check for Groq Compound Systems.
    Returns True if the Groq SDK is importable and an API key is present.
    We defer deeper checks to runtime to avoid import-time failures with older SDKs.
    """
    return Groq is not None and bool(GROQ_API_KEY)


def stream_compound_response(
    prompt: str,
    *,
    session_id: Optional[str] = None,
    system_id: Optional[str] = None,
    version: Optional[str] = None,
    simulate_chunk_bytes: int = 256,
) -> Generator[str, None, None]:
    """
    Stream tokens from Groq Compound System using the 'groq/compound' model.
    This replaces the legacy Systems API with the new chat completions model.
    """
    if Groq is None or not GROQ_API_KEY:
        raise RuntimeError("Groq SDK or API key not available for Systems.")

    client = Groq(api_key=GROQ_API_KEY)

    # The user requested to use model="groq/compound"
    # We map the prompt to a user message.
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    try:
        # Use the standard chat completions API with the compound model
        stream = client.chat.completions.create(
            model="groq/compound",
            messages=messages,
            stream=True
        )
        
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    except Exception as e:
        # Bubble up to trigger fallback
        raise RuntimeError(f"Groq Compound System invocation failed: {e}")
