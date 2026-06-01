import os
import anthropic

_SYSTEM_PROMPT = (
    "You are a financial market analyst specializing in US equity futures (NQ, ES, MES). "
    "Analyze the provided text and return a concise, professional assessment in the same "
    "language the user writes in. Be direct and actionable."
)

_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _client


def analyze_text(prompt: str, model: str = None, deep: bool = False) -> str:
    if model is None:
        model = "claude-sonnet-4-6" if deep else "claude-haiku-4-5-20251001"
    response = _get_client().messages.create(
        model=model,
        max_tokens=1024,
        system=[{
            "type": "text",
            "text": _SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": prompt}],
    )
    return next((b.text for b in response.content if b.type == "text"), "")
