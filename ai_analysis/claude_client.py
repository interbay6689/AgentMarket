"""
claude_client.py — Anthropic Claude wrapper for all AI analysis in AgentMarket.

Replaces the legacy openai_client.py. All AI calls in the project route through here.
Credentials come exclusively from the ANTHROPIC_API_KEY environment variable.
"""
import json
import logging
import os

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a financial market analyst specializing in US equity futures (NQ, ES, MES). "
    "Analyze the provided text and return a concise, professional assessment in the same "
    "language the user writes in. Be direct and actionable."
)

_client = None


def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _client


def analyze_text(
    prompt: str,
    model: str | None = None,
    deep: bool = False,
    system_override: str | None = None,
    max_tokens: int = 1024,
) -> str:
    """
    General-purpose text analysis via Claude.

    model      — explicit model ID; if None, chooses haiku (fast) or sonnet (deep=True)
    deep       — use sonnet-class model for complex reasoning
    system_override — replace the default financial-analyst system prompt
    """
    if model is None:
        model = "claude-sonnet-4-6" if deep else "claude-haiku-4-5-20251001"
    system = system_override or _SYSTEM_PROMPT
    try:
        response = _get_client().messages.create(
            model=model,
            max_tokens=max_tokens,
            system=[{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": prompt}],
        )
        return next((b.text for b in response.content if b.type == "text"), "")
    except Exception as e:
        logger.error(f"Claude analyze_text failed: {e}")
        return ""


def analyze_json(
    prompt: str,
    system_override: str | None = None,
    model: str | None = None,
    max_tokens: int = 1400,
) -> dict | None:
    """
    Call Claude and parse a JSON object from the response.

    Strips markdown code fences automatically. Returns None on parse failure.
    Use for structured outputs (category scoring, classification, extraction).
    """
    if model is None:
        model = "claude-haiku-4-5-20251001"

    system = system_override or (
        "You are a financial data extraction assistant. "
        "Return only valid JSON — no markdown, no prose outside the JSON object."
    )
    raw = analyze_text(prompt, model=model, system_override=system, max_tokens=max_tokens)
    return _parse_json(raw)


def analyze_article(
    title: str,
    body: str,
    language: str = "he",
) -> str:
    """
    Deep analysis of a single financial article.
    Returns a structured Hebrew/English assessment with impact and recommendation.
    """
    lang_note = "ענה בעברית." if language == "he" else "Respond in English."
    prompt = f"""נתח את כתבת החדשות הכלכלית הבאה וספק:
1. השפעה על חוזים עתידיים (NQ/ES): חיובית / שלילית / נייטרלית
2. ניקוד השפעה 1–10
3. המלצה: לונג / שורט / ללא פעולה
4. הסבר קצר (2-3 משפטים)

{lang_note}

כותרת: {title}
תוכן: {body[:2000]}
"""
    return analyze_text(prompt, deep=True, max_tokens=800)


def _parse_json(raw: str) -> dict | None:
    """Strip markdown fences and parse JSON. Returns None on failure."""
    if not raw:
        return None
    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
        raw = raw.rsplit("```", 1)[0].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass
    logger.debug(f"analyze_json: could not parse response: {raw[:200]}")
    return None
