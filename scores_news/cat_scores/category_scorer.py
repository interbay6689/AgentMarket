"""
category_scorer.py — Claude-based economic category analysis for news articles.

Ported from Drive's content_analyz.py (GPT-4o version) and adapted to use the
project's existing Anthropic/Claude stack (ai_analysis/openai_client.py pattern).

Falls back gracefully when ANTHROPIC_API_KEY is absent or the API call fails.
"""
import hashlib
import json
import logging
import os

logger = logging.getLogger(__name__)

_SCORE_CACHE: dict = {}
_MAX_CACHE = 150

_SYSTEM = (
    "You are a financial analyst specializing in US equity futures (NQ, ES). "
    "Analyze news articles for their impact on futures markets. "
    "Return only valid JSON — no markdown, no explanation outside the JSON."
)

_PROMPT_TEMPLATE = """\
Analyze the following {n} financial news articles for their impact on NQ/ES futures.

For EACH article return:
- net_impact: float -3.0 to +3.0 (negative = bearish for NQ, positive = bullish)
- categories: object with keys: inflation, interest_rates, equities, economic_growth,
  geopolitical, forex, energy, unemployment
  Each key maps to {{impact: float -3..+3, relevance: int 0..100}}

Also return aggregate_score: int 0..100 (50=neutral, >50=bullish NQ overall).

JSON schema:
{{
  "articles": [
    {{
      "idx": 0,
      "net_impact": 0.0,
      "categories": {{
        "inflation":      {{"impact": 0.0, "relevance": 0}},
        "interest_rates": {{"impact": 0.0, "relevance": 0}},
        "equities":       {{"impact": 0.0, "relevance": 0}},
        "economic_growth":{{"impact": 0.0, "relevance": 0}},
        "geopolitical":   {{"impact": 0.0, "relevance": 0}},
        "forex":          {{"impact": 0.0, "relevance": 0}},
        "energy":         {{"impact": 0.0, "relevance": 0}},
        "unemployment":   {{"impact": 0.0, "relevance": 0}}
      }}
    }}
  ],
  "aggregate_score": 50
}}

Articles:
{articles_text}"""


def _articles_to_text(articles: list) -> str:
    lines = []
    for i, a in enumerate(articles):
        full = a.get("full_text") or ""
        summary = a.get("summary") or ""
        title = a.get("title") or ""
        # Prefer full_text; fall back to summary; always include title
        body = (full if len(full) > len(summary) else summary)[:500]
        lines.append(f"[{i}] {title}. {body}")
    return "\n\n".join(lines)


def _cache_key(articles: list) -> str:
    tag = "".join((a.get("link") or a.get("title") or "")[:80] for a in articles)
    return hashlib.md5(tag.encode()).hexdigest()


def _parse_response(raw: str) -> dict | None:
    raw = raw.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
        raw = raw.rsplit("```", 1)[0].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to find a JSON object in the response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass
    return None


def score_articles_with_claude(articles: list) -> tuple[int | None, dict, str]:
    """
    Analyze articles using Claude for structured economic category impacts.

    Returns:
        (score, breakdown, explanation)
        score      — int 0-100 (None if unavailable)
        breakdown  — {category: avg_impact} for categories with relevance > 10
        explanation — human-readable summary string
    """
    if not articles:
        return None, {}, ""

    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None, {}, "ANTHROPIC_API_KEY not set — using VADER only"

    batch = articles[:8]  # Cap at 8 to keep prompt reasonable
    key = _cache_key(batch)
    if key in _SCORE_CACHE:
        return _SCORE_CACHE[key]

    articles_text = _articles_to_text(batch)
    prompt = _PROMPT_TEMPLATE.format(n=len(batch), articles_text=articles_text)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1400,
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = next((b.text for b in response.content if b.type == "text"), "")
        data = _parse_response(raw)
        if not data:
            logger.warning("category_scorer: could not parse Claude response")
            return None, {}, "parse_error"

        score = int(max(0, min(100, data.get("aggregate_score", 50))))

        # Aggregate category impacts across articles (weighted by relevance)
        cat_totals: dict[str, list[float]] = {}
        for art_data in data.get("articles", []):
            for cat, vals in art_data.get("categories", {}).items():
                rel = vals.get("relevance", 0)
                imp = vals.get("impact", 0.0)
                if rel > 10:
                    cat_totals.setdefault(cat, []).append(imp)

        breakdown = {
            cat: round(sum(vals) / len(vals), 1)
            for cat, vals in cat_totals.items()
            if vals
        }

        # Top 5 categories by absolute impact for explanation
        top = sorted(breakdown.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        explanation = ", ".join(f"{c}: {v:+.1f}" for c, v in top)

        result = (score, breakdown, explanation)
        if len(_SCORE_CACHE) < _MAX_CACHE:
            _SCORE_CACHE[key] = result
        return result

    except Exception as e:
        logger.warning(f"category_scorer Claude call failed: {e}")
        return None, {}, str(e)
