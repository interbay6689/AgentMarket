"""
Economic Calendar Guard — fetches high-impact USD events and enforces
trading blackout zones around major releases (NFP, CPI, FOMC, etc.).

Data source: ForexFactory public JSON (no API key required).
Cache: logs/economic_calendar_cache.json — refreshed once per day.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytz
import requests

_MD   = Path(__file__).resolve().parents[1]
_ROOT = _MD.parent
for p in (str(_MD), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

IL_TZ = pytz.timezone("Asia/Jerusalem")
ET_TZ = pytz.timezone("America/New_York")

_CACHE_PATH = _MD / "logs" / "economic_calendar_cache.json"
_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# Only these currencies affect NQ materially
_RELEVANT_CURRENCIES = {"USD"}

# Blackout settings (minutes)
BLACKOUT_BEFORE_MIN = 30
BLACKOUT_AFTER_MIN  = 15


# ─── Fetch & Cache ─────────────────────────────────────────────────────────────

def fetch_calendar_week(force: bool = False) -> list[dict]:
    """
    Fetch this week's calendar from ForexFactory.
    Returns list of event dicts, or [] on failure.
    Caches to disk for the day.
    """
    _MD.mkdir(parents=True, exist_ok=True)
    _cache_logs = _MD / "logs"
    _cache_logs.mkdir(parents=True, exist_ok=True)

    # Return cache if fresh (same calendar day, updated within last 6h)
    if not force and _CACHE_PATH.exists():
        try:
            cached = json.loads(_CACHE_PATH.read_text())
            cached_at = datetime.fromisoformat(cached.get("fetched_at", "2000-01-01"))
            if (datetime.now(IL_TZ) - cached_at.replace(tzinfo=IL_TZ)).total_seconds() < 21600:
                return cached.get("events", [])
        except Exception:
            pass

    try:
        resp = requests.get(_CALENDAR_URL, timeout=8)
        resp.raise_for_status()
        raw: list[dict] = resp.json()
    except Exception:
        # Return stale cache if available
        if _CACHE_PATH.exists():
            try:
                return json.loads(_CACHE_PATH.read_text()).get("events", [])
            except Exception:
                pass
        return []

    events = _parse_events(raw)
    _CACHE_PATH.write_text(json.dumps({
        "fetched_at": datetime.now(IL_TZ).isoformat(),
        "events":     events,
    }, indent=2, default=str))
    return events


def _parse_events(raw: list[dict]) -> list[dict]:
    """Parse FF JSON into normalised event dicts with IL + ET times."""
    parsed = []
    for item in raw:
        if item.get("country", "").upper() not in _RELEVANT_CURRENCIES:
            continue
        if item.get("impact", "").lower() not in ("high", "medium"):
            continue

        date_str  = item.get("date", "")
        time_str  = item.get("time", "")
        impact    = item.get("impact", "Medium")
        title     = item.get("title", "Unknown Event")

        et_dt = _parse_ff_datetime(date_str, time_str)
        if et_dt is None:
            continue

        il_dt = et_dt.astimezone(IL_TZ)
        parsed.append({
            "title":    title,
            "impact":   impact,
            "currency": item.get("country", "USD"),
            "forecast": item.get("forecast", ""),
            "previous": item.get("previous", ""),
            "et_time":  et_dt.strftime("%H:%M ET"),
            "il_time":  il_dt.strftime("%H:%M IL"),
            "datetime_et": et_dt.isoformat(),
            "datetime_il": il_dt.isoformat(),
            "date_str":  il_dt.strftime("%Y-%m-%d"),
        })
    return sorted(parsed, key=lambda e: e["datetime_et"])


def _parse_ff_datetime(date_str: str, time_str: str):
    """Parse ForexFactory date/time strings into an ET-aware datetime."""
    try:
        # FF format: "Jun 06 2025" + "8:30am"
        time_str = time_str.strip().lower()
        if not time_str or time_str in ("all day", "tentative", ""):
            # Default to 08:30 ET for all-day events
            time_str = "8:30am"
        dt_str = f"{date_str} {time_str}"
        naive  = datetime.strptime(dt_str, "%b %d %Y %I:%M%p")
        return ET_TZ.localize(naive)
    except Exception:
        return None


# ─── Query functions ───────────────────────────────────────────────────────────

def get_todays_events(impact_filter: str = "High") -> list[dict]:
    """Return today's events matching the given impact level."""
    events = fetch_calendar_week()
    today  = datetime.now(IL_TZ).strftime("%Y-%m-%d")
    return [
        e for e in events
        if e["date_str"] == today and
           (impact_filter.lower() == "all" or e["impact"].lower() == impact_filter.lower())
    ]


def get_upcoming_events(minutes_ahead: int = 60) -> list[dict]:
    """Return high-impact events occurring within the next N minutes."""
    events = fetch_calendar_week()
    now    = datetime.now(IL_TZ)
    cutoff = now + timedelta(minutes=minutes_ahead)
    return [
        e for e in events
        if e["impact"].lower() == "high" and
           now <= datetime.fromisoformat(e["datetime_il"]).replace(tzinfo=IL_TZ) <= cutoff
    ]


def is_blackout_zone(
    before_min: int = BLACKOUT_BEFORE_MIN,
    after_min:  int = BLACKOUT_AFTER_MIN,
) -> dict:
    """
    Check if current time is within a trading blackout window.

    Returns:
        {
          "blackout": bool,
          "reason":   str,
          "event":    dict | None,   — the triggering event
          "minutes_to_next": int | None,
          "next_clear_il":   str | None,
        }
    """
    events = fetch_calendar_week()
    now    = datetime.now(IL_TZ)

    for e in events:
        if e["impact"].lower() != "high":
            continue
        try:
            ev_dt = datetime.fromisoformat(e["datetime_il"]).replace(tzinfo=IL_TZ)
        except Exception:
            continue

        delta_s = (ev_dt - now).total_seconds()
        delta_m = delta_s / 60

        # Before window
        if 0 < delta_m <= before_min:
            clear_dt = ev_dt + timedelta(minutes=after_min)
            return {
                "blackout":         True,
                "reason":           f"{e['title']} in {int(delta_m)}m — blackout until {clear_dt.strftime('%H:%M IL')}",
                "event":            e,
                "minutes_to_event": int(delta_m),
                "next_clear_il":    clear_dt.strftime("%H:%M IL"),
            }

        # After window
        if -after_min <= delta_m <= 0:
            elapsed = abs(delta_m)
            clear_dt = ev_dt + timedelta(minutes=after_min)
            return {
                "blackout":         True,
                "reason":           f"{e['title']} released {int(elapsed)}m ago — blackout until {clear_dt.strftime('%H:%M IL')}",
                "event":            e,
                "minutes_to_event": -int(elapsed),
                "next_clear_il":    clear_dt.strftime("%H:%M IL"),
            }

    # Find next upcoming high-impact event
    future = [
        e for e in events
        if e["impact"].lower() == "high" and
           (datetime.fromisoformat(e["datetime_il"]).replace(tzinfo=IL_TZ) - now).total_seconds() > 0
    ]
    next_event = future[0] if future else None
    next_min   = None
    if next_event:
        ndt = datetime.fromisoformat(next_event["datetime_il"]).replace(tzinfo=IL_TZ)
        next_min = int((ndt - now).total_seconds() / 60)

    return {
        "blackout":         False,
        "reason":           "Clear",
        "event":            None,
        "minutes_to_next":  next_min,
        "next_event":       next_event,
    }


def calendar_summary() -> dict:
    """Compact summary for UI display and agent tools."""
    today_high   = get_todays_events("High")
    today_medium = get_todays_events("Medium")
    upcoming     = get_upcoming_events(60)
    blackout     = is_blackout_zone()

    return {
        "today_high_impact":   len(today_high),
        "today_high_events":   [{"title": e["title"], "time": e["il_time"]} for e in today_high],
        "today_medium_events": [{"title": e["title"], "time": e["il_time"]} for e in today_medium],
        "upcoming_60m":        [{"title": e["title"], "time": e["il_time"]} for e in upcoming],
        "blackout":            blackout["blackout"],
        "blackout_reason":     blackout["reason"],
        "minutes_to_next":     blackout.get("minutes_to_next"),
    }
