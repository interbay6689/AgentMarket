"""
Agent Tools — all tool functions available to Claude agents.
Each function maps to a Claude API tool_use call.
Also exports TOOL_SCHEMAS for use in messages.create(tools=...).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytz

_MD   = Path(__file__).resolve().parents[1]
_ROOT = _MD.parent
for p in (str(_MD), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from modules.nq_data import (
    get_todays_nq_data, load_nq_daily_cache, load_nq_hourly_cache,
    fetch_nq_15m, current_session_israel,
)
from modules.nq_calculations import (
    calculate_key_levels, cumulative_delta, approximate_delta,
    detect_fvg, detect_order_blocks, detect_stacked_imbalances,
    get_htf_bias, get_support_resistance_zones,
    calculate_atr_range_consumed, calculate_opening_range,
    classify_premium_discount, vwap_position, detect_eqh_eql,
    day_of_week_bias, gap_fill_stats, delta_zscore, volume_anomaly_score,
)
from modules.trade_tracker import (
    load_trade_log, load_trade_settings, factor_win_rates, summary_stats,
)
from agents.agent_safety import (
    apply_change, validate_change, validate_weights,
    dry_run, revert_last_change, get_change_history, PARAMETER_LIMITS,
)

IL_TZ = pytz.timezone("Asia/Jerusalem")
_CONFIG  = _ROOT / "scores_news" / "config"
_ML_DIR  = _ROOT / "scores_news" / "ml_model"
_LOGS    = _MD / "logs"

# ─── READ TOOLS ───────────────────────────────────────────────────────────────

def read_current_price() -> dict:
    """Current NQ price, session, 5m OHLCV last bar."""
    try:
        df = get_todays_nq_data()
        session = current_session_israel()
        if df.empty:
            return {"error": "no_data", "session": session}
        last = df.iloc[-1]
        return {
            "price":        round(float(last["close"]), 2),
            "open":         round(float(last["open"]), 2),
            "high_today":   round(float(df["high"].max()), 2),
            "low_today":    round(float(df["low"].min()), 2),
            "volume_last":  int(last["volume"]),
            "bars_today":   len(df),
            "session":      session,
            "timestamp_il": str(last.get("datetime_il", "")),
        }
    except Exception as e:
        return {"error": str(e)}


def read_technical_state() -> dict:
    """Full technical picture: HTF bias, zones, VWAP, EQH/EQL, delta, ATR."""
    try:
        df_5m     = get_todays_nq_data()
        df_daily  = load_nq_daily_cache()
        df_hourly = load_nq_hourly_cache()
        df_15m    = fetch_nq_15m(days_back=5)
        price     = float(df_5m["close"].iloc[-1]) if not df_5m.empty else None

        htf    = get_htf_bias(df_daily, df_hourly)
        levels = calculate_key_levels(df_daily, df_hourly)
        zones  = get_support_resistance_zones(df_daily, df_hourly, df_15m, price)
        vwap   = vwap_position(df_5m)
        eqhl   = detect_eqh_eql(df_5m) if not df_5m.empty else {}
        atr    = calculate_atr_range_consumed(df_daily, df_5m)
        or_    = calculate_opening_range(df_5m)
        pd_    = classify_premium_discount(
            price,
            float(df_daily.tail(5)["high"].max()) if len(df_daily) >= 5 else price or 0,
            float(df_daily.tail(5)["low"].min())  if len(df_daily) >= 5 else price or 0,
        ) if price else {}

        # Recent delta z-score
        dz = delta_zscore(df_5m).tail(3).tolist() if not df_5m.empty else []
        va = volume_anomaly_score(df_5m)

        # Stacked imbalances
        stacks = detect_stacked_imbalances(df_5m.tail(30).reset_index(drop=True)) \
                 if not df_5m.empty else []

        # Top 4 zones closest to price
        top_zones = [
            {"type": z["zone_type"], "score": z["score"],
             "mid": z["midpoint"], "dist": z["dist_pts"],
             "layers": z["components"][:3]}
            for z in zones[:4]
        ] if zones else []

        return {
            "htf_bias":        htf,
            "key_levels":      {k: round(v, 2) for k, v in levels.items()
                                if isinstance(v, float) and not np.isnan(v)},
            "top_zones":       top_zones,
            "vwap":            vwap,
            "eqh":             eqhl.get("eqh", []),
            "eql":             eqhl.get("eql", []),
            "atr_consumed":    atr,
            "opening_range":   or_,
            "premium_discount": pd_,
            "delta_zscore_last3": [round(x, 2) for x in dz],
            "volume_anomaly":  va,
            "stacked_imbalance": stacks[-1] if stacks else None,
        }
    except Exception as e:
        return {"error": str(e)}


def read_fundamental_state() -> dict:
    """Sentiment scores, ML prediction, category breakdown, score trend."""
    try:
        score_path = _CONFIG / "score_log.csv"
        ml_path    = _ML_DIR / "ml_performance_log.csv"

        result: dict = {}

        if score_path.exists():
            df = pd.read_csv(score_path)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date").tail(5)
            if not df.empty:
                latest = df.iloc[-1].to_dict()
                result["latest_scores"] = {k: round(float(v), 1)
                                            for k, v in latest.items()
                                            if k != "date" and isinstance(v, (int, float))}
                result["score_date"] = str(df.iloc[-1]["date"])
                if len(df) >= 3:
                    result["final_score_trend"] = round(
                        float(df["final_score"].iloc[-1]) - float(df["final_score"].iloc[-3]), 1
                    ) if "final_score" in df.columns else 0.0

        if ml_path.exists():
            ml_df = pd.read_csv(ml_path)
            if not ml_df.empty:
                last = ml_df.iloc[-1]
                result["ml_prediction"] = int(last.get("prediction", 0))
                result["ml_correct"]    = str(last.get("correct", "?"))
                result["ml_accuracy"]   = round(
                    float(ml_df["correct"].map({"✅": 1, "❌": 0}).mean()) * 100, 1
                ) if "correct" in ml_df.columns else None

        weights_path = _CONFIG / "weights.json"
        if weights_path.exists():
            result["current_weights"] = json.loads(weights_path.read_text())

        return result
    except Exception as e:
        return {"error": str(e)}


def read_trade_log_summary() -> dict:
    """Recent trades, win rate, factor performance, equity stats."""
    try:
        df    = load_trade_log()
        stats = summary_stats(df)
        wr    = factor_win_rates(df)
        recent = df.tail(5)[
            ["id", "date", "time_il", "direction", "entry",
             "outcome", "actual_rr", "nearest_zone_score", "htf_bias"]
        ].to_dict(orient="records") if not df.empty else []
        top_factors = (
            wr.head(3)[["Factor", "Value", "Win %", "Trades"]].to_dict(orient="records")
            if not wr.empty else []
        )
        return {
            "summary":          stats,
            "recent_trades":    recent,
            "top_win_factors":  top_factors,
            "open_count":       len(df[df["outcome"] == "OPEN"]) if not df.empty else 0,
        }
    except Exception as e:
        return {"error": str(e)}


def read_statistical_edge() -> dict:
    """Day-of-week bias, gap fill stats, ATR quartile, opening range stats."""
    try:
        df_daily = load_nq_daily_cache()
        dow  = day_of_week_bias(df_daily)
        gaps = gap_fill_stats(df_daily)
        today_dow = datetime.now(IL_TZ).weekday()
        dow_bias  = dow.get(today_dow)

        # ATR quartile (where does today's ATR sit vs last 60 days)
        atr_series: pd.Series | None = None
        if len(df_daily) >= 15:
            d = df_daily.copy().reset_index(drop=True)
            d["prev_close"] = d["close"].shift(1)
            d["tr"] = (
                pd.concat([d["high"], d["prev_close"]], axis=1).max(axis=1)
                - pd.concat([d["low"],  d["prev_close"]], axis=1).min(axis=1)
            )
            atr_series = d["tr"].rolling(14).mean()

        atr_percentile = None
        if atr_series is not None and len(atr_series.dropna()) >= 20:
            current_atr = float(atr_series.iloc[-1])
            hist_atrs   = atr_series.dropna().tail(60).values
            atr_percentile = round(float(np.sum(hist_atrs <= current_atr) / len(hist_atrs) * 100), 1)

        return {
            "today_dow":         today_dow,
            "today_dow_name":    ["Mon","Tue","Wed","Thu","Fri"].get(today_dow, "?")
                                  if isinstance(today_dow, int) else "?",
            "dow_long_winrate":  f"{dow_bias*100:.0f}%" if dow_bias else "?",
            "gap_stats":         gaps,
            "atr_percentile":    atr_percentile,
            "atr_note":          (
                "high volatility day — reduce size or widen stop"
                if atr_percentile and atr_percentile > 75 else
                "low volatility day — may need wider tolerance"
                if atr_percentile and atr_percentile < 25 else
                "normal volatility"
            ) if atr_percentile else "unknown",
        }
    except Exception as e:
        return {"error": str(e)}


def read_all_configs() -> dict:
    """All current configuration values across all config files."""
    result: dict = {}
    try:
        for fname in ("weights.json", "thresholds.json"):
            p = _CONFIG / fname
            if p.exists():
                result[fname.replace(".json", "")] = json.loads(p.read_text())
        ts = _LOGS / "trade_settings.json"
        if ts.exists():
            result["trade_settings"] = json.loads(ts.read_text())
        ac = _LOGS / "agent_config.json"
        if ac.exists():
            result["agent_config"] = json.loads(ac.read_text())
        result["allowed_parameters"] = list(PARAMETER_LIMITS.keys())
    except Exception as e:
        result["error"] = str(e)
    return result


def read_insights_log(n: int = 20) -> list:
    """Recent insights from Analysis Agent."""
    path = _LOGS / "insights_log.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return data[-n:] if isinstance(data, list) else []
    except Exception:
        return []


def read_agent_proposals(n: int = 10) -> list:
    """Recent trade proposals from Signal Agent."""
    path = _LOGS / "agent_proposals.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return data[-n:] if isinstance(data, list) else []
    except Exception:
        return []


def read_economic_calendar() -> dict:
    """Current economic calendar: today's high-impact events, upcoming in 60m, blackout status."""
    try:
        from modules.economic_calendar import calendar_summary
        return calendar_summary()
    except Exception as e:
        return {"error": str(e)}


def read_market_regime() -> dict:
    """Current market regime: trending/ranging/high_vol, ADX, ATR ratio, recommended strategies."""
    try:
        from modules.regime_detector import classify_regime
        from modules.nq_data import load_nq_daily_cache
        df = load_nq_daily_cache()
        return classify_regime(df)
    except Exception as e:
        return {"error": str(e)}


# ─── WRITE TOOLS ──────────────────────────────────────────────────────────────

def update_fundamental_weights(new_weights: dict, reasoning: str,
                                agent_name: str = "config_optimizer") -> dict:
    """Update fundamental scoring weights. Must sum to 1.0."""
    ok, reason = validate_weights(new_weights)
    if not ok:
        return {"success": False, "reason": reason}
    results = []
    # Read current weights first
    cur_path = _CONFIG / "weights.json"
    cur = json.loads(cur_path.read_text()) if cur_path.exists() else {}
    for key, val in new_weights.items():
        success, msg = apply_change(
            f"weights.{key}", val, cur.get(key), reasoning, agent_name
        )
        results.append({"param": f"weights.{key}", "success": success, "msg": msg})
    return {"results": results, "applied": sum(1 for r in results if r["success"])}


def update_entry_filter(param: str, value: float, reasoning: str,
                        agent_name: str = "config_optimizer") -> dict:
    """Update a single entry filter: min_confidence, min_zone_score, min_rr."""
    allowed = {"min_confidence", "min_zone_score", "min_rr", "min_pts", "min_margin"}
    if param not in allowed:
        return {"success": False, "reason": f"param must be one of {allowed}"}
    full_param = f"signal.{param}"
    # Read current value
    cur_path = _CONFIG / "thresholds.json"
    cur_data = json.loads(cur_path.read_text()) if cur_path.exists() else {}
    cur_val  = cur_data.get(param)
    success, msg = apply_change(full_param, value, cur_val, reasoning, agent_name)
    return {"success": success, "message": msg}


def update_agent_config(section: str, key: str, value: float, reasoning: str,
                        agent_name: str = "config_optimizer") -> dict:
    """Update ICT, VWAP, session, or risk config."""
    allowed_sections = {"ict", "vwap", "session", "risk"}
    if section not in allowed_sections:
        return {"success": False, "reason": f"section must be one of {allowed_sections}"}
    full_param = f"{section}.{key}"
    cur_path = _LOGS / "agent_config.json"
    cur_data = json.loads(cur_path.read_text()) if cur_path.exists() else {}
    cur_val  = cur_data.get(section, {}).get(key)
    success, msg = apply_change(full_param, value, cur_val, reasoning, agent_name)
    return {"success": success, "message": msg}


def write_trade_proposal(proposal: dict) -> dict:
    """Write a trade proposal from Signal Agent to agent_proposals.json."""
    path = _LOGS / "agent_proposals.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    proposals = []
    if path.exists():
        try:
            proposals = json.loads(path.read_text())
        except Exception:
            pass
    proposal["logged_at"] = datetime.now(IL_TZ).isoformat()
    proposals.append(proposal)
    # Keep last 100
    path.write_text(json.dumps(proposals[-100:], indent=2))
    return {"success": True, "proposal_id": proposal.get("logged_at", "")}


def write_insight(insight: dict) -> dict:
    """Write a post-trade insight from Analysis Agent to insights_log.json."""
    path = _LOGS / "insights_log.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    insights = []
    if path.exists():
        try:
            insights = json.loads(path.read_text())
        except Exception:
            pass
    insight["logged_at"] = datetime.now(IL_TZ).isoformat()
    insights.append(insight)
    path.write_text(json.dumps(insights[-200:], indent=2))
    return {"success": True}


def run_dry_run(proposed_changes: list) -> dict:
    """Simulate config changes without applying. Returns validation results."""
    return dry_run(proposed_changes)


def revert_change() -> dict:
    """Revert the most recent config change."""
    success, msg = revert_last_change()
    return {"success": success, "message": msg}


def get_config_history(n: int = 20) -> list:
    """Get recent config change history."""
    return get_change_history(n)


def run_backtest_sweep(param: str, test_values: list, lookback_days: int = 180) -> dict:
    """
    Backtest a single parameter across multiple values against historical NQ data.
    Returns ranked results (best profit_factor first) so Config Optimizer can
    compare proposed vs current before applying any change.
    """
    try:
        from modules.nq_data import load_nq_daily_cache
        from modules.backtest_engine import sweep_parameter, DEFAULT_PARAMS

        df = load_nq_daily_cache()
        if df.empty:
            return {"error": "no_daily_data"}

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
        end   = df.index.max()
        start = end - pd.Timedelta(days=lookback_days)
        df    = df[df.index >= start].copy()

        results = sweep_parameter(df, param, [float(v) for v in test_values], DEFAULT_PARAMS)
        return {
            "param":   param,
            "results": results[:8],
            "best":    results[0] if results else {},
        }
    except Exception as e:
        return {"error": str(e)}


# ─── Tool dispatcher ──────────────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, callable] = {
    "read_current_price":          read_current_price,
    "read_technical_state":        read_technical_state,
    "read_fundamental_state":      read_fundamental_state,
    "read_trade_log_summary":      read_trade_log_summary,
    "read_statistical_edge":       read_statistical_edge,
    "read_all_configs":            read_all_configs,
    "read_insights_log":           read_insights_log,
    "read_agent_proposals":        read_agent_proposals,
    "read_economic_calendar":      read_economic_calendar,
    "read_market_regime":          read_market_regime,
    "update_fundamental_weights":  update_fundamental_weights,
    "update_entry_filter":         update_entry_filter,
    "update_agent_config":         update_agent_config,
    "write_trade_proposal":        write_trade_proposal,
    "write_insight":               write_insight,
    "run_dry_run":                 run_dry_run,
    "revert_change":               revert_change,
    "get_config_history":          get_config_history,
    "run_backtest_sweep":          run_backtest_sweep,
}


def execute_tool(tool_name: str, tool_input: dict) -> Any:
    """Dispatch a Claude tool_use call to the correct function."""
    fn = TOOL_REGISTRY.get(tool_name)
    if fn is None:
        return {"error": f"Unknown tool: {tool_name}"}
    try:
        return fn(**tool_input) if tool_input else fn()
    except Exception as e:
        return {"error": str(e), "tool": tool_name}


# ─── Claude API tool schemas ──────────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "read_current_price",
        "description": "Get current NQ futures price, today's high/low, session info, last 5m bar.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "read_technical_state",
        "description": (
            "Full technical picture: HTF bias (Daily+1H), top S/R zones with scores, "
            "VWAP position + bands, EQH/EQL liquidity pools, ATR consumed, "
            "opening range, premium/discount, delta z-score, volume anomaly, stacked imbalances."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "read_fundamental_state",
        "description": (
            "Latest fundamental scores: sentiment, macro, bonds, VIX, sectors, MES, "
            "final_score, ML prediction, score trend vs 3 days ago, current category weights."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "read_trade_log_summary",
        "description": "Recent 5 trades, win/loss stats, top performing factors, open trade count.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "read_statistical_edge",
        "description": (
            "Today's statistical edge: day-of-week win rate, gap fill probability, "
            "ATR percentile vs 60-day history."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "read_all_configs",
        "description": "All current config values: weights, thresholds, trade settings, agent config.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "read_insights_log",
        "description": "Recent post-trade insights from Analysis Agent.",
        "input_schema": {
            "type": "object",
            "properties": {"n": {"type": "integer", "description": "Number of insights to return (default 20)"}},
            "required": [],
        },
    },
    {
        "name": "update_fundamental_weights",
        "description": (
            "Update fundamental category weights. All 6 keys required, must sum to 1.0. "
            "Safety: each weight checked against limits, 15% max change, 24h cooldown."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "new_weights": {
                    "type": "object",
                    "description": "Dict with keys: sentiment, macro, bonds, vix, sectors, mes",
                },
                "reasoning": {"type": "string", "description": "Evidence-based justification"},
            },
            "required": ["new_weights", "reasoning"],
        },
    },
    {
        "name": "update_entry_filter",
        "description": "Update a single entry filter: min_confidence (55-85), min_zone_score (4-9), min_rr (1.2-3.0).",
        "input_schema": {
            "type": "object",
            "properties": {
                "param":     {"type": "string", "enum": ["min_confidence","min_zone_score","min_rr","min_pts","min_margin"]},
                "value":     {"type": "number"},
                "reasoning": {"type": "string"},
            },
            "required": ["param", "value", "reasoning"],
        },
    },
    {
        "name": "update_agent_config",
        "description": "Update ICT/VWAP/session/risk parameters within allowed ranges.",
        "input_schema": {
            "type": "object",
            "properties": {
                "section":   {"type": "string", "enum": ["ict","vwap","session","risk"]},
                "key":       {"type": "string"},
                "value":     {"type": "number"},
                "reasoning": {"type": "string"},
            },
            "required": ["section", "key", "value", "reasoning"],
        },
    },
    {
        "name": "write_trade_proposal",
        "description": "Write a trade proposal (direction, entry, stop, target, reasoning, confidence).",
        "input_schema": {
            "type": "object",
            "properties": {
                "proposal": {
                    "type": "object",
                    "description": "Must include: direction, entry, stop, target, rr, confidence, reasoning, key_factors",
                }
            },
            "required": ["proposal"],
        },
    },
    {
        "name": "write_insight",
        "description": "Save a post-trade analysis insight to insights_log.json.",
        "input_schema": {
            "type": "object",
            "properties": {
                "insight": {
                    "type": "object",
                    "description": "Must include: trade_id, outcome, root_cause, factor_quality, recommendation",
                }
            },
            "required": ["insight"],
        },
    },
    {
        "name": "run_dry_run",
        "description": "Simulate a list of config changes without applying. Returns valid/invalid/warnings.",
        "input_schema": {
            "type": "object",
            "properties": {
                "proposed_changes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "param":         {"type": "string"},
                            "new_value":     {"type": "number"},
                            "current_value": {"type": "number"},
                            "reasoning":     {"type": "string"},
                        },
                    },
                }
            },
            "required": ["proposed_changes"],
        },
    },
    {
        "name": "revert_change",
        "description": "Revert the most recent config change made by any agent.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_config_history",
        "description": "Get recent config change history with timestamps and reasoning.",
        "input_schema": {
            "type": "object",
            "properties": {"n": {"type": "integer", "description": "Number of records (default 20)"}},
            "required": [],
        },
    },
    {
        "name": "read_economic_calendar",
        "description": (
            "Today's USD high-impact economic events, events in the next 60 minutes, "
            "and current blackout status. Always call before generating a trade proposal."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "read_market_regime",
        "description": (
            "Current market regime classification: trending_bull | trending_bear | "
            "ranging | high_vol | transitioning. Includes ADX, ATR ratio, recommended "
            "strategies, confidence threshold adjustment, and position size multiplier."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run_backtest_sweep",
        "description": (
            "Backtest a parameter across multiple values against historical NQ daily OHLCV. "
            "Returns ranked results (best profit_factor first). "
            "Use BEFORE applying any parameter change — confirms the new value outperforms current. "
            "Example: test signal.min_confidence at [55,60,65,70] over last 180 days."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "param": {
                    "type": "string",
                    "enum": [
                        "signal.min_confidence", "signal.min_zone_score",
                        "signal.min_rr", "signal.min_margin", "risk.atr_mult",
                    ],
                    "description": "Parameter to sweep",
                },
                "test_values": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "List of values to test (2-8 values recommended)",
                },
                "lookback_days": {
                    "type": "integer",
                    "description": "Days of history to use (default 180, max 730)",
                },
            },
            "required": ["param", "test_values"],
        },
    },
]
