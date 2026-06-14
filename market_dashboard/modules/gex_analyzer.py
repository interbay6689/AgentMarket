"""
gex_analyzer.py — Gamma Exposure (GEX) via QQQ options as NQ proxy.

GEX measures net dealer gamma positioning:
  positive regime → dealers are long gamma → they buy dips / sell rips (stabilizing)
  negative regime → dealers are short gamma → they sell dips / buy rips (amplifying)

Formula: GEX_dollar = Gamma × OI × 100 × Spot²
  Calls contribute positive GEX, puts contribute negative GEX.
  Convention: market makers are net short the options market.

QQQ is used as NQ proxy because NQ=F options have thin liquidity on yfinance.
NQ equivalent levels are derived via the live NQ=F / QQQ price ratio.
"""
from __future__ import annotations

import json
import math
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import pandas as pd
import pytz
import yfinance as yf

_MD        = Path(__file__).resolve().parents[1]
CACHE_PATH = _MD / "logs" / "gex_cache.json"

ET_TZ = pytz.timezone("America/New_York")
IL_TZ = pytz.timezone("Asia/Jerusalem")

NQ_RATIO_FALLBACK = 40.0
MAX_DTE           = 30    # options beyond this are irrelevant for near-term GEX
STRIKE_RANGE      = 0.10  # only strikes within ±10% of spot (avoids illiquid wings)
REGIME_THRESHOLD  = 50.0  # $50M net GEX boundary for positive/negative declaration
TTL_OPEN          = 15 * 60   # 15 minutes during NYSE hours
TTL_CLOSED        = 60 * 60   # 60 minutes outside hours

GexResult = dict[str, Any]

_EMPTY: GexResult = {
    "regime":           "neutral",
    "net_gex_value":    0.0,
    "zero_gamma_level": None,
    "call_wall":        None,
    "put_wall":         None,
    "call_wall_gex":    0.0,
    "put_wall_gex":     0.0,
    "dominant_expiry":  None,
    "gex_by_strike":    {},
    "gex_by_expiry":    {},
    "qqq_spot":         None,
    "nq_equivalent":    None,
    "atm_range":        None,
    "timestamp":        None,
    "from_cache":       False,
    "error":            None,
}


# ─── Market Hours ─────────────────────────────────────────────────────────────

def _is_market_open() -> bool:
    now_et = datetime.now(ET_TZ)
    if now_et.weekday() >= 5:
        return False
    return time(9, 30) <= now_et.time() <= time(16, 0)


# ─── Cache ────────────────────────────────────────────────────────────────────

def _load_cache() -> GexResult | None:
    if not CACHE_PATH.exists():
        return None
    try:
        with open(CACHE_PATH, encoding="utf-8") as fh:
            stored = json.load(fh)
        ts = datetime.fromisoformat(stored["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=IL_TZ)
        ttl = TTL_OPEN if _is_market_open() else TTL_CLOSED
        if (datetime.now(IL_TZ) - ts).total_seconds() < ttl:
            data = stored.get("data", {})
            # JSON keys are always strings; restore float keys for gex_by_strike
            if isinstance(data.get("gex_by_strike"), dict):
                data["gex_by_strike"] = {float(k): v for k, v in data["gex_by_strike"].items()}
            data["from_cache"] = True
            return data
    except Exception:
        pass
    return None


def _save_cache(result: GexResult) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump(
                {"timestamp": result["timestamp"], "data": result},
                fh, indent=2, default=str,
            )
    except Exception:
        pass


# ─── Math ─────────────────────────────────────────────────────────────────────

def _bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes gamma (∂²C/∂S²). Returns 0.0 on any math error."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return math.exp(-0.5 * d1 ** 2) / (math.sqrt(2 * math.pi) * S * sigma * math.sqrt(T))
    except (ValueError, ZeroDivisionError):
        return 0.0


# ─── Fetching ─────────────────────────────────────────────────────────────────

def _fetch_spot(qqq: yf.Ticker) -> float:
    """QQQ spot price. Tries fast_info first, then 5-day history close."""
    try:
        price = float(qqq.fast_info["last_price"])
        if price > 0:
            return price
    except Exception:
        pass
    hist = qqq.history(period="5d")
    if not hist.empty:
        return float(hist["Close"].iloc[-1])
    raise ValueError("Cannot determine QQQ spot price")


def _fetch_nq_ratio(qqq_spot: float) -> float:
    """Live NQ=F / QQQ ratio. Falls back to 40.0 if yfinance fails."""
    try:
        nq_price = float(yf.Ticker("NQ=F").fast_info["last_price"])
        if qqq_spot > 0:
            return nq_price / qqq_spot
    except Exception:
        pass
    return NQ_RATIO_FALLBACK


def _process_chain(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    spot: float,
    dte: float,
) -> dict[float, float]:
    """
    Net GEX per strike for a single expiry.
    Returns {strike: net_gex_millions}.

    0DTE uses T=0.5/365 minimum to avoid BS singularity at expiry.
    """
    r   = 0.045
    T   = max(dte / 365.0, 0.5 / 365.0)
    gex: dict[float, float] = {}

    for df, sign in ((calls, 1.0), (puts, -1.0)):
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            try:
                strike = float(row["strike"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (spot * (1 - STRIKE_RANGE) <= strike <= spot * (1 + STRIKE_RANGE)):
                continue

            oi = float(row.get("openInterest") or 0)
            if oi <= 0:
                continue

            # prefer exchange-supplied gamma; fall back to BS when NaN / missing
            raw_gamma = row.get("gamma")
            if raw_gamma is not None and not (isinstance(raw_gamma, float) and math.isnan(raw_gamma)):
                gamma = float(raw_gamma)
            else:
                iv    = float(row.get("impliedVolatility") or 0)
                gamma = _bs_gamma(spot, strike, T, r, iv) if iv > 0.01 else 0.0

            if gamma <= 0:
                continue

            dollar_gex = sign * gamma * oi * 100 * (spot ** 2) / 1_000_000
            gex[strike] = gex.get(strike, 0.0) + dollar_gex

    return gex


# ─── Main ─────────────────────────────────────────────────────────────────────

def get_gex_data() -> GexResult:
    """
    Compute QQQ-based GEX and return a GexResult dict.
    Uses JSON cache with TTL=15 min (open) / 60 min (closed).
    Never raises — all errors are captured in result["error"].
    """
    cached = _load_cache()
    if cached is not None:
        return cached

    result: GexResult = {**_EMPTY, "from_cache": False}
    result["timestamp"] = datetime.now(IL_TZ).isoformat()

    try:
        qqq  = yf.Ticker("QQQ")
        spot = _fetch_spot(qqq)
        result["qqq_spot"]    = round(spot, 2)
        result["nq_equivalent"] = round(spot * _fetch_nq_ratio(spot), 1)

        today       = date.today()
        expirations = qqq.options
        if not expirations:
            raise ValueError("yfinance returned no QQQ expirations")

        gex_by_strike: dict[float, float] = {}
        gex_by_expiry: dict[str, float]   = {}

        for expiry_str in expirations:
            dte = (date.fromisoformat(expiry_str) - today).days
            if dte < 0 or dte > MAX_DTE:
                continue
            try:
                chain      = qqq.option_chain(expiry_str)
                strike_gex = _process_chain(chain.calls, chain.puts, spot, float(dte))
            except Exception:
                continue
            if not strike_gex:
                continue
            gex_by_expiry[expiry_str] = round(sum(strike_gex.values()), 2)
            for k, v in strike_gex.items():
                gex_by_strike[k] = gex_by_strike.get(k, 0.0) + v

        if not gex_by_strike:
            raise ValueError("No usable GEX data after filtering strikes")

        gex_by_strike = {k: round(v, 3) for k, v in sorted(gex_by_strike.items())}
        net_gex       = sum(gex_by_strike.values())

        result["net_gex_value"] = round(net_gex, 2)
        result["gex_by_strike"] = gex_by_strike
        result["gex_by_expiry"] = gex_by_expiry

        if net_gex > REGIME_THRESHOLD:
            result["regime"] = "positive"
        elif net_gex < -REGIME_THRESHOLD:
            result["regime"] = "negative"
        else:
            result["regime"] = "neutral"

        # call wall: highest positive GEX above spot (dealer resistance)
        above = {k: v for k, v in gex_by_strike.items() if k >= spot and v > 0}
        below = {k: v for k, v in gex_by_strike.items() if k <= spot and v < 0}
        if above:
            cw = max(above, key=above.__getitem__)
            result["call_wall"]     = cw
            result["call_wall_gex"] = round(above[cw], 2)
        if below:
            pw = min(below, key=lambda k: below[k])
            result["put_wall"]     = pw
            result["put_wall_gex"] = round(below[pw], 2)

        if result["put_wall"] and result["call_wall"]:
            result["atm_range"] = [result["put_wall"], result["call_wall"]]

        # zero gamma level: strike where cumulative GEX (sorted low→high) crosses zero
        strikes  = sorted(gex_by_strike)
        cum      = 0.0
        prev_cum = 0.0
        for i, k in enumerate(strikes):
            prev_cum = cum
            cum += gex_by_strike[k]
            if i > 0 and prev_cum != 0.0 and (
                (prev_cum < 0 and cum >= 0) or (prev_cum > 0 and cum <= 0)
            ):
                denom = abs(prev_cum) + abs(cum)
                frac  = abs(prev_cum) / denom if denom > 0 else 0.5
                result["zero_gamma_level"] = round(strikes[i - 1] + frac * (k - strikes[i - 1]), 2)
                break

        if gex_by_expiry:
            result["dominant_expiry"] = max(gex_by_expiry, key=lambda k: abs(gex_by_expiry[k]))

    except Exception as exc:
        result["error"] = str(exc)

    _save_cache(result)
    return result


# ─── Streamlit Widget ─────────────────────────────────────────────────────────

def render_gex_widget(st: Any, result: GexResult) -> None:
    """
    Render GEX panel inside an existing Streamlit context.
    Accepts `st` as argument — no top-level streamlit import here,
    so this module is safe to import in CLI / non-Streamlit environments.
    """
    import plotly.graph_objects as go

    error     = result.get("error")
    qqq_spot  = result.get("qqq_spot")

    if error and not qqq_spot:
        st.warning(f"⚠️ GEX unavailable — {error}")
        return

    regime          = result.get("regime", "neutral")
    net_gex         = result.get("net_gex_value", 0.0)
    call_wall       = result.get("call_wall")
    put_wall        = result.get("put_wall")
    zero_gamma      = result.get("zero_gamma_level")
    nq_eq           = result.get("nq_equivalent")
    dom_expiry      = result.get("dominant_expiry", "—")
    from_cache      = result.get("from_cache", False)
    ts_str          = result.get("timestamp", "")
    gex_by_strike   = result.get("gex_by_strike", {})

    regime_icon  = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(regime, "⚪")
    regime_color = {"positive": "normal", "negative": "inverse", "neutral": "off"}.get(regime, "off")

    # ── row 1: summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "GEX Regime",
        f"{regime_icon} {regime.upper()}",
        help="Positive = dealers stabilize price · Negative = dealers amplify moves",
    )
    c2.metric(
        "Net GEX",
        f"${net_gex:+.0f}M",
        delta="stabilizing" if regime == "positive" else ("amplifying" if regime == "negative" else None),
        delta_color=regime_color,
        help="Aggregate dealer gamma in $M. <0 means expect wider intraday ranges.",
    )
    c3.metric(
        "Call Wall (resistance)",
        f"${call_wall:.2f}" if call_wall else "—",
        delta=f"{call_wall - qqq_spot:+.2f} pts" if call_wall and qqq_spot else None,
        delta_color="off",
        help="Strike with highest positive GEX above spot — dealers defend this level",
    )
    c4.metric(
        "Put Wall (support)",
        f"${put_wall:.2f}" if put_wall else "—",
        delta=f"{put_wall - qqq_spot:+.2f} pts" if put_wall and qqq_spot else None,
        delta_color="off",
        help="Strike with most negative GEX below spot — dealers defend this level",
    )

    # ── GEX by strike bar chart
    if gex_by_strike:
        strikes = list(gex_by_strike.keys())
        values  = list(gex_by_strike.values())
        colors  = ["#2ecc71" if v >= 0 else "#e74c3c" for v in values]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=values,
            y=strikes,
            orientation="h",
            marker_color=colors,
            hovertemplate="Strike $%{y:.2f}<br>GEX %{x:+.2f}M<extra></extra>",
        ))

        if qqq_spot:
            fig.add_hline(
                y=qqq_spot, line_dash="solid", line_color="#f39c12", line_width=2,
                annotation_text=f"  Spot ${qqq_spot:.2f}", annotation_position="right",
                annotation_font_color="#f39c12",
            )

        if zero_gamma is not None:
            fig.add_hline(
                y=zero_gamma, line_dash="dot", line_color="#9b59b6", line_width=1.5,
                annotation_text=f"  Zero Γ ${zero_gamma:.2f}", annotation_position="right",
                annotation_font_color="#9b59b6",
            )

        fig.update_layout(
            height=300,
            margin=dict(l=10, r=120, t=10, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(20,20,30,0.6)",
            font_color="#d0d0d0",
            xaxis=dict(title="Net GEX ($M)", gridcolor="#2a2a3a"),
            yaxis=dict(title="QQQ Strike", gridcolor="#2a2a3a"),
            showlegend=False,
            bargap=0.10,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── caption row
    ca1, ca2, ca3, ca4 = st.columns(4)
    ca1.caption(f"NQ≈ {nq_eq:,.0f}" if nq_eq else "NQ: N/A")
    ca2.caption(f"Dominant expiry: {dom_expiry}")
    ca3.caption(f"Zero Γ: ${zero_gamma:.2f}" if zero_gamma is not None else "")
    try:
        ts_dt    = datetime.fromisoformat(ts_str)
        ts_label = f"{'cached' if from_cache else 'live'} · {ts_dt.strftime('%H:%M')}"
    except Exception:
        ts_label = "cached" if from_cache else "live"
    ca4.caption(f"GEX {ts_label}")
    if error:
        st.caption(f"⚠️ partial data — {error}")
