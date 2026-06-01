"""
Backtest Engine — replays historical NQ signals against OHLCV data.

Design principles:
  - NO lookahead: each bar sees only data available at market close of previous day
  - Uses daily + hourly OHLCV (yfinance, no tick data)
  - Scoring mirrors signal_engine.py logic but adapted for historical replay
  - Trade outcome simulation: walk forward bar-by-bar until target/stop hit

Limitations (documented honestly):
  - HTF bias simplified (SMA-based) vs real-time multi-factor version
  - Entry assumed at next-day open (no slippage)
  - Stop/target resolution at daily or hourly granularity (not 5m)
  - Fundamental scores used if score_log.csv has that date; else default 50
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field, asdict
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

IL_TZ   = pytz.timezone("Asia/Jerusalem")
_CONFIG = _ROOT / "scores_news" / "config"

DEFAULT_PARAMS: dict[str, float] = {
    "signal.min_confidence":  60,
    "signal.min_zone_score":  5,
    "signal.min_rr":          1.5,
    "signal.min_margin":      15,
    "risk.atr_mult":          1.5,
}


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class BacktestSignal:
    date:        str
    direction:   str          # LONG | SHORT
    bias:        str
    score:       int
    zone_score:  int
    entry:       float
    stop:        float
    target:      float
    rr:          float
    atr:         float
    components:  dict = field(default_factory=dict)


@dataclass
class BacktestTrade:
    signal:      BacktestSignal
    outcome:     str           # WIN | LOSS | PARTIAL
    exit_price:  float
    actual_rr:   float
    bars_held:   int
    exit_date:   str


@dataclass
class BacktestResult:
    params:       dict
    start_date:   str
    end_date:     str
    signals_fired: int
    trades:       list[BacktestTrade] = field(default_factory=list)
    metrics:      dict = field(default_factory=dict)


# ─── Score log loader ──────────────────────────────────────────────────────────

def _load_score_log() -> pd.DataFrame:
    path = _CONFIG / "score_log.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def _get_fundamental_score(score_log: pd.DataFrame, date: pd.Timestamp) -> float:
    """Return final_score for a given date from score_log, or 50 (neutral) if missing."""
    if score_log.empty or "final_score" not in score_log.columns:
        return 50.0
    window = score_log[
        (score_log["date"] >= date - pd.Timedelta(days=3)) &
        (score_log["date"] <= date)
    ]
    if window.empty:
        return 50.0
    return float(window.iloc[-1]["final_score"])


# ─── ATR calculation ──────────────────────────────────────────────────────────

def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """True Range ATR series, aligned with df index."""
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([
        h - l,
        (h - prev_c).abs(),
        (l - prev_c).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period // 2).mean()


# ─── Historical signal scoring ────────────────────────────────────────────────

def _score_bar(
    bar_idx: int,
    df_daily: pd.DataFrame,
    score_log: pd.DataFrame,
    params: dict,
) -> BacktestSignal | None:
    """
    Score a single trading day using only data up to bar_idx - 1 (no lookahead).
    Returns BacktestSignal or None if no signal fires.
    """
    if bar_idx < 25 or bar_idx >= len(df_daily):
        return None

    today = df_daily.iloc[bar_idx]
    hist  = df_daily.iloc[:bar_idx]   # strictly past data

    # ── HTF Bias via SMA (mirrors get_htf_bias logic) ───────────────────────
    sma5  = float(hist["close"].tail(5).mean())
    sma20 = float(hist["close"].tail(20).mean())
    sma50 = float(hist["close"].tail(50).mean()) if len(hist) >= 50 else sma20

    last_close = float(hist["close"].iloc[-1])

    if sma5 > sma20 * 1.002 and last_close > sma50:
        bias = "bullish"
        htf_pts_long  = 20
        htf_pts_short = 0
    elif sma5 < sma20 * 0.998 and last_close < sma50:
        bias = "bearish"
        htf_pts_long  = 0
        htf_pts_short = 20
    else:
        bias = "neutral"
        htf_pts_long  = 8
        htf_pts_short = 8

    # ── Key levels for zone scoring ──────────────────────────────────────────
    yesterday = hist.iloc[-1]
    pdh = float(yesterday["high"])
    pdl = float(yesterday["low"])
    pdc = float(yesterday["close"])
    weekly_high = float(hist.tail(5)["high"].max())
    weekly_low  = float(hist.tail(5)["low"].min())

    # Entry: today's open (next bar open, no lookahead since we're at bar_idx - 1 EOD)
    entry = float(today.get("open", pdc)) if not pd.isna(today.get("open", np.nan)) else pdc

    # Zone score: proximity to key levels (0-10)
    key_levels = [pdh, pdl, weekly_high, weekly_low]
    dists = [abs(entry - lvl) / max(entry, 1) * 100 for lvl in key_levels]
    min_dist_pct = min(dists)
    zone_score = max(0, min(10, int(10 - min_dist_pct * 15)))

    zone_pts_long  = int(zone_score / 10 * 20) if entry <= pdh else 0
    zone_pts_short = int(zone_score / 10 * 20) if entry >= pdl else 0

    # ── Delta proxy (volume-weighted direction) ──────────────────────────────
    last5 = hist.tail(5).copy()
    if "volume" in last5.columns:
        last5["vol_dir"] = np.where(last5["close"] >= last5["open"], last5["volume"], -last5["volume"])
        net_vol = float(last5["vol_dir"].sum())
    else:
        net_vol = float((last5["close"] - last5["open"]).sum())

    if net_vol > 0:
        delta_pts_long  = 15
        delta_pts_short = 0
    elif net_vol < 0:
        delta_pts_long  = 0
        delta_pts_short = 15
    else:
        delta_pts_long  = 7
        delta_pts_short = 7

    # ── Session (assume NY Morning — most setups) ────────────────────────────
    session_pts = 10

    # ── Stacked imbalances proxy: consecutive directional days ───────────────
    consecutive_bull = 0
    consecutive_bear = 0
    for i in range(len(last5) - 1, -1, -1):
        if last5["close"].iloc[i] > last5["open"].iloc[i]:
            if consecutive_bear == 0:
                consecutive_bull += 1
        elif last5["close"].iloc[i] < last5["open"].iloc[i]:
            if consecutive_bull == 0:
                consecutive_bear += 1
        else:
            break
    stacked_pts_long  = min(10, consecutive_bull * 3)
    stacked_pts_short = min(10, consecutive_bear * 3)

    # ── Fundamental score ────────────────────────────────────────────────────
    today_date = pd.Timestamp(today.name) if hasattr(today.name, "year") else pd.Timestamp(str(today.name)[:10])
    fs = _get_fundamental_score(score_log, today_date)
    fund_pts_long  = int(min(15, fs / 100 * 15))       # bullish fundamental → longs
    fund_pts_short = int(min(15, (100 - fs) / 100 * 15))  # bearish → shorts

    # ── ML proxy: day-of-week edge (replaces actual ML) ──────────────────────
    dow = today_date.weekday()
    # Historical NQ tendencies: Mon/Wed/Thu good for long, Fri mixed
    dow_long  = {0: 8, 1: 7, 2: 9, 3: 9, 4: 5}.get(dow, 7)
    dow_short = {0: 5, 1: 6, 2: 5, 3: 5, 4: 9}.get(dow, 7)

    # ── Total scores ─────────────────────────────────────────────────────────
    long_pts = (htf_pts_long + zone_pts_long + delta_pts_long + session_pts
                + stacked_pts_long + fund_pts_long + dow_long)
    short_pts = (htf_pts_short + zone_pts_short + delta_pts_short + session_pts
                 + stacked_pts_short + fund_pts_short + dow_short)

    min_confidence = int(params.get("signal.min_confidence", 60))
    min_zone_score = int(params.get("signal.min_zone_score", 5))
    min_margin     = int(params.get("signal.min_margin", 15))
    min_rr         = float(params.get("signal.min_rr", 1.5))
    atr_mult       = float(params.get("risk.atr_mult", 1.5))

    # Determine firing direction
    if long_pts >= min_confidence and (long_pts - short_pts) >= min_margin and zone_score >= min_zone_score:
        direction  = "LONG"
        score      = long_pts
    elif short_pts >= min_confidence and (short_pts - long_pts) >= min_margin and zone_score >= min_zone_score:
        direction  = "SHORT"
        score      = short_pts
    else:
        return None   # no signal

    # ── ATR-based stop ────────────────────────────────────────────────────────
    atr_series = _compute_atr(hist)
    atr = float(atr_series.iloc[-1]) if not atr_series.empty and not pd.isna(atr_series.iloc[-1]) else (pdh - pdl) * 0.5

    if direction == "LONG":
        stop   = entry - atr * atr_mult
        target = entry + (entry - stop) * min_rr
        cmp    = {"htf": htf_pts_long, "zone": zone_pts_long, "delta": delta_pts_long,
                  "session": session_pts, "stacked": stacked_pts_long,
                  "fundamental": fund_pts_long, "ml_dow": dow_long}
    else:
        stop   = entry + atr * atr_mult
        target = entry - (stop - entry) * min_rr
        cmp    = {"htf": htf_pts_short, "zone": zone_pts_short, "delta": delta_pts_short,
                  "session": session_pts, "stacked": stacked_pts_short,
                  "fundamental": fund_pts_short, "ml_dow": dow_short}

    return BacktestSignal(
        date=str(today_date.date()),
        direction=direction,
        bias=bias,
        score=score,
        zone_score=zone_score,
        entry=round(entry, 2),
        stop=round(stop, 2),
        target=round(target, 2),
        rr=min_rr,
        atr=round(atr, 2),
        components=cmp,
    )


# ─── Trade outcome simulation ─────────────────────────────────────────────────

def _simulate_outcome(
    signal: BacktestSignal,
    df_daily: pd.DataFrame,
    bar_idx: int,
    max_bars: int = 3,
) -> BacktestTrade:
    """
    Walk forward up to max_bars daily bars to determine outcome.
    Uses each bar's High/Low — we don't know intraday order, so we apply
    a conservative rule: stop takes priority if both are hit in the same bar.
    """
    entry  = signal.entry
    stop   = signal.stop
    target = signal.target
    direction = signal.direction

    outcome    = "PARTIAL"
    exit_price = entry
    bars_held  = 0
    exit_date  = signal.date

    for i in range(1, min(max_bars + 1, len(df_daily) - bar_idx)):
        bar = df_daily.iloc[bar_idx + i]
        bar_high = float(bar["high"])
        bar_low  = float(bar["low"])
        bar_close = float(bar["close"])
        bars_held = i
        exit_date = str(pd.Timestamp(bar.name).date()) if hasattr(bar.name, "year") else str(bar.name)[:10]

        if direction == "LONG":
            stop_hit   = bar_low  <= stop
            target_hit = bar_high >= target
            if stop_hit and target_hit:
                # Conservative: stop wins (gap risk)
                outcome    = "LOSS"
                exit_price = stop
                break
            elif stop_hit:
                outcome    = "LOSS"
                exit_price = stop
                break
            elif target_hit:
                outcome    = "WIN"
                exit_price = target
                break
            # Neither hit yet — continue holding
            exit_price = bar_close  # mark at close for time-exit
        else:  # SHORT
            stop_hit   = bar_high >= stop
            target_hit = bar_low  <= target
            if stop_hit and target_hit:
                outcome    = "LOSS"
                exit_price = stop
                break
            elif stop_hit:
                outcome    = "LOSS"
                exit_price = stop
                break
            elif target_hit:
                outcome    = "WIN"
                exit_price = target
                break
            exit_price = bar_close
    else:
        # Time exit
        outcome = "PARTIAL"

    # Calculate actual R:R
    risk = abs(entry - stop)
    if risk > 0:
        if direction == "LONG":
            actual_rr = round((exit_price - entry) / risk, 2)
        else:
            actual_rr = round((entry - exit_price) / risk, 2)
    else:
        actual_rr = 0.0

    # Reclassify PARTIAL by actual_rr
    if outcome == "PARTIAL":
        if actual_rr >= signal.rr * 0.5:
            outcome = "PARTIAL"   # keep PARTIAL
        elif actual_rr < 0:
            outcome = "LOSS"

    return BacktestTrade(
        signal=signal,
        outcome=outcome,
        exit_price=round(exit_price, 2),
        actual_rr=actual_rr,
        bars_held=bars_held,
        exit_date=exit_date,
    )


# ─── Main backtest runner ─────────────────────────────────────────────────────

def run_backtest(
    df_daily: pd.DataFrame,
    params: dict | None = None,
    start_date: str | None = None,
    end_date:   str | None = None,
    max_bars_held: int = 3,
) -> BacktestResult:
    """
    Run a full backtest over the given daily OHLCV dataframe.

    Args:
        df_daily:     Daily OHLCV (close, open, high, low, volume). Index = DatetimeIndex.
        params:       Signal parameters dict. Defaults to DEFAULT_PARAMS.
        start_date:   "YYYY-MM-DD" — start of backtest window (default: 90 days ago).
        end_date:     "YYYY-MM-DD" — end (default: most recent bar).
        max_bars_held: Max trading days to hold before time exit.

    Returns:
        BacktestResult with trades and metrics.
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    if df_daily.empty or len(df_daily) < 30:
        return BacktestResult(params=params, start_date="", end_date="",
                              signals_fired=0, trades=[], metrics={"error": "insufficient_data"})

    # Normalize index to DatetimeIndex — keep the FULL df for lookback
    if not isinstance(df_daily.index, pd.DatetimeIndex):
        df_daily = df_daily.copy()
        df_daily.index = pd.to_datetime(df_daily.index, errors="coerce")
        df_daily = df_daily.dropna(how="all")

    full_df = df_daily.sort_index()

    # Determine which bar indices are in the signal window
    # We iterate only those bars, but use full_df for lookback — no filtering!
    all_indices = full_df.index
    first_signal_bar = 25   # need at least 25 bars of history

    # Signal window: find integer positions inside [start_date, end_date]
    if start_date:
        start_ts = pd.Timestamp(start_date)
        signal_start_i = max(first_signal_bar,
                             all_indices.searchsorted(start_ts, side="left"))
    else:
        signal_start_i = first_signal_bar

    if end_date:
        end_ts = pd.Timestamp(end_date)
        # last bar we generate signals for (leave 1+ bars for simulation)
        signal_end_i = min(len(full_df) - 2,
                           all_indices.searchsorted(end_ts, side="right") - 1)
    else:
        signal_end_i = len(full_df) - 2

    if signal_start_i > signal_end_i:
        n_bars_in_range = signal_end_i - signal_start_i + 1
        return BacktestResult(
            params=params,
            start_date=start_date or "",
            end_date=end_date or "",
            signals_fired=0,
            trades=[],
            metrics={
                "error": "insufficient_data_after_filter",
                "detail": (
                    f"Only {len(full_df)} total bars in cache. "
                    f"Selected window [{start_date} → {end_date}] maps to "
                    f"{max(0, n_bars_in_range)} bars — need ≥ 25 lookback + range bars. "
                    "Try expanding the date range or visit NQ Analysis to refresh the data cache."
                ),
            },
        )

    result_start = str(full_df.index[signal_start_i].date())
    result_end   = str(full_df.index[signal_end_i].date())

    score_log = _load_score_log()

    trades: list[BacktestTrade] = []
    signals_fired = 0

    # Iterate only the signal window, but pass full_df so _score_bar has full lookback
    for i in range(signal_start_i, signal_end_i + 1):
        sig = _score_bar(i, full_df, score_log, params)
        if sig is None:
            continue
        signals_fired += 1
        trade = _simulate_outcome(sig, full_df, i, max_bars=max_bars_held)
        trades.append(trade)

    from modules.backtest_metrics import compute_metrics
    metrics = compute_metrics(trades)

    return BacktestResult(
        params=params,
        start_date=result_start,
        end_date=result_end,
        signals_fired=signals_fired,
        trades=trades,
        metrics=metrics,
    )


# ─── Parameter sweep (for Config Optimizer) ──────────────────────────────────

def sweep_parameter(
    df_daily: pd.DataFrame,
    param_name: str,
    test_values: list[float],
    base_params: dict | None = None,
    start_date: str | None = None,
    end_date:   str | None = None,
) -> list[dict]:
    """
    Test a single parameter across multiple values.
    Returns a list of result dicts sorted by profit_factor descending.

    Example:
        sweep_parameter(df, "signal.min_confidence", [55, 60, 65, 70])
    """
    if base_params is None:
        base_params = DEFAULT_PARAMS.copy()

    results = []
    for val in test_values:
        p = {**base_params, param_name: val}
        bt = run_backtest(df_daily, params=p, start_date=start_date, end_date=end_date)
        results.append({
            "param":         param_name,
            "value":         val,
            "signals_fired": bt.signals_fired,
            **bt.metrics,
        })

    return sorted(results, key=lambda r: r.get("profit_factor", 0), reverse=True)


def results_to_records(result: BacktestResult) -> list[dict]:
    """Convert BacktestResult.trades to a list of plain dicts for DataFrame."""
    rows = []
    for t in result.trades:
        row = {
            "date":       t.signal.date,
            "direction":  t.signal.direction,
            "bias":       t.signal.bias,
            "score":      t.signal.score,
            "zone_score": t.signal.zone_score,
            "entry":      t.signal.entry,
            "stop":       t.signal.stop,
            "target":     t.signal.target,
            "rr_target":  t.signal.rr,
            "atr":        t.signal.atr,
            "outcome":    t.outcome,
            "exit_price": t.exit_price,
            "actual_rr":  t.actual_rr,
            "bars_held":  t.bars_held,
            "exit_date":  t.exit_date,
        }
        row.update({f"c_{k}": v for k, v in t.signal.components.items()})
        rows.append(row)
    return rows
