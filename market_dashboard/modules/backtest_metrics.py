"""
Backtest Metrics — computes all performance statistics from a list of BacktestTrade.
Used by backtest_engine and displayed in the Backtesting UI.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from modules.backtest_engine import BacktestTrade


def compute_metrics(trades: list) -> dict:
    """
    Compute full performance metrics from a list of BacktestTrade.
    Returns a dict compatible with both UI display and Config Optimizer comparison.
    """
    if not trades:
        return _empty_metrics()

    wins     = [t for t in trades if t.outcome == "WIN"]
    losses   = [t for t in trades if t.outcome == "LOSS"]
    partials = [t for t in trades if t.outcome == "PARTIAL"]

    n       = len(trades)
    n_win   = len(wins)
    n_loss  = len(losses)
    n_part  = len(partials)
    win_rate = round(n_win / n * 100, 1) if n > 0 else 0.0

    # R-multiples (actual_rr per trade)
    all_r   = [t.actual_rr for t in trades]
    win_r   = [t.actual_rr for t in wins]
    loss_r  = [t.actual_rr for t in losses]

    avg_win_r  = round(float(np.mean(win_r)),  2) if win_r  else 0.0
    avg_loss_r = round(float(np.mean(loss_r)), 2) if loss_r else 0.0
    avg_r      = round(float(np.mean(all_r)),  2) if all_r  else 0.0
    total_r    = round(float(np.sum(all_r)),   2)

    # Profit Factor = gross_profit / |gross_loss|
    gross_profit = sum(r for r in all_r if r > 0)
    gross_loss   = abs(sum(r for r in all_r if r < 0))
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0.0
    )

    # Expectancy = avg_r per trade
    expectancy = avg_r

    # Max Drawdown (in R)
    equity_curve = np.cumsum(all_r)
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = equity_curve - peak
    max_drawdown_r = round(float(drawdowns.min()), 2) if len(drawdowns) > 0 else 0.0

    # Sharpe (annualised, assuming ~252 trading days, using R as "return")
    sharpe = 0.0
    if len(all_r) >= 5:
        mean_r = float(np.mean(all_r))
        std_r  = float(np.std(all_r, ddof=1))
        if std_r > 0:
            # Scale to annual: sqrt(252) assumes one signal per day on avg
            sharpe = round(mean_r / std_r * math.sqrt(252), 2)

    # Average holding period
    avg_bars = round(float(np.mean([t.bars_held for t in trades])), 1) if trades else 0.0

    # Win rate by direction
    longs  = [t for t in trades if t.signal.direction == "LONG"]
    shorts = [t for t in trades if t.signal.direction == "SHORT"]
    long_wr  = round(sum(1 for t in longs  if t.outcome == "WIN") / len(longs)  * 100, 1) if longs  else 0.0
    short_wr = round(sum(1 for t in shorts if t.outcome == "WIN") / len(shorts) * 100, 1) if shorts else 0.0

    # Win rate by day of week
    dow_map: dict[int, list] = {i: [] for i in range(5)}
    for t in trades:
        try:
            dow = pd.Timestamp(t.signal.date).weekday()
            if 0 <= dow <= 4:
                dow_map[dow].append(1 if t.outcome == "WIN" else 0)
        except Exception:
            pass
    dow_winrates = {
        ["Mon","Tue","Wed","Thu","Fri"][d]: round(np.mean(v) * 100, 1) if v else None
        for d, v in dow_map.items()
    }

    # Win rate by score bucket
    low_score  = [t for t in trades if t.signal.score < 70]
    high_score = [t for t in trades if t.signal.score >= 70]
    score_wr = {
        "score<70":  round(sum(1 for t in low_score  if t.outcome == "WIN") / len(low_score)  * 100, 1) if low_score  else None,
        "score>=70": round(sum(1 for t in high_score if t.outcome == "WIN") / len(high_score) * 100, 1) if high_score else None,
    }

    # Consecutive wins / losses
    max_consec_wins  = _max_consecutive(trades, "WIN")
    max_consec_loss  = _max_consecutive(trades, "LOSS")

    return {
        "total_trades":    n,
        "wins":            n_win,
        "losses":          n_loss,
        "partials":        n_part,
        "win_rate":        win_rate,
        "avg_win_r":       avg_win_r,
        "avg_loss_r":      avg_loss_r,
        "avg_r":           avg_r,
        "total_r":         total_r,
        "profit_factor":   profit_factor,
        "expectancy":      expectancy,
        "max_drawdown_r":  max_drawdown_r,
        "sharpe":          sharpe,
        "avg_bars_held":   avg_bars,
        "long_win_rate":   long_wr,
        "short_win_rate":  short_wr,
        "dow_win_rates":   dow_winrates,
        "score_win_rates": score_wr,
        "max_consec_wins": max_consec_wins,
        "max_consec_loss": max_consec_loss,
        "gross_profit_r":  round(gross_profit, 2),
        "gross_loss_r":    round(gross_loss, 2),
    }


def _max_consecutive(trades: list, outcome: str) -> int:
    best = current = 0
    for t in trades:
        if t.outcome == outcome:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _empty_metrics() -> dict:
    return {
        "total_trades": 0, "wins": 0, "losses": 0, "partials": 0,
        "win_rate": 0.0, "avg_win_r": 0.0, "avg_loss_r": 0.0,
        "avg_r": 0.0, "total_r": 0.0, "profit_factor": 0.0,
        "expectancy": 0.0, "max_drawdown_r": 0.0, "sharpe": 0.0,
        "avg_bars_held": 0.0, "long_win_rate": 0.0, "short_win_rate": 0.0,
        "dow_win_rates": {}, "score_win_rates": {},
        "max_consec_wins": 0, "max_consec_loss": 0,
        "gross_profit_r": 0.0, "gross_loss_r": 0.0,
    }


def compare_params(result_a: dict, result_b: dict, label_a: str = "Current", label_b: str = "Proposed") -> pd.DataFrame:
    """
    Side-by-side comparison of two backtest metric dicts.
    Used by Config Optimizer to evaluate a proposed change.
    """
    keys = [
        "total_trades", "win_rate", "profit_factor", "expectancy",
        "total_r", "max_drawdown_r", "sharpe", "avg_r",
    ]
    rows = []
    for k in keys:
        va = result_a.get(k, "—")
        vb = result_b.get(k, "—")
        if isinstance(va, float) and isinstance(vb, float):
            delta = round(vb - va, 3)
            better = "✅" if (
                (k in ("win_rate","profit_factor","expectancy","total_r","sharpe","avg_r") and delta > 0) or
                (k in ("max_drawdown_r",) and delta > 0)   # drawdown less negative = better
            ) else ("❌" if delta != 0 else "—")
        else:
            delta = "—"
            better = "—"
        rows.append({
            "Metric": k.replace("_", " ").title(),
            label_a: va,
            label_b: vb,
            "Delta": delta,
            "Better?": better,
        })
    return pd.DataFrame(rows)


def equity_curve_df(trades: list) -> pd.DataFrame:
    """Build a DataFrame with cumulative R over time, suitable for Plotly chart."""
    if not trades:
        return pd.DataFrame(columns=["date", "r", "cumulative_r"])
    rows = [{"date": t.signal.date, "r": t.actual_rr} for t in trades]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna().sort_values("date").reset_index(drop=True)
    df["cumulative_r"] = df["r"].cumsum().round(2)
    return df
