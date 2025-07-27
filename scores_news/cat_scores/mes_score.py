# mes_score.py

import requests
from datetime import datetime
import pandas as pd


def fetch_mes_data(interval="1d", range_days="30d") -> pd.DataFrame:
    """משיכת נתוני MES=F מ־Yahoo Finance"""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/MES=F?interval={interval}&range={range_days}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()

    timestamps = data["chart"]["result"][0]["timestamp"]
    indicators = data["chart"]["result"][0]["indicators"]["quote"][0]
    closes = indicators["close"]
    highs = indicators["high"]
    lows = indicators["low"]

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps, unit="s"),
        "close": closes,
        "high": highs,
        "low": lows,
    }).dropna()

    return df

def calculate_mes_score(df: pd.DataFrame) -> tuple[int, str]:
    """
    Calculate a MES score based on daily percentage change and price range.

    The function expects a DataFrame with at least two rows representing
    consecutive days. It gracefully handles different column sets:

    * If the DataFrame has ``high`` and ``low`` columns, the price range is
      computed from those.
    * If ``high``/``low`` are missing but ``open`` and ``close`` are present,
      the range is estimated as the absolute difference between ``close`` and
      ``open``.

    If there are fewer than two rows or the necessary columns are missing,
    a neutral score of 50 is returned with an explanation.
    """
    # Need at least two data points to compute a change
    if len(df) < 2:
        return 50, "🔒 אין מספיק נתונים להשוואה יומית"

    # Work on lowercase column names for flexibility
    df_cols = {c.lower(): c for c in df.columns}
    today = df.iloc[-1]
    yesterday = df.iloc[-2]

    # Retrieve close values (fallback to uppercase if needed)
    close_col = df_cols.get("close", df_cols.get("close"))
    open_col = df_cols.get("open")

    if close_col is None:
        # Can't compute delta without close prices
        return 50, "🔒 אין נתוני close לחישוב MES"

    today_close = today[close_col]
    yesterday_close = yesterday[close_col]
    delta = ((today_close - yesterday_close) / yesterday_close) * 100

    # Determine price range
    high_col = df_cols.get("high")
    low_col = df_cols.get("low")
    if high_col and low_col:
        # Use high/low columns if available
        price_range = today[high_col] - today[low_col]
    elif open_col:
        # Fallback: use absolute difference between open and close as range
        price_range = abs(today_close - today[open_col])
    else:
        # No suitable columns, return neutral
        return 50, "🔒 אין נתוני high/low או open לחישוב טווח MES"

    # Score based on daily change
    if delta > 1:
        trend_score = 90
    elif delta > 0.3:
        trend_score = 70
    elif delta > -0.3:
        trend_score = 50
    elif delta > -1:
        trend_score = 35
    else:
        trend_score = 15

    # Score based on price range
    if price_range > 20:
        range_score = 90
    elif price_range > 10:
        range_score = 70
    elif price_range > 5:
        range_score = 50
    else:
        range_score = 30

    # Weighted average: trend contributes 70%, range contributes 30%
    final_score = round(trend_score * 0.7 + range_score * 0.3)

    explanation = (
        f"📈 שינוי: {delta:.2f}% | טווח: {price_range:.2f} נק׳ "
        f"(מגמה: {trend_score}, תנודתיות: {range_score})"
    )

    return final_score, explanation

if __name__ == "__main__":
    df = fetch_mes_data()
    score, explanation = calculate_mes_score(df)
    print(f"📉 MES Score: {score} | {explanation}")
