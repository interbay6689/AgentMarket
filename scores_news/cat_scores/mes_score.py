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
    """חישוב ציון MES על פי שינוי יומי + טווח תנועה"""
    if len(df) < 2:
        return 50, "🔒 אין מספיק נתונים להשוואה יומית"

    today = df.iloc[-1]
    yesterday = df.iloc[-2]

    delta = ((today["close"] - yesterday["close"]) / yesterday["close"]) * 100
    price_range = today["high"] - today["low"]

    # ניקוד לפי שינוי יומי
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

    # ניקוד לפי טווח
    if price_range > 20:
        range_score = 90
    elif price_range > 10:
        range_score = 70
    elif price_range > 5:
        range_score = 50
    else:
        range_score = 30

    # שקלול
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
