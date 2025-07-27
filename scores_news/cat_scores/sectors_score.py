# sectors_score.py


import requests
from datetime import datetime
import pandas as pd

SECTOR_SYMBOLS = {
    "tech": "XLK",
    "finance": "XLF",
    "health": "XLV",
    "energy": "XLE",
    "consumer": "XLP",
    "industrial": "XLI"
}


def fetch_sector_change(symbol: str) -> float:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=2d"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    closes = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
    if len(closes) < 2 or closes[-1] is None or closes[-2] is None:
        return 0.0
    change = ((closes[-1] - closes[-2]) / closes[-2]) * 100
    return round(change, 2)

def calculate_sectors_score() -> tuple[int, str]:
    results = {}
    for name, symbol in SECTOR_SYMBOLS.items():
        try:
            change = fetch_sector_change(symbol)
            results[name] = change
        except:
            results[name] = 0.0

    avg_change = sum(results.values()) / len(results)

    green = sum(1 for v in results.values() if v > 0.3)
    red = sum(1 for v in results.values() if v < -0.3)

    if avg_change > 1:
        score = 80
    elif avg_change > 0.3:
        score = 65
    elif avg_change > -0.3:
        score = 50
    elif avg_change > -1:
        score = 35
    else:
        score = 20

    explanation = f"שינוי ממוצע: {avg_change:.2f}% | 🟢 {green} סקטורים חיוביים, 🔴 {red} שליליים"
    return score, explanation


if __name__ == "__main__":
    score, explanation = calculate_sectors_score()
    print(f"📊 Sector Score: {score} | {explanation}")
