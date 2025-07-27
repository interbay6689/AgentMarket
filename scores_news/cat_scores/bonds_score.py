# bonds_score.py

import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta


def fetch_treasury_yield_xml(year_month: str = None) -> str:
    """מוריד את קובץ ה־XML של משרד האוצר לפי חודש"""
    if not year_month:
        year_month = datetime.now().strftime("%Y%m")
    url = (
        "https://home.treasury.gov/resource-center/data-chart-center/"
        f"interest-rates/pages/xmlview?data=daily_treasury_yield_curve&field_tdr_date_value_month={year_month}"
    )
    print(f"📡 מוריד XML מ: {url}")
    response = requests.get(url)
    response.raise_for_status()
    return response.content


def extract_yields_from_treasury_xml(xml_data: str) -> dict:
    root = ET.fromstring(xml_data)

    # השורות נמצאות תחת namespace ולכן נשתמש ב־{*}
    entries = root.findall(".//{*}entry")

    candidates = []

    for entry in entries:
        props = entry.find(".//{*}properties")
        if props is None:
            continue

        try:
            date_text = props.findtext(".//{*}NEW_DATE")
            y2_text = props.findtext(".//{*}BC_2YEAR")
            y10_text = props.findtext(".//{*}BC_10YEAR")

            print(f"🔍 {date_text} | 2Y: {y2_text} | 10Y: {y10_text}")

            if date_text and y2_text and y10_text:
                date_obj = datetime.strptime(date_text, "%Y-%m-%dT%H:%M:%S")
                y2 = float(y2_text)
                y10 = float(y10_text)
                candidates.append((date_obj, y2, y10))
        except Exception as e:
            continue

    if not candidates:
        return {}

    latest = max(candidates, key=lambda x: x[0])
    return {
        "2Y": latest[1],
        "10Y": latest[2]
    }


def calculate_bond_score(yields: dict) -> tuple[int, str]:
    if "2Y" not in yields or "10Y" not in yields:
        return 50, "🔒 אין נתונים מספיקים על תשואות 2Y ו־10Y"

    spread = yields["10Y"] - yields["2Y"]

    if spread > 1:
        score = 80
        explanation = f"🟢 שיפוע חיובי חזק: {spread:.2f}% → תנאים תומכים לונג"
    elif 0 < spread <= 1:
        score = 65
        explanation = f"🟡 שיפוע חיובי חלש: {spread:.2f}% → שוק חיובי זהיר"
    elif -1 < spread <= 0:
        score = 40
        explanation = f"🟠 אינברסיה קלה: {spread:.2f}% → חשש ממיתון"
    else:
        score = 20
        explanation = f"🔴 אינברסיה עמוקה: {spread:.2f}% → סיכון שוק גבוה"

    return score, explanation

import yfinance as yf

def detect_yield_inversion() -> float:
    """
    מושך את תשואת ה־2Y מה־2-Year Yield Futures ו־10Y מהמדד ^TNX,
    ומחזיר abs(10Y–2Y) אם התוצאה < 0 (אינברסיה), אחרת 0.0.
    """
    try:
        # שמוש ב-Yahoo Finance עבור 2Y ו־10Y
        hist2  = yf.Ticker('2YY=F').history(period='2d')['Close'].dropna()
        hist10 = yf.Ticker('^TNX').history(period='2d')['Close'].dropna()
        if len(hist2) < 1 or len(hist10) < 1:
            return 0.0
        val2  = hist2.iloc[-1]
        val10 = hist10.iloc[-1]
        spread = val10 - val2
        return abs(spread) if spread < 0 else 0.0
    except Exception as e:
        print(f"⚠️ שגיאה ב־detect_yield_inversion: {e}")
        return 0.0



if __name__ == "__main__":
    try:
        xml = fetch_treasury_yield_xml()
        yields = extract_yields_from_treasury_xml(xml)
        if not yields:  # אם אין תשואות – ננסה חודש קודם
            raise ValueError("אין תשואות")
    except:
        print("🔁 מנסה חודש קודם...")
        prev_month = (datetime.now().replace(day=1) - timedelta(days=1)).strftime("%Y%m")
        xml = fetch_treasury_yield_xml(prev_month)
        yields = extract_yields_from_treasury_xml(xml)

    score, explanation = calculate_bond_score(yields)
    print(f"📉 Bond Score: {score} | {explanation}")

