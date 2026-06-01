# final_score.py

from datetime import datetime
from pathlib import Path

from scores_news.cat_scores.sentiment_score import (
    load_sentiment_data,
    calculate_sentiment_score,
)
from scores_news.cat_scores.macro_score import (
    fetch_macro_news,
    calculate_macro_score,
)
from scores_news.cat_scores.bonds_score import (
    fetch_treasury_yield_xml,
    extract_yields_from_treasury_xml,
    calculate_bond_score,
)
from scores_news.cat_scores.futures_vix_score import (
    fetch_futures_news,
    calculate_futures_score,
)
from scores_news.cat_scores.mes_score import fetch_mes_data, calculate_mes_score
from scores_news.cat_scores.sectors_score import calculate_sectors_score
import pandas as pd
import csv
import json
import os

# ---------------------------------------------------------------
# נתיב בסיס - מחושב דינמית על בסיס מיקום הקובץ הנוכחי
# מאפשר להריץ את המערכת בכל סביבה ללא תלות בנתיבים אבסולוטיים.
# parents[2] מחזיר את תיקיית הפרויקט (AgentMarket-main) משום שהקובץ נמצא בתיקיה
# scores_news/cat_scores/.
BASE_DIR = Path(__file__).resolve().parents[2]

# Paths to data/config files relative to project root
CONFIG_DIR = BASE_DIR / "scores_news" / "config"
ML_MODEL_DIR = BASE_DIR / "scores_news" / "ml_model"


def load_mes_local_data() -> pd.DataFrame:
    """טוען נתוני MES מקובץ CSV תחת תיקיית config.

    משתמש בנתיב יחסי המוגדר ב-CONFIG_DIR וכך מאפשר ניידות בין סביבות.
    """
    mes_path = CONFIG_DIR / "MES_data.csv"
    if not mes_path.exists():
        raise FileNotFoundError(f"MES_data.csv לא נמצא בנתיב {mes_path}")
    df = pd.read_csv(mes_path)
    # Normalize column names and parse date
    df.columns = [col.strip().lower() for col in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    return df

def load_all_scores():
    """מריץ כל קטגוריה ומחזיר מילון ציונים"""
    scores = {}

    # === Sentiment ===
    try:
        df = load_sentiment_data()
        sentiment_score, explanation = calculate_sentiment_score(df)
        scores["sentiment"] = {
            "score": sentiment_score,
            "explanation": explanation
        }
    except Exception as e:
        print(f"⚠️ שגיאה בקטגוריית sentiment: {e}")
        scores["sentiment"] = {
            "score": 50,
            "explanation": "שגיאה או חוסר נתונים – ברירת מחדל"
        }

    # === Macro ===
    try:
        df_macro = fetch_macro_news()
        macro_score, explanation = calculate_macro_score(df_macro)
        scores["macro"] = {
            "score": macro_score,
            "explanation": explanation
        }
    except Exception as e:
        print(f"⚠️ שגיאה בקטגוריית macro: {e}")
        scores["macro"] = {
            "score": 50,
            "explanation": "שגיאה או חוסר נתונים – ברירת מחדל"
        }

    # === Bonds ===
    try:
        xml = fetch_treasury_yield_xml()
        yields = extract_yields_from_treasury_xml(xml)
        if not yields:
            raise ValueError("אין תשואות")
        bond_score, explanation = calculate_bond_score(yields)
        scores["bonds"] = {
            "score": bond_score,
            "explanation": explanation
        }
    except Exception as e:
        print(f"⚠️ שגיאה בקטגוריית bonds: {e}")
        scores["bonds"] = {
            "score": 50,
            "explanation": "שגיאה או חוסר נתונים – ברירת מחדל"
        }

    # === Futures/VIX ===
    try:
        df_fut = fetch_futures_news()
        fut_score, explanation = calculate_futures_score(df_fut)
        scores["futures_vix"] = {
            "score": fut_score,
            "explanation": explanation
        }
    except Exception as e:
        print(f"⚠️ שגיאה בקטגוריית futures_vix: {e}")
        scores["futures_vix"] = {
            "score": 50,
            "explanation": "שגיאה או חוסר נתונים – ברירת מחדל"
        }

    # === MES + שינוי יומי, כיוון, open/close ===
    try:
        df_mes = load_mes_local_data()
        mes_score, explanation = calculate_mes_score(df_mes)
        scores["mes"] = {
            "score": mes_score,
            "explanation": explanation
        }

        # חישוב שינוי יומי באחוזים לפי close/open
        daily_change = ((df_mes["close"].iloc[-1] - df_mes["open"].iloc[-1]) / df_mes["open"].iloc[-1]) * 100
        scores["daily_change_pct"] = {
            "score": round(daily_change, 2),
            "explanation": "שינוי יומי באחוזים לפי MES"
        }

        # הוספת ערכים של open ו-close
        scores["open"] = {
            "score": round(df_mes["open"].iloc[-1], 2),
            "explanation": "מחיר פתיחה יומי של MES"
        }
        scores["close"] = {
            "score": round(df_mes["close"].iloc[-1], 2),
            "explanation": "מחיר סגירה יומי של MES"
        }

        # יצירת כיוון תנועה על סמך שינוי באחוזים עם סף של 0.3%
        direction_value = 1 if daily_change > 0.3 else -1 if daily_change < -0.3 else 0
        scores["direction"] = {
            "score": direction_value,
            "explanation": "כיוון תנועה יומית על סמך MES: 1=עלייה, -1=ירידה, 0=נייטרלי"
        }

    except Exception as e:
        print(f"⚠️ שגיאה ב־MES או daily_change_pct: {e}")
        scores["mes"] = {
            "score": 0,
            "explanation": "שגיאה ב־calculate_mes_score או ב־df_mes"
        }
        scores["daily_change_pct"] = {
            "score": 0,
            "explanation": "שגיאה או חוסר נתונים"
        }

    # === Sectors ===
    try:
        sector_score, explanation = calculate_sectors_score()
        scores["sectors"] = {
            "score": sector_score,
            "explanation": explanation
        }
    except Exception as e:
        print(f"⚠️ שגיאה בקטגוריית sectors: {e}")
        scores["sectors"] = {
            "score": 50,
            "explanation": "שגיאה או חוסר נתונים – ברירת מחדל"
        }

    return scores


def calculate_weighted_score(score_dict: dict, weights_path: Path | None = None) -> int:
    """מקבל מילון ציונים ומחזיר ממוצע משוקלל.

    אם לא סופק נתיב חיצוני, יקרא את weights.json מתוך CONFIG_DIR.
    """
    # קבע נתיב ברירת מחדל למשקלים
    path = Path(weights_path) if weights_path else CONFIG_DIR / "weights.json"
    if not path.exists():
        raise FileNotFoundError(f"weights.json לא נמצא בנתיב {path}")
    with open(path, "r", encoding="utf-8") as f:
        weights = json.load(f)
    total_weight = 0
    weighted_sum = 0
    for cat, weight in weights.items():
        score = score_dict.get(cat, {}).get("score", 50)
        weighted_sum += score * weight
        total_weight += weight
    if total_weight == 0:
        return 50
    return round(weighted_sum / total_weight)

def determine_market_bias(score: int, thresholds_path: Path | None = None) -> str:
    """קובע המלצת מסחר יומית על בסיס ספים.

    אם לא סופק נתיב, יקרא את thresholds.json מתוך CONFIG_DIR.
    """
    path = Path(thresholds_path) if thresholds_path else CONFIG_DIR / "thresholds.json"
    if not path.exists():
        raise FileNotFoundError(f"thresholds.json לא נמצא בנתיב {path}")
    with open(path, "r", encoding="utf-8") as f:
        thresholds = json.load(f)
    long_threshold = thresholds.get("long", 60)
    short_threshold = thresholds.get("short", 40)
    if score >= long_threshold:
        return "✅ LONG"
    elif score <= short_threshold:
        return "❌ SHORT"
    return "🔒 NEUTRAL"

def save_scores_to_log(scores: dict, final_score: int, bias: str, path: Path | None = None) -> None:
    """Save today's scores to score_log.csv, updating existing row for same date."""
    log_path = Path(path) if path else CONFIG_DIR / "score_log.csv"
    date_str = datetime.now().strftime("%Y-%m-%d")

    if "sectors" not in scores:
        scores["sectors"] = {"score": 50, "explanation": "default"}

    row: dict[str, object] = {"date": date_str, "final_score": final_score, "bias": bias}
    for cat, result in scores.items():
        val = result.get("score")
        if cat in {"daily_change_pct", "open", "close", "direction", "long_short_ratio"}:
            row[cat] = val
        else:
            row[f"{cat}_score"] = val

    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing, normalise schema, replace today's row, rewrite atomically
    if log_path.exists():
        existing = pd.read_csv(log_path)
        # Unify columns: add any new columns the current row has
        for col in row:
            if col not in existing.columns:
                existing[col] = None
        # Drop today's old entries
        existing = existing[existing["date"].astype(str) != date_str]
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        updated = pd.DataFrame([row])

    updated.to_csv(log_path, index=False)
    print(f"Saved score_log.csv ({log_path})")



if __name__ == "__main__":
    all_scores = load_all_scores()
    weighted_score = calculate_weighted_score(all_scores)
    decision = determine_market_bias(weighted_score)

    print("\n📊 ציונים לפי קטגוריה:")
    for cat, result in all_scores.items():
        print(f"- {cat.capitalize()}: {result['score']} | {result['explanation']}")
    print(f"\n🎯 ציון סופי משוקלל: {weighted_score}/100")
    print(f"\n📌 המלצה יומית: {decision}")

    save_scores_to_log(all_scores, weighted_score, decision)
