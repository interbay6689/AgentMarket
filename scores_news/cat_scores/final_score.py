# final_score.py

from datetime import datetime
from scores_news.cat_scores.sentiment_score import load_sentiment_data, calculate_sentiment_score
from scores_news.cat_scores.macro_score import fetch_macro_news, calculate_macro_score
from scores_news.cat_scores.bonds_score import fetch_treasury_yield_xml, extract_yields_from_treasury_xml, calculate_bond_score
from scores_news.cat_scores.futures_vix_score import fetch_futures_news, calculate_futures_score
from scores_news.cat_scores.mes_score import fetch_mes_data, calculate_mes_score
from scores_news.cat_scores.sectors_score import calculate_sectors_score
import pandas as pd
import csv
import json
import os

def load_mes_local_data():
    path = r"/AgentMarket/scores_news/config/MES_data.csv"
    df = pd.read_csv(path)
    df.columns = [col.strip().lower() for col in df.columns]
    df["date"] = pd.to_datetime(df["date"]).dt.date
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


def calculate_weighted_score(score_dict, weights_path=r"C:\Users\inter\PycharmProjects\FuturesMarketAI\scores_news\config\weights.json"):
    """מקבל מילון עם ציונים + קובץ משקלים ומחזיר ממוצע משוקלל"""
    with open(weights_path, "r", encoding="utf-8") as f:
        weights = json.load(f)

    total_weight = 0
    weighted_sum = 0

    for cat, weight in weights.items():
        score = score_dict.get(cat, {}).get("score", 50)
        weighted_sum += score * weight
        total_weight += weight

    if total_weight == 0:
        return 50  # fallback
    return round(weighted_sum / total_weight)

def determine_market_bias(score, thresholds_path=r"C:\Users\inter\PycharmProjects\FuturesMarketAI\scores_news\config\thresholds.json"):
    """קובע המלצת מסחר יומית על בסיס ספים"""
    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    if score >= thresholds["long"]:
        return "✅ LONG"
    elif score <= thresholds["short"]:
        return "❌ SHORT"
    else:
        return "🔒 NEUTRAL"

def save_scores_to_log(scores: dict, final_score: int, bias: str, path=r"C:\Users\inter\PycharmProjects\FuturesMarketAI\scores_news\config\score_log.csv"):
    """שומר את תוצאות היום לקובץ score_log.csv"""
    date = datetime.now().strftime("%Y-%m-%d")
    row = {
        "date": date,
        "final_score": final_score,
        "bias": bias
    }

    # הוספת ציונים של כל הקטגוריות
    for cat, result in scores.items():
        score = result.get("score")
        if "sectors" not in scores:
            scores["sectors"] = {
                "score": 50,
                "explanation": "נתון לא זמין – ברירת מחדל"
            }
        if cat in ["daily_change_pct", "open", "close", "direction", "long_short_ratio"]:
            row[cat] = score  # שמירה בשם המקורי
        else:
            row[f"{cat}_score"] = score  # שאר הקטגוריות נשמרות עם _score

    file_exists = os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"💾 נשמר ל־score_log.csv ({path})")



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
