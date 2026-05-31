import pandas as pd
import os
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
SCORE_LOG_PATH = _ROOT / "scores_news" / "config" / "score_log.csv"
MES_DATA_PATH = _ROOT / "scores_news" / "config" / "MES_data.csv"
MERGED_OUTPUT_PATH = _ROOT / "scores_news" / "ml_model" / "merged_scores_mes.csv"

# === Threshold להגדרת מגמה ===
TREND_THRESHOLD = 5  # נקודות

def determine_direction(open_price, close_price):
    if close_price - open_price > TREND_THRESHOLD:
        return "UP"
    elif open_price - close_price > TREND_THRESHOLD:
        return "DOWN"
    else:
        return "NEUTRAL"

def load_score_log():
    score_df = pd.read_csv(SCORE_LOG_PATH)
    score_df['date'] = pd.to_datetime(score_df['date']).dt.date
    return score_df

def enrich_score_df(df):
    if 'open' in df.columns and 'close' in df.columns:
        if 'daily_change_pct' not in df.columns:
            df['daily_change_pct'] = ((df['close'] - df['open']) / df['open']) * 100

        if 'direction' not in df.columns:
            df['direction'] = df['daily_change_pct'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)

    return df


def load_mes_data():
    df = pd.read_csv(MES_DATA_PATH, index_col=False)

    # הפוך כל שמות העמודות לאותיות קטנות ונקה רווחים
    df.columns = [col.strip().lower() for col in df.columns]
    print("📋 עמודות לאחר ניקוי:", df.columns.tolist())

    if 'date' not in df.columns:
        raise KeyError("❌ עמודת 'date' לא קיימת בקובץ MES גם לאחר עיבוד.")

    df['date'] = pd.to_datetime(df['date']).dt.date
    df['direction'] = df.apply(lambda row: determine_direction(row['open'], row['close']), axis=1)
    df['daily_change_pct'] = ((df['close'] - df['open']) / df['open']) * 100

    return df[['date', 'open', 'close', 'direction']]


def merge_and_save(score_df, mes_df):
    # 🛠️ אם daily_change_pct קיימת אבל direction לא – צור אותה
    if "daily_change_pct" in score_df.columns and "direction" not in score_df.columns:
        score_df["direction"] = score_df["daily_change_pct"].apply(
            lambda x: 1 if x > 0 else -1 if x < 0 else 0
        )

    # ✅ ודא שהעמודות הדרושות קיימות
    expected_cols = ['date', 'sentiment_score', 'macro_score', 'bonds_score',
                     'futures_vix_score', 'sectors_score', 'mes_score', 'direction', 'daily_change_pct']

    print("📋 עמודות לאחר ניקוי:", list(score_df.columns))
    for col in expected_cols:
        if col not in score_df.columns:
            raise ValueError(f"❌ עמודת {col} לא קיימת ב־score_log.csv")

    # 🧠 מיזוג
    merged = pd.merge(score_df, mes_df, on='date', how='inner')

    # צור תיקייה אם לא קיימת
    os.makedirs(os.path.dirname(MERGED_OUTPUT_PATH), exist_ok=True)

    if os.path.exists(MERGED_OUTPUT_PATH):
        existing = pd.read_csv(MERGED_OUTPUT_PATH)
        existing['date'] = pd.to_datetime(existing['date']).dt.date
        merged = pd.concat([existing, merged]).drop_duplicates(subset='date').sort_values('date')

    merged.to_csv(MERGED_OUTPUT_PATH, index=False)
    print(f"✅ Merged data saved to {MERGED_OUTPUT_PATH} ({len(merged)} rows)")


def main():
    if not os.path.exists(SCORE_LOG_PATH):
        print("❌ Missing score_log.csv")
        return
    if not os.path.exists(MES_DATA_PATH):
        print("❌ Missing MES_data.csv")
        return

    score_df = load_score_log()
    score_df = enrich_score_df(score_df)
    mes_df = load_mes_data()
    merge_and_save(score_df, mes_df)




if __name__ == "__main__":
    main()
