import os
from datetime import datetime

import pandas as pd
import yfinance as yf


def enrich_mes_data(df):
    """המרת ערכים, חישוב שינוי יומי, ויצירת label"""
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df.dropna(subset=['open', 'close'], inplace=True)
    df['daily_change_pct'] = ((df['close'] - df['open']) / df['open']) * 100
    df['label'] = df['daily_change_pct'].apply(lambda x: 1 if x > 0.3 else -1 if x < -0.3 else 0)
    return df

def build_mes_csv(file_path="MES_data.csv", start_date="2025-01-01"):
    end_date = datetime.now().strftime("%Y-%m-%d")
    print("🔄 מוריד היסטוריית MES מ־Yahoo...")
    df = yf.download("MES=F", start=start_date, end=end_date, progress=False)
    df.reset_index(inplace=True)

    # ביטול MultiIndex אם קיים
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # בדיקה שעמודות חיוניות קיימות
    required_columns = ['date', 'open', 'close']
    for col in required_columns:
        if col not in df.columns:
            raise Exception(f"❌ העמודה '{col}' חסרה בנתונים שנמשכו מ־yfinance")

    df['date'] = pd.to_datetime(df['date']).dt.date
    df = df[['date', 'open', 'close']].dropna()
    df = enrich_mes_data(df)
    df.to_csv(file_path, index=False)
    print(f"✅ נשמרו {len(df)} שורות לקובץ {file_path}")
    return df


def update_mes_csv(file_path="MES_data.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError("❌ הקובץ MES_data.csv לא קיים. הפעל קודם את build_mes_csv()")

    existing_df = pd.read_csv(file_path)
    if existing_df.empty or 'date' not in existing_df.columns:
        print("⚠️ הקובץ קיים אך ריק או פגום — בונה אותו מחדש...")
        return build_mes_csv(file_path)

    existing_df['date'] = pd.to_datetime(existing_df['date'], errors='coerce').dt.date
    existing_df.dropna(subset=['date'], inplace=True)
    if existing_df.empty:
        print("⚠️ כל התאריכים לא תקינים — בונה מחדש...")
        return build_mes_csv(file_path)

    last_date = existing_df['date'].max()
    today = datetime.now().date()

    if last_date >= today:
        print("📅 הנתונים כבר מעודכנים עד היום.")
        return existing_df

    new_df = yf.download("MES=F", start=str(last_date), end=str(today + pd.Timedelta(days=1)), progress=False)
    new_df.reset_index(inplace=True)

    if isinstance(new_df.columns, pd.MultiIndex):
        new_df.columns = new_df.columns.get_level_values(0)

    if new_df.empty:
        print("⚠️ לא התקבלו נתונים חדשים מ־Yahoo Finance.")
        return existing_df

    print("📋 עמודות שהתקבלו מהנתונים החדשים:", new_df.columns.tolist())

    required_columns = ['date', 'open', 'close']
    for col in required_columns:
        if col not in new_df.columns:
            raise Exception(f"❌ העמודה '{col}' חסרה בנתונים שהתקבלו מ־yfinance")

    new_df['date'] = pd.to_datetime(new_df['date']).dt.date
    new_df = new_df[['date', 'open', 'close']].dropna()
    new_df = new_df[~new_df['date'].isin(existing_df['date'])]

    new_df = enrich_mes_data(new_df)
    updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    updated_df.drop_duplicates(subset=['date'], inplace=True)
    updated_df.sort_values('date', inplace=True)
    updated_df.to_csv(file_path, index=False)

    print(f"✅ נוסף {len(new_df)} רשומות חדשות. סה״כ: {len(updated_df)}")
    return updated_df

# הפעלה
if __name__ == "__main__":
    file_path = r"/AgentMarket/scores_news/config/MES_data.csv"
    if not os.path.exists(file_path):
        build_mes_csv(file_path)
    else:
        update_mes_csv(file_path)
