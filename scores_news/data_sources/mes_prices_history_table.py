import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

_ROOT = Path(__file__).resolve().parents[2]


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
    print("Downloading MES history from Yahoo...")
    df = yf.download("MES=F", start=start_date, end=end_date, progress=False)
    df.reset_index(inplace=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.lower() for c in df.columns]

    required_columns = ['date', 'open', 'close']
    for col in required_columns:
        if col not in df.columns:
            raise Exception(f"Missing column '{col}' in yfinance data. Got: {df.columns.tolist()}")

    df['date'] = pd.to_datetime(df['date']).dt.date
    df = df[['date', 'open', 'close']].dropna()
    df = enrich_mes_data(df)
    df.to_csv(file_path, index=False)
    print(f"Saved {len(df)} rows to {file_path}")
    return df


def update_mes_csv(file_path="MES_data.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError("MES_data.csv not found. Run build_mes_csv() first.")

    existing_df = pd.read_csv(file_path)
    if existing_df.empty or 'date' not in existing_df.columns:
        print("CSV empty or corrupt, rebuilding...")
        return build_mes_csv(file_path)

    existing_df['date'] = pd.to_datetime(existing_df['date'], errors='coerce').dt.date
    existing_df.dropna(subset=['date'], inplace=True)
    if existing_df.empty:
        print("All dates invalid, rebuilding...")
        return build_mes_csv(file_path)

    last_date = existing_df['date'].max()
    today = datetime.now().date()

    if last_date >= today:
        print("Data already up to date.")
        return existing_df

    new_df = yf.download("MES=F", start=str(last_date), end=str(today + pd.Timedelta(days=1)), progress=False)
    new_df.reset_index(inplace=True)

    if isinstance(new_df.columns, pd.MultiIndex):
        new_df.columns = new_df.columns.get_level_values(0)

    new_df.columns = [c.lower() for c in new_df.columns]

    if new_df.empty:
        print("No new data from Yahoo Finance.")
        return existing_df

    required_columns = ['date', 'open', 'close']
    for col in required_columns:
        if col not in new_df.columns:
            raise Exception(f"Missing column '{col}' in yfinance data. Got: {new_df.columns.tolist()}")

    new_df['date'] = pd.to_datetime(new_df['date']).dt.date
    new_df = new_df[['date', 'open', 'close']].dropna()
    new_df = new_df[~new_df['date'].isin(existing_df['date'])]

    new_df = enrich_mes_data(new_df)
    updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    updated_df.drop_duplicates(subset=['date'], inplace=True)
    updated_df.sort_values('date', inplace=True)
    updated_df.to_csv(file_path, index=False)

    print(f"Added {len(new_df)} new rows. Total: {len(updated_df)}")
    return updated_df

# הפעלה
if __name__ == "__main__":
    file_path = str(_ROOT / "scores_news" / "config" / "MES_data.csv")
    if not os.path.exists(file_path):
        build_mes_csv(file_path)
    else:
        update_mes_csv(file_path)
