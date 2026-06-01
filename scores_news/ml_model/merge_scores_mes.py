import pandas as pd
import os
from pathlib import Path

_ROOT             = Path(__file__).resolve().parents[2]
SCORE_LOG_PATH    = _ROOT / "scores_news" / "config" / "score_log.csv"
MES_DATA_PATH     = _ROOT / "scores_news" / "config" / "MES_data.csv"
MERGED_OUTPUT_PATH = _ROOT / "scores_news" / "ml_model" / "merged_scores_mes.csv"


def load_score_log() -> pd.DataFrame:
    df = pd.read_csv(SCORE_LOG_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df


def enrich_score_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily_change_pct and direction if not already present."""
    if 'open' in df.columns and 'close' in df.columns:
        if 'daily_change_pct' not in df.columns:
            df['daily_change_pct'] = ((df['close'] - df['open']) / df['open']) * 100

    if 'daily_change_pct' in df.columns and 'direction' not in df.columns:
        df['direction'] = df['daily_change_pct'].apply(
            lambda x: 1 if x > 0.3 else -1 if x < -0.3 else 0
        )
    return df


def load_mes_data() -> pd.DataFrame:
    df = pd.read_csv(MES_DATA_PATH, index_col=False)
    df.columns = [c.strip().lower() for c in df.columns]
    if 'date' not in df.columns:
        raise KeyError("'date' column missing from MES_data.csv")
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['daily_change_pct'] = ((df['close'] - df['open']) / df['open']) * 100
    return df[['date', 'open', 'close', 'daily_change_pct']]


def merge_and_save(score_df: pd.DataFrame, mes_df: pd.DataFrame) -> None:
    # If score_log is missing open/close, supplement from MES_data
    if 'open' not in score_df.columns or 'close' not in score_df.columns:
        score_df = pd.merge(score_df, mes_df[['date', 'open', 'close', 'daily_change_pct']],
                            on='date', how='left', suffixes=('', '_mes'))
        # Use mes values where score_log didn't have them
        for col in ['open', 'close', 'daily_change_pct']:
            mes_col = f'{col}_mes'
            if mes_col in score_df.columns:
                score_df[col] = score_df[col].fillna(score_df[mes_col])
                score_df.drop(columns=[mes_col], inplace=True)

    score_df = enrich_score_df(score_df)

    # Validate required columns
    required = ['date', 'sentiment_score', 'macro_score', 'bonds_score',
                'futures_vix_score', 'sectors_score', 'mes_score', 'direction', 'daily_change_pct']
    missing = [c for c in required if c not in score_df.columns]
    if missing:
        raise ValueError(f"Missing columns in score data: {missing}")

    # Append to existing merged file (dedup by date)
    os.makedirs(os.path.dirname(MERGED_OUTPUT_PATH), exist_ok=True)
    if os.path.exists(MERGED_OUTPUT_PATH):
        existing = pd.read_csv(MERGED_OUTPUT_PATH)
        existing['date'] = pd.to_datetime(existing['date']).dt.date
        merged = pd.concat([existing, score_df], ignore_index=True)
        merged = merged.drop_duplicates(subset='date', keep='last').sort_values('date')
    else:
        merged = score_df.sort_values('date')

    merged.to_csv(MERGED_OUTPUT_PATH, index=False)
    print(f"Merged data saved to {MERGED_OUTPUT_PATH} ({len(merged)} rows)")


def main():
    if not os.path.exists(SCORE_LOG_PATH):
        print("Missing score_log.csv")
        return
    if not os.path.exists(MES_DATA_PATH):
        print("Missing MES_data.csv — run mes_prices_history_table.py first")
        return

    score_df = load_score_log()
    mes_df   = load_mes_data()
    merge_and_save(score_df, mes_df)


if __name__ == "__main__":
    main()
