import pandas as pd
from pathlib import Path
from datetime import datetime
from modules.config import load_config

# ייבוא פונקציות חישוב מתוך scores_news
from scores_news.cat_scores.bonds_score import (
    fetch_treasury_yield_xml,
    extract_yields_from_treasury_xml,
    calculate_bond_score
)
from scores_news.cat_scores.macro_score import calculate_macro_score, fetch_macro_news
from scores_news.cat_scores.sentiment_score import calculate_sentiment_score, load_sentiment_data
from scores_news.cat_scores.futures_vix_score import fetch_futures_news, calculate_futures_score
from scores_news.cat_scores.sectors_score import calculate_sectors_score
from scores_news.cat_scores.mes_score import fetch_mes_data, calculate_mes_score

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'


def recommend_arrow(score: float) -> str:
    """
    מחזיר חץ המלצה על בסיס טווח הציון.
    """
    if score >= 60:
        return "🔼"
    elif score <= 40:
        return "🔽"
    return "➖"


def load_scores() -> pd.DataFrame:
    """
    Compute scores for each category using scores_news functions
    and save JSON to data/scores.json
    """
    # Bonds
    xml_data = fetch_treasury_yield_xml()
    yields = extract_yields_from_treasury_xml(xml_data)
    bonds_score, bonds_explanation = calculate_bond_score(yields)

    # Macro
    df_macro = fetch_macro_news()
    macro_score, macro_expl = calculate_macro_score(df_macro)

    # Sentiment
    try:
        df_sent = load_sentiment_data()
        sentiment_score, sentiment_expl = calculate_sentiment_score(df_sent)
    except Exception as e:
        sentiment_score, sentiment_expl = 50, f"🔒 אין נתוני סנטימנט: {e}"

    # VIX / Futures
    df_vix = fetch_futures_news()
    vix_score, vix_explanation = calculate_futures_score(df_vix)

    # Sectors
    sectors_score, sectors_expl = calculate_sectors_score()


    # MES
    df_mes = fetch_mes_data()
    mes_score, mes_expl = calculate_mes_score(df_mes)

    scores = [
        {'category': 'Bonds',     'score': bonds_score,     'explanation': bonds_explanation},
        {'category': 'Macro',     'score': macro_score,     'explanation': macro_expl},
        {'category': 'Sentiment', 'score': sentiment_score, 'explanation': sentiment_expl},
        {'category': 'VIX',       'score': vix_score,       'explanation': vix_explanation},
        {'category': 'Sectors', 'score': sectors_score, 'explanation': sectors_expl},
        {'category': 'MES',       'score': mes_score,       'explanation': mes_expl}
    ]

    print("DEBUG scores list:")
    for x in scores:
        print(f"{x['category']} — {type(x['score'])}: {x['score']}")

    df = pd.DataFrame(scores)
    DATA_DIR.mkdir(exist_ok=True)
    df.to_json(DATA_DIR / 'scores.json', force_ascii=False, orient='records', date_format='iso')
    return df



def load_time_series() -> pd.DataFrame:
    """
    Load historical time series from data/time_series.json else create dummy
    """
    ts_file = DATA_DIR / 'time_series.json'
    if ts_file.exists():
        df = pd.read_json(ts_file, convert_dates=['timestamp'])
        return df.set_index('timestamp')
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    cols = [r['category'] for r in load_scores().to_dict('records')]
    df = pd.DataFrame(index=dates, columns=cols).fillna(50)
    return df


def load_significant_changes(category: str) -> pd.DataFrame:
    """
    Load significant changes from data/{category}_changes.json
    """
    changes_file = DATA_DIR / f'{category.lower()}_changes.json'
    if changes_file.exists():
        return pd.read_json(changes_file)
    return pd.DataFrame(columns=['timestamp', 'source', 'old_score', 'new_score'])
