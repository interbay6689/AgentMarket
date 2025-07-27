# modules/alerts.py
import pandas as pd
from datetime import datetime
from pathlib import Path
from modules.config import load_config
from scores_news.cat_scores.bonds_score import fetch_treasury_yield_xml, extract_yields_from_treasury_xml
from scores_news.cat_scores.futures_vix_score import detect_vix_spike

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'


def detect_yield_inversion() -> float:
    """
    משלב fetch_treasury_yield_xml ו-extract_yields_from_treasury_xml כדי לחשב inversion
    """
    try:
        xml = fetch_treasury_yield_xml()
        yields = extract_yields_from_treasury_xml(xml)
        spread = yields.get('10Y', 0.0) - yields.get('2Y', 0.0)
        return abs(spread) if spread < 0 else 0.0
    except Exception:
        return 0.0


def load_alerts() -> pd.DataFrame:
    """
    Generate alerts based on thresholds and save to data/alerts.json
    """
    cfg = load_config()
    now = datetime.utcnow().isoformat()
    alerts = []

    inv = detect_yield_inversion()
    if inv >= cfg['thresholds'].get('yield_inversion', 0):
        alerts.append({
            'time': now,
            'type': 'Yield Inversion',
            'description': f'Inversion: {inv:.2f}%',
            'value': inv
        })

    vix = detect_vix_spike()
    if vix >= cfg['thresholds'].get('vix_spike', 0):
        alerts.append({
            'time': now,
            'type': 'VIX Spike',
            'description': f'Spike: {vix:.2f}%',
            'value': vix
        })

    df = pd.DataFrame(alerts)
    DATA_DIR.mkdir(exist_ok=True)
    df.to_json(DATA_DIR / 'alerts.json', force_ascii=False, orient='records', date_format='iso')
    return df
