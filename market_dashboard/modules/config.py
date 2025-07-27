# modules/config.py
from pathlib import Path
import yaml
import json

# מיקום תיקיית config של scores_news
CONFIG_DIR = Path(__file__).parent.parent.parent / 'scores_news' / 'config'
SOURCES_FILE = CONFIG_DIR / 'sources.yaml'
THRESHOLDS_FILE = CONFIG_DIR / 'thresholds.json'
WEIGHTS_FILE = CONFIG_DIR / 'weights.json'


def load_rss_sources() -> list:
    """
    טוען כתובות RSS מתוך sources.yaml שב־scores_news/config
    """
    if SOURCES_FILE.exists():
        with open(SOURCES_FILE, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        rss_feeds = data.get('rss_feeds', {})
        sources = []
        for lst in rss_feeds.values():
            if isinstance(lst, list):
                sources.extend(lst)
        return sources
    return []


def load_thresholds() -> dict:
    """
    טוען thresholds מתוך thresholds.json
    """
    if THRESHOLDS_FILE.exists():
        with open(THRESHOLDS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def load_weights() -> dict:
    """
    טוען weights מתוך weights.json
    """
    if WEIGHTS_FILE.exists():
        with open(WEIGHTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def load_config() -> dict:
    """
    מחזיר dict עם keys: thresholds, weights, rss_list
    """
    return {
        'thresholds': load_thresholds(),
        'weights': load_weights(),
        'rss_list': load_rss_sources()
    }


def save_config(cfg: dict) -> None:
    """
    שומר thresholds ו-weights לנתיב JSON תחת scores_news/config
    """
    # ודא שתיקיית הקונפיג קיימת
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    thresholds = cfg.get('thresholds', {})
    with open(THRESHOLDS_FILE, 'w', encoding='utf-8') as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)
    weights = cfg.get('weights', {})
    with open(WEIGHTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(weights, f, ensure_ascii=False, indent=2)