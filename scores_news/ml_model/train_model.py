import pandas as pd
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

_ROOT            = Path(__file__).resolve().parents[2]
MERGED_DATA_PATH = _ROOT / "scores_news" / "ml_model" / "merged_scores_mes.csv"
MODEL_PATH       = _ROOT / "scores_news" / "ml_model" / "model.pkl"

FEATURE_COLS = ['sentiment', 'macro', 'bonds', 'futures_vix', 'sectors', 'mes']
RENAME_MAP = {
    'sentiment_score': 'sentiment', 'macro_score': 'macro', 'bonds_score': 'bonds',
    'futures_vix_score': 'futures_vix', 'sectors_score': 'sectors', 'mes_score': 'mes'
}


def load_data():
    df = pd.read_csv(MERGED_DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df.rename(columns=RENAME_MAP, inplace=True)

    if not all(c in df.columns for c in FEATURE_COLS + ['direction']):
        missing = [c for c in FEATURE_COLS + ['direction'] if c not in df.columns]
        raise ValueError(f"Missing required columns: {missing}")

    # Handle all direction formats: "UP"/"DOWN"/"NEUTRAL", "1"/"-1"/"0", 1/-1/0
    _str_map = {"UP": 1, "DOWN": -1, "NEUTRAL": 0}

    def _to_label(val):
        if isinstance(val, str):
            if val in _str_map:
                return _str_map[val]
            try:
                v = float(val)
                return 1 if v > 0.3 else -1 if v < -0.3 else 0
            except (ValueError, TypeError):
                return None
        try:
            v = float(val)
            return 1 if v > 0.3 else -1 if v < -0.3 else 0
        except (ValueError, TypeError):
            return None

    df['label'] = df['direction'].apply(_to_label)

    df = df.dropna(subset=['label'] + FEATURE_COLS)
    df['label'] = df['label'].astype(int)

    X = df[FEATURE_COLS]
    y = df['label']

    return train_test_split(X, y, test_size=0.2, random_state=42) if len(df) > 3 else (X, X, y, y)


def train_model():
    X_train, X_test, y_train, y_test = load_data()

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    if len(X_test) > 0:
        y_pred = model.predict(X_test)
        print("\nEvaluation Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model


def predict_today(model) -> int:
    df = pd.read_csv(MERGED_DATA_PATH)
    df.rename(columns=RENAME_MAP, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    latest = df.sort_values("date").iloc[-1]
    features = pd.DataFrame([latest[FEATURE_COLS].values], columns=FEATURE_COLS)
    pred = int(model.predict(features)[0])
    labels = {1: "LONG", -1: "SHORT", 0: "NEUTRAL"}
    print(f"\nToday's Forecast: {labels.get(pred, str(pred))}")
    return pred


def main():
    if not os.path.exists(MERGED_DATA_PATH):
        print("Data not found. Run merge_scores_mes.py first.")
        return
    model = train_model()
    predict_today(model)


if __name__ == "__main__":
    main()
