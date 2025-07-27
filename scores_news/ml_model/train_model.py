import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

MERGED_DATA_PATH = r"/AgentMarket/scores_news/ml_model/merged_scores_mes.csv"
MODEL_PATH = r"/AgentMarket/scores_news/ml_model/model.pkl"

def load_data():
    df = pd.read_csv(MERGED_DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])

    # שינוי שמות עמודות לסטנדרט אחיד
    rename_map = {
        'sentiment_score': 'sentiment',
        'macro_score': 'macro',
        'bonds_score': 'bonds',
        'futures_vix_score': 'futures_vix',
        'sectors_score': 'sectors',
        'mes_score': 'mes'
    }
    df.rename(columns=rename_map, inplace=True)

    feature_cols = ['sentiment', 'macro', 'bonds', 'futures_vix', 'sectors', 'mes']
    if not all(col in df.columns for col in feature_cols + ['direction']):
        raise ValueError("❌ עמודות חובה חסרות בקובץ הקלט. נדרש לכלול את: " + ', '.join(feature_cols + ['direction']))

    # המרת תוצאה למספר
    df['label'] = df['direction'].map({"UP": 1, "DOWN": -1, "NEUTRAL": 0})
    X = df[feature_cols]
    y = df['label']

    return train_test_split(X, y, test_size=0.2, random_state=42) if len(df) > 3 else (X, X, y, y)

def train_model():
    X_train, X_test, y_train, y_test = load_data()

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    if len(X_test) > 0:
        y_pred = model.predict(X_test)
        print("\n📊 Evaluation Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")
    return model

def predict_today(model):
    df = pd.read_csv(MERGED_DATA_PATH)
    df.rename(columns={
        'sentiment_score': 'sentiment',
        'macro_score': 'macro',
        'bonds_score': 'bonds',
        'futures_vix_score': 'futures_vix',
        'sectors_score': 'sectors',
        'mes_score': 'mes'
    }, inplace=True)

    df['date'] = pd.to_datetime(df['date'])
    latest = df.sort_values("date").iloc[-1]

    feature_cols = ['sentiment', 'macro', 'bonds', 'futures_vix', 'sectors', 'mes']
    features = pd.DataFrame([latest[feature_cols].values], columns=feature_cols)

    pred = model.predict(features)[0]

    print("\n🎯 Today's Forecast:")
    if pred == 1:
        print("✅ LONG")
    elif pred == -1:
        print("❌ SHORT")
    else:
        print("🔒 NEUTRAL")


def main():
    if not os.path.exists(MERGED_DATA_PATH):
        print("❌ Data not found. Run merge_scores_mes.py first.")
        return

    model = train_model()
    predict_today(model)


if __name__ == "__main__":
    main()
