import pandas as pd
import json
import joblib
from pathlib import Path

_ROOT        = Path(__file__).resolve().parents[2]
merged_path  = _ROOT / "scores_news" / "ml_model" / "merged_scores_mes.csv"
model_path   = _ROOT / "scores_news" / "ml_model" / "model.pkl"
log_path     = _ROOT / "scores_news" / "ml_model" / "ml_performance_log.csv"
weights_path = _ROOT / "scores_news" / "config" / "weights.json"
history_path = _ROOT / "scores_news" / "ml_model" / "weights_history.csv"

FEATURE_COLS = ['sentiment', 'macro', 'bonds', 'futures_vix', 'sectors', 'mes']
RENAME_MAP   = {
    'sentiment_score': 'sentiment', 'macro_score': 'macro', 'bonds_score': 'bonds',
    'futures_vix_score': 'futures_vix', 'sectors_score': 'sectors', 'mes_score': 'mes'
}
RAW_COLS = ['sentiment_score', 'macro_score', 'bonds_score',
            'futures_vix_score', 'sectors_score', 'mes_score']


def _make_prediction(df_merged: pd.DataFrame, model) -> int:
    df = df_merged.rename(columns=RENAME_MAP)
    latest = df.sort_values("date").iloc[-1]
    features = pd.DataFrame([latest[FEATURE_COLS].values], columns=FEATURE_COLS)
    return int(model.predict(features)[0])


def _label_result(true_move: float, prediction: int) -> str:
    if true_move > 0.3 and prediction == 1:
        return "✅"
    if true_move < -0.3 and prediction == -1:
        return "✅"
    if abs(true_move) <= 0.3 and prediction == 0:
        return "✅"
    return "❌"


def main():
    if not merged_path.exists():
        print("Missing merged_scores_mes.csv — run merge_scores_mes.py first")
        return
    if not model_path.exists():
        print("Missing model.pkl — run train_model.py first")
        return

    df_merged = pd.read_csv(merged_path)
    df_merged["date"] = pd.to_datetime(df_merged["date"])
    df_merged = df_merged.sort_values("date")

    model = joblib.load(model_path)
    today = df_merged["date"].max().date()

    # Load or create performance log (accept legacy "actual" column name)
    if log_path.exists():
        df_log = pd.read_csv(log_path)
        if "actual" in df_log.columns and "true_move" not in df_log.columns:
            df_log.rename(columns={"actual": "true_move"}, inplace=True)
        df_log["date"] = pd.to_datetime(df_log["date"])
    else:
        df_log = pd.DataFrame(columns=["date", "prediction", "true_move", "correct"])

    # Backfill true_move for prior predictions that are still pending
    for idx, row in df_log.iterrows():
        if pd.isna(row.get("true_move")):
            match = df_merged[df_merged["date"] == pd.Timestamp(row["date"])]
            if not match.empty and "daily_change_pct" in match.columns:
                true_move = float(match.iloc[0]["daily_change_pct"])
                pred = int(row["prediction"])
                df_log.at[idx, "true_move"] = true_move
                df_log.at[idx, "correct"]   = _label_result(true_move, pred)

    # Record today's prediction
    pred_today = _make_prediction(df_merged, model)
    today_ts   = pd.Timestamp(today)
    today_mask = df_log["date"] == today_ts
    if today_mask.any():
        df_log.loc[today_mask, "prediction"] = pred_today
    else:
        new_row = pd.DataFrame([{"date": today_ts, "prediction": pred_today,
                                  "true_move": None, "correct": None}])
        df_log = pd.concat([df_log, new_row], ignore_index=True)

    df_log["date"] = pd.to_datetime(df_log["date"])
    df_log = df_log.sort_values("date")
    df_log.to_csv(log_path, index=False)
    labels = {1: "LONG", -1: "SHORT", 0: "NEUTRAL"}
    print(f"Prediction saved for {today}: {labels.get(pred_today, str(pred_today))}")

    # Update weights.json based on correctly-predicted days
    correct_dates = df_log[df_log["correct"] == "✅"]["date"].dt.date.values
    avail_cols    = [c for c in RAW_COLS if c in df_merged.columns]
    df_success    = df_merged[df_merged["date"].dt.date.isin(correct_dates)]

    if avail_cols and len(df_success) >= 3:
        averages = df_success[avail_cols].mean()
        total    = averages.sum()
        if total > 0:
            clean = {k.replace("_score", ""): round(v / total, 3) for k, v in averages.items()}
            with open(weights_path, "w") as f:
                json.dump(clean, f, indent=2)
            print(f"Updated weights.json: {clean}")

    # Append to weights history
    current_weights = json.loads(weights_path.read_text()) if weights_path.exists() else {}
    row_h = {"date": str(today), **current_weights}
    if history_path.exists():
        df_h = pd.read_csv(history_path)
        df_h = df_h[df_h["date"] != str(today)]
        df_h = pd.concat([df_h, pd.DataFrame([row_h])], ignore_index=True)
    else:
        df_h = pd.DataFrame([row_h])
    df_h.to_csv(history_path, index=False)


if __name__ == "__main__":
    main()
