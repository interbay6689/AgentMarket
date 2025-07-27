import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# נתיבים
merged_path = Path(r"C:\Users\inter\PycharmProjects\FuturesMarketAI\scores_news\ml_model\merged_scores_mes.csv")
log_path = Path(r"C:\Users\inter\PycharmProjects\FuturesMarketAI\scores_news\ml_model\ml_performance_log.csv")
weights_path = Path(r"C:\Users\inter\PycharmProjects\FuturesMarketAI\scores_news\config\weights.json")

# קריאת נתונים
df_merged = pd.read_csv(merged_path)
df_merged["date"] = pd.to_datetime(df_merged["date"])
today = df_merged["date"].max().date()

row = df_merged[df_merged["date"].dt.date == today].iloc[-1]
true_move = row["daily_change_pct"]
prediction = row["prediction"]

# הגדרת הצלחה
if true_move > 0.3 and prediction == 1:
    result = "✅"
elif true_move < -0.3 and prediction == -1:
    result = "✅"
elif abs(true_move) <= 0.3 and prediction == 0:
    result = "✅"
else:
    result = "❌"

# טוען לוג קיים
if log_path.exists():
    df_log = pd.read_csv(log_path)
    df_log["date"] = pd.to_datetime(df_log["date"])
else:
    df_log = pd.DataFrame(columns=["date", "prediction", "true_move", "correct"])

# אם התאריך קיים – נעדכן, אחרת נוסיף
if today in df_log["date"].dt.date.values:
    df_log.loc[df_log["date"].dt.date == today, ["prediction", "true_move", "correct"]] = [prediction, true_move, result]
else:
    df_log = pd.concat([
        df_log,
        pd.DataFrame([{
            "date": today,
            "prediction": prediction,
            "true_move": true_move,
            "correct": result
        }])
    ])

df_log = df_log.sort_values("date")
df_log.to_csv(log_path, index=False)
print(f"✅ תחזית {result} עודכנה עבור {today}")

# 🧠 עדכון משקלים לפי הצלחות
df_success = df_merged[df_merged["date"].dt.date.isin(df_log[df_log["correct"] == "✅"]["date"].dt.date)]
features = ['sentiment_score', 'macro_score', 'bonds_score', 'futures_vix_score', 'sectors_score', 'mes_score']
averages = df_success[features].mean()
norm_weights = (averages / averages.sum()).round(3).to_dict()

# עדכון weights.json
with open(weights_path, "w") as f:
    json.dump(norm_weights, f, indent=2)

print("🎯 weights.json עודכן על בסיס התחזיות המוצלחות:")
print(norm_weights)


# === עדכון היסטוריית משקלים ===
history_path = Path(r"C:\Users\inter\PycharmProjects\FuturesMarketAI\scores_news\ml_model\weights_history.csv")
weights_row = {"date": today}
weights_row.update(norm_weights)

if history_path.exists():
    df_history = pd.read_csv(history_path)
    df_history = df_history[df_history["date"] != str(today)]
    df_history = pd.concat([df_history, pd.DataFrame([weights_row])])
else:
    df_history = pd.DataFrame([weights_row])

df_history.to_csv(history_path, index=False)
