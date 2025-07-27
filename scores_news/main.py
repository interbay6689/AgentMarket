
import subprocess
import datetime
import os

LOG_FILE = r"/AgentMarket/scores_news/logs/system_run.txt"

def log(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_message + "\n")

def run_step(script_path):
    log(f"Running: {script_path}")
    try:
        subprocess.run(["python", script_path], check=True)
        log(f"Completed: {script_path}\n")
    except subprocess.CalledProcessError as e:
        log(f"Failed: {script_path} with error: {e}\n")

def main():
    log("=== Starting Full System Run ===")

    # === שלב 1: משיכת נתונים ===
    run_step("data_sources/fetch_news_rss.py")
    run_step("data_sources/fetch_macro_data.py")
    run_step("data_sources/fetch_bonds_data.py")
    run_step("data_sources/fetch_sector_data.py")
    run_step("data_sources/fetch_vix_dxy.py")
    run_step("data_sources/fetch_mes_data.py")

    # === שלב 2: חישוב ציונים ===
    run_step("cat_scores/sentiment_score.py")
    run_step("cat_scores/macro_score.py")
    run_step("cat_scores/bonds_score.py")
    run_step("cat_scores/sectors_score.py")
    run_step("cat_scores/futures_vix_score.py")
    run_step("cat_scores/mes_score.py")

    # === שלב 3: חישוב ציון סופי ===
    run_step("cat_scores/final_score.py")

    # === שלב 4: מיזוג נתונים עבור המודל ===
    run_step("ml_model/merge_scores_mes.py")

    # === שלב 5: חיזוי על ידי המודל ===
    run_step("ml_model/train_model.py")

    # === שלב 6: שמירת תחזית יומית מול תוצאה בפועל ===
    run_step("ml_model/performance_tracker.py")


    # ניתוח התוצאה מהחיזוי האחרון
    try:
        import pandas as pd
        import joblib

        MODEL_PATH = r"/AgentMarket/scores_news/ml_model/model.pkl"
        MERGED_PATH = r"/AgentMarket/scores_news/ml_model/merged_scores_mes.csv"

        model = joblib.load(MODEL_PATH)
        df = pd.read_csv(MERGED_PATH)
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

        features = pd.DataFrame([latest[['sentiment', 'macro', 'bonds', 'futures_vix', 'sectors', 'mes']].values],
                                columns=['sentiment', 'macro', 'bonds', 'futures_vix', 'sectors', 'mes'])

        pred = model.predict(features)[0]
        if pred == 1:
            log("🤖 ML Forecast: ✅ LONG")
        elif pred == -1:
            log("🤖 ML Forecast: ❌ SHORT")
        else:
            log("🤖 ML Forecast: 🔒 NEUTRAL")
    except Exception as e:
        log(f"⚠️ ML prediction failed: {e}")

    log("=== System Run Completed ===\n")

if __name__ == "__main__":
    main()
