
import subprocess
import datetime
import os
from pathlib import Path

# ---------------------------------------------------------------
# נתיב בסיס - מחושב דינמית על בסיס מיקום הקובץ הנוכחי
# קובץ זה נמצא בתיקייה scores_news ולכן parent הוא תיקיית הפרויקט
BASE_DIR = Path(__file__).resolve().parent  # scores_news directory

# נתיב לקובץ לוג המריץ ריצות מערכת מלאות
LOG_FILE = BASE_DIR / "logs" / "system_run.txt"

def log(message: str) -> None:
    """Write a timestamped message to stdout and the system log file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    # Ensure log directory exists
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_message + "\n")

def run_step(script_path: str) -> None:
    """Run a Python script located within the scores_news package.

    The provided ``script_path`` is relative to the ``scores_news`` directory.
    A fully qualified path is constructed to ensure correct execution from any working directory.
    """
    # Construct absolute path to the script inside scores_news
    abs_path = (BASE_DIR / script_path).resolve()
    log(f"Running: {abs_path}")
    try:
        subprocess.run(["python", str(abs_path)], check=True)
        log(f"Completed: {script_path}\n")
    except subprocess.CalledProcessError as e:
        log(f"Failed: {script_path} with error: {e}\n")

def main() -> None:
    """Execute the full data collection, scoring, merging, and model training pipeline."""
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

        model_path = BASE_DIR / "ml_model" / "model.pkl"
        merged_path = BASE_DIR / "ml_model" / "merged_scores_mes.csv"
        if not model_path.exists() or not merged_path.exists():
            raise FileNotFoundError("Model or merged data file missing. Run previous steps first.")

        model = joblib.load(model_path)
        df = pd.read_csv(merged_path)
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

        features = pd.DataFrame([
            latest[['sentiment', 'macro', 'bonds', 'futures_vix', 'sectors', 'mes']].values
        ], columns=['sentiment', 'macro', 'bonds', 'futures_vix', 'sectors', 'mes'])

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
