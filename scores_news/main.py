import subprocess
import datetime
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent        # scores_news/
_ROOT    = BASE_DIR.parent                         # AgentMarket/
LOG_FILE = BASE_DIR / "logs" / "system_run.txt"


def log(message: str) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_message + "\n")


def run_step(script_path: str) -> bool:
    """Run a Python script within the scores_news package. Returns True on success."""
    abs_path = (BASE_DIR / script_path).resolve()
    log(f"Running: {abs_path}")
    env = os.environ.copy()
    env["PYTHONPATH"]      = str(_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        subprocess.run(["python", str(abs_path)], check=True, env=env, cwd=str(_ROOT))
        log(f"OK: {script_path}")
        return True
    except subprocess.CalledProcessError as e:
        log(f"FAILED: {script_path} — {e}")
        return False


def main() -> None:
    log("=== Starting Full System Run ===")

    # Step 1: Fetch RSS sentiment articles → sentiment_raw_DATE.csv
    run_step("data_sources/fetch_news_rss.py")

    # Step 2: Update MES price history → MES_data.csv
    run_step("data_sources/mes_prices_history_table.py")

    # Step 3: Calculate all category scores → score_log.csv
    if not run_step("cat_scores/final_score.py"):
        log("Critical step failed — aborting pipeline")
        return

    # Step 4: Merge scores with MES data → merged_scores_mes.csv
    run_step("ml_model/merge_scores_mes.py")

    # Step 5: Train ML model → model.pkl
    run_step("ml_model/train_model.py")

    # Step 6: Track prediction performance → ml_performance_log.csv
    run_step("ml_model/performance_tracker.py")

    log("=== System Run Completed ===\n")


if __name__ == "__main__":
    main()
