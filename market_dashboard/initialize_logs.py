import os
from pathlib import Path
import csv

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# הגדרות קבצים והנתונים הראשוניים
log_files = {
    "error_log.txt": "=== Error Log Initialized ===\n",
    "alerts_log.txt": "=== Alerts Log Initialized ===\n",
    "score_log.csv": None  # נטפל בו בנפרד כקובץ CSV
}

# יצירת הקבצים אם לא קיימים
for file_name, initial_content in log_files.items():
    path = Path(LOG_DIR) / file_name
    if not path.exists():
        with open(path, "w", encoding="utf-8") as f:
            if initial_content is not None:
                f.write(initial_content)
            else:
                writer = csv.writer(f)
                writer.writerow(["date", "macro", "sentiment", "bonds", "sectors", "vix", "total_score"])

print("✅ כל קבצי הלוג הוכנו בהצלחה.")
