import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from datetime import datetime, timedelta


def clean_articles(df: pd.DataFrame) -> pd.DataFrame:
    # סינון כותרות ריקות
    df = df[df["title"].str.strip().astype(bool)]

    # המרה לפורמט תאריך
    df["published"] = pd.to_datetime(df["published"], errors="coerce")

    # סינון לפי 24 שעות אחרונות
    one_day_ago = datetime.now() - timedelta(days=1)
    df = df[df["published"] > one_day_ago]

    # הסרת כפילויות לפי כותרת + לינק
    df = df.drop_duplicates(subset=["title", "link"])

    return df.reset_index(drop=True)
