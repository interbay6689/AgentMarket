# AgentMarket — Setup Guide

## Required Knowledge (כישורים נדרשים)

### Python
- Python 3.9+ מותקן
- הבנה בסיסית של virtual environments
- הכרות עם pip ו-requirements.txt

### Trading Knowledge
- הבנה בסיסית של חוזים עתידיים (NQ, MES)
- הכרות עם מושגי Order Flow: Fair Value Gap (FVG), Delta, Order Blocks (OB)
- הכרות עם ICT/SMC methodology: AMD Cycle, Judas Swing, Killzones (לא חובה)

### APIs & Accounts
- **yfinance** — חינמי, ללא חשבון (Yahoo Finance)
- **OpenAI API** — נדרש API key (gpt-4) לניתוח sentiment
- **RSS Feeds** — חינמי, ללא חשבון

---

## Prerequisites (לפני ההתקנה)

| דרישה | גרסה מינימלית | קישור |
|-------|--------------|-------|
| Python | 3.9+ | https://python.org/downloads |
| Git (אופציונלי) | כלשהי | https://git-scm.com |
| Windows / Mac / Linux | כלשהי | — |

---

## Installation — Windows (הדרך המהירה)

**לחץ פעמיים על `install.bat`** — הכל יתקין אוטומטית.

אם מעדיף ידנית:

```powershell
# 1. חלץ את ה-zip — חשוב: תיקיית הפרויקט חייבת להיות AgentMarket\
#    אם GitHub יצר תת-תיקייה כפולה כמו AgentMarket-main\AgentMarket-main\
#    — היכנס לתיקייה הפנימית.
cd C:\path\to\AgentMarket

# 2. צור virtual environment
python -m venv venv

# 3. הפעל את הסביבה
venv\Scripts\activate

# 4. התקן את כל התלויות (requirements.txt ברמת root)
pip install -r requirements.txt
```

## Installation — Mac / Linux

```bash
cd /path/to/AgentMarket
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Configuration (משתני סביבה)

צור קובץ `.env` בתיקיית הפרויקט הראשית:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

> **שים לב:** הקובץ `.env` מוגן על-ידי `.gitignore` ולא יועלה ל-GitHub.

---

## Running the App

```powershell
# Windows — הפעל תמיד עם streamlit run (לא python!)
venv\Scripts\activate
streamlit run market_dashboard/app.py
```

```bash
# Mac / Linux
source venv/bin/activate
streamlit run market_dashboard/app.py
```

**חשוב:** לעולם אל תריץ `python app.py` — חייב `streamlit run`!

לנוחות — קובץ `run.bat` בתיקייה הראשית מריץ הכל בלחיצה כפולה (Windows בלבד).

---

## Architecture Overview (מבנה הפרויקט)

```
AgentMarket/
├── market_dashboard/         ← Streamlit UI (הדשבורד)
│   ├── app.py                ← ENTRY POINT — streamlit run market_dashboard/app.py
│   ├── config.yaml           ← Thresholds, weights
│   ├── requirements.txt      ← כל התלויות
│   ├── modules/
│   │   ├── config.py         ← load/save config.yaml
│   │   ├── data.py           ← load_time_series
│   │   ├── alerts.py         ← load_alerts
│   │   ├── nq_data.py        ← yfinance fetching עבור NQ
│   │   └── nq_calculations.py← Order Flow calculations
│   └── pages/
│       ├── dashboard.py      ← ציון יומי + ML prediction
│       ├── nq_order_flow.py  ← NQ dashboard (7 טאבים)
│       ├── category_detail.py← drill-down לקטגוריה
│       ├── alerts_page.py    ← התראות פעילות
│       ├── settings.py       ← הגדרות
│       ├── system_info.py    ← לוגים
│       └── ml_learning.py   ← ביצועי מודל ML
│
├── scores_news/              ← מנוע ניקוד (backend)
│   ├── cat_scores/           ← 6 מודולי ציון
│   │   ├── final_score.py    ← אגרגציה + החלטה סופית
│   │   ├── sentiment_score.py
│   │   ├── macro_score.py
│   │   ├── bonds_score.py
│   │   ├── mes_score.py
│   │   ├── sectors_score.py
│   │   └── futures_vix_score.py
│   ├── data_sources/         ← Fetchers
│   ├── ml_model/             ← RandomForest
│   └── config/               ← weights.json, score_log.csv, NQ_data.csv
│
├── ai_analysis/
│   └── openai_client.py      ← GPT-4 wrapper (API key מ-.env)
├── .env                      ← API keys (לא ב-git!)
├── run.bat                   ← הפעלה מהירה ל-Windows
└── SETUP.md                  ← קובץ זה
```

### Data Flow (זרימת נתונים)

```
[yfinance] → nq_data.py → nq_calculations.py → nq_order_flow.py (UI)
[RSS + APIs] → scores_news/cat_scores/ → final_score.py → score_log.csv → dashboard.py (UI)
                                               ↓
                                     ml_model → LONG/SHORT/NEUTRAL
```

---

## NQ Order Flow Dashboard — 7 טאבים

| טאב | תוכן |
|-----|------|
| 1. War Plan | מחיר live, session נוכחי, bias יומי, assets מתואמים |
| 2. Session Timeline | Gantt chart סשנים (שעון ישראל + ET) |
| 3. Key Levels | גרף נרות 5m + FVG zones + Volume Profile |
| 4. Order Flow | Cumulative Delta, Absorption, Divergence |
| 5. ICT/SMC | Killzones, AMD Cycle, Judas Swing, Order Blocks, BOS/MSS |
| 6. Historical | Similar Days (cosine similarity), Win Rate by Hour |
| 7. Checklist | 11 תנאים לפני כניסה לעסקה |

---

## Common Errors & Fixes (שגיאות נפוצות)

| שגיאה | סיבה | תיקון |
|-------|------|-------|
| `ModuleNotFoundError: feedparser` | חבילות לא הותקנו | הרץ `install.bat` או: `pip install -r requirements.txt` |
| `ModuleNotFoundError: yaml` | חבילות לא הותקנו | הרץ `install.bat` או: `pip install -r requirements.txt` |
| `ModuleNotFoundError: streamlit` | venv לא מופעל | `venv\Scripts\activate` |
| `ModuleNotFoundError: AgentMarket` | import path שגוי | הרץ מ-`AgentMarket/` (לא מ-`pages/`) |
| שתי תיקיות כפולות בzip | GitHub מייצר `Repo-branch\Repo-branch\` | היכנס לתיקייה הפנימית |
| `No data available` / `yfinance empty` | חסימת Yahoo Finance | בדוק חיבור אינטרנט, נסה שוב אחרי כמה דקות |
| `OpenAI API error: 401` | API key חסר/שגוי | ודא שיש קובץ `.env` עם `OPENAI_API_KEY=...` |
| `score_log.csv not found` | final_score.py לא הורץ | הרץ `python scores_news/cat_scores/final_score.py` |
| `model.pkl not found` | מודל לא אומן | הרץ `python scores_news/ml_model/train_model.py` |
| Port 8501 in use | שרת streamlit כבר רץ | סגור את הטאב/טרמינל הישן |

---

## Daily Workflow (שגרה יומית)

```powershell
# 1. הפעל את הסביבה
venv\Scripts\activate

# 2. (אופציונלי) הרץ ניתוח יומי
python scores_news/cat_scores/final_score.py

# 3. הפעל את הדשבורד
streamlit run market_dashboard/app.py
```

---

## Dependencies (רשימת תלויות מלאה)

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.0.0
yfinance>=0.2.0
pytz
scikit-learn>=1.3.0
joblib>=1.3.0
pyyaml>=6.0
matplotlib>=3.5.0
seaborn>=0.12.0
feedparser>=6.0.0
vaderSentiment>=3.3.2
requests>=2.28.0
openai>=1.0.0
```
