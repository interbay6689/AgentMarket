import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import subprocess
import joblib
import matplotlib
from datetime import datetime, date, timedelta

matplotlib.use('Agg')

_ROOT = Path(__file__).resolve().parents[2]
_SCORE_PATH  = _ROOT / "scores_news" / "config" / "score_log.csv"
_MERGED_PATH = _ROOT / "scores_news" / "ml_model" / "merged_scores_mes.csv"
_MODEL_PATH  = _ROOT / "scores_news" / "ml_model" / "model.pkl"
_PERF_PATH   = _ROOT / "scores_news" / "ml_model" / "ml_performance_log.csv"
_FINAL_SCORE = _ROOT / "scores_news" / "cat_scores" / "final_score.py"
_MES_UPDATE  = _ROOT / "scores_news" / "data_sources" / "mes_prices_history_table.py"
_RSS_FETCH   = _ROOT / "scores_news" / "data_sources" / "fetch_news_rss.py"


@st.cache_data(ttl=300)
def _load_score_log():
    df = pd.read_csv(_SCORE_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date")


@st.cache_data(ttl=300)
def _load_ml_prediction():
    try:
        df = pd.read_csv(_MERGED_PATH)
        df['date'] = pd.to_datetime(df['date'])
        latest = df.sort_values("date").iloc[-1]
        model = joblib.load(_MODEL_PATH)
        X = pd.DataFrame(
            [[latest[c] for c in ['sentiment_score', 'macro_score', 'bonds_score',
                                  'futures_vix_score', 'sectors_score', 'mes_score']]],
            columns=['sentiment', 'macro', 'bonds', 'futures_vix', 'sectors', 'mes']
        )
        pred = model.predict(X)[0]
        return {1: "✅ LONG", -1: "❌ SHORT", 0: "🔒 NEUTRAL"}.get(pred, "🔒")
    except Exception as e:
        return f"לא זמין ({e})"


@st.cache_data(ttl=300)
def _load_model_accuracy():
    try:
        df = pd.read_csv(_PERF_PATH)
        acc = (df['correct'] == '✅').mean() * 100
        return f"{acc:.1f}% ({len(df)} ימים)"
    except Exception:
        return "לא זמין"


def _make_env() -> dict:
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run_script(script: Path, env: dict) -> tuple[bool, str]:
    try:
        r = subprocess.run(
            ["python", str(script)],
            capture_output=True, text=True, timeout=120,
            cwd=str(_ROOT), env=env
        )
        return r.returncode == 0, (r.stdout + r.stderr).strip()[:800]
    except subprocess.TimeoutExpired:
        return False, f"timeout (>2 min)"
    except Exception as e:
        return False, str(e)


def _run_daily_pipeline() -> list[dict]:
    """
    Pipeline מלא ביום מסחר:
      1. עדכון MES_data.csv    — נדרש לחישוב ציון MES
      2. משיכת RSS סנטימנט    — אופציונלי (ברירת מחדל 50 אם נכשל)
      3. final_score.py        — חישוב + כתיבה ל-score_log.csv
    """
    env = _make_env()
    steps = [
        {"name": "📈 עדכון MES_data.csv",        "script": _MES_UPDATE,  "required": True},
        {"name": "📰 משיכת חדשות RSS לסנטימנט",  "script": _RSS_FETCH,   "required": False},
        {"name": "🧮 חישוב ציונים (final_score)", "script": _FINAL_SCORE, "required": True},
    ]
    results = []
    for step in steps:
        ok, out = _run_script(step["script"], env)
        results.append({"name": step["name"], "ok": ok,
                        "required": step["required"], "output": out})
        if not ok and step["required"]:
            break
    st.cache_data.clear()
    return results


def _data_age_days(df: pd.DataFrame) -> int:
    last = df["date"].max().date()
    return (date.today() - last).days


def _show_run_button(label: str = "▶️ הרץ ניתוח יומי עכשיו"):
    """כפתור שמריץ את ה-pipeline המלא ומציג תוצאה לכל שלב."""
    if st.button(label, type="primary"):
        results = []
        with st.spinner("מריץ pipeline... (עד 5 דקות)"):
            results = _run_daily_pipeline()

        all_ok = True
        for r in results:
            icon = "✅" if r["ok"] else ("⚠️" if not r["required"] else "❌")
            st.markdown(f"{icon} **{r['name']}**")
            if r["output"]:
                with st.expander("פלט", expanded=not r["ok"]):
                    st.code(r["output"])
            if not r["ok"] and r["required"]:
                all_ok = False

        if all_ok:
            st.success("✅ הניתוח הסתיים — הנתונים עודכנו")
            st.rerun()
        else:
            st.error("❌ שלב קריטי נכשל — בדוק את הפלט למעלה")


def app():
    # ─── כותרת + רענן ───────────────────────────────────────────
    col_title, col_refresh = st.columns([6, 1])
    with col_title:
        st.title("📊 Market Sentiment Dashboard – החלטת מסחר יומית רשמית")
    with col_refresh:
        st.write("")
        if st.button("🔄 רענן", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    st.caption(f"זמן נוכחי: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

    # ─── זיהוי נתונים ישנים ───────────────────────────────────────
    try:
        df_check = _load_score_log()
        age = _data_age_days(df_check)
        last_date = df_check["date"].max().date()

        if age >= 1:
            st.warning(
                f"⚠️ **נתוני הציונים אינם מעודכנים** — "
                f"הנתון האחרון מתאריך **{last_date}** ({age} ימים לאחור)."
            )
            _show_run_button()
    except FileNotFoundError:
        st.error("❌ score_log.csv לא נמצא — הרץ ניתוח ראשוני.")
        _show_run_button(label="▶️ הרץ ניתוח ראשוני")
        return

    # ─── תחזית ML ───────────────────────────────────────────────
    st.subheader("🧠 תחזית בינה מלאכותית")
    col1, col2 = st.columns(2)
    col1.metric("📍 המלצת AI יומית", _load_ml_prediction())
    col2.metric("🎯 דיוק מצטבר", _load_model_accuracy())

    with st.expander("🧠 ניהול חיזוי יומי"):
        if st.button("📊 בדוק הצלחת חיזוי"):
            if datetime.now().hour >= 20:
                try:
                    result = subprocess.run(
                        ["python", str(_ROOT / "scores_news" / "ml_model" / "performance_tracker.py")],
                        capture_output=True, text=True
                    )
                    st.success("📈 הושוותה בהצלחה!")
                    st.code(result.stdout)
                except Exception as e:
                    st.error(f"שגיאה: {e}")
            else:
                st.warning("⏳ ניתן לבדוק רק לאחר השעה 20:00")

    # ─── ציוני הניתוח ───────────────────────────────────────────
    try:
        df = _load_score_log()
        last_day = df.iloc[-1]
        score_columns = [col for col in df.columns if col.endswith("_score")]

        st.subheader(f"📆 {last_day['date'].date()} — החלטה יומית")
        col1, col2 = st.columns(2)
        col1.metric("🎯 ציון משוקלל", last_day.get("final_score", "—"))
        col2.metric("📌 המלצה", last_day.get("bias", "—"))

        st.markdown("### 📊 ציוני קטגוריות")
        cols = st.columns(len(score_columns))
        for i, col_name in enumerate(score_columns):
            score = last_day[col_name]
            color = "🟢" if score >= 60 else "🟡" if score >= 40 else "🔴"
            with cols[i]:
                st.metric(label=col_name.replace("_score", "").capitalize(),
                          value=f"{score:.0f}", delta=color)

        st.markdown("### 📈 מגמות לפי קטגוריות")
        st.line_chart(df.set_index("date")[score_columns], height=250)

        st.markdown("### 🔁 השוואת Final Score ל־MES")
        plot_cols = [c for c in ["final_score", "daily_change_pct"] if c in df.columns]
        if plot_cols:
            st.line_chart(df.set_index("date")[plot_cols], height=250)
        else:
            st.info("📁 daily_change_pct יוצג לאחר הרצת הניתוח")

        st.markdown("### 🌡 Heatmap ציונים אחרונים")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(df.set_index("date")[score_columns].tail(15),
                    cmap="RdYlGn", annot=True, fmt=".0f", linewidths=0.5, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("### 📊 קטגוריות – היום האחרון")
        latest_scores = {c.replace("_score", ""): last_day[c] for c in score_columns}
        st.bar_chart(pd.DataFrame.from_dict(latest_scores, orient="index", columns=["score"]))

    except Exception as e:
        st.error(f"❌ שגיאה בטעינת הנתונים: {e}")
