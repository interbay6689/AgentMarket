"""
ab_tracking.py — A/B Performance Tracking page.

Visualises signal accuracy across experiments so you can verify whether
a config change actually improved signal quality over 14 rolling days.
"""
import sys
from pathlib import Path

_MD   = Path(__file__).resolve().parents[1]
_ROOT = _MD.parent
for p in (str(_MD), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from modules import ab_tracker
from modules.nq_data import get_todays_nq_data


@st.cache_data(ttl=300)
def _get_5m():
    try:
        return get_todays_nq_data()
    except Exception:
        return None


def _oc(outcome: str) -> str:
    return {"WIN": "#00c853", "LOSS": "#d50000", "NEUTRAL": "#9e9e9e", "PENDING": "#78909c"}.get(outcome, "#9e9e9e")


def _oe(outcome: str) -> str:
    return {"WIN": "✅", "LOSS": "❌", "NEUTRAL": "⚪", "PENDING": "⏳"}.get(outcome, "❓")


def _metrics_row(m: dict, prefix: str = "") -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(f"{prefix}סה״כ",      m.get("total", 0))
    c2.metric(f"{prefix}Win Rate",   f"{m.get('win_rate', 0):.1f}%")
    c3.metric(f"{prefix}Accuracy",   f"{m.get('accuracy', 0):.1f}%")
    c4.metric(f"{prefix}Avg P&L",    f"{m.get('avg_pnl', 0):+.1f} pts")
    c5.metric(f"{prefix}Pending",    m.get("pending", 0))


# ─── Tab: Active experiment ────────────────────────────────────────────────────

def _tab_active(active: dict):
    st.subheader(f"🔬 {active['name']}")
    st.caption(f"פתוח מ: {active.get('created_at','')[:10]}  ·  {active.get('description','')}")

    # Auto-resolve pending outcomes from today's 5m data
    df_5m = _get_5m()
    n_res = ab_tracker.resolve_pending_outcomes(df_5m)
    if n_res > 0:
        st.toast(f"📊 {n_res} תוצאות נפתרו אוטומטית", icon="✅")

    m = ab_tracker.get_experiment_metrics(active["exp_id"], days=14)
    _metrics_row(m)
    st.caption("חלון: 14 ימים אחרונים")

    signals = active.get("signals", [])
    if not signals:
        st.info(
            "אין אותות מוקלטים עדיין. "
            "עבור לעמוד **🎯 Signals** — האות ייוצר ויירשם אוטומטית."
        )
        return

    # Pie chart
    n_res_total = m.get("resolved", 0)
    if n_res_total > 0:
        fig_pie = go.Figure(go.Pie(
            labels=["Win", "Loss", "Neutral"],
            values=[m["wins"], m["losses"], m["neutrals"]],
            marker_colors=["#00c853", "#d50000", "#9e9e9e"],
            hole=0.55,
            textinfo="label+percent",
        ))
        fig_pie.update_layout(
            height=220, template="plotly_dark",
            margin=dict(l=0, r=0, t=16, b=0), showlegend=False,
        )
        pie_col, _ = st.columns([1, 2])
        with pie_col:
            st.plotly_chart(fig_pie, use_container_width=True)

    # Cumulative P&L line
    resolved_sigs = [s for s in signals if s.get("outcome") != "PENDING"]
    if len(resolved_sigs) >= 2:
        cum, running = [], 0.0
        for s in resolved_sigs:
            running += s.get("pnl_points", 0)
            cum.append(running)

        final_color = "#00c853" if cum[-1] >= 0 else "#d50000"
        fig_pnl = go.Figure(go.Scatter(
            x=list(range(1, len(cum) + 1)), y=cum,
            mode="lines+markers",
            line=dict(color=final_color, width=2),
            fill="tozeroy",
            fillcolor=f"rgba({'0,200,83' if cum[-1] >= 0 else '213,0,0'},0.10)",
        ))
        fig_pnl.add_hline(y=0, line_color="#555", line_dash="dash")
        fig_pnl.update_layout(
            height=200, template="plotly_dark",
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Signal #", yaxis_title="Cumulative P&L (pts)",
        )
        st.markdown("**Cumulative P&L**")
        st.plotly_chart(fig_pnl, use_container_width=True)

    # Signal table (most recent first)
    st.markdown("**אותות אחרונים**")
    rows = []
    for s in reversed(signals[-30:]):
        oc = s.get("outcome", "PENDING")
        rows.append({
            "תאריך":      s.get("date", ""),
            "שעה (IL)":  s.get("time_il", ""),
            "כיוון":      s.get("direction", ""),
            "Confidence": f"{s.get('confidence', 0):.0f}%",
            "Session":    s.get("session", ""),
            "Regime":     s.get("regime", ""),
            "תוצאה":      f"{_oe(oc)} {oc}",
            "P&L (pts)":  f"{s.get('pnl_points', 0):+.1f}" if oc != "PENDING" else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ─── Tab: History ─────────────────────────────────────────────────────────────

def _tab_history():
    exps = ab_tracker.list_experiments(30)
    if not exps:
        st.info("אין ניסויים עדיין.")
        return

    for exp in exps:
        tag = "🟢 פעיל" if exp.get("active") else "⚫"
        m   = ab_tracker.get_experiment_metrics(exp["exp_id"])
        with st.expander(
            f"{tag}  {exp['name']}  —  {exp.get('created_at','')[:10]}  "
            f"({m.get('total', 0)} אותות · Win {m.get('win_rate', 0):.0f}%)",
            expanded=bool(exp.get("active")),
        ):
            _metrics_row(m)
            if exp.get("description"):
                st.caption(f"📝 {exp['description']}")


# ─── Tab: Compare ─────────────────────────────────────────────────────────────

def _tab_compare():
    exps = ab_tracker.list_experiments(30)
    if len(exps) < 2:
        st.info("צריך לפחות 2 ניסויים להשוואה.")
        return

    names = [f"{e['name']}  ({e.get('created_at','')[:10]})" for e in exps]

    col1, col2 = st.columns(2)
    with col1:
        idx_a = st.selectbox(
            "ניסוי A — בסיס / לפני",
            range(len(names)),
            index=min(1, len(names) - 1),
            format_func=lambda i: names[i],
        )
    with col2:
        idx_b = st.selectbox(
            "ניסוי B — חדש / אחרי",
            range(len(names)),
            index=0,
            format_func=lambda i: names[i],
        )

    if idx_a == idx_b:
        st.warning("בחר שני ניסויים שונים.")
        return

    days = st.slider("חלון (ימים)", 7, 60, 14)
    comp = ab_tracker.compare_experiments(
        exps[idx_a]["exp_id"], exps[idx_b]["exp_id"], days
    )
    m_a, m_b, delta, verdict = comp["a"], comp["b"], comp["delta"], comp["verdict"]

    # Verdict banner
    vc = "#00c853" if "better ✅" in verdict else "#d50000" if "⚠️" in verdict else "#ffd600"
    st.markdown(
        f"""<div style="background:{vc}20; border-left:4px solid {vc};
                        padding:10px 16px; border-radius:6px; margin:8px 0 16px 0;">
            <b style="color:{vc}; font-size:1.05em;">⚖️ {verdict}</b>
        </div>""",
        unsafe_allow_html=True,
    )

    # Side-by-side metrics
    ca, cb = st.columns(2)
    with ca:
        st.markdown(f"**A — {m_a.get('exp_name','')}**")
        _metrics_row(m_a)
    with cb:
        st.markdown(f"**B — {m_b.get('exp_name','')}**")
        _metrics_row(m_b)

    st.divider()

    # Delta comparison table
    metric_labels = {
        "win_rate":       "Win Rate (%)",
        "accuracy":       "Accuracy (%)",
        "avg_pnl":        "Avg P&L (pts)",
        "avg_conf_wins":  "Avg Confidence on Wins (%)",
    }
    rows = []
    for k, label in metric_labels.items():
        d = delta.get(k, 0)
        arrow = "⬆️" if d > 0 else "⬇️" if d < 0 else "➡️"
        rows.append({
            "מדד":     label,
            "A":       m_a.get(k, 0),
            "B":       m_b.get(k, 0),
            "Δ (B−A)": f"{arrow} {d:+.1f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Delta bar chart
    vals   = [delta.get(k, 0) for k in metric_labels]
    labels = list(metric_labels.values())
    fig    = go.Figure(go.Bar(
        x=labels, y=vals,
        marker_color=["#00c853" if v > 0 else "#d50000" for v in vals],
        text=[f"{v:+.1f}" for v in vals],
        textposition="outside",
    ))
    fig.add_hline(y=0, line_color="#555")
    fig.update_layout(
        height=280, template="plotly_dark",
        title="Δ = B − A  (ירוק = שיפור)",
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis_title="Delta",
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── Tab: New experiment ───────────────────────────────────────────────────────

def _tab_new():
    st.markdown("#### יצירת ניסוי חדש")
    st.caption(
        "כל שינוי משמעותי (פרמטרים, אסטרטגיה, הפעלת/כיבוי פיצ'ר) ראוי לניסוי נפרד. "
        "הניסוי הקודם נסגר אוטומטית ומשמר לצורך השוואה."
    )

    with st.form("new_exp_form", clear_on_submit=True):
        name = st.text_input(
            "שם הניסוי *",
            placeholder="למשל: הפעלת Claude category scorer",
        )
        desc = st.text_area(
            "מה שינית?",
            placeholder="תאר את השינוי ומטרתו — יעזור לפרשנות התוצאות בעתיד",
        )
        include_cfg = st.checkbox("שמור snapshot של config.yaml", value=True)
        submitted = st.form_submit_button("➕ התחל ניסוי חדש", type="primary")

        if submitted:
            if not name.strip():
                st.error("שם הניסוי חובה.")
                return
            cfg_snap: dict = {}
            if include_cfg:
                try:
                    from modules.config import load_config
                    cfg_snap = load_config()
                except Exception:
                    pass
            exp_id = ab_tracker.create_experiment(name.strip(), desc.strip(), cfg_snap)
            st.success(f"✅ ניסוי חדש נוצר: `{exp_id}`")
            st.info(
                "עבור לעמוד **🎯 Signals** — האות הבא שייווצר ייוצג אוטומטית. "
                "חזור לטאב **📊 ניסוי פעיל** לראות את ההתקדמות."
            )


# ─── Entry point ─────────────────────────────────────────────────────────────

def app():
    st.title("🧪 A/B Performance Tracking")
    st.caption(
        "מעקב אוטומטי אחר דיוק האותות לפני ואחרי שינויים בתצורה — "
        "תוצאות מחושבות מנתוני NQ 5m (WIN ≥ 0.12% תוך שעה)"
    )

    active = ab_tracker.get_active_experiment()
    if active:
        st.markdown(
            f"""<div style="background:#1a2e4a; border-radius:6px;
                            padding:8px 14px; margin-bottom:8px;">
                🟢 ניסוי פעיל: <b>{active['name']}</b>
                &nbsp;·&nbsp;
                <span style="color:#aaa; font-size:.9em;">
                    מ-{active.get('created_at','')[:10]}
                </span>
            </div>""",
            unsafe_allow_html=True,
        )

    t_active, t_history, t_compare, t_new = st.tabs(
        ["📊 ניסוי פעיל", "📜 היסטוריה", "⚖️ השוואה", "➕ ניסוי חדש"]
    )

    with t_active:
        if active:
            _tab_active(active)
        else:
            st.info("אין ניסוי פעיל. עבור לטאב **➕ ניסוי חדש** כדי להתחיל.")

    with t_history:
        _tab_history()

    with t_compare:
        _tab_compare()

    with t_new:
        _tab_new()
