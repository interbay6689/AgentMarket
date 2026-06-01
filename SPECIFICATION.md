# AgentMarket — תוכנית איפיון מלאה
## Multi-Agent AI Trading System v2.0

> **מסמך זה מגדיר את ארכיטקטורת המערכת המלאה, שילוב 5 אסטרטגיות מסחר מובילות, ומערך סוכני AI עם שליטה מלאה בפרמטרי המערכת.**

---

## 1. סקירת המערכת הנוכחית

### 1.1 שכבות המערכת

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT DASHBOARD                       │
│  Signals │ Dashboard │ NQ Analysis │ Learning │ Journal │ System │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                  SIGNAL ENGINE (75% Tech / 25% Fund)         │
│  HTF Bias(20) + Zone(20) + Delta(15) + Session(10) +         │
│  Stacked(10) + Sentiment(15) + ML(10) = 100 pts              │
└────────────┬───────────────────────────┬────────────────────┘
             │                           │
┌────────────▼──────────┐   ┌────────────▼──────────────────┐
│  TECHNICAL ENGINE      │   │  FUNDAMENTAL ENGINE            │
│  nq_data.py           │   │  scores_news/cat_scores/       │
│  nq_calculations.py   │   │  sentiment, macro, bonds,      │
│  (Daily/1H/15m/5m)    │   │  vix, sectors, mes             │
│  FVG, OB, BOS, Delta  │   │  → final_score.py (weighted)   │
│  Zones, ATR Stop      │   │  → ML RandomForest prediction  │
└────────────────────────┘   └───────────────────────────────┘
             │
┌────────────▼──────────────────────────────────────────────┐
│  TRADE TRACKER                                              │
│  log_trade → evaluate_open → factor_win_rates → equity_curve│
└─────────────────────────────────────────────────────────────┘
```

### 1.2 קבצי מפתח

| קובץ | תפקיד | ניתן לשינוי ע"י AI |
|------|--------|-------------------|
| `scores_news/config/weights.json` | משקלות 6 קטגוריות פנדמנטל | ✅ |
| `scores_news/config/thresholds.json` | סף LONG/SHORT | ✅ |
| `market_dashboard/config.yaml` | הגדרות dashboard | ✅ |
| `market_dashboard/logs/trade_settings.json` | הגדרות מעקב עסקאות | ✅ |
| `market_dashboard/logs/trade_log.csv` | לוג עסקאות | 📖 קריאה בלבד |
| `scores_news/config/score_log.csv` | לוג ציונים יומי | 📖 קריאה בלבד |
| `scores_news/ml_model/model.pkl` | מודל ML | 🔄 רה-אימון בלבד |
| `scores_news/config/sources.yaml` | מקורות RSS | ✅ |

---

## 2. חמש אסטרטגיות מסחר מובילות — שילוב במערכת

### אסטרטגיה 1: ICT (Inner Circle Trader) — שילוב 90%

**מה קיים כבר:**
- ✅ Fair Value Gaps (FVG) — bullish/bearish
- ✅ Order Blocks (OB) — daily + 15m
- ✅ BOS / MSS (Break of Structure / Market Structure Shift)
- ✅ Killzones (London, NY AM, NY PM)
- ✅ AMD Cycle (Accumulation, Manipulation, Distribution)
- ✅ Judas Swing detector
- ✅ Premium / Discount classification
- ✅ Opening Range

**מה חסר לשלב:**

| פיצ'ר | לוגיקה | שילוב בסיגנל |
|-------|--------|--------------|
| **EQH/EQL** — Equal Highs/Lows | חיפוש 2+ highs/lows ב-tolerance של 3pt | +8 pts אם מחיר קרוב ל-EQH/EQL |
| **Breaker Blocks** | OB שהתבטל (price traded through) → flip לכיוון הפוך | +6 pts כשמחיר חוזר ל-Breaker |
| **Optimal Trade Entry (OTE)** | Fibonacci 61.8%–78.6% של impulse wave האחרון | +10 pts אם כניסה ב-OTE zone |
| **ICT Macros** | חלונות 2-דקות: 09:50-10:10, 10:50-11:10, 13:50-14:10 ET | x1.2 multiplier לסיגנל בחלון |
| **Silver Bullet** | 03:00-04:00 AM ET window — FVG fill trade | סיגנל נפרד בחלון זה |

**פרמטרים לשינוי ע"י AI:**
```json
"ict_config": {
  "ote_fib_lo": 0.618,
  "ote_fib_hi": 0.786,
  "eqh_eql_tolerance_pts": 3.0,
  "macro_window_multiplier": 1.2,
  "silver_bullet_enabled": true
}
```

---

### אסטרטגיה 2: VWAP Trading — אינדיקטור מוסדי

**לוגיקה:**
VWAP = נקודת ייחוס מוסדית. מחיר מעל VWAP = bias קנייה, מתחת = מכירה.
Band 1σ/2σ = אזורי extreme deviation לreversion trades.

**שילוב במערכת:**

```
VWAP Components:
├── Standard VWAP (daily, resets at 00:00 ET)
├── Anchored VWAP (from: prev day close / weekly open / major swing)
├── VWAP Bands ±1σ, ±2σ
└── VWAP Slope (trend direction)

Signal Integration:
├── Price above VWAP + bullish HTF bias → +8 pts LONG
├── Price below VWAP + bearish HTF bias → +8 pts SHORT
├── Price at VWAP band 2σ (extreme) → +12 pts reversion
└── VWAP slope direction agrees → +5 pts confirmation
```

**חישוב:**
```python
# נוסיף ל-nq_calculations.py
def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * df["volume"]).cumsum() / df["volume"].cumsum()

def calculate_vwap_bands(df: pd.DataFrame, n_std: float = 1.0) -> tuple:
    vwap = calculate_vwap(df)
    deviation = (df["close"] - vwap).rolling(20).std()
    return vwap + n_std * deviation, vwap - n_std * deviation
```

**פרמטרים:**
```json
"vwap_config": {
  "enabled": true,
  "band1_weight_pts": 8,
  "band2_weight_pts": 12,
  "anchor_points": ["prev_close", "weekly_open"],
  "slope_confirmation_pts": 5
}
```

---

### אסטרטגיה 3: Quantitative Order Flow — מדד עוצמת מחויבות

**לוגיקה:**
מעבר מ-"יש delta bullish" ל-"כמה חזק/מובהק ה-delta?" עם z-score סטטיסטי.

**שילוב:**

| מדד | חישוב | משמעות |
|-----|--------|---------|
| **Delta Z-Score** | (delta - mean20) / std20 | Z > 2 = קנייה חריגה |
| **Buy/Sell Ratio** | buy_volume / sell_volume | >1.5 = dominant buyers |
| **Volume Anomaly** | current_vol / avg_vol_5sessions | >2.0 = institutional activity |
| **Delta Exhaustion** | cumulative delta שינוי כיוון אחרי extreme | סיגנל reversal |

```python
# חדש ב-nq_calculations.py
def delta_zscore(df: pd.DataFrame, window: int = 20) -> pd.Series:
    delta = approximate_delta(df)
    return (delta - delta.rolling(window).mean()) / delta.rolling(window).std()

def volume_anomaly_score(df: pd.DataFrame) -> float:
    if len(df) < 5:
        return 1.0
    recent_vol = df["volume"].tail(3).mean()
    hist_vol   = df["volume"].tail(30).mean()
    return round(recent_vol / hist_vol, 2) if hist_vol > 0 else 1.0
```

**שילוב בסיגנל:** מחליף את ה"Order Flow Delta (15 pts)" הפשוט בציון מורכב יותר (0-20 pts) לפי z-score + anomaly.

---

### אסטרטגיה 4: Statistical Edge — יתרון סטטיסטי מצטבר

**לוגיקה:** שוק הוא חוזרת — ימים מסוימים, שעות מסוימות, ומצבים מסוימים יש להם הסתברות statistically significant.

**מדדים לשלב:**

| מדד | כיצד מחשבים | שימוש בסיגנל |
|-----|------------|-------------|
| **Day-of-Week Bias** | % ימי עלייה לפי יום שבוע (NQ) | Monday short bias, Friday high volatility |
| **Gap Fill Probability** | % גפים שנסגרים ב-RTH | Gap fill > 75% = target |
| **ORB Statistics** | % breakouts מ-Opening Range שמתקיימים | מגדיר אמינות OR break |
| **Session Continuation Rate** | % sessions שמסיימים בכיוון הפתיחה | Session bias multiplier |
| **ATR Quartile** | אתמול ATR rank ב-60 יום | Q4 ATR = היזהר ממוסן |

```python
# חדש ב-nq_calculations.py
def day_of_week_bias(df_daily: pd.DataFrame) -> dict:
    """Returns win% per day-of-week for NQ long trades."""
    df = df_daily.copy()
    df["dow"]    = pd.to_datetime(df["date"]).dt.dayofweek
    df["pct"]    = df["close"].pct_change()
    df["is_up"]  = (df["pct"] > 0).astype(int)
    return df.groupby("dow")["is_up"].mean().to_dict()  # 0=Mon ... 4=Fri

def gap_fill_stats(df_daily: pd.DataFrame, lookback: int = 60) -> dict:
    """Probability that today's gap fills during RTH."""
    df = df_daily.tail(lookback).copy()
    df["gap"]       = df["open"] - df["close"].shift(1)
    df["next_low"]  = df["low"].shift(-1)
    df["next_high"] = df["high"].shift(-1)
    gap_up   = df[df["gap"] > 0]
    gap_down = df[df["gap"] < 0]
    fill_up   = (gap_up["next_low"]  <= gap_up["close"].shift(1)).mean()
    fill_down = (gap_down["next_high"] >= gap_down["close"].shift(1)).mean()
    return {"gap_up_fill_pct": round(fill_up * 100, 1),
            "gap_down_fill_pct": round(fill_down * 100, 1)}
```

---

### אסטרטגיה 5: Wyckoff Method — מחזורי מוסד

**לוגיקה:** מחזור Wyckoff: Accumulation → Markup → Distribution → Markdown.
זיהוי Phase מאפשר לסחור עם הכסף החכם.

```
Wyckoff Accumulation:              Wyckoff Distribution:
├── PS  (Preliminary Support)      ├── PSY (Preliminary Supply)
├── SC  (Selling Climax)           ├── BC  (Buying Climax)
├── AR  (Automatic Rally)          ├── AR  (Automatic Reaction)
├── ST  (Secondary Test)           ├── UT  (Upthrust)
├── Spring (שבירה מזויפת לכיוון מטה)├── UTAD (Upthrust After Distribution)
└── LPS + SOS (Entry)              └── LPSY + SOW (Entry)
```

**שילוב:**
```python
def detect_wyckoff_phase(df_daily: pd.DataFrame, df_volume: pd.Series) -> dict:
    """
    Identifies current Wyckoff phase based on:
    - Price range compression (Accumulation/Distribution range)
    - Volume characteristics (climax vs dry-up)
    - Swing structure vs range boundaries
    Returns: phase, bias, confidence
    """
    ...
```

**שילוב בסיגנל:** +15 pts אם Wyckoff phase מאשר כיוון (Accumulation → LONG, Distribution → SHORT).

---

## 3. ארכיטקטורת Multi-Agent AI

### 3.1 תרשים כולל

```
                    ┌─────────────────────────────┐
                    │    ORCHESTRATOR AGENT         │
                    │  claude-sonnet-4-6            │
                    │  (מנהל, מתזמן, מחליט)         │
                    └──┬────┬────┬────┬────────────┘
                       │    │    │    │
          ┌────────────┘    │    │    └─────────────┐
          │                 │    │                   │
    ┌─────▼──────┐   ┌──────▼──┐ │   ┌─────────────▼──┐
    │  SIGNAL    │   │INDICATORS│ │   │   ANALYSIS     │
    │  AGENT     │   │ AGENT    │ │   │   AGENT        │
    │haiku-4-5   │   │haiku-4-5 │ │   │ sonnet-4-6     │
    │כל 5 דקות  │   │כל 15 דק' │ │   │ post-trade     │
    └─────┬──────┘   └──────┬──┘ │   └─────────┬──────┘
          │                 │    │              │
          └────────────┐    │    │   ┌──────────┘
                       │    │    │   │
                    ┌──▼────▼────▼───▼───────────────┐
                    │      CONFIG OPTIMIZER AGENT      │
                    │      claude-sonnet-4-6           │
                    │      (adaptive thinking)         │
                    │                                  │
                    │  📖 קורא: כל קבצי המערכת        │
                    │  ✏️  כותב: weights, thresholds,  │
                    │           config, settings       │
                    │  🔄 מאמן: ML model              │
                    │  📋 מתעד: כל שינוי + reasoning  │
                    └──────────────────────────────────┘
```

---

### 3.2 Signal Agent — סוכן הסיגנל

**מטרה:** לצלב את כל מקורות הנתונים ולהחזיר המלצת עסקה מנומקת.

**תזמון:** כל 5 דקות בשעות מסחר (14:30–23:00 IL).

**Tools:**
```python
@tool
def read_current_price() -> dict:
    """מחיר NQ נוכחי, session, 5m OHLCV אחרון"""

@tool
def read_technical_state() -> dict:
    """HTF bias, zone scores, delta, FVGs, OBs, VWAP position,
       Wyckoff phase, EQH/EQL proximity, OTE zone"""

@tool
def read_fundamental_state() -> dict:
    """final_score, ML prediction, category breakdown,
       score trend (עולה/יורד 3 ימים אחרונים)"""

@tool
def read_trade_log_summary() -> dict:
    """5 עסקאות אחרונות, win rate שבועי, factor performance"""

@tool
def read_statistical_edge() -> dict:
    """day_of_week_bias, gap_status, ORB_stats, ATR_quartile"""

@tool
def write_trade_proposal(proposal: dict) -> None:
    """כותב הצעת עסקה ל-agent_proposals.json"""
```

**System Prompt (עם prompt caching):**
```
You are a professional NQ futures day trader. Your job is to analyze
current market conditions and propose a trade ONLY when there is genuine
confluence across multiple timeframes and data sources.

Trading style: ICT/SMC with quantitative order flow confirmation.
Risk rules: Never propose a trade with R:R < 1.5 or zone_score < 6.
Session rules: Only during NY Morning (IL 17:00-20:00), PM Session
               (21:30-22:30), Power Hour (22:30-23:00).

Be conservative. A "no trade" is a valid output.
Always return structured JSON: {direction, entry, stop, target,
confidence, reasoning, key_factors[]}
```

---

### 3.3 Indicators Agent — סוכן מעקב אינדיקטורים

**מטרה:** מפקח רציף על אירועי שוק — לא מחליט, רק מדווח.

**תזמון:** כל 15 דקות (או real-time trigger לפי events).

**Tools:**
```python
@tool
def check_key_level_breaks() -> list:
    """בדיקה אם מחיר שבר PDH/PDL/POC/Weekly H-L"""

@tool
def check_vwap_deviation() -> dict:
    """מרחק ממחיר ל-VWAP ב-σ, האם פרץ band"""

@tool
def check_delta_divergence() -> dict:
    """האם יש divergence בין price ל-cumulative delta"""

@tool
def check_liquidity_sweep() -> dict:
    """האם EQH/EQL נפרץ ב-5 דקות האחרונות (Judas/Stop Hunt)"""

@tool
def check_session_transition() -> dict:
    """האם עברנו ל-session חדש, מה ה-risk rating"""

@tool
def alert_orchestrator(event_type: str, data: dict) -> None:
    """שולח alert ל-Orchestrator"""
```

**פלט:** JSON עם רשימת events מסווגים:
```json
{
  "events": [
    {"type": "key_level_break", "level": "PDH", "price": 21430, "direction": "bullish"},
    {"type": "delta_divergence", "severity": "moderate", "bars": 4}
  ],
  "action_required": true,
  "urgency": "medium"
}
```

---

### 3.4 Analysis Agent — סוכן ניתוח

**מטרה:** לאחר כל עסקה סגורה — ניתוח מעמיק של מה עבד ומה לא.

**תזמון:** Triggered אחרי כל שינוי outcome ב-trade_log.csv.

**Tools:**
```python
@tool
def read_closed_trade(trade_id: str) -> dict:
    """כל פרטי העסקה + outcome + factor breakdown"""

@tool
def read_similar_historical_trades(trade: dict, n: int = 10) -> list:
    """עסקאות דומות מה-log (same direction, similar zone/session/HTF)"""

@tool
def read_market_context_at_time(timestamp: str) -> dict:
    """מה היה המצב הטכני/פנדמנטלי בזמן הכניסה"""

@tool
def read_factor_win_rates() -> pd.DataFrame:
    """win rate per factor bucket מ-trade_log"""

@tool
def write_insight(insight: dict) -> None:
    """שומר תובנה ל-insights_log.json"""

@tool
def notify_config_optimizer(recommendation: dict) -> None:
    """שולח המלצת שינוי ל-Config Optimizer"""
```

**System Prompt:**
```
You are a trading performance analyst. After each completed trade,
perform a root-cause analysis:

For WINS: What specifically worked? Which factors were decisive?
For LOSSES: What failed? Was the entry thesis wrong, or correct but
            stopped out by noise? Could a wider/ATR-based stop have saved it?

Output a structured insight:
{
  "trade_id": "...",
  "outcome": "WIN/LOSS/PARTIAL",
  "root_cause": "...",
  "factor_quality": {factor: "helped/neutral/hurt"},
  "entry_quality": "good/premature/late",
  "exit_quality": "...",
  "recommendation": {
    "type": "adjust_weight/adjust_threshold/adjust_timing",
    "parameter": "...",
    "current_value": ...,
    "suggested_value": ...,
    "confidence": "low/medium/high",
    "min_trades_required": 20
  }
}
```

---

### 3.5 Orchestrator Agent — מנהל המערכת

**מטרה:** לתאם בין כל הסוכנים, להחליט מתי להפעיל כל אחד, ומה לעשות עם הפלט שלהם.

**תזמון:** Event-driven + כל 5 דקות.

**Tools:** כל ה-tools של כל הסוכנים + יכולת לקרוא ל-sub-agents.

**לוגיקת קבלת החלטות:**
```python
# Orchestrator decision logic:
if indicators_agent.urgency == "high":
    → activate Signal Agent immediately
    → if signal >= 70 pts: propose trade

if signal_agent.proposal.confidence >= 75:
    → validate with current market_context
    → if validated: log_trade() + notify_user

if new_closed_trade detected:
    → activate Analysis Agent
    → if insight.confidence == "high" and insight.min_trades met:
        → notify Config Optimizer

every 24h (09:00 IL):
    → run daily health check
    → update statistical edge tables
    → if ML model performance < 55%: trigger retrain
```

---

## 4. Config Optimizer Agent — הסוכן החושב

### 4.1 מהות

זהו הסוכן החשוב ביותר במערכת. הוא לא סוחר — הוא **משפר את הסוחרים האחרים**.
מקבל תובנות מה-Analysis Agent, בודק patterns בנתונים היסטוריים, ומחליט בזהירות **מה לשנות במערכת**.

### 4.2 ארכיטקטורה

```python
model = "claude-sonnet-4-6"
thinking = {"type": "adaptive"}  # חשיבה עמוקה לפני כל שינוי
```

### 4.3 כל ה-Tools שלו

**קריאה (📖):**
```python
read_all_configs()          # weights.json, thresholds.json, config.yaml, trade_settings.json
read_trade_log()            # כל ה-trade log
read_score_log()            # ציונים יומיים
read_factor_win_rates()     # win rate per factor
read_ml_performance()       # דיוק ML לאורך זמן
read_insights_log()         # כל התובנות מ-Analysis Agent
read_change_history()       # לוג כל שינוי קודם
read_statistical_edge()     # day/week patterns
```

**כתיבה (✏️) — עם validation:**
```python
update_fundamental_weights(new_weights: dict) -> bool
update_signal_thresholds(new_thresholds: dict) -> bool
update_entry_filters(min_confidence, min_zone, min_rr) -> bool
update_ict_config(param, value) -> bool
update_vwap_config(param, value) -> bool
update_session_weights(session_name, multiplier) -> bool
update_rss_sources(add=[], remove=[]) -> bool
trigger_ml_retrain() -> bool
```

**בקרה (🔒):**
```python
dry_run(proposed_changes: list) -> dict    # סימולציה ללא שמירה
revert_last_change() -> bool               # ביטול שינוי אחרון
revert_to_defaults() -> bool              # חזרה לברירת מחדל
log_change(param, old, new, reasoning)    # תמיד מתועד
```

### 4.4 System Prompt

```
You are the AgentMarket Configuration Optimizer. Your role is to
carefully tune trading system parameters based on evidence from
completed trades and market patterns.

CORE PRINCIPLES:
1. Evidence over intuition — only change if data supports it
2. Minimum samples — never change a weight without 20+ relevant trades
3. Small steps — maximum 10% change per parameter per session
4. Document everything — log every change with full reasoning
5. Conservative bias — when uncertain, do nothing
6. Never disable safety — min R:R cannot go below 1.2, min confidence below 55

WHAT YOU CAN CHANGE:
- fundamental_weights: 6 category weights (must sum to 1.0)
- signal_thresholds: LONG/SHORT score thresholds
- entry_filters: min_confidence, min_zone_score, min_rr
- ict_config: OTE levels, EQH tolerance, macro multipliers
- vwap_config: band weights, slope confirmation
- session_multipliers: per-session signal weight
- rss_sources: add/remove news feeds based on sentiment accuracy

WHAT YOU CANNOT CHANGE:
- Core ATR calculation logic
- Historical trade records
- ML model architecture (only retrain)

OUTPUT FORMAT:
Always respond with:
{
  "analysis": "...",
  "proposed_changes": [{param, current, proposed, reasoning, confidence, evidence}],
  "dry_run_results": {...},
  "decision": "apply/defer/reject",
  "review_in_trades": N
}
```

### 4.5 Safety Guardrails — גדרות בטיחות

```python
PARAMETER_LIMITS = {
    # Fundamental weights — כל משקל בין min-max, סכום חייב = 1.0
    "weights.sentiment":    (0.10, 0.45),
    "weights.macro":        (0.05, 0.30),
    "weights.bonds":        (0.05, 0.20),
    "weights.vix":          (0.05, 0.25),
    "weights.sectors":      (0.02, 0.15),
    "weights.mes":          (0.10, 0.40),

    # Signal thresholds
    "signal.min_confidence": (55, 85),
    "signal.min_zone_score": (4, 9),
    "signal.min_rr":         (1.2, 3.0),

    # Change rate limits
    "max_change_pct_per_session": 0.10,  # 10% מקסימום
    "min_trades_before_change": 20,
    "cooldown_hours_between_changes": 24,
}

def validate_change(param: str, new_value: float) -> tuple[bool, str]:
    """Returns (is_valid, reason)"""
    limits = PARAMETER_LIMITS.get(param)
    if not limits:
        return False, f"Unknown parameter: {param}"
    lo, hi = limits
    if not (lo <= new_value <= hi):
        return False, f"{param} must be between {lo} and {hi}, got {new_value}"
    return True, "OK"
```

### 4.6 Change History Log

כל שינוי נרשם ל-`market_dashboard/logs/config_changes.log`:
```
2026-06-01 09:15 | CONFIG_OPTIMIZER | weights.sentiment: 0.30 → 0.35
  Reasoning: Sentiment factor showed 71% win rate over last 28 trades
             vs system average of 62%. Macro factor underperformed at 48%.
  Evidence: factor_win_rates[sentiment=0.71, n=28], [macro=0.48, n=25]
  Confidence: high | Approved: auto | Reviewer: n/a
  Dry-run: +2.3% expected improvement in signal quality
```

---

## 5. זרימת נתונים מלאה — Agent Flow

```
[כל 5 דקות בשעות מסחר]
     │
     ▼
Indicators Agent ──────────────────────────────────┐
  → check_key_level_breaks()                        │
  → check_vwap_deviation()                          │ no events
  → check_delta_divergence()                        ├──→ sleep 15m
  → check_liquidity_sweep()                         │
     │ events detected                              │
     ▼
Orchestrator Agent
  → קיבל events מ-Indicators
  → מחליט: להפעיל Signal Agent?
     │ yes (urgency >= medium OR scheduled 5m)
     ▼
Signal Agent
  → read_current_price()
  → read_technical_state()      [HTF + zones + ICT + VWAP + Wyckoff]
  → read_fundamental_state()    [final_score + ML + categories]
  → read_statistical_edge()     [DoW + Gap + ORB stats]
  → read_trade_log_summary()    [recent performance context]
  → Claude generates proposal
     │
     ▼
Orchestrator validates:
  → confidence >= 70? zone_score >= 6? session quality?
  → R:R >= 1.5? not in Judas window?
     │ all green
     ▼
  log_trade() → trade_log.csv
  notify_user() → st.toast + alerts_log

[לאחר 2-4 שעות]
     ▼
evaluate_open_trades(df_price) → WIN / LOSS / PARTIAL
     │
     ▼
Analysis Agent triggered
  → ניתוח root cause
  → comparison to similar trades
  → write_insight(insights_log.json)
     │ confidence == "high" AND min_trades met
     ▼
Config Optimizer Agent
  → reads: insights + trade_log + current_configs + change_history
  → adaptive thinking (חשיבה עמוקה)
  → dry_run(proposed_changes)
  → if expected improvement > threshold:
       apply changes
       log_change()
  → schedule next review in N trades
```

---

## 6. מבנה קבצים חדש — שמתווסף למערכת

```
AgentMarket/
├── market_dashboard/
│   ├── agents/                          ← NEW
│   │   ├── __init__.py
│   │   ├── orchestrator.py              ← Orchestrator Agent
│   │   ├── signal_agent.py              ← Signal Agent
│   │   ├── indicators_agent.py          ← Indicators Agent
│   │   ├── analysis_agent.py            ← Analysis Agent
│   │   ├── config_optimizer.py          ← Config Optimizer Agent
│   │   ├── agent_tools.py               ← כל ה-tools
│   │   ├── agent_runner.py              ← scheduler + APScheduler
│   │   └── agent_safety.py             ← validation + guardrails
│   ├── logs/
│   │   ├── trade_log.csv                (קיים)
│   │   ├── trade_settings.json          (קיים)
│   │   ├── insights_log.json            ← NEW: Analysis Agent output
│   │   ├── agent_proposals.json         ← NEW: Signal Agent proposals
│   │   ├── config_changes.log           ← NEW: Config Optimizer history
│   │   └── agent_activity.log           ← NEW: כל פעילות הסוכנים
│   └── pages/
│       └── agents_dashboard.py          ← NEW: עמוד ניהול סוכנים
└── market_dashboard/modules/
    └── nq_calculations.py               ← להוסיף: VWAP, EQH/EQL, OTE, Wyckoff
```

---

## 7. עמוד Agent Dashboard — ניהול שליטה

### טאבים:

**Tab 1: Agent Status**
```
┌──────────────────────────────────────────────────────────┐
│  Orchestrator  🟢 Active    Last run: 2 min ago           │
│  Signal        🟢 Active    Last proposal: 14 min ago     │
│  Indicators    🟢 Active    Last alert: 47 min ago        │
│  Analysis      🟡 Waiting   Next trigger: on trade close  │
│  Config Opt.   🔵 Idle      Last change: 3 days ago       │
├──────────────────────────────────────────────────────────┤
│  [Start All] [Stop All] [Force Signal] [Force Analysis]  │
└──────────────────────────────────────────────────────────┘
```

**Tab 2: Agent Log**
- Activity log עם timestamps לכל פעולה של כל סוכן

**Tab 3: Config Changes History**
- טבלה: מתי / מה שינוי / למה / תוצאה

**Tab 4: Current Config (Visualized)**
- Bar chart של משקלות נוכחיות vs ברירת מחדל
- Score כרגע לפי כל component

**Tab 5: Manual Override**
- אפשרות לסגור כל סוכן
- Dry-run Config Optimizer ידנית
- Revert לכל שינוי קודם

---

## 8. עלויות מפורטות

### תרחיש מסחר רגיל (יום אחד, 8 שעות מסחר)

| סוכן | מודל | טוקנים/ריצה | ריצות/יום | עלות/יום |
|------|------|------------|-----------|---------|
| Signal Agent | haiku-4-5 | 1,500 | 96 | $0.14 |
| Indicators Agent | haiku-4-5 | 600 | 32 | $0.02 |
| Orchestrator | sonnet-4-6 | 800 | 50 | $0.04 |
| Analysis Agent | sonnet-4-6 | 3,500 | 4 | $0.05 |
| Config Optimizer | sonnet-4-6 | 5,000 | 1 | $0.08 |
| **סה"כ בלי caching** | | | | **~$0.33/יום** |
| **עם prompt caching (85%)** | | | | **~$0.07/יום** |

### עלות חודשית (22 ימי מסחר)

| תרחיש | עלות/חודש |
|--------|----------|
| פעיל מלא + caching | **~$1.5–3** |
| פעיל מלא ללא caching | **~$7–10** |
| + Opus לניתוחים מורכבים (שבועי) | **+$3–5/חודש** |

> ✅ **Bottom line: $5–15/חודש** לתפעול מלא של מערכת AI המנהלת עסקאות, מנתחת ביצועים, ומייעלת את עצמה באופן אוטומטי.

---

## 9. Roadmap מימוש — 4 Sprints

### Sprint 1: Infrastructure (שבוע 1)
```
□ agent_tools.py — כל ה-tools עם interfaces ברורים
□ agent_safety.py — validation + guardrails + change_history
□ agent_runner.py — APScheduler + event system
□ VWAP + EQH/EQL ב-nq_calculations.py
□ agents_dashboard.py — Agent Status tab בלבד
```

### Sprint 2: Signal + Indicators (שבוע 2)
```
□ indicators_agent.py — key level breaks + vwap + divergence
□ signal_agent.py — tool use loop + structured proposal
□ orchestrator.py — coordination logic
□ שילוב proposals עם trade_tracker.py הקיים
□ agents_dashboard.py — Agent Log + proposals view
```

### Sprint 3: Analysis Agent (שבוע 3)
```
□ analysis_agent.py — root cause + structured insights
□ insights_log.json schema + writer
□ trigger mechanism מ-trade_tracker
□ Factor win rates → insights pipeline
□ agents_dashboard.py — Insights tab
```

### Sprint 4: Config Optimizer (שבוע 4)
```
□ config_optimizer.py — adaptive thinking + dry_run + apply
□ כל write tools עם validation
□ Change history log + revert capability
□ agents_dashboard.py — Config Changes + Manual Override tabs
□ Integration test: full loop מ-trade close עד config change
```

---

## 10. עקרונות עיצוב — מה AI לא יעשה לעולם

```
❌ לא יבצע עסקה אמיתית — רק יציע (execution נשאר ידני)
❌ לא ישנה core calculation logic — רק parameters
❌ לא יוריד min_rr מתחת ל-1.2
❌ לא ישנה 2 פרמטרים ביחד בלי הפסקה של 24 שעות ביניהם
❌ לא יפעל עם פחות מ-20 עסקאות closed per factor
❌ לא ימחק trade history
❌ לא יבצע שינוי ללא dry_run ולוג מפורט
```

---

*מסמך זה מתעדכן אוטומטית כחלק מ-Config Optimizer — כל שינוי פרמטר מתועד עם timestamp ו-reasoning.*
