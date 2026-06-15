import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
import pytz

DAILY_CACHE = Path(__file__).resolve().parents[2] / "scores_news" / "config" / "NQ_data.csv"
HOURLY_CACHE = Path(__file__).resolve().parents[2] / "scores_news" / "config" / "NQ_hourly_cache.csv"

ET_TZ = pytz.timezone("America/New_York")
IL_TZ = pytz.timezone("Asia/Jerusalem")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    return df


def fetch_nq_daily(start_date: str = "2024-01-01", end_date: str = None) -> pd.DataFrame:
    end = end_date or datetime.now().strftime("%Y-%m-%d")
    df = yf.download("NQ=F", start=start_date, end=end, interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        return pd.DataFrame()
    df = _normalize_columns(df)
    df = df.reset_index()
    df = df.rename(columns={"date": "date", "index": "date", "Date": "date", "Datetime": "date"})
    if "date" not in df.columns and df.columns[0] in ("Date", "Datetime"):
        df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.date
    return df[["date", "open", "high", "low", "close", "volume"]].dropna()


def load_nq_daily_cache() -> pd.DataFrame:
    if DAILY_CACHE.exists():
        existing = pd.read_csv(DAILY_CACHE, parse_dates=["date"])
        existing["date"] = pd.to_datetime(existing["date"]).dt.date
        last_date = existing["date"].max()
        start = (datetime.combine(last_date, datetime.min.time()) + timedelta(days=1)).strftime("%Y-%m-%d")
        fresh = fetch_nq_daily(start_date=start)
        if not fresh.empty:
            combined = pd.concat([existing, fresh], ignore_index=True)
            combined = combined.drop_duplicates(subset=["date"]).sort_values("date")
            combined.to_csv(DAILY_CACHE, index=False)
            return combined
        return existing.sort_values("date")
    else:
        df = fetch_nq_daily(start_date="2024-01-01")
        if not df.empty:
            DAILY_CACHE.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(DAILY_CACHE, index=False)
        return df


def fetch_nq_hourly(days_back: int = 60) -> pd.DataFrame:
    df = yf.download("NQ=F", period=f"{days_back}d", interval="1h", auto_adjust=True, progress=False)
    if df.empty:
        return pd.DataFrame()
    df = _normalize_columns(df)
    df = df.reset_index()
    col0 = df.columns[0]
    df = df.rename(columns={col0: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["datetime_et"] = df["datetime"].dt.tz_convert(ET_TZ)
    df["datetime_il"] = df["datetime"].dt.tz_convert(IL_TZ)
    return df[["datetime", "datetime_et", "datetime_il", "open", "high", "low", "close", "volume"]].dropna()


def load_nq_hourly_cache() -> pd.DataFrame:
    fresh = fetch_nq_hourly(days_back=60)
    if fresh.empty:
        if HOURLY_CACHE.exists():
            return pd.read_csv(HOURLY_CACHE, parse_dates=["datetime"])
        return pd.DataFrame()
    if HOURLY_CACHE.exists():
        existing = pd.read_csv(HOURLY_CACHE, parse_dates=["datetime"])
        combined = pd.concat([existing, fresh], ignore_index=True)
        combined["datetime"] = pd.to_datetime(combined["datetime"], utc=True)
        combined = combined.drop_duplicates(subset=["datetime"]).sort_values("datetime")
    else:
        fresh["datetime"] = pd.to_datetime(fresh["datetime"], utc=True)
        combined = fresh.sort_values("datetime")
    HOURLY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(HOURLY_CACHE, index=False)
    return combined


def _fetch_intraday(interval: str, period: str = "1d") -> pd.DataFrame:
    """Base intraday fetcher — normalizes columns and converts to ET/IL timezones."""
    df = yf.download("NQ=F", period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        return pd.DataFrame()
    df = _normalize_columns(df)
    df = df.reset_index()
    col0 = df.columns[0]
    df = df.rename(columns={col0: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize("UTC")
    df["datetime_et"] = df["datetime"].dt.tz_convert(ET_TZ)
    df["datetime_il"] = df["datetime"].dt.tz_convert(IL_TZ)
    return df[["datetime", "datetime_et", "datetime_il", "open", "high", "low", "close", "volume"]].dropna()


def get_todays_nq_data() -> pd.DataFrame:
    """5-minute NQ bars — entry / execution timeframe."""
    return _fetch_intraday("5m", period="1d")


def get_nq_1m() -> pd.DataFrame:
    """1-minute NQ bars for today — finest scalping execution timeframe.
    yfinance only provides 7 days of 1m data; we request today only."""
    return _fetch_intraday("1m", period="1d")


def get_nq_3m() -> pd.DataFrame:
    """3-minute NQ bars — resampled from 1m (yfinance has no native 3m).
    Used for scalp entry timing between 1m noise and 5m structure."""
    df_1m = get_nq_1m()
    if df_1m.empty:
        return pd.DataFrame()
    df_1m = df_1m.copy()
    df_1m["datetime"] = pd.to_datetime(df_1m["datetime"])
    df_1m = df_1m.set_index("datetime")
    df_3m = df_1m[["open", "high", "low", "close", "volume"]].resample("3min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna().reset_index()
    df_3m["datetime_et"] = df_3m["datetime"].dt.tz_convert(ET_TZ)
    df_3m["datetime_il"] = df_3m["datetime"].dt.tz_convert(IL_TZ)
    return df_3m[["datetime", "datetime_et", "datetime_il", "open", "high", "low", "close", "volume"]]


def fetch_nq_15m(days_back: int = 5) -> pd.DataFrame:
    """15-minute NQ data for MTF analysis (yfinance supports ≤60d for 15m)."""
    df = yf.download("NQ=F", period=f"{days_back}d", interval="15m", auto_adjust=True, progress=False)
    if df.empty:
        return pd.DataFrame()
    df = _normalize_columns(df)
    df = df.reset_index()
    col0 = df.columns[0]
    df = df.rename(columns={col0: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize("UTC")
    df["datetime_et"] = df["datetime"].dt.tz_convert(ET_TZ)
    df["datetime_il"] = df["datetime"].dt.tz_convert(IL_TZ)
    return df[["datetime", "datetime_et", "datetime_il", "open", "high", "low", "close", "volume"]].dropna()


def fetch_correlated_assets() -> dict:
    symbols = {"VIX": "^VIX", "DXY": "DX-Y.NYB", "TNX": "^TNX", "ES": "ES=F", "Gold": "GC=F"}
    result = {}
    for name, sym in symbols.items():
        try:
            df = yf.download(sym, period="5d", interval="1h", auto_adjust=True, progress=False)
            if not df.empty:
                df = _normalize_columns(df)
                df = df.reset_index()
                col0 = df.columns[0]
                df = df.rename(columns={col0: "datetime"})
                df["datetime"] = pd.to_datetime(df["datetime"])
                result[name] = df
        except Exception:
            result[name] = pd.DataFrame()
    return result


def et_to_israel(et_time_str: str, ref_date=None) -> str:
    if ref_date is None:
        ref_date = datetime.now().date()
    try:
        naive = datetime.combine(ref_date, datetime.strptime(et_time_str, "%H:%M").time())
        et_dt = ET_TZ.localize(naive)
        il_dt = et_dt.astimezone(IL_TZ)
        return il_dt.strftime("%H:%M")
    except Exception:
        return ""


def current_session_israel() -> dict:
    now_il = datetime.now(IL_TZ)
    hour_min = now_il.hour * 60 + now_il.minute

    sessions = [
        {"name": "RTH Open",    "il_start": "16:30", "il_end": "17:00", "risk": "high"},
        {"name": "NY Morning",  "il_start": "17:00", "il_end": "20:00", "risk": "high"},
        {"name": "Lunch Lull",  "il_start": "20:00", "il_end": "21:30", "risk": "low"},
        {"name": "PM Session",  "il_start": "21:30", "il_end": "22:30", "risk": "high"},
        {"name": "Power Hour",  "il_start": "22:30", "il_end": "23:00", "risk": "high"},
        {"name": "Cash Close",  "il_start": "23:00", "il_end": "23:10", "risk": "medium"},
        {"name": "Pre-Market",  "il_start": "14:30", "il_end": "16:30", "risk": "medium"},
        {"name": "Overnight",   "il_start": "02:00", "il_end": "11:00", "risk": "low"},
    ]

    def hm(s):
        h, m = map(int, s.split(":"))
        return h * 60 + m

    for s in sessions:
        start = hm(s["il_start"])
        end = hm(s["il_end"])
        if start <= hour_min < end:
            return {**s, "active": True, "current_il_time": now_il.strftime("%H:%M")}

    return {"name": "Globex / Off-Hours", "risk": "low", "active": False, "current_il_time": now_il.strftime("%H:%M")}
