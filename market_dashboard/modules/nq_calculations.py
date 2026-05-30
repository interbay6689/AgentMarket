import numpy as np
import pandas as pd


def approximate_delta(df: pd.DataFrame) -> pd.Series:
    bar_range = df["high"] - df["low"]
    buy_frac = (df["close"] - df["low"]) / bar_range.replace(0, np.nan)
    sell_frac = 1 - buy_frac
    delta = (buy_frac - sell_frac) * df["volume"]
    return delta.fillna(0)


def cumulative_delta(df: pd.DataFrame) -> pd.Series:
    return approximate_delta(df).cumsum()


def detect_fvg(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 3:
        return pd.DataFrame()
    fvgs = []
    idx = df.index.tolist()
    for i in range(2, len(df)):
        prev2 = df.iloc[i - 2]
        curr = df.iloc[i]
        if curr["low"] > prev2["high"]:
            fvgs.append({
                "datetime": idx[i],
                "type": "bullish",
                "top": curr["low"],
                "bottom": prev2["high"],
                "size": curr["low"] - prev2["high"],
            })
        elif curr["high"] < prev2["low"]:
            fvgs.append({
                "datetime": idx[i],
                "type": "bearish",
                "top": prev2["low"],
                "bottom": curr["high"],
                "size": prev2["low"] - curr["high"],
            })
    return pd.DataFrame(fvgs) if fvgs else pd.DataFrame()


def calculate_volume_profile(df: pd.DataFrame, price_bins: int = 50) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["price", "volume"])
    price_min = df["low"].min()
    price_max = df["high"].max()
    if price_min == price_max:
        return pd.DataFrame({"price": [price_min], "volume": [df["volume"].sum()]})
    bins = np.linspace(price_min, price_max, price_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    vol_at_price = np.zeros(price_bins)
    for _, row in df.iterrows():
        lo_idx = np.searchsorted(bins, row["low"], side="left")
        hi_idx = np.searchsorted(bins, row["high"], side="right")
        lo_idx = max(0, min(lo_idx, price_bins - 1))
        hi_idx = max(0, min(hi_idx, price_bins - 1))
        span = max(hi_idx - lo_idx, 1)
        vol_at_price[lo_idx: hi_idx + 1] += row["volume"] / span
    return pd.DataFrame({"price": bin_centers, "volume": vol_at_price})


def get_poc(vp_df: pd.DataFrame) -> float:
    if vp_df.empty:
        return np.nan
    return float(vp_df.loc[vp_df["volume"].idxmax(), "price"])


def get_value_area(vp_df: pd.DataFrame, pct: float = 0.70):
    if vp_df.empty:
        return np.nan, np.nan
    total_vol = vp_df["volume"].sum()
    target = total_vol * pct
    poc_idx = int(vp_df["volume"].idxmax())
    lo_idx = hi_idx = poc_idx
    accumulated = float(vp_df.loc[poc_idx, "volume"])
    while accumulated < target:
        lo_try = max(lo_idx - 1, 0)
        hi_try = min(hi_idx + 1, len(vp_df) - 1)
        lo_vol = float(vp_df.loc[lo_try, "volume"]) if lo_try != lo_idx else 0
        hi_vol = float(vp_df.loc[hi_try, "volume"]) if hi_try != hi_idx else 0
        if lo_vol >= hi_vol and lo_try != lo_idx:
            lo_idx = lo_try
            accumulated += lo_vol
        elif hi_try != hi_idx:
            hi_idx = hi_try
            accumulated += hi_vol
        else:
            break
    return float(vp_df.loc[lo_idx, "price"]), float(vp_df.loc[hi_idx, "price"])


def detect_absorption(df: pd.DataFrame, volume_pct: int = 80, range_pct: int = 30) -> pd.Series:
    result = pd.Series(0, index=df.index)
    if len(df) < 5:
        return result
    vol_thresh = df["volume"].quantile(volume_pct / 100)
    bar_range = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    range_thresh = bar_range.quantile(range_pct / 100)
    high_vol = df["volume"] >= vol_thresh
    tight_range = bar_range <= range_thresh
    span = (df["high"] - df["low"]).replace(0, np.nan)
    close_pos = (df["close"] - df["low"]) / span
    result[high_vol & tight_range & (close_pos > 0.7)] = 1
    result[high_vol & tight_range & (close_pos < 0.3)] = -1
    return result


def detect_stacked_imbalances(df: pd.DataFrame, min_stack: int = 3) -> list:
    stacks = []
    if len(df) < min_stack:
        return stacks
    direction = (df["close"] > df["open"]).astype(int) - (df["close"] < df["open"]).astype(int)
    count = 1
    for i in range(1, len(direction)):
        if direction.iloc[i] == direction.iloc[i - 1] and direction.iloc[i] != 0:
            count += 1
        else:
            if count >= min_stack:
                stacks.append({
                    "end_idx": i - 1,
                    "direction": "bullish" if direction.iloc[i - 1] > 0 else "bearish",
                    "candle_count": count,
                    "price_start": float(df.iloc[i - count]["open"]),
                    "price_end": float(df.iloc[i - 1]["close"]),
                })
            count = 1
    if count >= min_stack:
        stacks.append({
            "end_idx": len(direction) - 1,
            "direction": "bullish" if direction.iloc[-1] > 0 else "bearish",
            "candle_count": count,
            "price_start": float(df.iloc[-count]["open"]),
            "price_end": float(df.iloc[-1]["close"]),
        })
    return stacks


def detect_delta_divergence(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    result = pd.Series(0, index=df.index)
    if len(df) < lookback + 2:
        return result
    cd = cumulative_delta(df)
    price_hh = df["close"].rolling(lookback).max()
    delta_hh = cd.rolling(lookback).max()
    price_ll = df["close"].rolling(lookback).min()
    delta_ll = cd.rolling(lookback).min()
    bearish_div = (df["close"] >= price_hh.shift(1)) & (cd < delta_hh.shift(1))
    bullish_div = (df["close"] <= price_ll.shift(1)) & (cd > delta_ll.shift(1))
    result[bearish_div] = 1
    result[bullish_div] = -1
    return result


def detect_judas_swing(df_5m: pd.DataFrame, overnight_high: float, overnight_low: float,
                       sweep_pts: float = 10.0) -> dict:
    if df_5m.empty or overnight_high == overnight_low:
        return {"detected": False}
    try:
        col = "datetime_et" if "datetime_et" in df_5m.columns else "datetime"
        df_5m = df_5m.copy()
        df_5m["_et"] = pd.to_datetime(df_5m[col])
        if df_5m["_et"].dt.tz is not None:
            df_5m["_et_time"] = df_5m["_et"].dt.strftime("%H:%M")
        else:
            df_5m["_et_time"] = df_5m["_et"].dt.strftime("%H:%M")
        judas = df_5m[df_5m["_et_time"].between("09:15", "09:29")]
        post = df_5m[df_5m["_et_time"].between("09:30", "10:00")]
        if judas.empty or post.empty:
            return {"detected": False}
        swept_high = judas["high"].max() > overnight_high + sweep_pts
        swept_low = judas["low"].min() < overnight_low - sweep_pts
        if swept_high:
            reversed_down = post["close"].iloc[-1] < overnight_high
            if reversed_down:
                return {"detected": True, "direction": "bearish", "level": overnight_high,
                        "sweep_high": float(judas["high"].max())}
        if swept_low:
            reversed_up = post["close"].iloc[-1] > overnight_low
            if reversed_up:
                return {"detected": True, "direction": "bullish", "level": overnight_low,
                        "sweep_low": float(judas["low"].min())}
    except Exception:
        pass
    return {"detected": False}


def calculate_key_levels(df_daily: pd.DataFrame, df_hourly: pd.DataFrame = None) -> dict:
    levels = {}
    if len(df_daily) >= 2:
        prev = df_daily.iloc[-2]
        levels["pdh"] = float(prev["high"])
        levels["pdl"] = float(prev["low"])
        levels["pdc"] = float(prev["close"])
        levels["pdo"] = float(prev["open"])
    if len(df_daily) >= 5:
        last5 = df_daily.tail(5)
        levels["weekly_high"] = float(last5["high"].max())
        levels["weekly_low"] = float(last5["low"].min())
    if len(df_daily) >= 20:
        vp20 = calculate_volume_profile(df_daily.tail(20))
        levels["poc_20d"] = get_poc(vp20)
        levels["val_20d"], levels["vah_20d"] = get_value_area(vp20)
    if len(df_daily) >= 5:
        vp5 = calculate_volume_profile(df_daily.tail(5))
        levels["poc_5d"] = get_poc(vp5)

    if df_hourly is not None and not df_hourly.empty:
        today = pd.Timestamp.now().date()
        try:
            col = "datetime_et" if "datetime_et" in df_hourly.columns else "datetime"
            df_h = df_hourly.copy()
            df_h["_dt"] = pd.to_datetime(df_h[col])
            df_h["_date"] = df_h["_dt"].dt.date
            yesterday = df_h[df_h["_date"] < today]
            overnight_mask = df_h["_dt"].dt.hour < 9
            overnight = df_h[overnight_mask & (df_h["_date"] == today)]
            if not overnight.empty:
                levels["overnight_high"] = float(overnight["high"].max())
                levels["overnight_low"] = float(overnight["low"].min())
        except Exception:
            pass
    return levels


def find_similar_setups(df_daily: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if len(df_daily) < 30:
        return pd.DataFrame()
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics.pairwise import cosine_similarity
        df = df_daily.copy()
        df["norm_range"] = (df["high"] - df["low"]) / df["close"]
        df["pct_change"] = df["close"].pct_change()
        df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
        df["above_sma20"] = (df["close"] > df["close"].rolling(20).mean()).astype(int)
        df["next_change"] = df["close"].pct_change().shift(-1)
        feature_cols = ["norm_range", "pct_change", "vol_ratio", "above_sma20"]
        feat = df[feature_cols].dropna()
        if len(feat) < 10:
            return pd.DataFrame()
        scaler = StandardScaler()
        X = scaler.fit_transform(feat)
        today_vec = X[-1].reshape(1, -1)
        history = X[:-1]
        sims = cosine_similarity(today_vec, history)[0]
        top_idx = np.argsort(sims)[-top_n:][::-1]
        similar = df.iloc[top_idx].copy()
        similar["similarity"] = sims[top_idx]
        similar["next_day_change"] = similar["next_change"] * 100
        return similar[["date", "similarity", "close", "next_day_change"]].reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def session_win_rates(df_hourly: pd.DataFrame) -> pd.DataFrame:
    if df_hourly.empty:
        return pd.DataFrame()
    try:
        import pytz
        ET_TZ = pytz.timezone("America/New_York")
        df = df_hourly.copy()
        col = "datetime_et" if "datetime_et" in df.columns else "datetime"
        df["_dt"] = pd.to_datetime(df[col])
        df["_hour"] = df["_dt"].dt.hour
        df["_date"] = df["_dt"].dt.date
        df["_up"] = (df["close"] > df["open"]).astype(int)
        result = df.groupby("_hour")["_up"].agg(["mean", "count"]).reset_index()
        result.columns = ["hour_et", "win_rate", "n_bars"]
        result["win_rate_pct"] = (result["win_rate"] * 100).round(1)
        return result
    except Exception:
        return pd.DataFrame()
