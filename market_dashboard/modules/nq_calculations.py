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


# ─────────────────────────────────────────────────────────────────────────────
# Sprint 1 — Order Blocks / BOS-MSS / Opening Range
# ─────────────────────────────────────────────────────────────────────────────

def detect_swings(df: pd.DataFrame, n: int = 3):
    """Returns (swing_highs_df, swing_lows_df) each with columns [idx, price, datetime]."""
    highs, lows = [], []
    for i in range(n, len(df) - n):
        h = float(df.iloc[i]["high"])
        l = float(df.iloc[i]["low"])
        if all(h > float(df.iloc[i - j]["high"]) for j in range(1, n + 1)) and \
           all(h > float(df.iloc[i + j]["high"]) for j in range(1, n + 1)):
            highs.append({"idx": i, "price": h, "datetime": df.index[i]})
        if all(l < float(df.iloc[i - j]["low"]) for j in range(1, n + 1)) and \
           all(l < float(df.iloc[i + j]["low"]) for j in range(1, n + 1)):
            lows.append({"idx": i, "price": l, "datetime": df.index[i]})
    return (
        pd.DataFrame(highs) if highs else pd.DataFrame(columns=["idx", "price", "datetime"]),
        pd.DataFrame(lows)  if lows  else pd.DataFrame(columns=["idx", "price", "datetime"]),
    )


def detect_order_blocks(df: pd.DataFrame, min_move_factor: float = 1.5, lookback: int = 3) -> pd.DataFrame:
    """
    Identifies Order Blocks: the last opposing candle before a strong impulse move.
    Bullish OB = last bearish candle before strong upward impulse.
    Bearish OB = last bullish candle before strong downward impulse.
    An OB is invalidated once price closes beyond it.
    """
    if len(df) < lookback + 3:
        return pd.DataFrame()

    avg_body = (df["close"] - df["open"]).abs().mean()
    if avg_body == 0:
        return pd.DataFrame()

    obs = {}  # keyed by candle index to deduplicate

    for i in range(1, len(df) - lookback):
        future = df.iloc[i: i + lookback]
        move = float(future["close"].iloc[-1]) - float(df.iloc[i - 1]["close"])

        if abs(move) < avg_body * min_move_factor:
            continue

        is_bullish_impulse = move > 0
        for j in range(i - 1, max(i - 6, -1), -1):
            c = df.iloc[j]
            if is_bullish_impulse and float(c["close"]) < float(c["open"]):
                if j not in obs:
                    obs[j] = {
                        "type": "bullish",
                        "idx": j,
                        "high": float(c["high"]),
                        "low": float(c["low"]),
                        "body_top": float(c["open"]),
                        "body_bottom": float(c["close"]),
                        "datetime": df.index[j],
                        "valid": True,
                    }
                break
            elif not is_bullish_impulse and float(c["close"]) > float(c["open"]):
                if j not in obs:
                    obs[j] = {
                        "type": "bearish",
                        "idx": j,
                        "high": float(c["high"]),
                        "low": float(c["low"]),
                        "body_top": float(c["close"]),
                        "body_bottom": float(c["open"]),
                        "datetime": df.index[j],
                        "valid": True,
                    }
                break

    if not obs:
        return pd.DataFrame()

    ob_df = pd.DataFrame(list(obs.values()))

    # Invalidate OBs price has already closed through
    for i, ob in ob_df.iterrows():
        after = df.iloc[ob["idx"] + lookback:]
        if ob["type"] == "bullish":
            if (after["close"] < ob["low"]).any():
                ob_df.at[i, "valid"] = False
        else:
            if (after["close"] > ob["high"]).any():
                ob_df.at[i, "valid"] = False

    return ob_df.sort_values("idx").reset_index(drop=True)


def detect_market_structure(df: pd.DataFrame, swing_n: int = 3) -> pd.DataFrame:
    """
    Detects Break of Structure (BOS) and Market Structure Shift (MSS).
    BOS = break in trend direction (continuation signal).
    MSS = break against trend direction (reversal signal).
    Returns DataFrame with columns: datetime, type, price, label.
    """
    if len(df) < swing_n * 4:
        return pd.DataFrame()

    sh_df, sl_df = detect_swings(df, n=swing_n)
    if sh_df.empty or sl_df.empty:
        return pd.DataFrame()

    events = []
    trend = "neutral"  # track: "bullish" | "bearish" | "neutral"

    for i in range(swing_n + 1, len(df)):
        close = float(df.iloc[i]["close"])
        prev_close = float(df.iloc[i - 1]["close"])

        prev_sh = sh_df[sh_df["idx"] < i]
        prev_sl = sl_df[sl_df["idx"] < i]

        if prev_sh.empty or prev_sl.empty:
            continue

        last_sh_price = float(prev_sh.iloc[-1]["price"])
        last_sl_price = float(prev_sl.iloc[-1]["price"])

        # Break above swing high
        if prev_close <= last_sh_price < close:
            label = "BOS ↑" if trend == "bullish" else "MSS ↑"
            event_type = "BOS_bullish" if trend == "bullish" else "MSS_bullish"
            events.append({
                "datetime": df.index[i],
                "type": event_type,
                "price": last_sh_price,
                "label": label,
            })
            trend = "bullish"

        # Break below swing low
        elif prev_close >= last_sl_price > close:
            label = "BOS ↓" if trend == "bearish" else "MSS ↓"
            event_type = "BOS_bearish" if trend == "bearish" else "MSS_bearish"
            events.append({
                "datetime": df.index[i],
                "type": event_type,
                "price": last_sl_price,
                "label": label,
            })
            trend = "bearish"

    return pd.DataFrame(events) if events else pd.DataFrame()


def calculate_opening_range(df_5m: pd.DataFrame, minutes: int = 15) -> dict:
    """
    Opening Range = high/low of the first `minutes` after RTH open (09:30 ET).
    Returns dict with or_high, or_low, or_mid, or_range, or_complete.
    """
    if df_5m.empty:
        return {}
    try:
        col = "datetime_et" if "datetime_et" in df_5m.columns else "datetime"
        df = df_5m.copy()
        df["_et"] = pd.to_datetime(df[col])
        df["_et_time"] = df["_et"].dt.strftime("%H:%M")

        open_start = "09:30"
        open_end_h = 9 + (30 + minutes) // 60
        open_end_m = (30 + minutes) % 60
        open_end = f"{open_end_h:02d}:{open_end_m:02d}"

        or_bars = df[df["_et_time"].between(open_start, open_end)]
        if or_bars.empty:
            return {}

        or_high = float(or_bars["high"].max())
        or_low = float(or_bars["low"].min())
        expected_bars = minutes // 5
        return {
            "or_high": or_high,
            "or_low": or_low,
            "or_mid": round((or_high + or_low) / 2, 2),
            "or_range": round(or_high - or_low, 2),
            "or_complete": len(or_bars) >= expected_bars,
            "or_bars": len(or_bars),
        }
    except Exception:
        return {}


def classify_premium_discount(current_price: float, range_high: float, range_low: float) -> dict:
    """
    Classify where current price sits within a range.
    0% = Deep Discount (buy zone), 50% = Equilibrium, 100% = Deep Premium (sell zone).
    """
    if range_high <= range_low or current_price is None:
        return {"pct": 50.0, "zone": "Equilibrium", "bias": "neutral"}
    pct = (current_price - range_low) / (range_high - range_low) * 100
    pct = max(0.0, min(100.0, pct))
    if pct < 25:
        zone, bias = "Deep Discount", "bullish"
    elif pct < 45:
        zone, bias = "Discount", "bullish"
    elif pct < 55:
        zone, bias = "Equilibrium", "neutral"
    elif pct < 75:
        zone, bias = "Premium", "bearish"
    else:
        zone, bias = "Deep Premium", "bearish"
    return {"pct": round(pct, 1), "zone": zone, "bias": bias}


def calculate_atr_range_consumed(df_daily: pd.DataFrame, df_5m: pd.DataFrame,
                                  atr_period: int = 14) -> dict:
    """
    How much of the expected daily range (ATR) has been consumed today?
    Returns atr, today_range, consumed_pct, remaining_pts.
    """
    if len(df_daily) < atr_period + 1:
        return {}
    try:
        d = df_daily.copy().reset_index(drop=True)
        d["prev_close"] = d["close"].shift(1)
        d["tr"] = (
            pd.concat([d["high"], d["prev_close"]], axis=1).max(axis=1) -
            pd.concat([d["low"],  d["prev_close"]], axis=1).min(axis=1)
        )
        atr = float(d["tr"].rolling(atr_period).mean().iloc[-1])

        today_high = float(df_5m["high"].max()) if not df_5m.empty else float(d.iloc[-1]["high"])
        today_low  = float(df_5m["low"].min())  if not df_5m.empty else float(d.iloc[-1]["low"])
        today_range = today_high - today_low
        consumed_pct = min(today_range / atr * 100, 100.0) if atr > 0 else 0.0

        return {
            "atr": round(atr, 1),
            "today_range": round(today_range, 1),
            "consumed_pct": round(consumed_pct, 1),
            "remaining_pts": round(max(atr - today_range, 0), 1),
            "today_high": today_high,
            "today_low": today_low,
        }
    except Exception:
        return {}


def get_htf_bias(df_daily: pd.DataFrame, df_hourly: pd.DataFrame = None) -> dict:
    """
    Higher Timeframe bias from Daily structure (HH/HL vs LH/LL) + SMA20 + 1H trend slope.
    Returns dict: bias ('bullish'/'bearish'/'neutral'), strength (0-100).
    """
    pts = 0
    max_pts = 0

    if len(df_daily) >= 10:
        d = df_daily.tail(10)
        highs = d["high"].values.astype(float)
        lows  = d["low"].values.astype(float)
        hh = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i - 1])
        hl = sum(1 for i in range(1, len(lows))  if lows[i]  > lows[i - 1])
        lh = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i - 1])
        ll = sum(1 for i in range(1, len(lows))  if lows[i]  < lows[i - 1])
        bull = hh + hl
        total = bull + lh + ll
        max_pts += 40
        if total > 0:
            pts += int(40 * bull / total)

    if len(df_daily) >= 20:
        sma20 = float(df_daily["close"].tail(20).mean())
        cur   = float(df_daily["close"].iloc[-1])
        max_pts += 20
        if cur > sma20:
            pts += 20

    if df_hourly is not None and len(df_hourly) >= 24:
        close = df_hourly["close"].tail(24).values.astype(float)
        slope = np.polyfit(np.arange(len(close)), close, 1)[0]
        max_pts += 40
        if slope > 0:
            pts += int(min(40, slope / max(close[-1], 1) * 10_000))

    if max_pts == 0:
        return {"bias": "neutral", "strength": 50}

    strength = int(pts / max_pts * 100)
    bias = "bullish" if strength >= 60 else ("bearish" if strength <= 40 else "neutral")
    return {"bias": bias, "strength": strength}


def get_support_resistance_zones(
    df_daily: pd.DataFrame,
    df_hourly: pd.DataFrame = None,
    df_15m: pd.DataFrame = None,
    current_price: float = None,
    n_zones: int = 8,
) -> list:
    """
    Combines Key Levels, Order Blocks (daily + 15m), and FVGs (15m) into scored Zone objects.
    Each zone: price_low, price_high, midpoint, zone_type, score (0-10), components, dist_pts.
    """
    clusters: dict = {}  # keyed by rounded midpoint (10-pt grid) to merge nearby levels

    def _add(low: float, high: float, tag: str):
        mid = round(((low + high) / 2) / 10) * 10  # snap to nearest 10-pt grid
        if mid in clusters:
            if tag not in clusters[mid]["components"]:
                clusters[mid]["components"].append(tag)
            clusters[mid]["price_low"]  = min(clusters[mid]["price_low"], low)
            clusters[mid]["price_high"] = max(clusters[mid]["price_high"], high)
        else:
            clusters[mid] = {"price_low": low, "price_high": high, "components": [tag]}

    # Key levels
    try:
        kl = calculate_key_levels(df_daily, df_hourly)
        for name, val in kl.items():
            if isinstance(val, float) and not np.isnan(val):
                _add(val - 4, val + 4, name.upper())
    except Exception:
        pass

    # Order Blocks — daily
    if len(df_daily) >= 10:
        try:
            obs = detect_order_blocks(df_daily.reset_index(drop=True))
            if not obs.empty:
                for _, ob in obs[obs["valid"] == True].iterrows():
                    _add(float(ob["low"]), float(ob["high"]), "OB_D")
        except Exception:
            pass

    # Order Blocks — 15m (last 10 valid)
    if df_15m is not None and len(df_15m) >= 20:
        try:
            obs15 = detect_order_blocks(df_15m.reset_index(drop=True))
            if not obs15.empty:
                for _, ob in obs15[obs15["valid"] == True].tail(10).iterrows():
                    _add(float(ob["low"]), float(ob["high"]), "OB_15m")
        except Exception:
            pass

    # FVG — 15m (last 8)
    if df_15m is not None and len(df_15m) >= 3:
        try:
            fvgs = detect_fvg(df_15m.reset_index(drop=True))
            if not fvgs.empty:
                for _, fvg in fvgs.tail(8).iterrows():
                    _add(float(fvg["bottom"]), float(fvg["top"]), "FVG_15m")
        except Exception:
            pass

    # Score each cluster
    KEY_LEVEL_TAGS = {"PDH", "PDL", "PDC", "POC_20D", "POC_5D",
                      "WEEKLY_HIGH", "WEEKLY_LOW", "OVERNIGHT_HIGH", "OVERNIGHT_LOW",
                      "VAL_20D", "VAH_20D"}
    result = []
    for mid_snap, z in clusters.items():
        comps  = z["components"]
        raw    = len(comps)
        has_ob = any("OB" in c for c in comps)
        has_kl = any(c in KEY_LEVEL_TAGS for c in comps)
        has_fvg = any("FVG" in c for c in comps)
        if has_ob and has_kl:
            raw += 2
        if has_ob and has_fvg:
            raw += 1

        midpoint = (z["price_low"] + z["price_high"]) / 2
        if current_price is not None:
            zone_type = "support" if midpoint < current_price else "resistance"
            dist_pts  = abs(midpoint - current_price)
        else:
            zone_type = "unknown"
            dist_pts  = 0.0

        result.append({
            "price_low":  round(z["price_low"],  1),
            "price_high": round(z["price_high"], 1),
            "midpoint":   round(midpoint, 1),
            "zone_type":  zone_type,
            "score":      min(10, raw * 2),
            "components": comps,
            "dist_pts":   round(dist_pts, 1),
        })

    result.sort(key=lambda x: x["dist_pts"])
    return result[:n_zones]


def calculate_dynamic_stop(
    df_daily: pd.DataFrame,
    direction: str,
    entry: float,
    atr_period: int = 14,
    atr_mult: float = 1.5,
) -> float:
    """ATR(14) × atr_mult dynamic stop. Falls back to 15-pt fixed if data is insufficient."""
    try:
        if len(df_daily) >= atr_period + 1:
            d = df_daily.copy().reset_index(drop=True)
            d["prev_close"] = d["close"].shift(1)
            d["tr"] = (
                pd.concat([d["high"], d["prev_close"]], axis=1).max(axis=1)
                - pd.concat([d["low"],  d["prev_close"]], axis=1).min(axis=1)
            )
            atr = float(d["tr"].rolling(atr_period).mean().iloc[-1])
            risk = atr * atr_mult
            return round((entry - risk) if direction == "LONG" else (entry + risk), 2)
    except Exception:
        pass
    fallback = 15.0
    return round((entry - fallback) if direction == "LONG" else (entry + fallback), 2)


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
