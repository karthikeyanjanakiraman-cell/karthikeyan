import os
import sys
import logging
from datetime import datetime, timedelta, time
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from fyers_apiv3 import fyersModel
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

class UTF8Formatter(logging.Formatter):
    def format(self, record):
        msg = record.getMessage()
        record.msg = msg.encode("ascii", "ignore").decode("ascii")
        return super().format(record)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = UTF8Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)

DAILY_LOOKBACK_DAYS = 60
INTRADAY_LOOKBACK_DAYS = 20
fyers: Optional[fyersModel.FyersModel] = None
EMAIL_DISPLAY_COLS = [
    "Symbol", "LTP", "% Change",
    "15m_Signal", "30m_Signal", "75m_Signal",
    "3h_Signal", "6h_Signal", "2D_Signal",
    "Bull_Signal", "Bear_Signal", "Overall_Signal",
    "HigherTF Confirm", "Last Iteration Time"
]


def init_fyers():
    global fyers
    try:
        client_id = os.environ.get("CLIENT_ID") or os.environ.get("CLIENTID")
        access_token = os.environ.get("ACCESS_TOKEN") or os.environ.get("ACCESSTOKEN")
        if not client_id or not access_token:
            logger.warning("INIT Missing Fyers credentials.")
            fyers = None
            return
        fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=access_token, log_path="")
        logger.info("INIT FyersModel initialized successfully.")
    except Exception as e:
        logger.warning(f"INIT Failed: {e}")
        fyers = None


def load_fno_symbols_from_sectors(root_dir: str = "sectors") -> List[str]:
    symbols = set()
    if not os.path.isdir(root_dir):
        return []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.lower().endswith('.csv'):
                continue
            try:
                df = pd.read_csv(os.path.join(dirpath, fname))
                col = next((c for c in df.columns if c.lower() in ["symbol", "symbols", "ticker"]), None)
                if col is None:
                    continue
                for s in df[col].dropna().astype(str):
                    s = s.strip()
                    if s:
                        symbols.add(s)
            except Exception:
                pass
    return sorted(symbols)


def format_fyers_symbol(symbol: str) -> str:
    if symbol.startswith("NSE:") and symbol.endswith("-EQ"):
        return symbol
    return f"NSE:{symbol}-EQ"


def get_fyers_history(symbol: str, resolution: str, days_back: int) -> Optional[pd.DataFrame]:
    if not fyers:
        return None
    try:
        now = datetime.now()
        start_date = now - timedelta(days=days_back)
        data = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": "1",
            "range_from": start_date.strftime("%Y-%m-%d"),
            "range_to": now.strftime("%Y-%m-%d"),
            "cont_flag": "1"
        }
        res = fyers.history(data=data)
        if res and res.get("s") == "ok" and "candles" in res and res["candles"]:
            df = pd.DataFrame(res["candles"], columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s").dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
            df.sort_values("timestamp", inplace=True)
            df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df
    except Exception as e:
        logger.error(f"FYERS Error fetching {resolution} data for {symbol}: {e}")
    return None


def compute_cumulative_directional_metrics(curr_df: pd.DataFrame) -> pd.DataFrame:
    df = curr_df.copy().sort_values("time").reset_index(drop=True)
    if df.empty:
        return df
    h = df["high"].astype(float).to_numpy()
    l = df["low"].astype(float).to_numpy()
    c = df["close"].astype(float).to_numpy()
    o = df["open"].astype(float).to_numpy()
    n = len(df)
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr[0] = max(h[0] - l[0], 0)
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        up = h[i] - h[i - 1]
        dn = l[i - 1] - l[i]
        plus_dm[i] = up if up > dn and up > 0 else 0.0
        minus_dm[i] = dn if dn > up and dn > 0 else 0.0
    out = []
    qualified = 0
    for i in range(n):
        if i == 0:
            out.append([np.nan, np.nan, np.nan, np.nan, "0/0", 0.0])
            continue
        prior_kers = []
        for j in range(1, i + 1):
            pj = abs(c[j] - o[0])
            wj = abs(c[0] - o[0]) + np.sum(np.abs(np.diff(c[: j + 1])))
            prior_kers.append(pj / wj if wj > 0 else 0.0)
        cum_ker = float(np.mean(prior_kers)) if prior_kers else np.nan
        tr_sum = tr[1:i + 1].sum()
        plus_sum = plus_dm[1:i + 1].sum()
        minus_sum = minus_dm[1:i + 1].sum()
        pdi = 100 * plus_sum / tr_sum if tr_sum > 0 else 0.0
        mdi = 100 * minus_sum / tr_sum if tr_sum > 0 else 0.0
        dxs = []
        for k in range(1, i + 1):
            ktr = tr[1:k + 1].sum()
            kp = plus_dm[1:k + 1].sum()
            km = minus_dm[1:k + 1].sum()
            kpdi = 100 * kp / ktr if ktr > 0 else 0.0
            kmdi = 100 * km / ktr if ktr > 0 else 0.0
            dxs.append(100 * abs(kpdi - kmdi) / (kpdi + kmdi) if (kpdi + kmdi) > 0 else 0.0)
        adx = float(np.mean(dxs)) if dxs else np.nan
        if c[i] > c[i - 1]:
            qualified += 1
        length_so_far = i + 1
        survival_ratio = qualified / length_so_far if length_so_far > 0 else 0.0
        out.append([cum_ker, pdi, mdi, adx, f"{qualified}/{length_so_far}", survival_ratio])
    cols = ["Cumulative KER", "Cumulative +DI", "Cumulative -DI", "Cumulative ADX", "Survival Score", "Survival_Num"]
    return pd.concat([df, pd.DataFrame(out, columns=cols)], axis=1)


def compute_cumulative_flow_metrics(curr_df: pd.DataFrame) -> pd.DataFrame:
    df = curr_df.copy().sort_values("time").reset_index(drop=True)
    if df.empty:
        return df
    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    high = pd.to_numeric(df["high"], errors="coerce").astype(float)
    low = pd.to_numeric(df["low"], errors="coerce").astype(float)
    volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).astype(float)
    delta = close.diff().fillna(0.0)
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    period = 14
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    zero_loss = (avg_loss == 0) & avg_loss.notna()
    rsi = rsi.mask(zero_loss, 100.0).fillna(0.0)
    direction = close.diff().fillna(0.0).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv = (direction * volume).cumsum()
    typical_price = (high + low + close) / 3.0
    cum_pv = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum().replace(0, float("nan"))
    vwap = (cum_pv / cum_vol).fillna(0.0)
    vwap_variance = (volume * (typical_price - vwap) ** 2).cumsum() / cum_vol
    vwap_std = np.sqrt(vwap_variance).fillna(0.0)
    vwap_z_score = np.where(vwap_std > 0, (close - vwap) / vwap_std, 0.0)
    out = pd.DataFrame({
        "Cumulative RSI": rsi,
        "Cumulative OBV": obv,
        "Cumulative VWAP": vwap,
        "VWAP Z-Score": pd.Series(vwap_z_score, index=df.index).fillna(0.0)
    })
    return pd.concat([df.reset_index(drop=True), out.reset_index(drop=True)], axis=1)


def calculate_hybrid_freshness(df_intraday: pd.DataFrame) -> pd.DataFrame:
    if df_intraday is None or df_intraday.empty:
        return df_intraday
    work = df_intraday.copy()
    if "time" not in work.columns and "timestamp" in work.columns:
        work["time"] = pd.to_datetime(work["timestamp"])
    work = work.sort_values("time").reset_index(drop=True)
    if len(work) < 3:
        work["Freshness_Score"] = 50.0
        work["Is_Fresh"] = False
        work["Fresh_State"] = ""
        work["Fresh_Since"] = ""
        return work
    work["rolling_HOD"] = work["high"].cummax()
    work["vol_sma_10"] = work["volume"].rolling(window=10, min_periods=1).mean().shift(1)
    work["vol_sma_10"] = work["vol_sma_10"].fillna(work["volume"].mean())
    work["vol_ratio"] = np.where(work["vol_sma_10"] > 0, work["volume"] / work["vol_sma_10"], 0.0)
    open_price = float(work.iloc[0]["open"])
    work["close_vs_open_pct"] = (work["close"] - open_price) / open_price * 100
    ib_high = float(work.iloc[0:3]["high"].max())
    ib_vol_avg = float(work.iloc[0:3]["volume"].mean()) if len(work.iloc[0:3]) > 0 else 0.0
    freshness_scores, is_fresh_flags = [], []
    for i in range(len(work)):
        row = work.iloc[i]
        base_score = min(100.0, max(0.0, row["close_vs_open_pct"] / 3.0 * 100.0))
        score = base_score
        is_fresh = False
        if i < 3:
            score = base_score
        elif 3 <= i < 11:
            dist_to_ib_high = ((ib_high - row["close"]) / ib_high * 100.0) if ib_high > 0 else 999.0
            vol_vs_ib = (row["volume"] / ib_vol_avg) if ib_vol_avg > 0 else 0.0
            if dist_to_ib_high < 0.3 or row["close"] > ib_high:
                score = score * 0.2 if vol_vs_ib < 0.8 else min(100.0, score * 1.5)
                is_fresh = vol_vs_ib >= 0.8
        else:
            prev_hod = float(work.iloc[i - 1]["rolling_HOD"])
            dist_to_prev_hod = ((prev_hod - row["close"]) / prev_hod * 100.0) if prev_hod > 0 else 999.0
            if dist_to_prev_hod < 0.3 or row["close"] > prev_hod:
                score = score * 0.2 if row["vol_ratio"] < 1.3 else min(100.0, score * 1.5)
                is_fresh = row["vol_ratio"] >= 1.3
            elif row["vol_ratio"] > 2.0 and row["close"] > row["open"]:
                score = min(100.0, score * 1.2)
                is_fresh = True
        freshness_scores.append(round(score, 1))
        is_fresh_flags.append(is_fresh)
    work["Freshness_Score"] = freshness_scores
    work["Is_Fresh"] = is_fresh_flags
    prev_fresh = work["Is_Fresh"].shift(1).fillna(False)
    fresh_states, fresh_since = [], []
    fresh_start_time = ""
    fresh_cycle = 0
    for i in range(len(work)):
        curr = bool(work.loc[i, "Is_Fresh"])
        prev = bool(prev_fresh.iloc[i])
        tstr = pd.to_datetime(work.loc[i, "time"]).strftime("%H:%M")
        if curr and not prev:
            fresh_cycle += 1
            fresh_start_time = tstr
            state = "Fresh" if fresh_cycle == 1 else "Re-Ignited"
            since = tstr
        elif curr and prev:
            state = "Fresh"
            since = fresh_start_time
        elif (not curr) and prev:
            state = "Fresh Lost"
            since = tstr
        else:
            state = ""
            since = ""
        fresh_states.append(state)
        fresh_since.append(since)
    work["Fresh_State"] = fresh_states
    work["Fresh_Since"] = fresh_since
    return work


def compute_iteration_volume_profile(intra_df: Optional[pd.DataFrame]) -> Tuple[Dict, pd.DataFrame]:
    if intra_df is None or intra_df.empty:
        return {}, pd.DataFrame()
    df = calculate_hybrid_freshness(intra_df.copy())
    if "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        df["time_only"] = pd.to_datetime(df["timestamp"]).dt.time
    else:
        df["date"] = pd.to_datetime(df["time"]).dt.date
        df["time_only"] = pd.to_datetime(df["time"]).dt.time
    dates = sorted(df["date"].unique())
    if len(dates) < 2:
        return {}, pd.DataFrame()
    current_date = dates[-1]
    hist_dates_10 = dates[-11:-1] if len(dates) >= 11 else dates[:-1]
    hist_dates_20 = dates[-21:-1] if len(dates) >= 21 else dates[:-1]
    curr_df = df[df["date"] == current_date].copy()
    hist_df_10 = df[df["date"].isin(hist_dates_10)].copy()
    hist_df_20 = df[df["date"].isin(hist_dates_20)].copy()
    if curr_df.empty:
        return {}, pd.DataFrame()
    curr_df.sort_values("time_only", inplace=True)
    curr_df["cum_vol"] = curr_df["volume"].cumsum()
    metric_df = compute_cumulative_directional_metrics(curr_df[["time", "open", "high", "low", "close", "volume"]].copy())
    flow_df = compute_cumulative_flow_metrics(curr_df[["time", "high", "low", "close", "volume"]].copy())
    rows = []
    total_iters = 0
    last_iter_mins = None
    last_iter_time = None
    last_cum_vol = last_rvol10 = last_rvol20 = 0
    for i in range(len(curr_df)):
        total_iters += 1
        row = curr_df.iloc[i]
        t = row["time_only"]
        cum_vol = float(row["cum_vol"])
        h10 = hist_df_10[hist_df_10["time_only"] <= t]
        avg_cum_10 = h10.groupby("date")["volume"].sum().mean() if not h10.empty else 0
        rvol10 = cum_vol / avg_cum_10 if avg_cum_10 > 0 else 0
        h20 = hist_df_20[hist_df_20["time_only"] <= t]
        avg_cum_20 = h20.groupby("date")["volume"].sum().mean() if not h20.empty else 0
        rvol20 = cum_vol / avg_cum_20 if avg_cum_20 > 0 else 0
        dt_time = datetime.combine(current_date, t)
        market_open = datetime.combine(current_date, time(9, 15))
        iter_mins = int((dt_time - market_open).total_seconds() / 60)
        rows.append({
            "Iteration No": total_iters,
            "Iteration Minutes": iter_mins,
            "Iteration Time": t.strftime("%H:%M"),
            "Current Volume": cum_vol,
            "10 Day Relative Volume": rvol10,
            "20 Day Relative Volume": rvol20,
            "Cumulative RSI": float(flow_df["Cumulative RSI"].iloc[i]) if not flow_df.empty else float("nan"),
            "Cumulative OBV": float(flow_df["Cumulative OBV"].iloc[i]) if not flow_df.empty else float("nan"),
            "Cumulative VWAP": float(flow_df["Cumulative VWAP"].iloc[i]) if not flow_df.empty else float("nan"),
            "VWAP Z-Score": float(flow_df["VWAP Z-Score"].iloc[i]) if not flow_df.empty else float("nan"),
            "Freshness_Score": float(curr_df["Freshness_Score"].iloc[i]) if "Freshness_Score" in curr_df.columns else float("nan"),
            "Is_Fresh": bool(curr_df["Is_Fresh"].iloc[i]) if "Is_Fresh" in curr_df.columns else False
        })
        last_cum_vol, last_rvol10, last_rvol20 = cum_vol, rvol10, rvol20
        last_iter_mins = iter_mins
        last_iter_time = t.strftime("%H:%M")
    detail_df = pd.DataFrame(rows)
    ltp = float(curr_df["close"].iloc[-1]) if not curr_df.empty else np.nan
    hod = float(curr_df["high"].max()) if not curr_df.empty else float("nan")
    strike_distance = (hod - ltp) / hod if hod and hod > 0 and ltp is not None else 1.0
    last_5m_volume = float(curr_df["volume"].iloc[-1]) if not curr_df.empty else 0.0
    recent_12 = curr_df["volume"].tail(12) if not curr_df.empty else pd.Series(dtype=float)
    vol_1h_avg_5m = float(recent_12.mean()) if len(recent_12) > 0 else last_5m_volume
    obv_30m_delta = 0.0
    rsi_30m_delta = 0.0
    if not flow_df.empty and len(flow_df) >= 7:
        obv_30m_delta = float(flow_df["Cumulative OBV"].iloc[-1]) - float(flow_df["Cumulative OBV"].iloc[-7])
        rsi_30m_delta = float(flow_df["Cumulative RSI"].iloc[-1]) - float(flow_df["Cumulative RSI"].iloc[-7])
    adx_now = float(metric_df["Cumulative ADX"].iloc[-1]) if not metric_df.empty else float("nan")
    ker_now = float(metric_df["Cumulative KER"].iloc[-1]) if not metric_df.empty else float("nan")
    fresh_score = float(curr_df["Freshness_Score"].iloc[-1]) if "Freshness_Score" in curr_df.columns else 0.0
    is_fresh = bool(fresh_score >= 60.0) and bool(adx_now > 20.0) and bool(ker_now > 0.40)
    summary = {
        "LTP": ltp,
        "Current Volume": last_cum_vol,
        "10 Day Relative Volume": last_rvol10,
        "20 Day Relative Volume": last_rvol20,
        "Cumulative RSI": float(flow_df["Cumulative RSI"].iloc[-1]) if not flow_df.empty else float("nan"),
        "Cumulative OBV": float(flow_df["Cumulative OBV"].iloc[-1]) if not flow_df.empty else float("nan"),
        "Cumulative VWAP": float(flow_df["Cumulative VWAP"].iloc[-1]) if not flow_df.empty else float("nan"),
        "VWAP Z-Score": float(flow_df["VWAP Z-Score"].iloc[-1]) if not flow_df.empty else float("nan"),
        "Total Iterations": total_iters,
        "Last Iteration Minutes": last_iter_mins,
        "Last Iteration Time": last_iter_time,
        "Cumulative KER": float(metric_df["Cumulative KER"].iloc[-1]) if not metric_df.empty else np.nan,
        "Cumulative +DI": float(metric_df["Cumulative +DI"].iloc[-1]) if not metric_df.empty else np.nan,
        "Cumulative -DI": float(metric_df["Cumulative -DI"].iloc[-1]) if not metric_df.empty else np.nan,
        "Cumulative ADX": float(metric_df["Cumulative ADX"].iloc[-1]) if not metric_df.empty else np.nan,
        "Survival Score": str(metric_df["Survival Score"].iloc[-1]) if not metric_df.empty else "0/0",
        "Survival_Num": float(metric_df["Survival_Num"].iloc[-1]) if not metric_df.empty else 0.0,
        "HOD": hod,
        "Strike_Distance": strike_distance,
        "Last_5m_Volume": last_5m_volume,
        "Volume_1h_Avg_5m": vol_1h_avg_5m,
        "OBV_30m_Delta": obv_30m_delta,
        "RSI_30m_Delta": rsi_30m_delta,
        "Freshness_Score": float(fresh_score),
        "Fresh_State": str(df["Fresh_State"].iloc[-1]) if "Fresh_State" in df.columns else "",
        "Fresh_Since": str(df["Fresh_Since"].iloc[-1]) if "Fresh_Since" in df.columns else "",
        "Is_Fresh": bool(is_fresh)
    }
    return summary, detail_df


def rank_delta_to_label(delta: float) -> str:
    if pd.isna(delta):
        return ""
    if delta >= 7:
        return "Buy++"
    if delta >= 4:
        return "Buy+"
    if delta >= 1:
        return "Buy"
    if delta == 0:
        return "Neutral"
    if delta <= -7:
        return "Sell++"
    if delta <= -4:
        return "Sell+"
    return "Sell"


def compute_tf_rank(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    out = df.copy()
    close = pd.to_numeric(out[price_col], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    volume = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).fillna(0)
    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr14 = tr.rolling(14, min_periods=14).sum()
    pdi = 100 * pd.Series(plus_dm, index=out.index).rolling(14, min_periods=14).sum() / atr14.replace(0, np.nan)
    mdi = 100 * pd.Series(minus_dm, index=out.index).rolling(14, min_periods=14).sum() / atr14.replace(0, np.nan)
    typical = (high + low + close) / 3.0
    obv = (np.sign(close.diff().fillna(0)) * volume).cumsum()
    vwap = (typical * volume).cumsum() / volume.cumsum().replace(0, np.nan)
    bull = pd.Series(0, index=out.index, dtype=float)
    bear = pd.Series(0, index=out.index, dtype=float)
    bull += (close > ema9).astype(int)
    bull += (ema9 > ema21).astype(int)
    bull += (rsi >= 55).astype(int)
    bull += (pdi > mdi).astype(int)
    bull += (close > vwap).astype(int)
    bull += (obv > obv.rolling(5, min_periods=1).mean()).astype(int)
    bear += (close < ema9).astype(int)
    bear += (ema9 < ema21).astype(int)
    bear += (rsi <= 45).astype(int)
    bear += (mdi > pdi).astype(int)
    bear += (close < vwap).astype(int)
    bear += (obv < obv.rolling(5, min_periods=1).mean()).astype(int)
    out["Bull Rank"] = bull
    out["Bear Rank"] = bear
    out["Rank Delta"] = bull - bear
    return out


def build_mtf_signals_from_intraday(intra_df: Optional[pd.DataFrame]) -> Dict[str, object]:
    if intra_df is None or intra_df.empty or "timestamp" not in intra_df.columns:
        return {}
    df = intra_df.copy().sort_values("timestamp").reset_index(drop=True)
    tdf = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]].copy()
    tf_map = {
        "15m": "15min",
        "30m": "30min",
        "75m": "75min",
        "3h": "180min",
        "6h": "360min",
        "2D": "2D"
    }
    signals = {}
    bull_total = 0.0
    bear_total = 0.0
    weights = {"15m": 1.0, "30m": 1.1, "75m": 1.2, "3h": 1.3, "6h": 1.4, "2D": 1.5}
    for label, rule in tf_map.items():
        rdf = tdf.resample(rule).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna().reset_index()
        if len(rdf) < 20:
            signals[f"{label}_Signal"] = ""
            continue
        ranked = compute_tf_rank(rdf)
        last = ranked.iloc[-1]
        delta = float(last["Rank Delta"])
        signals[f"{label}_Signal"] = rank_delta_to_label(delta)
        bull_total += float(last["Bull Rank"]) * weights[label]
        bear_total += float(last["Bear Rank"]) * weights[label]
    overall_delta = bull_total - bear_total
    signals["Bull_Signal"] = rank_delta_to_label(bull_total)
    signals["Bear_Signal"] = rank_delta_to_label(-bear_total)
    signals["Overall_Signal"] = rank_delta_to_label(overall_delta)
    higher = [signals.get("75m_Signal", ""), signals.get("3h_Signal", ""), signals.get("6h_Signal", ""), signals.get("2D_Signal", "")]
    if sum(1 for x in higher if str(x).startswith("Buy")) >= 2:
        signals["HigherTF Confirm"] = "Bullish"
    elif sum(1 for x in higher if str(x).startswith("Sell")) >= 2:
        signals["HigherTF Confirm"] = "Bearish"
    else:
        signals["HigherTF Confirm"] = "Mixed"
    return signals


def scan_fno_universe() -> Tuple[pd.DataFrame, pd.DataFrame]:
    symbols = load_fno_symbols_from_sectors("sectors")
    if not symbols:
        logger.error("CORE No F&O symbols found.")
        return pd.DataFrame(), pd.DataFrame()
    rows, iteration_rows = [], []
    total = len(symbols)
    for idx, sym in enumerate(symbols, start=1):
        logger.info(f"CORE [{idx}/{total}] Processing {sym}")
        fyers_sym = format_fyers_symbol(sym)
        daily_df = get_fyers_history(fyers_sym, resolution="D", days_back=DAILY_LOOKBACK_DAYS)
        intra_df = get_fyers_history(fyers_sym, resolution="5", days_back=INTRADAY_LOOKBACK_DAYS)
        iter_summary, iter_detail = compute_iteration_volume_profile(intra_df)
        mtf_info = build_mtf_signals_from_intraday(intra_df)
        prev_close = float(daily_df["close"].iloc[-2]) if daily_df is not None and len(daily_df) >= 2 else None
        ltp = iter_summary.get("LTP")
        pct_change = ((ltp - prev_close) / prev_close * 100) if (ltp is not None and prev_close and prev_close != 0) else 0.0
        if not iter_detail.empty:
            iter_detail.insert(0, "Symbol", sym)
            iter_detail.insert(1, "% Change", pct_change)
            iteration_rows.append(iter_detail)
        row = {
            "Symbol": sym,
            "LTP": ltp,
            "% Change": pct_change,
            "Current Volume": iter_summary.get("Current Volume"),
            "10 Day Relative Volume": iter_summary.get("10 Day Relative Volume"),
            "20 Day Relative Volume": iter_summary.get("20 Day Relative Volume"),
            "Cumulative RSI": iter_summary.get("Cumulative RSI"),
            "Cumulative OBV": iter_summary.get("Cumulative OBV"),
            "Cumulative VWAP": iter_summary.get("Cumulative VWAP"),
            "VWAP Z-Score": iter_summary.get("VWAP Z-Score"),
            "Total Iterations": iter_summary.get("Total Iterations"),
            "Last Iteration Minutes": iter_summary.get("Last Iteration Minutes"),
            "Last Iteration Time": iter_summary.get("Last Iteration Time"),
            "Cumulative KER": iter_summary.get("Cumulative KER"),
            "Cumulative +DI": iter_summary.get("Cumulative +DI"),
            "Cumulative -DI": iter_summary.get("Cumulative -DI"),
            "Cumulative ADX": iter_summary.get("Cumulative ADX"),
            "Survival Score": iter_summary.get("Survival Score"),
            "Survival_Num": iter_summary.get("Survival_Num"),
            "HOD": iter_summary.get("HOD"),
            "Strike_Distance": iter_summary.get("Strike_Distance"),
            "Last_5m_Volume": iter_summary.get("Last_5m_Volume"),
            "Volume_1h_Avg_5m": iter_summary.get("Volume_1h_Avg_5m"),
            "OBV_30m_Delta": iter_summary.get("OBV_30m_Delta"),
            "RSI_30m_Delta": iter_summary.get("RSI_30m_Delta"),
            "Freshness_Score": iter_summary.get("Freshness_Score"),
            "Fresh_State": iter_summary.get("Fresh_State"),
            "Fresh_Since": iter_summary.get("Fresh_Since"),
            "Is_Fresh": iter_summary.get("Is_Fresh")
        }
        row.update(mtf_info)
        rows.append(row)
    return pd.DataFrame(rows), (pd.concat(iteration_rows, ignore_index=True) if iteration_rows else pd.DataFrame())


def signal_color(label: str) -> str:
    label = str(label).strip()
    if label == "Buy++": return "#0b6623"
    if label == "Buy+": return "#1e8449"
    if label == "Buy": return "#27ae60"
    if label == "Bullish": return "#145a32"
    if label == "Neutral": return "#555555"
    if label == "Mixed": return "#555555"
    if label == "Sell++": return "#922b21"
    if label == "Sell+": return "#c0392b"
    if label == "Sell": return "#e74c3c"
    if label == "Bearish": return "#7b241c"
    return "#2c2c2c"


def format_value(col: str, val):
    if pd.isna(val):
        return ""
    if col == "% Change":
        return f"{float(val):.2f}%"
    if isinstance(val, (int, float, np.integer, np.floating)):
        return f"{float(val):.2f}"
    return str(val)


def build_candidate_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty:
        return pd.DataFrame(columns=EMAIL_DISPLAY_COLS), pd.DataFrame(columns=EMAIL_DISPLAY_COLS)
    base = df.copy()
    strict_long = base[
        (base["% Change"] > 0) &
        (base["Cumulative +DI"] > base["Cumulative -DI"]) &
        (base["HigherTF Confirm"] == "Bullish")
    ].copy()
    strict_short = base[
        (base["% Change"] < 0) &
        (base["Cumulative -DI"] > base["Cumulative +DI"]) &
        (base["HigherTF Confirm"] == "Bearish")
    ].copy()
    long_df = strict_long.sort_values(by=["Cumulative KER", "Survival_Num", "Cumulative ADX", "% Change"], ascending=[False, False, False, False], na_position="last").drop_duplicates(subset=["Symbol"]).head(15)
    short_df = strict_short.sort_values(by=["Cumulative KER", "Survival_Num", "Cumulative ADX", "% Change"], ascending=[False, False, False, True], na_position="last").drop_duplicates(subset=["Symbol"]).head(15)
    long_df = long_df[[c for c in EMAIL_DISPLAY_COLS if c in long_df.columns]] if not long_df.empty else pd.DataFrame(columns=EMAIL_DISPLAY_COLS)
    short_df = short_df[[c for c in EMAIL_DISPLAY_COLS if c in short_df.columns]] if not short_df.empty else pd.DataFrame(columns=EMAIL_DISPLAY_COLS)
    return long_df, short_df


def df_to_html_table(df: pd.DataFrame, max_rows: int = 15) -> str:
    if df is None or df.empty:
        return '<p style="font-family:Arial,sans-serif;color:#dddddd;">No candidates found.</p>'
    df_slice = df.head(max_rows).copy()
    cols = [c for c in EMAIL_DISPLAY_COLS if c in df_slice.columns]
    if not cols:
        return '<p style="font-family:Arial,sans-serif;color:#dddddd;">No candidates found.</p>'
    html = ['<table role="presentation" cellpadding="0" cellspacing="0" border="0" style="border-collapse:collapse;font-family:Arial,sans-serif;font-size:12px;background:#111111;color:#f5f5f5;">']
    html.append('<tr>')
    for c in cols:
        html.append(f'<th bgcolor="#202225" style="border:1px solid #333333;padding:6px 8px;color:#f5f5f5;text-align:center;font-weight:700;white-space:nowrap;">{c}</th>')
    html.append('</tr>')
    for _, row in df_slice.iterrows():
        html.append('<tr>')
        for c in cols:
            val = format_value(c, row[c])
            bgcolor = '#181a1b'
            color = '#f5f5f5'
            extra = ''
            if c == 'Symbol':
                color = '#ff7b72'
                extra = 'font-weight:700;'
            elif c == 'LTP':
                color = '#6ee7a8'
                extra = 'font-weight:700;'
            elif c == '% Change':
                try:
                    color = '#6ee7a8' if float(row[c]) >= 0 else '#ff7b72'
                    extra = 'font-weight:700;'
                except Exception:
                    pass
            if c.endswith('_Signal') or c in ('Bull_Signal', 'Bear_Signal', 'Overall_Signal', 'HigherTF Confirm'):
                bgcolor = signal_color(val)
                color = '#ffffff'
                extra = 'font-weight:700;'
            html.append(f'<td bgcolor="{bgcolor}" style="border:1px solid #333333;padding:5px 7px;text-align:center;color:{color};{extra}white-space:nowrap;">{val}</td>')
        html.append('</tr>')
    html.append('</table>')
    return ''.join(html)


def send_email_with_tables(long_df: pd.DataFrame, short_df: pd.DataFrame, csv_filename: str, detail_csv_filename: str) -> bool:
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "465"))
    sender_email = os.environ.get("SENDER_EMAIL")
    sender_password = os.environ.get("SENDER_PASSWORD")
    recipient_email = os.environ.get("RECIPIENT_EMAIL") or sender_email
    if not sender_email or not sender_password or not recipient_email:
        logger.error("EMAIL Missing sender/recipient SMTP credentials in environment.")
        return False
    long_table = df_to_html_table(long_df)
    short_table = df_to_html_table(short_df)
    scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
    html_body = (
        '<html><body style="font-family:Arial,sans-serif;background:#0f1115;color:#f5f5f5;padding:16px;">'
        '<h2 style="color:#f5f5f5;">Long Candidates</h2>' + long_table +
        '<br><h2 style="color:#f5f5f5;">Short Candidates</h2>' + short_table +
        f'<p style="margin-top:16px;color:#cccccc;">Scan completed at {scan_time}.</p>' +
        '</body></html>'
    )
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = f"Intraday Vol Iteration Alert - {datetime.now().strftime('%d %b %H:%M')}"
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    for filename in [csv_filename, detail_csv_filename]:
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(filename)}")
            msg.attach(part)
    try:
        if smtp_port == 465:
            with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=40) as server:
                server.login(sender_email, sender_password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=40) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(sender_email, sender_password)
                server.send_message(msg)
        logger.info(f"EMAIL Sent successfully to {recipient_email}")
        return True
    except Exception as e:
        logger.error(f"EMAIL Failed to send email: {type(e).__name__}: {e}")
        return False


def main():
    logger.info("Starting F&O Iteration Volume Volatility Scan")
    init_fyers()
    df_all, df_iter = scan_fno_universe()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = f"fo_fyers_iteration_summary_{timestamp}.csv"
    detail_csv = f"fo_fyers_iteration_details_{timestamp}.csv"
    if df_all is not None and not df_all.empty:
        df_all.to_csv(summary_csv, index=False)
        long_df, short_df = build_candidate_tables(df_all)
    else:
        pd.DataFrame().to_csv(summary_csv, index=False)
        long_df, short_df = pd.DataFrame(columns=EMAIL_DISPLAY_COLS), pd.DataFrame(columns=EMAIL_DISPLAY_COLS)
    if df_iter is not None and not df_iter.empty:
        df_iter.to_csv(detail_csv, index=False)
    else:
        pd.DataFrame().to_csv(detail_csv, index=False)
    send_email_with_tables(long_df, short_df, summary_csv, detail_csv)
    logger.info("Scan Pipeline Completed")


if __name__ == "__main__":
    main()
