import os
import smtplib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, time as dtime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders

try:
    from fyers_apiv3 import fyersModel
except Exception:
    try:
        import fyersModel
    except Exception:
        from fyersapi import fyersModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DAILY_LOOKBACK_DAYS   = 252
INTRADAY_LOOKBACK_DAYS = 20
IVP_LOOKBACK_DAYS     = 252
OPTION_PAIRS_TO_KEEP  = 5
ITERATIONS_TO_KEEP    = 75
CSV_DIR               = os.environ.get("CSV_DIR", "sectors")
OUTPUT_DIR            = os.environ.get("OUTPUT_DIR", ".")
MIN_OPTION_LTP        = 10.0
EMAIL_MAX_ROWS_LONG   = int(os.environ.get("EMAIL_MAX_ROWS_LONG",  "14"))
EMAIL_MAX_ROWS_SHORT  = int(os.environ.get("EMAIL_MAX_ROWS_SHORT", "14"))
EMAIL_SAFE_WIDTH      = int(os.environ.get("EMAIL_SAFE_WIDTH",     "600"))
TOP_N_UNDERLYINGS     = int(os.environ.get("TOP_N_UNDERLYINGS",    "60"))
MAX_WORKERS           = int(os.environ.get("MAX_WORKERS",          "4"))

OPTION_EMAIL_COLS = [
    "Underlying","Option Type","Option Symbol","Strike","LTP","% Change",
    "OI","Volume","OBV","OI+Volume+OBV Score","EMAIL_RANK_SCORE",
    "Rank Delta","Cumulative ADX","5m_Signal","15m_Signal","30m_Signal",
    "60m_Signal","Bull_Signal","Bear_Signal","Overall_Signal",
    "Price_Lead_Status","IVP","Volatility State","Last Iteration Time",
]

OPTION_EMAIL_COL_RENAME = {
    "OI+Volume+OBV Score": "Liq Score",
    "EMAIL_RANK_SCORE":    "Rank",
    "% Change":            "% Chg",
    "Last Iteration Time": "Time",
}

fyers = None


# â”€â”€â”€ Fyers init â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def init_fyers():
    global fyers
    client_id    = os.environ.get("CLIENT_ID")    or os.environ.get("CLIENTID")
    access_token = os.environ.get("ACCESS_TOKEN") or os.environ.get("ACCESSTOKEN")
    if not client_id or not access_token:
        logger.error("Missing CLIENT_ID / ACCESS_TOKEN")
        return None
    fyers = fyersModel.FyersModel(client_id=client_id, token=access_token,
                                   is_async=False, log_path="")
    return fyers


# â”€â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def safe_float(value, default=np.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_series(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if frame is not None and col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce").fillna(default)
    if frame is None:
        return pd.Series(dtype=float)
    return pd.Series(default, index=frame.index, dtype=float)


# â”€â”€â”€ Sector / symbol loading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def discover_csvs(root_dir: str = CSV_DIR) -> List[str]:
    if not os.path.isdir(root_dir):
        return []
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(".csv"):
                paths.append(os.path.join(dirpath, fname))
    return sorted(set(paths))


def load_fno_symbols_from_csvs(root_dir: str = CSV_DIR) -> List[str]:
    symbols = set()
    for path in discover_csvs(root_dir):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        lowered = {str(c).strip().lower(): c for c in df.columns}
        symbol_col = None
        for key in ["symbol","symbols","ticker","tradingsymbol"]:
            if key in lowered:
                symbol_col = lowered[key]
                break
        if symbol_col is None:
            continue
        for raw in df[symbol_col].dropna().astype(str):
            sym = raw.strip().upper()
            if sym and sym not in {"NAN","NONE"}:
                symbols.add(sym)
    return sorted(symbols)


def format_eq_symbol(symbol: str) -> str:
    symbol = str(symbol).strip().upper()
    if symbol.startswith("NSE:"):
        return symbol if symbol.endswith("-EQ") else f"{symbol}-EQ"
    return f"NSE:{symbol}-EQ"


# â”€â”€â”€ Data fetch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def get_history(symbol: str, resolution: str, days_back: int) -> pd.DataFrame:
    if fyers is None:
        return pd.DataFrame()
    now   = datetime.now()
    start = now - timedelta(days=days_back)
    payload = {
        "symbol":     symbol,
        "resolution": resolution,
        "date_format":"1",
        "range_from": start.strftime("%Y-%m-%d"),
        "range_to":   now.strftime("%Y-%m-%d"),
        "cont_flag":  "1",
    }
    try:
        res = fyers.history(data=payload)
    except Exception:
        return pd.DataFrame()
    candles = (res or {}).get("candles", [])
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], unit="s", utc=True)
          .dt.tz_convert("Asia/Kolkata")
          .dt.tz_localize(None)
    )
    return (df.sort_values("timestamp")
              .drop_duplicates(subset=["timestamp"], keep="last")
              .reset_index(drop=True))


# â”€â”€â”€ OBV (true cumulative) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def compute_obv(df: pd.DataFrame) -> float:
    """True cumulative OBV: +vol when close rises, -vol when falls, unchanged otherwise."""
    if df is None or df.empty or len(df) < 2:
        return np.nan
    d     = df.copy().sort_values("timestamp").reset_index(drop=True)
    close = pd.to_numeric(d["close"],  errors="coerce")
    vol   = pd.to_numeric(d["volume"], errors="coerce").fillna(0.0)
    obv   = [0.0]
    for i in range(1, len(d)):
        prev, curr = close.iat[i-1], close.iat[i]
        if pd.isna(prev) or pd.isna(curr):
            obv.append(obv[-1])
        elif curr > prev:
            obv.append(obv[-1] + float(vol.iat[i]))
        elif curr < prev:
            obv.append(obv[-1] - float(vol.iat[i]))
        else:
            obv.append(obv[-1])
    return round(float(obv[-1]), 2)


def compute_today_obv(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return np.nan
    d = df.copy().sort_values("timestamp")
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    if len(d) < 2:
        return np.nan
    return compute_obv(d)


# â”€â”€â”€ IVP â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def compute_ivp(history_df: pd.DataFrame, min_bars: int = 10) -> Tuple[float, str]:
    if history_df is None or history_df.empty or len(history_df) < min_bars:
        return np.nan, "Neutral Vol"
    close  = pd.to_numeric(history_df["close"], errors="coerce")
    high   = pd.to_numeric(history_df["high"],  errors="coerce")
    low    = pd.to_numeric(history_df["low"],   errors="coerce")
    proxy  = ((high - low) / close.replace(0, np.nan) * 100.0
              ).replace([np.inf, -np.inf], np.nan).dropna()
    if proxy.empty:
        return np.nan, "Neutral Vol"
    lookback = proxy.tail(min(IVP_LOOKBACK_DAYS, len(proxy)))
    current  = float(lookback.iloc[-1])
    ivp      = round((lookback.lt(current).sum() / len(lookback)) * 100, 2)
    if ivp < 30:
        return ivp, "Buyer Zone"
    if ivp > 50:
        return ivp, "Avoid Buy Premium"
    return ivp, "Neutral Vol"


# â”€â”€â”€ Signal helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def score_label(delta: float) -> str:
    if pd.isna(delta): return "Neutral"
    if delta >=  7: return "Buy++"
    if delta >=  4: return "Buy+"
    if delta >=  1: return "Buy"
    if delta <= -7: return "Sell++"
    if delta <= -4: return "Sell+"
    if delta <= -1: return "Sell"
    return "Neutral"


def directional_label(raw_label: str, side: str) -> str:
    raw = str(raw_label).strip().upper()
    if side == "long":
        return {"BUY":"LONG","BUY+":"LONG+","BUY++":"LONG++",
                "SELL":"SHORT","SELL+":"SHORT+","SELL++":"SHORT++"}.get(raw, raw)
    return {"SELL":"SHORT","SELL+":"SHORT+","SELL++":"SHORT++",
            "BUY":"LONG","BUY+":"LONG+","BUY++":"LONG++"}.get(raw, raw)


def apply_display_labels(df: pd.DataFrame, side: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in ["5m_Signal","15m_Signal","30m_Signal","60m_Signal",
                "Bull_Signal","Bear_Signal","Overall_Signal"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: directional_label(str(x), side))
    return out


# â”€â”€â”€ Window scoring â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def intraday_window_score(df: pd.DataFrame, window_minutes: int = 5) -> float:
    if df is None or df.empty or len(df) < 2:
        return np.nan
    d      = df.copy().sort_values("timestamp")
    end_ts = pd.to_datetime(d["timestamp"].iloc[-1])
    cur    = d[d["timestamp"] >= end_ts - timedelta(minutes=window_minutes)]
    if len(cur) < 2:
        return np.nan
    f = safe_float(cur["close"].iloc[0])
    l = safe_float(cur["close"].iloc[-1])
    if pd.isna(f) or f == 0:
        return np.nan
    return round(((l - f) / f) * 100.0, 2)


def previous_trading_day_same_time_score(full_df: pd.DataFrame,
                                          end_ts: Optional[pd.Timestamp] = None,
                                          window_minutes: int = 5) -> float:
    if full_df is None or full_df.empty or len(full_df) < 4:
        return np.nan
    d = full_df.copy().sort_values("timestamp").reset_index(drop=True)
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    end_ts = pd.to_datetime(end_ts) if end_ts is not None else d["timestamp"].iloc[-1]
    trading_days = sorted(d["timestamp"].dt.date.unique())
    prev_days    = [day for day in trading_days if day < end_ts.date()]
    if not prev_days:
        return np.nan
    prev_day      = prev_days[-1]
    prev_day_data = d[d["timestamp"].dt.date == prev_day]
    if prev_day_data.empty:
        return np.nan
    same = prev_day_data[(prev_day_data["timestamp"].dt.hour   == end_ts.hour) &
                          (prev_day_data["timestamp"].dt.minute == end_ts.minute)]
    prev_end   = pd.to_datetime(same.iloc[-1]["timestamp"]) if not same.empty else prev_day_data["timestamp"].iloc[-1]
    prev_win   = prev_day_data[prev_day_data["timestamp"] >= prev_end - timedelta(minutes=window_minutes)]
    if len(prev_win) < 2:
        return np.nan
    f = safe_float(prev_win["close"].iloc[0])
    l = safe_float(prev_win["close"].iloc[-1])
    if pd.isna(f) or f == 0:
        return np.nan
    return round(((l - f) / f) * 100.0, 2)


def compare_window_signal(current: float, previous: float) -> Tuple[float, str]:
    if pd.isna(current) or pd.isna(previous):
        return np.nan, "Neutral"
    d = round(current - previous, 2)
    if d >=  0.50: return d, "Buy++"
    if d >=  0.20: return d, "Buy+"
    if d >=  0.05: return d, "Buy"
    if d <= -0.50: return d, "Sell++"
    if d <= -0.20: return d, "Sell+"
    if d <= -0.05: return d, "Sell"
    return d, "Neutral"


# â”€â”€â”€ Iteration history â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def build_iteration_history(intra_df: pd.DataFrame,
                             iterations: int = ITERATIONS_TO_KEEP) -> pd.DataFrame:
    if intra_df is None or intra_df.empty:
        return pd.DataFrame()
    full = intra_df.copy().sort_values("timestamp").reset_index(drop=True)
    full["timestamp"] = pd.to_datetime(full["timestamp"])
    last_day = full["timestamp"].dt.date.max()
    d = full[full["timestamp"].dt.date == last_day].reset_index(drop=True)
    if d.empty:
        return pd.DataFrame()
    rows = []
    for i in range(len(d)):
        end_ts  = pd.to_datetime(d.loc[i, "timestamp"])
        start_ts = end_ts - timedelta(minutes=5)
        cur = d[d["timestamp"] >= start_ts]
        if len(cur) < 2:
            continue
        f = safe_float(cur["close"].iloc[0])
        l = safe_float(cur["close"].iloc[-1])
        if pd.isna(f) or f == 0:
            continue
        cs   = round(((l - f) / f) * 100.0, 2)
        ps   = previous_trading_day_same_time_score(full, end_ts, 5)
        dlt, sig = compare_window_signal(cs, ps)
        rows.append({
            "iteration":   len(rows) + 1,
            "timestamp":   end_ts.strftime("%H:%M"),
            "window_start":start_ts.strftime("%H:%M"),
            "window_end":  end_ts.strftime("%H:%M"),
            "current_window_score": cs,
            "previous_trading_day_same_time_score": ps,
            "window_delta":  dlt,
            "window_signal": sig,
            "close":         l,
        })
        if len(rows) >= iterations:
            break
    return pd.DataFrame(rows)


# â”€â”€â”€ Intraday summarise â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def summarize_intraday(intra_df: pd.DataFrame,
                        reference_df: pd.DataFrame) -> Dict[str, object]:
    if intra_df is None or intra_df.empty:
        return {}
    df     = intra_df.copy().sort_values("timestamp").reset_index(drop=True)
    close  = pd.to_numeric(df["close"],  errors="coerce")
    high   = pd.to_numeric(df["high"],   errors="coerce")
    low    = pd.to_numeric(df["low"],    errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

    dlt       = close.diff().fillna(0.0)
    avg_gain  = dlt.clip(lower=0.0).rolling(14, min_periods=14).mean()
    avg_loss  = (-dlt).clip(lower=0.0).rolling(14, min_periods=14).mean()
    rsi       = (100 - 100 / (1 + avg_gain / avg_loss.replace(0, np.nan))).fillna(50)

    tr       = pd.concat([(high - low).abs(),(high - close.shift()).abs(),(low - close.shift()).abs()], axis=1).max(axis=1)
    up_mv    = high.diff()
    dn_mv    = -low.diff()
    plus_dm  = np.where((up_mv > dn_mv) & (up_mv > 0), up_mv, 0.0)
    minus_dm = np.where((dn_mv > up_mv) & (dn_mv > 0), dn_mv, 0.0)
    atr      = pd.Series(tr).rolling(14, min_periods=14).mean()
    plus_di  = 100 * pd.Series(plus_dm).rolling(14, min_periods=14).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm).rolling(14, min_periods=14).mean() / atr.replace(0, np.nan)
    dx       = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    adx      = dx.rolling(14, min_periods=14).mean().fillna(0)

    typical  = (high + low + close) / 3.0
    cum_vol  = volume.cumsum().replace(0, np.nan)
    vwap     = ((typical * volume).cumsum() / cum_vol).ffill().fillna(close)
    vwap_std = (((typical - vwap) ** 2 * volume).cumsum() / cum_vol).pow(0.5).replace(0, np.nan)
    vwap_z   = ((close - vwap) / vwap_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    prev_close = safe_float(reference_df["close"].iloc[-2]) if reference_df is not None and len(reference_df) >= 2 else np.nan
    ltp        = safe_float(close.iloc[-1])
    pct_change = ((ltp - prev_close) / prev_close * 100.0) if pd.notna(prev_close) and prev_close != 0 else 0.0

    cur_win  = intraday_window_score(df)
    prev_win = previous_trading_day_same_time_score(intra_df)
    _, win_signal = compare_window_signal(cur_win, prev_win)
    iter_hist     = build_iteration_history(intra_df)
    ivp, vol_state = compute_ivp(reference_df)

    bull = bear = 0
    if pct_change > 0: bull += 1
    if pct_change < 0: bear += 1
    if safe_float(vwap_z.iloc[-1], 0) >=  0.30: bull += 1
    if safe_float(vwap_z.iloc[-1], 0) <= -0.30: bear += 1
    if safe_float(plus_di.iloc[-1],  0) > safe_float(minus_di.iloc[-1], 0): bull += 1
    if safe_float(minus_di.iloc[-1], 0) > safe_float(plus_di.iloc[-1],  0): bear += 1
    if safe_float(adx.iloc[-1], 0) >= 20:
        if safe_float(plus_di.iloc[-1],  0) > safe_float(minus_di.iloc[-1], 0): bull += 1
        elif safe_float(minus_di.iloc[-1],0) > safe_float(plus_di.iloc[-1], 0): bear += 1
    if safe_float(rsi.iloc[-1], 50) >= 55: bull += 1
    if safe_float(rsi.iloc[-1], 50) <= 45: bear += 1
    if win_signal.startswith("Buy"):  bull += 2
    elif win_signal.startswith("Sell"): bear += 2

    rank_delta = bull - bear
    last_ts    = pd.to_datetime(df["timestamp"].iloc[-1])

    return {
        "LTP":              round(ltp, 2),
        "% Change":         round(pct_change, 2),
        "5m_Signal":        score_label(rank_delta),
        "15m_Signal":       win_signal,
        "30m_Signal":       score_label(rank_delta * 0.8),
        "60m_Signal":       score_label(rank_delta * 0.7),
        "Bull_Signal":      score_label(bull),
        "Bear_Signal":      score_label(-bear),
        "Overall_Signal":   score_label(rank_delta),
        "Price_Lead_Status":"NORMAL",
        "IVP":              ivp,
        "Volatility State": vol_state,
        "Last Iteration Time": last_ts.strftime("%H:%M"),
        "Bull Rank":        bull,
        "Bear Rank":        bear,
        "Rank Delta":       rank_delta,
        "Cumulative ADX":   round(safe_float(adx.iloc[-1], np.nan), 2),
        "Iteration History":iter_hist,
    }


# â”€â”€â”€ Candidate selection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def choose_top_candidates(summary_df: pd.DataFrame,
                           top_n: int = TOP_N_UNDERLYINGS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if summary_df is None or summary_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    rd      = safe_series(summary_df, "Rank Delta", 0)
    long_df  = summary_df[rd > 0].sort_values(["Rank Delta","Cumulative ADX","% Change"],
                                               ascending=[False,False,False]).head(top_n)
    short_df = summary_df[rd < 0].sort_values(["Rank Delta","Cumulative ADX","% Change"],
                                               ascending=[True, False,True]).head(top_n)
    return long_df.reset_index(drop=True), short_df.reset_index(drop=True)


# â”€â”€â”€ Option chain â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def nearest_step(value: float) -> int:
    val = abs(safe_float(value, 0))
    if val >= 20000: return 100
    if val >= 10000: return 50
    if val >= 2000:  return 20
    if val >= 500:   return 10
    if val >= 100:   return 5
    return 1


def fetch_option_pairs(symbol: str, pair_count: int = OPTION_PAIRS_TO_KEEP) -> pd.DataFrame:
    if fyers is None:
        return pd.DataFrame()
    eq = format_eq_symbol(symbol)
    try:
        quote     = fyers.quotes({"symbols": eq})
        ltp       = safe_float(quote.get("d",[{}])[0].get("v",{}).get("lp"), np.nan)
        chain_res = fyers.optionchain(data={"symbol": eq, "strikecount": 50})
    except Exception:
        return pd.DataFrame()
    chain = ((chain_res or {}).get("data") or {}).get("optionsChain", [])
    if not chain:
        return pd.DataFrame()
    rows = []
    for item in chain:
        strike = safe_float(item.get("strike_price") or item.get("strike"), np.nan)
        typ    = str(item.get("option_type") or item.get("type") or "").upper()
        if pd.isna(strike) or typ not in {"CE","PE"}:
            continue
        rows.append({
            "Strike":        strike,
            "Option Type":   typ,
            "Option Symbol": str(item.get("symbol","")),
            "OI":            safe_float(item.get("oi") or item.get("open_interest"), np.nan),
            "Chain Volume":  safe_float(item.get("volume"), np.nan),
        })
    oc = pd.DataFrame(rows)
    if oc.empty:
        return pd.DataFrame()
    step   = nearest_step(ltp if pd.notna(ltp) else oc["Strike"].median())
    atm    = round(ltp / step) * step if pd.notna(ltp) else oc["Strike"].median()
    strikes = sorted(oc["Strike"].dropna().unique(), key=lambda x: abs(x - atm))[:pair_count]
    return oc[oc["Strike"].isin(strikes)].reset_index(drop=True)


# â”€â”€â”€ Single option scan â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def scan_single_option(option_symbol: str, option_type: str,
                        strike: float, underlying: str) -> Optional[Dict]:
    hist_sym = option_symbol if option_symbol.startswith("NSE:") else f"NSE:{option_symbol}"
    daily_df = get_history(hist_sym, "D", max(DAILY_LOOKBACK_DAYS, IVP_LOOKBACK_DAYS))
    intra_df = get_history(hist_sym, "5", INTRADAY_LOOKBACK_DAYS)
    if daily_df.empty or intra_df.empty:
        return None
    summary = summarize_intraday(intra_df, daily_df)
    if not summary:
        return None
    obv_val = compute_today_obv(intra_df)
    if pd.isna(obv_val):
        obv_val = 0.0
    summary.update({
        "Underlying":   underlying,
        "Option Type":  option_type,
        "Option Symbol":option_symbol,
        "Strike":       strike,
        "OBV":          obv_val,
        "OI":           np.nan,
        "Volume":       float(intra_df["volume"].sum()) if "volume" in intra_df.columns else 0.0,
        "IVP":          float(summary.get("IVP", 0.0)) if pd.notna(summary.get("IVP", np.nan)) else 0.0,
        "Volatility State": summary.get("Volatility State", "Neutral Vol") or "Neutral Vol",
    })
    return summary


def option_liquidity_score(oi, volume, obv) -> float:
    return round(
        np.log1p(max(safe_float(oi,     0), 0)) * 0.45 +
        np.log1p(max(safe_float(volume, 0), 0)) * 0.35 +
        np.log1p(max(abs(safe_float(obv,0)),0)) * 0.20, 4)


# â”€â”€â”€ Ranking (no option-type bonus) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def rank_option_candidates(df: pd.DataFrame, side: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    liq = safe_series(out, "OI+Volume+OBV Score", 0)
    rd  = safe_series(out, "Rank Delta", 0)
    adx = safe_series(out, "Cumulative ADX", 0)
    pct = safe_series(out, "% Change", 0)
    out["EMAIL_RANK_SCORE"] = liq * 0.40 + rd * 0.30 + adx * 0.18 + pct * 0.12
    if side == "long":
        out = out[out["EMAIL_RANK_SCORE"] > 0].copy()
        out = out[out["Rank Delta"] >= 0].copy()
        out = out[out["Overall_Signal"].astype(str).str.upper().str.contains("BUY|LONG", na=False)].copy()
        out = out.sort_values(["EMAIL_RANK_SCORE","OI+Volume+OBV Score","Rank Delta","Cumulative ADX","% Change"],
                              ascending=[False,False,False,False,False])
    else:
        out["EMAIL_RANK_SCORE"] = liq * 0.40 + (-rd) * 0.30 + adx * 0.18 + (-pct) * 0.12
        out = out[out["EMAIL_RANK_SCORE"] > 0].copy()
        out = out[out["Rank Delta"] <= 0].copy()
        out = out[out["Overall_Signal"].astype(str).str.upper().str.contains("SELL|SHORT", na=False)].copy()
        out = out.sort_values(["EMAIL_RANK_SCORE","OI+Volume+OBV Score","Rank Delta","Cumulative ADX","% Change"],
                              ascending=[False,False,True,False,True])
    return out.reset_index(drop=True)


# â”€â”€â”€ Threaded option task â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _scan_option_task(underlying: str, row_dict: dict):
    strike   = safe_float(row_dict.get("Strike"), np.nan)
    opt_type = str(row_dict.get("Option Type","")).upper()
    sym      = str(row_dict.get("Option Symbol",""))
    if not sym or opt_type not in {"CE","PE"}:
        return None
    scanned = scan_single_option(sym, opt_type, strike, underlying)
    if not scanned:
        return None
    scanned["OI"]              = safe_float(row_dict.get("OI"), 0)
    scanned["Chain_Volume"]    = safe_float(row_dict.get("Chain Volume"), 0)
    scanned["OI+Volume+OBV Score"] = option_liquidity_score(
        scanned.get("OI",0), scanned.get("Volume",0), scanned.get("OBV",np.nan))
    if safe_float(scanned.get("LTP"), 0.0) < MIN_OPTION_LTP:
        return None
    hist = scanned.get("Iteration History")
    if isinstance(hist, pd.DataFrame) and not hist.empty:
        tmp = hist.tail(20).copy()
        tmp.insert(0, "Option Symbol", sym)
        tmp.insert(1, "Underlying",    underlying)
        tmp.insert(2, "Strike",        strike)
        tmp.insert(3, "Option Type",   opt_type)
        return scanned, tmp
    return scanned, None


def build_option_candidates(candidates_df: pd.DataFrame,
                              side: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if candidates_df is None or candidates_df.empty or "Symbol" not in candidates_df.columns:
        return pd.DataFrame(), pd.DataFrame()
    tasks = []
    for underlying in candidates_df["Symbol"].dropna().astype(str):
        pair_df = fetch_option_pairs(underlying)
        if pair_df.empty:
            continue
        for _, row in pair_df.iterrows():
            tasks.append((underlying, row.to_dict()))
    if not tasks:
        return pd.DataFrame(), pd.DataFrame()

    rows, iter_rows = [], []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_scan_option_task, u, r): (u, r) for u, r in tasks}
        for fut in as_completed(futures):
            try:
                result = fut.result()
            except Exception:
                u, r = futures[fut]
                logger.exception("Option scan failed: %s %s", u, r.get("Option Symbol"))
                continue
            if not result:
                continue
            scanned, hist = result
            rows.append(scanned)
            if hist is not None and not hist.empty:
                iter_rows.append(hist)

    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    out = pd.DataFrame(rows)
    out = rank_option_candidates(out, side)
    if out.empty:
        return pd.DataFrame(), pd.DataFrame()
    rd  = safe_series(out, "Rank Delta", 0)
    pct = safe_series(out, "% Change",  0)
    out = out[(rd > 0) & (pct >= 0)].copy() if side == "long" else out[(rd < 0) & (pct <= 0)].copy()
    final_cols = [c for c in OPTION_EMAIL_COLS if c in out.columns]
    final_out  = out[final_cols].reset_index(drop=True)

    iter_df = pd.DataFrame()
    if iter_rows and not final_out.empty:
        all_iters = pd.concat(iter_rows, ignore_index=True)
        all_iters = all_iters[all_iters["Option Symbol"].isin(final_out["Option Symbol"])].copy()
        if not all_iters.empty:
            scols = [c for c in ["Underlying","Option Type","Strike","Option Symbol","iteration"] if c in all_iters.columns]
            if scols:
                all_iters = all_iters.sort_values(scols).reset_index(drop=True)
            gcols = [c for c in ["Underlying","Option Type","Strike","Option Symbol"] if c in all_iters.columns]
            if gcols:
                all_iters["iteration"] = all_iters.groupby(gcols).cumcount() + 1
        iter_df = all_iters.reset_index(drop=True)
    return final_out, iter_df


# â”€â”€â”€ Email helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def format_cell(col: str, val) -> str:
    if pd.isna(val):
        return ""
    if col in {"% Change","% Chg"}:
        return f"{float(val):.2f}%"
    if col in {"OI","Volume","OBV"}:
        try:
            return f"{int(float(val)):,}"
        except Exception:
            return ""
    if col in {"Rank","Liq Score","LTP","Strike","IVP"}:
        try:
            return f"{float(val):.2f}"
        except Exception:
            return ""
    if isinstance(val, (int, float, np.integer, np.floating)):
        return f"{float(val):.2f}"
    return str(val)


def compact_table_html(df: pd.DataFrame, title: str, max_rows: int) -> str:
    cols = ["Underlying","Option Type","Strike","LTP","% Change","OI","Volume","OBV",
            "EMAIL_RANK_SCORE","5m_Signal","15m_Signal","Overall_Signal","IVP","Last Iteration Time"]
    cols = [c for c in cols if c in df.columns]
    html = [f"<tr><td style='padding:10px 12px 4px 12px;font-family:Arial,Helvetica,sans-serif;font-size:13px;color:#111;'><b>{title}</b></td></tr>"]
    if df is None or df.empty:
        html.append("<tr><td style='padding:0 12px 12px 12px;font-family:Arial,Helvetica,sans-serif;font-size:12px;color:#111;'>No data found.</td></tr>")
        return "".join(html)
    view = df[cols].head(max_rows).copy().rename(columns=OPTION_EMAIL_COL_RENAME)
    if "Rank" in view.columns:
        view["Rank"] = pd.to_numeric(view["Rank"], errors="coerce")
    if "OBV" in view.columns:
        view["OBV"] = pd.to_numeric(view["OBV"], errors="coerce")
    html.append("<tr><td style='padding:0 12px 12px 12px;'>"
                "<table width='100%' border='0' cellpadding='0' cellspacing='1' "
                "style='border-collapse:separate;background:#ffffff;'>")
    html.append("<tr>")
    for c in view.columns:
        html.append(f"<th style='background:#2f3b59;color:#ffffff;padding:5px 4px;"
                    f"font-size:10px;font-family:Arial,Helvetica,sans-serif;"
                    f"white-space:nowrap;'>{c}</th>")
    html.append("</tr>")
    for _, row in view.iterrows():
        html.append("<tr>")
        for c in view.columns:
            html.append(f"<td style='background:#2d3651;color:#ffffff;padding:4px 4px;"
                        f"font-size:10px;font-family:Arial,Helvetica,sans-serif;"
                        f"white-space:nowrap;text-align:center;'>{format_cell(c, row[c])}</td>")
        html.append("</tr>")
    html.append("</table></td></tr>")
    return "".join(html)


def prepare_option_email_view(df: pd.DataFrame, side: str, max_rows: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=OPTION_EMAIL_COLS)
    out = df.copy()
    if "LTP" in out.columns:
        out = out[pd.to_numeric(out["LTP"], errors="coerce") >= MIN_OPTION_LTP].copy()
    out = apply_display_labels(out, side)
    out = rank_option_candidates(out, side)
    if out.empty:
        return pd.DataFrame(), pd.DataFrame()
    final_cols = [c for c in OPTION_EMAIL_COLS if c in out.columns]
    return out[final_cols].head(max_rows).reset_index(drop=True)


def build_email_html(view_df: pd.DataFrame, title: str,
                      scan_time: str, max_rows: int) -> str:
    body = compact_table_html(view_df, title, max_rows)
    return (f"<html><body style='margin:0;padding:0;background:#f4f4f4;'>"
            f"<table width='100%' border='0' cellpadding='0' cellspacing='0' "
            f"style='background:#f4f4f4;'><tr><td align='center' style='padding:12px;'>"
            f"<table width='{EMAIL_SAFE_WIDTH}' border='0' cellpadding='0' cellspacing='0' "
            f"style='width:{EMAIL_SAFE_WIDTH}px;max-width:{EMAIL_SAFE_WIDTH}px;"
            f"background:#ffffff;border-collapse:collapse;'>"
            f"<tr><td style='padding:12px 12px 6px 12px;font-family:Arial,Helvetica,sans-serif;"
            f"font-size:14px;color:#111;'><b>{title}</b></td></tr>"
            f"<tr><td style='padding:0 12px 10px 12px;font-family:Arial,Helvetica,sans-serif;"
            f"font-size:12px;color:#111;'>Scan completed at {scan_time}</td></tr>"
            f"{body}</table></td></tr></table></body></html>")


def send_single_email(subject: str, html_body: str, attachments: list) -> bool:
    sender   = os.environ.get("SENDER_EMAIL")
    recipient= os.environ.get("RECIPIENT_EMAIL")
    password = os.environ.get("SENDER_PASSWORD")
    if not sender or not recipient or not password:
        logger.error("Missing email env vars.")
        return False
    msg = MIMEMultipart()
    msg["From"]    = sender
    msg["To"]      = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    for path in attachments:
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                part = MIMEBase("application","octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition",
                            f'attachment; filename="{os.path.basename(path)}"')
            msg.attach(part)
    try:
        smtp_host = os.environ.get("SMTP_HOST","smtp.gmail.com")
        with smtplib.SMTP_SSL(smtp_host, 465, timeout=40) as s:
            s.login(sender, password)
            s.send_message(msg)
        logger.info("Email sent: %s", subject)
        return True
    except Exception:
        logger.exception("Email failed: %s", subject)
        return False


def send_direction_email(df: pd.DataFrame, direction: str, attachments: list) -> bool:
    subject_time = datetime.now().strftime("%d %b %H:%M")
    scan_time    = datetime.now().strftime("%d %b %Y, %H:%M")
    side     = "long" if direction.upper() == "LONG" else "short"
    max_rows = EMAIL_MAX_ROWS_LONG if side == "long" else EMAIL_MAX_ROWS_SHORT
    view     = prepare_option_email_view(df, side, max_rows=max_rows)
    html     = build_email_html(view, f"{direction} Candidates", scan_time, max_rows)
    return send_single_email(f"{direction} Candidates - {subject_time}", html, attachments)


# â”€â”€â”€ Underlying scan â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def scan_symbol(symbol: str) -> Optional[Dict]:
    eq       = format_eq_symbol(symbol)
    daily_df = get_history(eq, "D", max(DAILY_LOOKBACK_DAYS, IVP_LOOKBACK_DAYS))
    intra_df = get_history(eq, "5", INTRADAY_LOOKBACK_DAYS)
    if daily_df.empty or intra_df.empty:
        return None
    summary = summarize_intraday(intra_df, daily_df)
    if not summary:
        return None
    summary["Symbol"] = symbol
    return summary


# â”€â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    init_fyers()
    symbols = load_fno_symbols_from_csvs(CSV_DIR)
    logger.info("Scanning %s symbols from CSVs | MAX_WORKERS=%s", len(symbols), MAX_WORKERS)

    rows = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(scan_symbol, sym): sym for sym in symbols}
        for i, fut in enumerate(as_completed(futures), 1):
            sym = futures[fut]
            try:
                row = fut.result()
                if row:
                    rows.append(row)
                logger.info("[%s/%s] %s", i, len(symbols), sym)
            except Exception:
                logger.exception("Underlying scan failed: %s", sym)

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        raise RuntimeError("No symbols returned usable market data.")

    summary_df   = summary_df.sort_values(["Rank Delta","% Change"],
                                            ascending=[False,False]).reset_index(drop=True)
    long_seed, short_seed = choose_top_candidates(summary_df, top_n=TOP_N_UNDERLYINGS)
    long_df,  long_iter   = build_option_candidates(long_seed,  side="long")
    short_df, short_iter  = build_option_candidates(short_seed, side="short")
    iter_df = pd.concat([long_iter, short_iter], ignore_index=True) if (not long_iter.empty or not short_iter.empty) else pd.DataFrame()

    ts           = datetime.now().strftime("%Y%m%d_%H%M")
    summary_csv  = os.path.join(OUTPUT_DIR, f"fo_summary_{ts}.csv")
    long_csv     = os.path.join(OUTPUT_DIR, f"fo_long_candidates_{ts}.csv")
    short_csv    = os.path.join(OUTPUT_DIR, f"fo_short_candidates_{ts}.csv")
    iter_csv     = os.path.join(OUTPUT_DIR, f"fo_iteration_history_{ts}.csv")

    summary_df.to_csv(summary_csv, index=False)
    long_df.to_csv(long_csv,        index=False)
    short_df.to_csv(short_csv,      index=False)
    if iter_df.empty:
        iter_df = pd.DataFrame(columns=["iteration","Underlying","Option Type","Strike",
                                         "Option Symbol","timestamp","window_start","window_end",
                                         "current_window_score","previous_trading_day_same_time_score",
                                         "window_delta","window_signal","close"])
    iter_df.to_csv(iter_csv, index=False)

    logger.info("LONG  rows: %s", len(long_df))
    logger.info("SHORT rows: %s", len(short_df))

    attachments = [summary_csv, long_csv, short_csv, iter_csv]
    send_direction_email(long_df,  "LONG",  attachments)
    send_direction_email(short_df, "SHORT", attachments)


if __name__ == "__main__":
    main()
