import os
import json
import time
import smtplib
import logging
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

# â”€â”€ Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DAILY_LOOKBACK_DAYS     = 252
INTRADAY_LOOKBACK_DAYS  = 20
IVP_LOOKBACK_DAYS       = 252
OPTION_PAIRS_TO_KEEP    = 5
SIGNAL_WINDOW_MINUTES   = 5
ITERATIONS_TO_KEEP      = 75
SECTORS_DIR             = os.environ.get("SECTORS_DIR", "sectors")
OUTPUT_DIR              = os.environ.get("OUTPUT_DIR", ".")
MIN_OPTION_LTP          = 10.0
MIN_ATM_CHAIN_VOLUME    = int(os.environ.get("MIN_ATM_CHAIN_VOLUME", "100000"))
PER_SYMBOL_SLEEP_SEC    = float(os.environ.get("PER_SYMBOL_SLEEP_SEC", "0.25"))
EMAIL_MAX_ROWS_LONG     = int(os.environ.get("EMAIL_MAX_ROWS_LONG", "25"))
EMAIL_MAX_ROWS_SHORT    = int(os.environ.get("EMAIL_MAX_ROWS_SHORT", "25"))
EMAIL_SAFE_WIDTH        = int(os.environ.get("EMAIL_SAFE_WIDTH", "900"))
TOP_N_UNDERLYINGS       = int(os.environ.get("TOP_N_UNDERLYINGS", "60"))
OBV_BREAKOUT_WINDOW     = int(os.environ.get("OBV_BREAKOUT_WINDOW", "5"))

# â”€â”€ Chain Signal Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Entry : 5m CONFIRMED + 30m CONFIRMED + T30 >= T30_MIN
# Exit  : 30m BROKEN   OR  T30 < T30_MIN
T30_MIN          = int(os.environ.get("T30_MIN", "10"))
DAILY_STATE_FILE = os.path.join(OUTPUT_DIR, "chain_signal_state.json")

OPTION_EMAIL_COLS = [
    "Underlying", "Option Type", "Option Symbol", "Strike", "LTP", "% Change",
    "OI", "Volume", "OBV", "OI+Volume+OBV Score", "EMAIL_RANK_SCORE",
    "Rank Delta", "Cumulative ADX", "5m_Signal", "15m_Signal", "30m_Signal",
    "60m_Signal", "Bull_Signal", "Bear_Signal", "Overall_Signal",
    "Price_Lead_Status", "IVP", "Volatility State", "Last Iteration Time",
]

OPTION_EMAIL_COL_RENAME = {
    "OI+Volume+OBV Score": "Liq Score",
    "EMAIL_RANK_SCORE":    "Rank",
    "% Change":            "% Chg",
    "Last Iteration Time": "Time",
    "Price_Lead_Status":   "Lead",
    "Volatility State":    "Vol State",
    "Option Symbol":       "Opt Symbol",
}

fyers = None


# â”€â”€ Fyers init â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def init_fyers() -> Optional[object]:
    global fyers
    client_id    = os.environ.get("CLIENT_ID") or os.environ.get("CLIENTID")
    access_token = os.environ.get("ACCESS_TOKEN") or os.environ.get("ACCESSTOKEN")
    if not client_id or not access_token:
        logger.error("Missing CLIENT_ID / ACCESS_TOKEN environment variables.")
        fyers = None
        return None
    try:
        fyers = fyersModel.FyersModel(client_id=client_id, token=access_token,
                                      is_async=False, log_path="")
    except Exception:
        fyers = fyersModel.FyersModel(client_id=client_id, token=access_token,
                                      is_async=False, log_path="")
    return fyers


# â”€â”€ Utility helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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


def discover_sector_csvs(root_dir: str = SECTORS_DIR) -> List[str]:
    if not os.path.isdir(root_dir):
        return []
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(".csv"):
                paths.append(os.path.join(dirpath, fname))
    return sorted(set(paths))


def load_fno_symbols_from_sectors(root_dir: str = SECTORS_DIR) -> List[str]:
    symbols = set()
    for path in discover_sector_csvs(root_dir):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        lowered = {str(c).strip().lower(): c for c in df.columns}
        symbol_col = next(
            (lowered[k] for k in ["symbol", "symbols", "ticker", "tradingsymbol"] if k in lowered),
            None,
        )
        if symbol_col is None:
            continue
        for raw in df[symbol_col].dropna().astype(str):
            sym = raw.strip().upper()
            if sym and sym not in {"NAN", "NONE"}:
                symbols.add(sym)
    return sorted(symbols)


def format_eq_symbol(symbol: str) -> str:
    symbol = str(symbol).strip().upper()
    if symbol.startswith("NSE:"):
        return symbol if symbol.endswith("-EQ") else f"{symbol}-EQ"
    return f"NSE:{symbol}-EQ"


def get_history(symbol: str, resolution: str, days_back: int) -> pd.DataFrame:
    if fyers is None:
        return pd.DataFrame()
    now   = datetime.now()
    start = now - timedelta(days=days_back)
    payload = {
        "symbol": symbol, "resolution": resolution, "date_format": "1",
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
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], unit="s", utc=True)
        .dt.tz_convert("Asia/Kolkata")
        .dt.tz_localize(None)
    )
    return (
        df.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


def nearest_step(value: float) -> int:
    val = abs(safe_float(value, 0))
    if val >= 20000: return 100
    if val >= 10000: return 50
    if val >= 2000:  return 20
    if val >= 500:   return 10
    if val >= 100:   return 5
    return 1


# â”€â”€ OBV â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def compute_obv(df: pd.DataFrame) -> float:
    if df is None or df.empty or len(df) < 2:
        return np.nan
    close     = pd.to_numeric(df["close"],  errors="coerce")
    vol       = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    direction = np.sign(close.diff())
    obv       = (vol * direction).cumsum()
    return round(safe_float(obv.iloc[-1], np.nan), 2)


def compute_today_obv(df: pd.DataFrame) -> float:
    if df is None or df.empty or len(df) < 2:
        return np.nan
    d = df.copy().sort_values("timestamp")
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    today    = pd.Timestamp.now(tz=None).date()
    today_df = d[d["timestamp"].dt.date == today].copy()
    if not today_df.empty:
        today_df = today_df[
            today_df["timestamp"] >= pd.Timestamp.combine(today, dtime(9, 15))
        ].copy()
        if len(today_df) >= 2:
            return compute_obv(today_df)
    return compute_obv(d)


# â”€â”€ IVP â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def compute_ivp(history_df: pd.DataFrame, min_bars: int = 10) -> Tuple[float, str]:
    if history_df is None or history_df.empty or len(history_df) < min_bars:
        return np.nan, "Neutral Vol"
    close = pd.to_numeric(history_df["close"], errors="coerce")
    high  = pd.to_numeric(history_df["high"],  errors="coerce")
    low   = pd.to_numeric(history_df["low"],   errors="coerce")
    proxy = (
        ((high - low) / close.replace(0, np.nan) * 100.0)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if proxy.empty:
        return np.nan, "Neutral Vol"
    lookback = proxy.tail(min(IVP_LOOKBACK_DAYS, len(proxy)))
    current  = float(lookback.iloc[-1])
    ivp      = round((lookback.lt(current).sum() / len(lookback)) * 100, 2)
    if ivp < 30:  return ivp, "Buyer Zone"
    if ivp > 50:  return ivp, "Avoid Buy Premium"
    return ivp, "Neutral Vol"


# â”€â”€ Score / label helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def score_label(delta: float) -> str:
    if pd.isna(delta): return "Neutral"
    if delta >= 7:  return "Buy++"
    if delta >= 4:  return "Buy+"
    if delta >= 1:  return "Buy"
    if delta <= -7: return "Sell++"
    if delta <= -4: return "Sell+"
    if delta <= -1: return "Sell"
    return "Neutral"


def directional_label(raw_label: str, side: str) -> str:
    raw = str(raw_label).strip().upper()
    if side == "long":
        return {"BUY": "LONG", "BUY+": "LONG+", "BUY++": "LONG++",
                "SELL": "SHORT", "SELL+": "SHORT+", "SELL++": "SHORT++"}.get(raw, raw)
    return {"SELL": "SHORT", "SELL+": "SHORT+", "SELL++": "SHORT++",
            "BUY": "LONG", "BUY+": "LONG+", "BUY++": "LONG++"}.get(raw, raw)


def apply_display_labels(df: pd.DataFrame, side: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in ["5m_Signal", "15m_Signal", "30m_Signal", "60m_Signal",
                "Bull_Signal", "Bear_Signal", "Overall_Signal"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: directional_label(str(x), side))
    return out


# â”€â”€ Window score helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def intraday_window_score(df: pd.DataFrame,
                          window_minutes: int = SIGNAL_WINDOW_MINUTES) -> float:
    if df is None or df.empty or len(df) < 2:
        return np.nan
    d        = df.copy().sort_values("timestamp")
    end_ts   = pd.to_datetime(d["timestamp"].iloc[-1])
    start_ts = end_ts - timedelta(minutes=window_minutes)
    cur      = d[(d["timestamp"] >= start_ts) & (d["timestamp"] <= end_ts)]
    if cur.empty or len(cur) < 2:
        return np.nan
    fc = safe_float(cur["close"].iloc[0])
    lc = safe_float(cur["close"].iloc[-1])
    if pd.isna(fc) or fc == 0:
        return np.nan
    return round(((lc - fc) / fc) * 100.0, 2)


def previous_trading_day_same_time_score(
    full_df: pd.DataFrame,
    end_ts: Optional[pd.Timestamp] = None,
    window_minutes: int = SIGNAL_WINDOW_MINUTES,
) -> float:
    if full_df is None or full_df.empty or len(full_df) < 4:
        return np.nan
    d  = full_df.copy().sort_values("timestamp").reset_index(drop=True)
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    end_ts = pd.to_datetime(end_ts) if end_ts is not None else pd.to_datetime(d["timestamp"].iloc[-1])
    trading_days = sorted(d["timestamp"].dt.date.unique())
    prev_days    = [day for day in trading_days if day < end_ts.date()]
    if not prev_days:
        return np.nan
    prev_day      = prev_days[-1]
    prev_day_data = d[d["timestamp"].dt.date == prev_day].copy()
    if prev_day_data.empty:
        return np.nan
    same_time_rows = prev_day_data[
        (prev_day_data["timestamp"].dt.hour   == end_ts.hour) &
        (prev_day_data["timestamp"].dt.minute == end_ts.minute)
    ]
    prev_end   = (
        pd.to_datetime(same_time_rows.iloc[-1]["timestamp"])
        if not same_time_rows.empty
        else prev_day_data["timestamp"].iloc[-1]
    )
    prev_start  = prev_end - timedelta(minutes=window_minutes)
    prev_window = prev_day_data[
        (prev_day_data["timestamp"] >= prev_start) &
        (prev_day_data["timestamp"] <= prev_end)
    ]
    if prev_window.empty or len(prev_window) < 2:
        return np.nan
    fc = safe_float(prev_window["close"].iloc[0])
    lc = safe_float(prev_window["close"].iloc[-1])
    if pd.isna(fc) or fc == 0:
        return np.nan
    return round(((lc - fc) / fc) * 100.0, 2)


def compare_window_signal(current_score: float,
                          previous_score: float) -> Tuple[float, str]:
    if pd.isna(current_score) or pd.isna(previous_score):
        return np.nan, "Neutral"
    delta = round(current_score - previous_score, 2)
    if delta >= 0.50:   return delta, "Buy++"
    if delta >= 0.20:   return delta, "Buy+"
    if delta >= 0.05:   return delta, "Buy"
    if delta <= -0.50:  return delta, "Sell++"
    if delta <= -0.20:  return delta, "Sell+"
    if delta <= -0.05:  return delta, "Sell"
    return delta, "Neutral"


# â”€â”€ Iteration history â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def build_iteration_history(
    intra_df: pd.DataFrame,
    window_minutes: int = SIGNAL_WINDOW_MINUTES,
    iterations: int = ITERATIONS_TO_KEEP,
) -> pd.DataFrame:
    if intra_df is None or intra_df.empty:
        return pd.DataFrame()
    full_df = intra_df.copy().sort_values("timestamp").reset_index(drop=True)
    full_df["timestamp"] = pd.to_datetime(full_df["timestamp"])
    last_day = full_df["timestamp"].dt.date.max()
    d = full_df[full_df["timestamp"].dt.date == last_day].copy().reset_index(drop=True)
    if d.empty:
        return pd.DataFrame()
    start_anchor = pd.Timestamp.combine(pd.Timestamp(last_day).date(), dtime(9, 15))
    end_anchor   = pd.Timestamp.combine(pd.Timestamp(last_day).date(), dtime(15, 30))
    d = d[(d["timestamp"] >= start_anchor) & (d["timestamp"] <= end_anchor)].copy().reset_index(drop=True)
    if d.empty:
        return pd.DataFrame()
    rows = []
    for i in range(len(d)):
        end_ts   = pd.to_datetime(d.loc[i, "timestamp"])
        start_ts = end_ts - timedelta(minutes=window_minutes)
        cur      = d[(d["timestamp"] >= start_ts) & (d["timestamp"] <= end_ts)]
        if cur.empty or len(cur) < 2:
            continue
        fc = safe_float(cur["close"].iloc[0])
        lc = safe_float(cur["close"].iloc[-1])
        if pd.isna(fc) or fc == 0:
            continue
        current_score = round(((lc - fc) / fc) * 100.0, 2)
        prev_score    = previous_trading_day_same_time_score(full_df, end_ts, window_minutes)
        delta, signal = compare_window_signal(current_score, prev_score)
        rows.append({
            "iteration":                            len(rows) + 1,
            "timestamp":                            end_ts.strftime("%H:%M"),
            "window_minutes":                       window_minutes,
            "window_start":                         start_ts.strftime("%H:%M"),
            "window_end":                           end_ts.strftime("%H:%M"),
            "current_window_score":                 current_score,
            "previous_trading_day_same_time_score": prev_score,
            "window_delta":                         delta,
            "window_signal":                        signal,
            "close":                                lc,
        })
        if len(rows) >= iterations:
            break
    return pd.DataFrame(rows)


# â”€â”€ summarize_intraday â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def summarize_intraday(intra_df: pd.DataFrame,
                       reference_df: pd.DataFrame) -> Dict[str, object]:
    if intra_df is None or intra_df.empty:
        return {}
    df     = intra_df.copy().sort_values("timestamp").reset_index(drop=True)
    close  = pd.to_numeric(df["close"],  errors="coerce")
    high   = pd.to_numeric(df["high"],   errors="coerce")
    low    = pd.to_numeric(df["low"],    errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

    delta    = close.diff().fillna(0.0)
    avg_gain = delta.clip(lower=0.0).rolling(14, min_periods=14).mean()
    avg_loss = (-delta).clip(lower=0.0).rolling(14, min_periods=14).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    rsi      = (100 - (100 / (1 + rs))).fillna(50)

    tr       = pd.concat(
        [(high - low).abs(), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    up_move  = high.diff()
    down_move = -low.diff()
    plus_dm  = np.where((up_move > down_move)  & (up_move  > 0), up_move,  0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr      = pd.Series(tr).rolling(14, min_periods=14).mean()
    plus_di  = 100 * pd.Series(plus_dm).rolling(14,  min_periods=14).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm).rolling(14, min_periods=14).mean() / atr.replace(0, np.nan)
    dx       = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    adx      = dx.rolling(14, min_periods=14).mean().fillna(0)

    typical  = (high + low + close) / 3.0
    cum_vol  = volume.cumsum().replace(0, np.nan)
    vwap     = ((typical * volume).cumsum() / cum_vol).ffill().fillna(close)
    vwap_std = (((typical - vwap) ** 2 * volume).cumsum() / cum_vol).pow(0.5).replace(0, np.nan)
    vwap_z   = ((close - vwap) / vwap_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    range_now   = (high - low).clip(lower=0)
    avg_range5  = range_now.rolling(5, min_periods=3).mean()
    avg_vol5    = volume.rolling(5, min_periods=3).mean()
    price_lead_flag = (
        (range_now / avg_range5.replace(0, np.nan)) >= 1.5
    ) & (
        (volume / avg_vol5.replace(0, np.nan)) <= 1.0
    )
    streak = []
    run    = 0
    for flag in price_lead_flag.fillna(False).astype(bool):
        run = run + 1 if flag else 0
        streak.append(run)
    streak      = pd.Series(streak, index=df.index)
    lead_status = pd.Series(
        np.select(
            [price_lead_flag & (streak >= 3),
             price_lead_flag & (streak >= 2),
             price_lead_flag],
            ["STRONG_PRICE_LEAD_FADE", "PRICE_LEADING_FADE_RISK", "EARLY_PRICE_LEAD"],
            default="NORMAL",
        ),
        index=df.index,
    )

    prev_close = (
        safe_float(reference_df["close"].iloc[-2])
        if reference_df is not None and len(reference_df) >= 2
        else np.nan
    )
    ltp        = safe_float(close.iloc[-1])
    pct_change = (
        ((ltp - prev_close) / prev_close * 100.0)
        if pd.notna(prev_close) and prev_close != 0
        else 0.0
    )
    current_win   = intraday_window_score(df)
    prev_win      = previous_trading_day_same_time_score(intra_df)
    _, win_signal = compare_window_signal(current_win, prev_win)
    iteration_history = build_iteration_history(intra_df)

    bull = bear = 0
    if pct_change > 0:                                              bull += 1
    if pct_change < 0:                                              bear += 1
    if safe_float(vwap_z.iloc[-1], 0) >= 0.30:                     bull += 1
    if safe_float(vwap_z.iloc[-1], 0) <= -0.30:                    bear += 1
    if safe_float(plus_di.iloc[-1],  0) > safe_float(minus_di.iloc[-1], 0): bull += 1
    if safe_float(minus_di.iloc[-1], 0) > safe_float(plus_di.iloc[-1],  0): bear += 1
    if safe_float(adx.iloc[-1], 0) >= 20:
        if   safe_float(plus_di.iloc[-1],  0) > safe_float(minus_di.iloc[-1], 0): bull += 1
        elif safe_float(minus_di.iloc[-1], 0) > safe_float(plus_di.iloc[-1],  0): bear += 1
    if safe_float(rsi.iloc[-1], 50) >= 55: bull += 1
    if safe_float(rsi.iloc[-1], 50) <= 45: bear += 1
    if win_signal.startswith("Buy"):  bull += 2
    elif win_signal.startswith("Sell"): bear += 2
    rank_delta = bull - bear

    ivp, vol_state = compute_ivp(reference_df, min_bars=10)
    last_ts        = pd.to_datetime(df["timestamp"].iloc[-1])
    return {
        "LTP":                  round(ltp, 2),
        "% Change":             round(pct_change, 2),
        "5m_Signal":            score_label(rank_delta),
        "15m_Signal":           win_signal,
        "30m_Signal":           score_label(rank_delta * 0.8),
        "60m_Signal":           score_label(rank_delta * 0.7),
        "Bull_Signal":          score_label(bull),
        "Bear_Signal":          score_label(-bear),
        "Overall_Signal":       score_label(rank_delta),
        "Price_Lead_Status":    str(lead_status.iloc[-1]),
        "IVP":                  ivp,
        "Volatility State":     vol_state,
        "Last Iteration Time":  last_ts.strftime("%H:%M"),
        "Bull Rank":            bull,
        "Bear Rank":            bear,
        "Rank Delta":           rank_delta,
        "Cumulative ADX":       round(safe_float(adx.iloc[-1], np.nan), 2),
        "Iteration History":    iteration_history,
    }


# â”€â”€ Candidate selection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def choose_top_candidates(
    summary_df: pd.DataFrame, top_n: int = TOP_N_UNDERLYINGS
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if summary_df is None or summary_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    rank_delta = safe_series(summary_df, "Rank Delta", 0)
    long_df  = summary_df[rank_delta > 0].copy()
    short_df = summary_df[rank_delta < 0].copy()
    long_df  = long_df.sort_values(
        ["Rank Delta", "Cumulative ADX", "% Change"], ascending=[False, False, False]
    ).head(top_n)
    short_df = short_df.sort_values(
        ["Rank Delta", "Cumulative ADX", "% Change"], ascending=[True, False, True]
    ).head(top_n)
    return long_df.reset_index(drop=True), short_df.reset_index(drop=True)


# â”€â”€ Option chain fetch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def fetch_option_pairs(symbol: str,
                       pair_count: int = OPTION_PAIRS_TO_KEEP) -> pd.DataFrame:
    if fyers is None:
        return pd.DataFrame()
    eq_symbol = format_eq_symbol(symbol)
    try:
        quote     = fyers.quotes({"symbols": eq_symbol})
        ltp       = safe_float(quote.get("d", [{}])[0].get("v", {}).get("lp"), np.nan)
        chain_res = fyers.optionchain(data={"symbol": eq_symbol, "strikecount": 50})
    except Exception:
        return pd.DataFrame()
    chain = ((chain_res or {}).get("data") or {}).get("optionsChain", [])
    if not chain:
        return pd.DataFrame()
    rows = []
    for item in chain:
        strike = safe_float(item.get("strike_price") or item.get("strike"), np.nan)
        typ    = str(item.get("option_type") or item.get("type") or "").upper()
        if pd.isna(strike) or typ not in {"CE", "PE"}:
            continue
        rows.append({
            "Strike":        strike,
            "Type":          typ,
            "OptionSymbol":  str(item.get("symbol", "")),
            "OptionLTP":     safe_float(item.get("ltp") or item.get("lp"), 0.0),
            "OI":            safe_float(item.get("oi") or item.get("open_interest"), np.nan),
            "Volume":        safe_float(item.get("volume"), np.nan),
        })
    if not rows:
        return pd.DataFrame()
    oc   = pd.DataFrame(rows)
    step = nearest_step(ltp if pd.notna(ltp) else oc["Strike"].median())
    atm  = round(ltp / step) * step if pd.notna(ltp) else oc["Strike"].median()
    strikes = sorted(oc["Strike"].dropna().unique(), key=lambda x: abs(x - atm))[:pair_count]
    final_rows = []
    for strike in sorted(strikes):
        sub = oc[oc["Strike"] == strike]
        for opt_type in ["CE", "PE"]:
            leg = sub[sub["Type"] == opt_type]
            if leg.empty:
                continue
            final_rows.append({
                "Strike":        strike,
                "Option Type":   opt_type,
                "Option Symbol": leg["OptionSymbol"].iloc[0],
                "OI":            safe_float(leg["OI"].iloc[0], 0),
                "Chain Volume":  safe_float(leg["Volume"].iloc[0], 0),
            })
    return pd.DataFrame(final_rows)


# â”€â”€ Single option scan â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def scan_single_option(option_symbol: str, option_type: str,
                       strike: float, underlying: str) -> Optional[Dict]:
    hist_symbol = option_symbol if option_symbol.startswith("NSE:") else f"NSE:{option_symbol}"
    daily_df    = get_history(hist_symbol, "D",  max(DAILY_LOOKBACK_DAYS, IVP_LOOKBACK_DAYS))
    intra_df    = get_history(hist_symbol, "5",  INTRADAY_LOOKBACK_DAYS)
    if daily_df.empty or intra_df.empty:
        return None
    summary = summarize_intraday(intra_df, daily_df)
    if not summary:
        return None
    summary.update({
        "Underlying":    underlying,
        "Option Type":   option_type,
        "Option Symbol": option_symbol,
        "Strike":        strike,
        "OBV":           compute_today_obv(intra_df),
        "OI":            np.nan,
        "Volume":        intra_df["volume"].sum() if "volume" in intra_df.columns else 0,
    })
    return summary


def option_liquidity_score(oi, volume, obv) -> float:
    return round(
        (np.log1p(max(safe_float(oi,     0), 0)) * 0.45) +
        (np.log1p(max(safe_float(volume, 0), 0)) * 0.35) +
        (np.log1p(max(abs(safe_float(obv, 0)), 0)) * 0.20),
        4,
    )


def rank_option_candidates(df: pd.DataFrame, side: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out  = df.copy()
    liq  = safe_series(out, "OI+Volume+OBV Score", 0)
    rd   = safe_series(out, "Rank Delta",    0)
    adx  = safe_series(out, "Cumulative ADX", 0)
    pct  = safe_series(out, "% Change",      0)
    otyp = (
        out["Option Type"].astype(str).str.upper()
        if "Option Type" in out.columns
        else pd.Series("", index=out.index)
    )
    out["Liq"] = liq; out["RD"] = rd; out["ADX"] = adx; out["PCT"] = pct
    if side == "long":
        type_bonus = np.where(otyp.eq("CE"), 0.30, 0.10)
        out["EMAIL_RANK_SCORE"] = liq * 0.40 + rd * 0.30 + adx * 0.18 + pct * 0.10 + type_bonus
        out = out.sort_values(
            ["EMAIL_RANK_SCORE", "Liq", "RD", "ADX", "PCT"],
            ascending=[False, False, False, False, False],
        )
    else:
        type_bonus = np.where(otyp.eq("PE"), 0.30, 0.10)
        out["EMAIL_RANK_SCORE"] = liq * 0.40 + (-rd) * 0.30 + adx * 0.18 + (-pct) * 0.10 + type_bonus
        out = out.sort_values(
            ["EMAIL_RANK_SCORE", "Liq", "RD", "ADX", "PCT"],
            ascending=[False, False, True, False, True],
        )
    return out.reset_index(drop=True)


def build_option_candidates(
    candidates_df: pd.DataFrame, side: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if candidates_df is None or candidates_df.empty or "Symbol" not in candidates_df.columns:
        return pd.DataFrame(), pd.DataFrame()
    rows, iter_rows = [], []
    for underlying in candidates_df["Symbol"].dropna().astype(str):
        pair_df = fetch_option_pairs(underlying)
        if pair_df.empty:
            continue
        atm_strike = pair_df["Strike"].iloc[0]
        req_type   = "CE" if side == "long" else "PE"
        atm_rows   = pair_df[
            (pair_df["Strike"] == atm_strike) & (pair_df["Option Type"] == req_type)
        ]
        atm_vol = safe_float(atm_rows["Chain Volume"].iloc[0] if not atm_rows.empty else 0, 0)
        if atm_vol < MIN_ATM_CHAIN_VOLUME:
            logger.debug("SKIP %s: ATM %s vol %.0f < %d",
                         underlying, req_type, atm_vol, MIN_ATM_CHAIN_VOLUME)
            continue
        for _, row in pair_df.iterrows():
            strike   = safe_float(row.get("Strike"), np.nan)
            opt_type = str(row.get("Option Type", "")).upper()
            sym      = str(row.get("Option Symbol", ""))
            if not sym or opt_type not in {"CE", "PE"}:
                continue
            scanned = scan_single_option(sym, opt_type, strike, underlying)
            if not scanned:
                continue
            scanned["OI"]               = safe_float(row.get("OI"), 0)
            scanned["Chain_Volume"]     = safe_float(row.get("Chain Volume"), 0)
            scanned["OI+Volume+OBV Score"] = option_liquidity_score(
                scanned.get("OI", 0), scanned.get("Volume", 0), scanned.get("OBV", 0)
            )
            if safe_float(scanned.get("LTP"), 0.0) < MIN_OPTION_LTP:
                continue
            if safe_float(scanned.get("Volume"), 0) < MIN_ATM_CHAIN_VOLUME:
                logger.debug("SKIP option %s: vol %.0f < %d",
                             sym, safe_float(scanned.get("Volume"), 0), MIN_ATM_CHAIN_VOLUME)
                continue
            rows.append(scanned)
            hist = scanned.get("Iteration History")
            if isinstance(hist, pd.DataFrame) and not hist.empty:
                tmp = hist.copy()
                tmp.insert(0, "Option Symbol", sym)
                tmp.insert(1, "Underlying",    underlying)
                tmp.insert(2, "Strike",        strike)
                tmp.insert(3, "Option Type",   opt_type)
                iter_rows.append(tmp)
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    out = pd.DataFrame(rows)
    out = rank_option_candidates(out, side)
    rd  = safe_series(out, "Rank Delta", 0)
    pct = safe_series(out, "% Change",   0)
    if side == "long":
        out = out[(rd > 0) | (pct > 0)].copy()
        out = out.sort_values(
            ["EMAIL_RANK_SCORE", "OI+Volume+OBV Score", "Rank Delta", "Cumulative ADX", "% Change"],
            ascending=[False, False, False, False, False],
        )
    else:
        out = out[(rd < 0) | (pct < 0)].copy()
        out = out.sort_values(
            ["EMAIL_RANK_SCORE", "OI+Volume+OBV Score", "Rank Delta", "Cumulative ADX", "% Change"],
            ascending=[False, False, True, False, True],
        )
    final_cols = [c for c in OPTION_EMAIL_COLS if c in out.columns]
    final_out  = out[final_cols].reset_index(drop=True)
    iter_df    = pd.DataFrame()
    if iter_rows and not final_out.empty:
        all_iters = pd.concat(iter_rows, ignore_index=True)
        all_iters = all_iters[
            all_iters["Option Symbol"].isin(final_out["Option Symbol"])
        ].copy()
        sort_cols = [c for c in
                     ["Underlying", "Option Type", "Strike", "Option Symbol", "iteration"]
                     if c in all_iters.columns]
        if sort_cols:
            all_iters = all_iters.sort_values(sort_cols).reset_index(drop=True)
        group_cols = [c for c in
                      ["Underlying", "Option Type", "Strike", "Option Symbol"]
                      if c in all_iters.columns]
        if group_cols and not all_iters.empty:
            all_iters["iteration"] = all_iters.groupby(group_cols).cumcount() + 1
        if "iteration" in all_iters.columns:
            all_iters["iteration"] = (
                pd.to_numeric(all_iters["iteration"], errors="coerce").astype("Int64")
            )
            all_iters = all_iters[all_iters["iteration"].between(1, ITERATIONS_TO_KEEP)]
        iter_df = all_iters.reset_index(drop=True)
    return final_out, iter_df


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CHAIN SIGNAL SYSTEM
# Entry : 5m CONFIRMED + 30m CONFIRMED + T30 >= T30_MIN
# Exit  : 30m BROKEN   OR  T30 < T30_MIN
# Sticky: once a row fires ENTER it is retained all day
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def compute_chain_status(window_signals: list) -> Tuple[str, int, int, int]:
    """
    Given a list of window_signal strings from iteration history,
    returns (chain_status, buy_count, sell_count, total_count).

    chain_status:
      CONFIRMED  â€” >= 65% Buy signals in the block
      BROKEN     â€” <= 35% Buy signals (mostly Sell)
      MIXED      â€” in between
    """
    b = sum(1 for s in window_signals if str(s).startswith("Buy"))
    s = sum(1 for s in window_signals if str(s).startswith("Sell"))
    t = b + s
    if t == 0:
        return "MIXED", 0, 0, 0
    ratio = b / t
    if ratio >= 0.65:
        status = "CONFIRMED"
    elif ratio <= 0.35:
        status = "BROKEN"
    else:
        status = "MIXED"
    return status, b, s, t


def latest_block_chain(iter_df: pd.DataFrame,
                       block_minutes: int) -> Tuple[str, int, int, int]:
    """
    Compute chain status for the CURRENT (latest) block from iteration history.
    block_minutes: 5, 15, or 30
    Returns (status, b, s, t)
    """
    if iter_df is None or iter_df.empty or "iteration" not in iter_df.columns:
        return "MIXED", 0, 0, 0
    iters_per_block = max(1, block_minutes // SIGNAL_WINDOW_MINUTES)
    last_it    = int(iter_df["iteration"].max())
    block_num  = ((last_it - 1) // iters_per_block) + 1
    bstart     = (block_num - 1) * iters_per_block + 1
    bend       = block_num * iters_per_block
    block_rows = iter_df[
        (iter_df["iteration"] >= bstart) & (iter_df["iteration"] <= bend)
    ]
    sigs = block_rows["window_signal"].tolist() if not block_rows.empty else []
    return compute_chain_status(sigs)


def scan_option_chain_signals(option_symbol: str, option_type: str,
                               strike: float, underlying: str,
                               iter_df: pd.DataFrame) -> Optional[Dict]:
    """
    Given the already-built iteration history (iter_df from build_option_candidates),
    compute 5m / 15m / 30m chain status and Entry/Exit signals.

    Returns a dict suitable for the chain signal email table.
    """
    c5,  b5,  s5,  t5  = latest_block_chain(iter_df,  5)
    c15, b15, s15, t15 = latest_block_chain(iter_df, 15)
    c30, b30, s30, t30 = latest_block_chain(iter_df, 30)

    entry_signal = (c5 == "CONFIRMED" and c30 == "CONFIRMED" and t30 >= T30_MIN)
    exit_signal  = (c30 == "BROKEN"   or  t30 < T30_MIN)

    chain_signal = (
        "ðŸŸ¢ ENTER" if entry_signal else
        "ðŸ”´ EXIT"  if exit_signal  else
        "â³ WAIT"
    )
    exit_label = "âŒ EXIT NOW" if exit_signal else "âœ… HOLD"

    return {
        "Underlying":    underlying,
        "Option Type":   option_type,
        "Strike":        strike,
        "5m":            c5,
        "T5":            t5,
        "15m":           c15,
        "T15":           t15,
        "30m":           c30,
        "T30":           t30,
        "Chain Signal":  chain_signal,
        "Exit Signal":   exit_label,
    }


# â”€â”€ Sticky-row state (persist all day in JSON) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def load_daily_state() -> Dict:
    today = datetime.now().strftime("%Y-%m-%d")
    if os.path.exists(DAILY_STATE_FILE):
        try:
            with open(DAILY_STATE_FILE) as f:
                state = json.load(f)
            if state.get("date") == today:
                return state
        except Exception:
            pass
    return {"date": today, "rows": {}}


def save_daily_state(state: Dict):
    try:
        os.makedirs(os.path.dirname(DAILY_STATE_FILE) or ".", exist_ok=True)
        with open(DAILY_STATE_FILE, "w") as f:
            json.dump(state, f, default=str)
    except Exception as e:
        logger.warning("Could not save chain state: %s", e)


def update_sticky_rows(state: Dict, new_rows: List[Dict]) -> List[Dict]:
    """
    Merge new scan results into the daily sticky state.
    - New rows that show ENTER get added with Entry Time.
    - Existing rows always get their LTP + chain columns updated.
    Returns sorted list (by Entry Time).
    """
    existing = state.get("rows", {})
    for row in new_rows:
        key = f"{row['Underlying']}|{row['Option Type']}|{row['Strike']}"
        if key not in existing:
            if "ENTER" in str(row.get("Chain Signal", "")):
                row["Entry Time"] = datetime.now().strftime("%H:%M")
                existing[key]     = {k: v for k, v in row.items()}
        else:
            for col in ["LTP", "5m", "T5", "15m", "T15", "30m", "T30",
                        "Chain Signal", "Exit Signal"]:
                if col in row:
                    existing[key][col] = row[col]
    state["rows"] = existing
    save_daily_state(state)
    return sorted(existing.values(), key=lambda r: r.get("Entry Time", "99:99"))


# â”€â”€ Signal colour helpers (used in both old + new email tables) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_SIGNAL_COLORS = {
    "LONG":    ("#d4edda", "#155724"),
    "LONG+":   ("#c3e6cb", "#0b4619"),
    "LONG++":  ("#a8d5b5", "#083010"),
    "SHORT":   ("#f8d7da", "#721c24"),
    "SHORT+":  ("#f5c6cb", "#5c1721"),
    "SHORT++": ("#f1aeb5", "#491219"),
    "BUY":     ("#d4edda", "#155724"),
    "BUY+":    ("#c3e6cb", "#0b4619"),
    "BUY++":   ("#a8d5b5", "#083010"),
    "SELL":    ("#f8d7da", "#721c24"),
    "SELL+":   ("#f5c6cb", "#5c1721"),
    "SELL++":  ("#f1aeb5", "#491219"),
    "NEUTRAL": ("#fff3cd", "#856404"),
}

_SIGNAL_COLS = {
    "5m_Signal", "15m_Signal", "30m_Signal", "60m_Signal",
    "Bull_Signal", "Bear_Signal", "Overall_Signal",
}

OBV_TABLE_COLS = [
    "Underlying", "Option Type", "Strike", "LTP", "% Change",
    "OI", "Volume", "OBV", "EMAIL_RANK_SCORE",
    "5m_Signal", "15m_Signal", "Overall_Signal", "IVP",
    "Last Iteration Time",
]


def _signal_td(val: str) -> str:
    v       = str(val).strip().upper()
    bg, fg  = _SIGNAL_COLORS.get(v, ("#ffffff", "#212529"))
    return (
        '<td style="background:' + bg + ';color:' + fg + ';font-weight:bold;'
        'padding:3px 6px;text-align:center;border:1px solid #dee2e6;'
        'white-space:nowrap;">' + val + '</td>'
    )


def _plain_td(val: str) -> str:
    return (
        '<td style="padding:3px 6px;text-align:right;'
        'border:1px solid #dee2e6;white-space:nowrap;">' + val + '</td>'
    )


def format_cell(col: str, val) -> str:
    if pd.isna(val):
        return ""
    if col in {"% Change", "% Chg"}:
        return f"{float(val):.2f}%"
    if col in {"OI", "Volume", "OBV"}:
        try:    return f"{int(float(val)):,}"
        except: return str(val)
    if col in {"Rank", "Liq Score", "LTP", "Strike", "IVP", "EMAIL_RANK_SCORE"}:
        try:    return f"{float(val):.2f}"
        except: return str(val)
    if isinstance(val, (int, float, np.integer, np.floating)):
        return f"{float(val):.2f}"
    return str(val)


# â”€â”€ Email styles â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
EMAIL_STYLE = """
<style>
body { font-family: Arial, sans-serif; font-size: 11px; }
table { border-collapse: collapse; width: 100%; }
th { background: #1a1a2e; color: #e0e0e0; padding: 5px 7px; font-size: 11px;
     text-align: center; border: 1px solid #333; white-space: nowrap; }
td { padding: 4px 6px; border: 1px solid #ddd; text-align: center;
     font-size: 11px; white-space: nowrap; }
tr:nth-child(even) { background: #f9f9f9; }
.confirmed { color: #006600; font-weight: bold; }
.broken    { color: #cc0000; font-weight: bold; }
.mixed     { color: #ff8800; font-weight: bold; }
.enter     { background: #d4edda; color: #155724; font-weight: bold; }
.exit_now  { background: #f8d7da; color: #721c24; font-weight: bold; }
.hold      { background: #fff3cd; color: #856404; }
.wait      { background: #e2e3e5; color: #383d41; }
.long_head { background: #155724; color: white; padding: 8px;
             font-size: 13px; font-weight: bold; margin: 12px 0 4px 0; }
.short_head{ background: #721c24; color: white; padding: 8px;
             font-size: 13px; font-weight: bold; margin: 12px 0 4px 0; }
.ts        { color: #888; font-size: 10px; }
</style>
"""


# â”€â”€ compact_table_html (original OBV / ranked candidates) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def compact_table_html(df: pd.DataFrame, title: str, max_rows: int) -> str:
    cols = [
        "Underlying", "Option Type", "Strike", "LTP", "% Change",
        "OI", "Volume", "OBV", "EMAIL_RANK_SCORE",
        "5m_Signal", "15m_Signal", "Overall_Signal", "IVP",
        "Last Iteration Time",
    ]
    cols    = [c for c in cols if c in df.columns]
    display = df[cols].head(max_rows).copy()
    rename  = dict(list(OPTION_EMAIL_COL_RENAME.items()) + [("EMAIL_RANK_SCORE", "Rank")])
    header_cells = "".join(
        '<th>' + rename.get(c, c) + '</th>'
        for c in cols
    )
    rows_html = []
    for _, row in display.iterrows():
        cells = []
        for c in cols:
            val = format_cell(c, row[c])
            cells.append(_signal_td(val) if c in _SIGNAL_COLS else _plain_td(val))
        rows_html.append("<tr>" + "".join(cells) + "</tr>")
    ts      = datetime.now().strftime("%H:%M")
    is_long = "long" in title.lower() or "bull" in title.lower()
    color   = "#155724" if is_long else "#721c24"
    icon    = "&#128200;" if is_long else "&#128201;"
    return (
        '<h3 style="color:' + color + ';margin:16px 0 6px;">' + icon + ' ' + title
        + ' &nbsp;<small style="font-size:11px;color:#6c757d;">' + ts + '</small></h3>'
        '<div style="overflow-x:auto;">'
        '<table>'
        '<thead><tr>' + header_cells + '</tr></thead>'
        '<tbody>' + "".join(rows_html) + '</tbody>'
        '</table></div>'
    )


# â”€â”€ Chain signal HTML table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_CHAIN_COLS = ["Underlying", "Option Type", "Strike", "LTP",
               "5m", "T5", "15m", "T15", "30m", "T30",
               "Chain Signal", "Exit Signal", "Entry Time"]


def _chain_td(col: str, val: str) -> str:
    v = str(val).strip()
    css = ""
    if col in ("5m", "15m", "30m"):
        if "CONFIRMED" in v: css = "confirmed"
        elif "BROKEN"  in v: css = "broken"
        else:                css = "mixed"
    elif col == "Chain Signal":
        if "ENTER" in v:     css = "enter"
        elif "EXIT" in v:    css = "exit_now"
        else:                css = "wait"
    elif col == "Exit Signal":
        if "EXIT NOW" in v:  css = "exit_now"
        elif "HOLD" in v:    css = "hold"
    return f'<td class="{css}">{v}</td>'


def build_chain_table_html(rows: List[Dict], title: str, css_class: str) -> str:
    header = "".join(f"<th>{c}</th>" for c in _CHAIN_COLS)
    html_rows = []
    for row in rows:
        cells = []
        for c in _CHAIN_COLS:
            val = str(row.get(c, ""))
            cells.append(_chain_td(c, val))
        html_rows.append("<tr>" + "".join(cells) + "</tr>")
    no_data = (
        f'<tr><td colspan="{len(_CHAIN_COLS)}" '
        'style="color:#999;text-align:center;">â€” No signals yet â€”</td></tr>'
    )
    body = "".join(html_rows) if html_rows else no_data
    ts   = datetime.now().strftime("%H:%M")
    return (
        f'<h3 class="{css_class}">{title}'
        f' <span class="ts">{ts}</span></h3>'
        '<div style="overflow-x:auto;"><table>'
        f'<thead><tr>{header}</tr></thead>'
        f'<tbody>{body}</tbody>'
        '</table></div><br>'
    )


def build_chain_email_html(long_rows: List[Dict], short_rows: List[Dict]) -> str:
    ts          = datetime.now().strftime("%d %b %Y  %H:%M")
    long_table  = build_chain_table_html(long_rows,  "ðŸ“ˆ LONG  CANDIDATES (CE)", "long_head")
    short_table = build_chain_table_html(short_rows, "ðŸ“‰ SHORT CANDIDATES (PE)", "short_head")
    return (
        "<html><head>" + EMAIL_STYLE + "</head><body>"
        '<p class="ts">Chain Signal Report â€” ' + ts + "</p>"
        '<p style="font-size:11px;color:#555;">'
        "<b>Entry:</b> 5m CONFIRMED + 30m CONFIRMED + T30 &ge; " + str(T30_MIN) + "&nbsp;|&nbsp;"
        "<b>Exit:</b> 30m BROKEN OR T30 &lt; " + str(T30_MIN) + "&nbsp;|&nbsp;"
        "<b>Sticky:</b> rows retained all day"
        "</p>"
        + long_table + short_table
        + "</body></html>"
    )


# â”€â”€ Original LONG/SHORT candidate email â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def prepare_option_email_view(df: pd.DataFrame, side: str,
                               max_rows: int = 25) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    view = apply_display_labels(df.copy(), side)
    view = view.rename(columns=OPTION_EMAIL_COL_RENAME)
    return view.head(max_rows).reset_index(drop=True)


def build_email_html(view_df: pd.DataFrame, title: str,
                     scan_time: str, max_rows: int) -> str:
    table_html = compact_table_html(view_df, title, max_rows)
    return (
        "<html><head>" + EMAIL_STYLE + "</head>"
        "<body style='margin:0;padding:0;background:#f4f4f4;'>"
        "<table width='100%' border='0' cellpadding='0' cellspacing='0' "
        "style='background:#f4f4f4;'>"
        "<tr><td align='center' style='padding:12px;'>"
        "<table width='" + str(EMAIL_SAFE_WIDTH) + "' border='0' cellpadding='0' "
        "cellspacing='0' style='width:" + str(EMAIL_SAFE_WIDTH) + "px;"
        "max-width:" + str(EMAIL_SAFE_WIDTH) + "px;background:#ffffff;"
        "border-collapse:collapse;'>"
        "<tr><td style='padding:12px;font-family:Arial,Helvetica,sans-serif;'>"
        "<p style='font-size:11px;color:#6c757d;margin:0 0 8px 0;'>Scan time: " + scan_time + "</p>"
        + table_html
        + "</td></tr></table></td></tr></table></body></html>"
    )


# â”€â”€ Email send â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def send_single_email(subject: str, html_body: str,
                      attachments: list = None) -> bool:
    sender          = os.environ.get("EMAIL_SENDER", "")
    password        = os.environ.get("EMAIL_PASSWORD", "")
    recipients_raw  = os.environ.get("EMAIL_RECIPIENTS", "")
    recipients      = [r.strip() for r in recipients_raw.split(",") if r.strip()]
    if not sender or not password or not recipients:
        logger.warning("Email credentials/recipients not configured; skipping.")
        return False
    msg             = MIMEMultipart("mixed")
    msg["Subject"]  = subject
    msg["From"]     = sender
    msg["To"]       = ", ".join(recipients)
    msg.attach(MIMEText(html_body, "html"))
    for path in (attachments or []):
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, "rb") as fh:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(fh.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition",
                            f"attachment; filename={os.path.basename(path)}")
            msg.attach(part)
        except Exception as e:
            logger.warning("Could not attach %s: %s", path, e)
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, recipients, msg.as_string())
        logger.info("Email sent: %s", subject)
        return True
    except Exception as e:
        logger.error("Failed to send email '%s': %s", subject, e)
        return False


def send_direction_email(df: pd.DataFrame, direction: str,
                         attachments: list) -> bool:
    subject_time = datetime.now().strftime("%d %b %H:%M")
    scan_time    = datetime.now().strftime("%d %b %Y, %H:%M")
    side         = "long" if direction.upper() == "LONG" else "short"
    max_rows     = EMAIL_MAX_ROWS_LONG if side == "long" else EMAIL_MAX_ROWS_SHORT
    view         = prepare_option_email_view(df, side, max_rows=max_rows)
    logger.info("%s email rows: %s", direction, len(view))
    html         = build_email_html(view, f"{direction} Candidates", scan_time, max_rows)
    return send_single_email(
        f"{direction} Candidates - {subject_time}", html, attachments
    )


def send_chain_signal_email(long_rows: List[Dict],
                             short_rows: List[Dict],
                             attachments: list = None) -> bool:
    subject_time = datetime.now().strftime("%d %b %H:%M")
    html         = build_chain_email_html(long_rows, short_rows)
    return send_single_email(
        f"Chain Signal Report - {subject_time}", html, attachments
    )


# â”€â”€ Symbol-level scan â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def scan_symbol(symbol: str) -> Optional[Dict]:
    eq  = format_eq_symbol(symbol)
    daily_df = get_history(eq, "D", max(DAILY_LOOKBACK_DAYS, IVP_LOOKBACK_DAYS))
    intra_df = get_history(eq, "5", INTRADAY_LOOKBACK_DAYS)
    if daily_df.empty or intra_df.empty:
        return None
    summary = summarize_intraday(intra_df, daily_df)
    if not summary:
        return None
    summary["Symbol"] = symbol
    return summary


# â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# POST-MARKET: Run chain signal from saved iteration CSV (no API calls needed)
# Usage: python OPTIONS_OI.py --chain-from-csv fo_iteration_history_YYYYMMDD_HHMM.csv
#        [--long-csv fo_long_candidates_*.csv] [--short-csv fo_short_candidates_*.csv]
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def run_chain_from_csv(iter_csv_path: str,
                       long_csv_path: str  = None,
                       short_csv_path: str = None) -> None:
    """
    Rebuild chain signals from a saved iteration history CSV.
    No Fyers API call needed â€” works entirely from disk.
    """
    if not os.path.isfile(iter_csv_path):
        logger.error("Iteration CSV not found: %s", iter_csv_path)
        return

    iteration_df = pd.read_csv(iter_csv_path)
    logger.info("Loaded iteration CSV: %d rows from %s", len(iteration_df), iter_csv_path)

    if iteration_df.empty or "window_signal" not in iteration_df.columns:
        logger.error("iteration_df is empty or missing 'window_signal' column.")
        return

    # Load LTP lookup from long/short CSVs if provided
    ltp_lookup: Dict[str, float] = {}
    for csv_path in [long_csv_path, short_csv_path]:
        if csv_path and os.path.isfile(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if "Option Symbol" in df.columns and "LTP" in df.columns:
                    for _, row in df.iterrows():
                        ltp_lookup[str(row["Option Symbol"])] = safe_float(row["LTP"])
            except Exception as e:
                logger.warning("Could not load LTP from %s: %s", csv_path, e)

    state      = load_daily_state()
    chain_rows = []
    group_cols = [c for c in
                  ["Underlying", "Option Type", "Strike", "Option Symbol"]
                  if c in iteration_df.columns]

    for keys, grp in iteration_df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map     = dict(zip(group_cols, keys))
        underlying  = key_map.get("Underlying", "")
        option_type = key_map.get("Option Type", "")
        opt_symbol  = key_map.get("Option Symbol", "")
        strike      = safe_float(key_map.get("Strike", np.nan))
        ltp         = ltp_lookup.get(opt_symbol, np.nan)

        sig = scan_option_chain_signals(opt_symbol, option_type, strike, underlying, grp)
        if sig:
            sig["LTP"] = round(ltp, 2) if pd.notna(ltp) else ""
            chain_rows.append(sig)

    long_chain_new  = [r for r in chain_rows if r.get("Option Type") == "CE"]
    short_chain_new = [r for r in chain_rows if r.get("Option Type") == "PE"]

    long_sticky  = [v for v in state["rows"].values() if v.get("Option Type") == "CE"]
    short_sticky = [v for v in state["rows"].values() if v.get("Option Type") == "PE"]

    def _make_state(sticky_list):
        return {
            "date": state["date"],
            "rows": {
                f"{r['Underlying']}|{r['Option Type']}|{r['Strike']}": r
                for r in sticky_list
            },
        }

    long_merged  = update_sticky_rows(_make_state(long_sticky),  long_chain_new)
    short_merged = update_sticky_rows(_make_state(short_sticky), short_chain_new)

    # Persist updated state
    state2 = load_daily_state()
    for r in long_merged + short_merged:
        key = f"{r['Underlying']}|{r['Option Type']}|{r['Strike']}"
        state2["rows"][key] = r
    save_daily_state(state2)

    # Save chain CSVs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    if long_merged:
        out_path = os.path.join(OUTPUT_DIR, f"chain_long_{timestamp}.csv")
        pd.DataFrame(long_merged).to_csv(out_path, index=False)
        logger.info("Saved: %s", out_path)
    if short_merged:
        out_path = os.path.join(OUTPUT_DIR, f"chain_short_{timestamp}.csv")
        pd.DataFrame(short_merged).to_csv(out_path, index=False)
        logger.info("Saved: %s", out_path)

    logger.info("Chain LONG: %d | Chain SHORT: %d", len(long_merged), len(short_merged))
    send_chain_signal_email(long_merged, short_merged)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    init_fyers()
    symbols = load_fno_symbols_from_sectors(SECTORS_DIR)

    rows = []
    for i, symbol in enumerate(symbols, start=1):
        logger.info("[%s/%s] Scanning %s", i, len(symbols), symbol)
        row = scan_symbol(symbol)
        if row:
            rows.append(row)
        time.sleep(PER_SYMBOL_SLEEP_SEC)

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        raise RuntimeError("No symbols returned usable market data.")

    summary_df = summary_df.sort_values(
        ["Rank Delta", "% Change"], ascending=[False, False]
    ).reset_index(drop=True)

    long_seed_df, short_seed_df = choose_top_candidates(summary_df, top_n=TOP_N_UNDERLYINGS)

    long_df,  long_iter_df  = build_option_candidates(long_seed_df,  side="long")
    short_df, short_iter_df = build_option_candidates(short_seed_df, side="short")

    iteration_df = (
        pd.concat([long_iter_df, short_iter_df], ignore_index=True)
        if (not long_iter_df.empty or not short_iter_df.empty)
        else pd.DataFrame()
    )

    # â”€â”€ Save CSVs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M")
    summary_csv = os.path.join(OUTPUT_DIR, f"fo_summary_{timestamp}.csv")
    long_csv    = os.path.join(OUTPUT_DIR, f"fo_long_candidates_{timestamp}.csv")
    short_csv   = os.path.join(OUTPUT_DIR, f"fo_short_candidates_{timestamp}.csv")
    iter_csv    = os.path.join(OUTPUT_DIR, f"fo_iteration_history_{timestamp}.csv")

    summary_df.to_csv(summary_csv, index=False)
    long_df.to_csv(long_csv,  index=False)
    short_df.to_csv(short_csv, index=False)

    if iteration_df.empty:
        iteration_df = pd.DataFrame(columns=[
            "iteration", "Underlying", "Option Type", "Strike", "Option Symbol",
            "timestamp", "window_minutes", "window_start", "window_end",
            "current_window_score", "previous_trading_day_same_time_score",
            "window_delta", "window_signal", "close",
        ])
    iteration_df.to_csv(iter_csv, index=False)

    logger.info("LONG: %d rows | SHORT: %d rows | Iter: %d rows",
                len(long_df), len(short_df), len(iteration_df))

    # â”€â”€ Send standard LONG / SHORT emails â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    attachments = [summary_csv, long_csv, short_csv, iter_csv]
    send_direction_email(long_df,  "LONG",  attachments)
    send_direction_email(short_df, "SHORT", attachments)

    # â”€â”€ Build Chain Signal rows from iteration history â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # iteration_df has columns: Option Symbol, Underlying, Strike, Option Type,
    #   iteration, window_signal, ...
    state       = load_daily_state()
    chain_rows  = []

    if not iteration_df.empty and "Option Symbol" in iteration_df.columns:
        # group by option â€” compute chain status per option
        group_cols = ["Underlying", "Option Type", "Strike", "Option Symbol"]
        for keys, grp in iteration_df.groupby(
            [c for c in group_cols if c in iteration_df.columns]
        ):
            # keys may be a tuple or scalar depending on number of group cols
            if not isinstance(keys, tuple):
                keys = (keys,)
            key_map = dict(zip(
                [c for c in group_cols if c in iteration_df.columns], keys
            ))
            underlying  = key_map.get("Underlying", "")
            option_type = key_map.get("Option Type", "")
            strike      = safe_float(key_map.get("Strike", np.nan))

            # Also grab LTP from the candidate df if available
            ltp = np.nan
            src_df = long_df if option_type == "CE" else short_df
            if not src_df.empty and "Option Symbol" in src_df.columns:
                match = src_df[src_df["Option Symbol"] == key_map.get("Option Symbol", "")]
                if not match.empty and "LTP" in match.columns:
                    ltp = safe_float(match["LTP"].iloc[0])

            sig = scan_option_chain_signals(
                key_map.get("Option Symbol", ""),
                option_type, strike, underlying, grp
            )
            if sig:
                sig["LTP"] = round(ltp, 2) if pd.notna(ltp) else ""
                chain_rows.append(sig)

    # â”€â”€ Merge with sticky state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    long_chain_new  = [r for r in chain_rows if r.get("Option Type") == "CE"]
    short_chain_new = [r for r in chain_rows if r.get("Option Type") == "PE"]

    long_sticky  = [v for v in state["rows"].values() if v.get("Option Type") == "CE"]
    short_sticky = [v for v in state["rows"].values() if v.get("Option Type") == "PE"]

    def _make_state(sticky_list):
        return {
            "date": state["date"],
            "rows": {
                f"{r['Underlying']}|{r['Option Type']}|{r['Strike']}": r
                for r in sticky_list
            },
        }

    long_merged  = update_sticky_rows(_make_state(long_sticky),  long_chain_new)
    short_merged = update_sticky_rows(_make_state(short_sticky), short_chain_new)

    # Persist combined state
    state2 = load_daily_state()
    for r in long_merged + short_merged:
        key = f"{r['Underlying']}|{r['Option Type']}|{r['Strike']}"
        state2["rows"][key] = r
    save_daily_state(state2)

    # â”€â”€ Save chain CSVs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if long_merged:
        pd.DataFrame(long_merged).to_csv(
            os.path.join(OUTPUT_DIR, f"chain_long_{timestamp}.csv"), index=False
        )
    if short_merged:
        pd.DataFrame(short_merged).to_csv(
            os.path.join(OUTPUT_DIR, f"chain_short_{timestamp}.csv"), index=False
        )

    logger.info("Chain LONG: %d | Chain SHORT: %d",
                len(long_merged), len(short_merged))

    # â”€â”€ Send chain signal email â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    send_chain_signal_email(long_merged, short_merged, attachments)


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if "--chain-from-csv" in args:
        idx       = args.index("--chain-from-csv")
        iter_csv  = args[idx + 1] if idx + 1 < len(args) else None
        long_csv  = args[args.index("--long-csv")  + 1] if "--long-csv"  in args else None
        short_csv = args[args.index("--short-csv") + 1] if "--short-csv" in args else None
        if not iter_csv:
            print("Usage: python OPTIONS_OI.py --chain-from-csv <iter_csv> "
                  "[--long-csv <long_csv>] [--short-csv <short_csv>]")
            sys.exit(1)
        run_chain_from_csv(iter_csv, long_csv, short_csv)
    else:
        main()
