import os
import time
import smtplib
import logging
from concurrent.futures import ThreadPoolExecutor
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

DAILY_LOOKBACK_DAYS = 252
INTRADAY_LOOKBACK_DAYS = 20
IVP_LOOKBACK_DAYS = 252
OPTION_PAIRS_TO_KEEP = 5
SIGNAL_WINDOW_MINUTES = 5
ITERATIONS_TO_KEEP = 75
SECTORS_DIR = os.environ.get("SECTORS_DIR", "sectors")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", ".")
MIN_OPTION_LTP = 10.0
PER_SYMBOL_SLEEP_SEC = float(os.environ.get("PER_SYMBOL_SLEEP_SEC", "0.25"))
EMAIL_MAX_ROWS_LONG = int(os.environ.get("EMAIL_MAX_ROWS_LONG", "14"))
EMAIL_MAX_ROWS_SHORT = int(os.environ.get("EMAIL_MAX_ROWS_SHORT", "14"))
EMAIL_SAFE_WIDTH = int(os.environ.get("EMAIL_SAFE_WIDTH", "600"))
TOP_N_UNDERLYINGS = int(os.environ.get("TOP_N_UNDERLYINGS", "60"))

OPTION_EMAIL_COLS = [
    "Underlying", "Option Type", "Option Symbol", "Strike", "LTP", "% Change", "OI", "Volume", "OBV",
    "OI+Volume+OBV Score", "EMAIL_RANK_SCORE", "Rank Delta", "Cumulative ADX",
    "5m_Signal", "15m_Signal", "30m_Signal", "60m_Signal",
    "Bull_Signal", "Bear_Signal", "Overall_Signal", "Price_Lead_Status", "IVP", "Volatility State",
    "Last Iteration Time",
]

OPTION_EMAIL_COL_RENAME = {
    "OI+Volume+OBV Score": "Liq Score",
    "EMAIL_RANK_SCORE": "Rank",
    "% Change": "% Chg",
    "Last Iteration Time": "Time",
    "Price_Lead_Status": "Lead",
    "Volatility State": "Vol State",
    "Option Symbol": "Opt Symbol",
}

fyers = None

def init_fyers() -> Optional[object]:
    global fyers
    client_id = os.environ.get("CLIENT_ID") or os.environ.get("CLIENTID")
    access_token = os.environ.get("ACCESS_TOKEN") or os.environ.get("ACCESSTOKEN")
    if not client_id or not access_token:
        logger.error("Missing CLIENT_ID / ACCESS_TOKEN environment variables.")
        return None
    fyers = fyersModel.FyersModel(client_id=client_id, token=access_token, is_async=False, log_path="")
    return fyers

def safe_float(value, default=np.nan) -> float:
    try:
        return float(value) if value is not None and value != "" else default
    except: return default

def safe_series(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if frame is not None and col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index if frame is not None else None, dtype=float)

def load_fno_symbols_from_sectors(root_dir: str = SECTORS_DIR) -> List[str]:
    symbols = set()
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(".csv"):
                try:
                    df = pd.read_csv(os.path.join(dirpath, fname))
                    sym_col = next((c for c in df.columns if c.lower() in ["symbol", "ticker", "tradingsymbol"]), None)
                    if sym_col:
                        for s in df[sym_col].dropna().astype(str):
                            if s.strip().upper() not in {"NAN", "NONE"}: symbols.add(s.strip().upper())
                except: continue
    return sorted(symbols)

def format_eq_symbol(symbol: str) -> str:
    s = symbol.strip().upper()
    return s if s.startswith("NSE:") else f"NSE:{s}-EQ"

def get_history(symbol: str, resolution: str, days_back: int) -> pd.DataFrame:
    if fyers is None: return pd.DataFrame()
    payload = {
        "symbol": symbol, "resolution": resolution, "date_format": "1",
        "range_from": (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d"),
        "range_to": datetime.now().strftime("%Y-%m-%d"), "cont_flag": "1"
    }
    try:
        res = fyers.history(data=payload)
        candles = (res or {}).get("candles", [])
        if not candles: return pd.DataFrame()
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        return df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    except: return pd.DataFrame()

def compute_today_obv(df: pd.DataFrame) -> float:
    if df is None or df.empty or len(df) < 2: return np.nan
    d = df.copy().sort_values("timestamp")
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    last_day = d["timestamp"].dt.date.max()
    d = d[d["timestamp"].dt.date == last_day].copy()
    d = d[d["timestamp"] >= pd.Timestamp.combine(last_day, dtime(9, 15))].reset_index(drop=True)
    if len(d) < 2: return np.nan
    close = pd.to_numeric(d["close"], errors="coerce")
    vol = pd.to_numeric(d["volume"], errors="coerce").fillna(0.0)
    obv = (vol * np.sign(close.diff().fillna(0.0))).cumsum()
    return round(float(obv.iloc[-1]), 2)

def compute_ivp(history_df: pd.DataFrame, min_bars: int = 10) -> Tuple[float, str]:
    if history_df is None or history_df.empty or len(history_df) < min_bars: return np.nan, "Neutral Vol"
    close = pd.to_numeric(history_df["close"], errors="coerce")
    high = pd.to_numeric(history_df["high"], errors="coerce")
    low = pd.to_numeric(history_df["low"], errors="coerce")
    proxy = ((high - low) / close.replace(0, np.nan) * 100.0).dropna()
    if proxy.empty: return np.nan, "Neutral Vol"
    lookback = proxy.tail(min(IVP_LOOKBACK_DAYS, len(proxy)))
    ivp = round((lookback.lt(float(lookback.iloc[-1])).sum() / len(lookback)) * 100, 2)
    return (ivp, "Buyer Zone") if ivp < 30 else ((ivp, "Avoid Buy Premium") if ivp > 50 else (ivp, "Neutral Vol"))

def score_label(delta: float) -> str:
    if pd.isna(delta): return "Neutral"
    if delta >= 7: return "Buy++"
    if delta >= 4: return "Buy+"
    if delta >= 1: return "Buy"
    if delta <= -7: return "Sell++"
    if delta <= -4: return "Sell+"
    if delta <= -1: return "Sell"
    return "Neutral"

def directional_label(raw, side):
    r = str(raw).strip().upper()
    mapping = {"BUY": "LONG", "BUY+": "LONG+", "BUY++": "LONG++", "SELL": "SHORT", "SELL+": "SHORT+", "SELL++": "SHORT++"}
    if side == "short": mapping = {v: k for k, v in mapping.items()}
    return mapping.get(r, r)

def summarize_intraday(intra_df: pd.DataFrame, reference_df: pd.DataFrame) -> Dict:
    df = intra_df.copy().sort_values("timestamp").reset_index(drop=True)
    close, high, low, vol = [pd.to_numeric(df[c], errors="coerce") for c in ["close", "high", "low", "volume"]]
    delta = close.diff().fillna(0)
    rsi = (100 - (100 / (1 + (delta.clip(0).rolling(14).mean() / (-delta).clip(0).rolling(14).mean().replace(0, np.nan))))).fillna(50)
    tr = pd.concat([(high-low).abs(), (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    plus_di = 100 * (np.where(high.diff() > -low.diff(), high.diff().clip(0), 0)).rolling(14).mean() / atr.replace(0, np.nan)
    minus_di = 100 * (np.where(-low.diff() > high.diff(), -low.diff().clip(0), 0)).rolling(14).mean() / atr.replace(0, np.nan)
    adx = (100 * (plus_di-minus_di).abs() / (plus_di+minus_di).replace(0, np.nan)).rolling(14).mean().fillna(0)

    prev_close = safe_float(reference_df["close"].iloc[-2], np.nan)
    pct = ((close.iloc[-1] - prev_close)/prev_close * 100) if pd.notna(prev_close) else 0

    bull = (pct > 0) + (plus_di.iloc[-1] > minus_di.iloc[-1]) + (rsi.iloc[-1] >= 55)
    bear = (pct < 0) + (minus_di.iloc[-1] > plus_di.iloc[-1]) + (rsi.iloc[-1] <= 45)

    return {
        "LTP": round(close.iloc[-1], 2), "% Change": round(pct, 2),
        "Bull_Signal": score_label(bull), "Bear_Signal": score_label(-bear),
        "Rank Delta": bull - bear, "Cumulative ADX": round(safe_float(adx.iloc[-1], 0), 2)
    }

def scan_symbol(symbol: str) -> Optional[Dict]:
    eq = format_eq_symbol(symbol)
    d, i = get_history(eq, "D", 252), get_history(eq, "5", 20)
    if d.empty or i.empty: return None
    s = summarize_intraday(i, d)
    s["Symbol"] = symbol
    return s

def fetch_option_pairs(symbol: str) -> pd.DataFrame:
    if fyers is None: return pd.DataFrame()
    try:
        chain = fyers.optionchain(data={"symbol": format_eq_symbol(symbol), "strikecount": 50})
        oc = pd.DataFrame([(i["strike"], i["type"], i["symbol"], i.get("lp", 0), i.get("oi", 0), i.get("volume", 0)) for i in chain["data"]["optionsChain"]], 
                          columns=["Strike", "Type", "Sym", "OptionLTP", "OI", "Volume"])
        return oc
    except: return pd.DataFrame()

def scan_single_option(sym, typ, strike, underlying) -> Optional[Dict]:
    i = get_history(sym if sym.startswith("NSE:") else f"NSE:{sym}", "5", 7)
    if i.empty: return None
    return {"Underlying": underlying, "Option Type": typ, "Option Symbol": sym, "Strike": strike, "OBV": compute_today_obv(i)}

def build_option_candidates(seed, side):
    rows = []
    for underlying in seed["Symbol"].dropna().astype(str):
        for _, row in fetch_option_pairs(underlying).iterrows():
            s = scan_single_option(row["Sym"], row["Type"], row["Strike"], underlying)
            if s:
                s.update({"OI": row["OI"], "Volume": row["Chain Volume"], "LTP": row["OptionLTP"]})
                rows.append(s)
    return pd.DataFrame(rows)

def main():
    init_fyers()
    symbols = load_fno_symbols_from_sectors(SECTORS_DIR)
    with ThreadPoolExecutor(max_workers=8) as ex:
        summary = pd.DataFrame(filter(None, ex.map(scan_symbol, symbols)))

    summary = summary.sort_values("Rank Delta", ascending=False)
    longs, shorts = choose_top_candidates(summary, TOP_N_UNDERLYINGS)

    # ... (rest of main remains similar) ...

if __name__ == "__main__":
    main()
