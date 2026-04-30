import os
import smtplib
import logging
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from email.message import EmailMessage

try:
    from fyers_apiv3 import fyersModel
except Exception:
    try:
        import fyersModel
    except Exception:
        from fyersapi import fyersModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DAILY_LOOKBACK_DAYS = 90
INTRADAY_LOOKBACK_DAYS = 20
IVP_LOOKBACK_DAYS = 252
OPTION_PAIRS_TO_KEEP = 5
ITERATIONS_TO_KEEP = 75
SECTORS_DIR = os.environ.get("SECTORS_DIR", "sectors")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", ".")
MIN_OPTION_LTP = 10.0
INDEX_UNDERLYINGS = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX"]

EMAIL_DISPLAY_COLS = [
    "Symbol", "LTP", "% Change", "5m_Signal", "15m_Signal", "30m_Signal", "60m_Signal",
    "Bull_Signal", "Bear_Signal", "Overall_Signal", "Price_Lead_Status", "IVP",
    "Volatility State", "Last Iteration Time",
]

OPTION_EMAIL_COLS = [
    "Underlying", "Option Type", "Option Symbol", "Strike", "LTP", "% Change", "OI", "Volume", "OBV",
    "OI+Volume+OBV Score", "5m_Signal", "15m_Signal", "30m_Signal", "60m_Signal", "Bull_Signal",
    "Bear_Signal", "Overall_Signal", "Price_Lead_Status", "IVP", "Volatility State", "Last Iteration Time",
]

fyers = None

def init_fyers() -> Optional[object]:
    global fyers
    client_id = os.environ.get("CLIENT_ID") or os.environ.get("CLIENTID")
    access_token = os.environ.get("ACCESS_TOKEN") or os.environ.get("ACCESSTOKEN")
    if not client_id or not access_token:
        logger.error("Missing CLIENT_ID / ACCESS_TOKEN environment variables.")
        fyers = None
        return None
    try:
        fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=access_token, log_path="")
    except Exception:
        fyers = fyersModel.FyersModel(client_id=client_id, token=access_token, is_async=False, log_path="")
    return fyers

def safe_float(value, default=np.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default

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
        symbol_col = None
        for key in ["symbol", "symbols", "ticker", "tradingsymbol"]:
            if key in lowered:
                symbol_col = lowered[key]
                break
        if symbol_col is None:
            continue
        for raw in df[symbol_col].dropna().astype(str):
            sym = raw.strip().upper()
            if sym and sym not in {"NAN", "NONE"}:
                symbols.add(sym)
    return sorted(symbols)

def load_scan_universe(root_dir: str = SECTORS_DIR) -> List[str]:
    syms = load_fno_symbols_from_sectors(root_dir)
    syms.extend(INDEX_UNDERLYINGS)
    seen, out = set(), []
    for s in syms:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def format_eq_symbol(symbol: str) -> str:
    symbol = str(symbol).strip().upper()
    if symbol.startswith("NSE:"):
        return symbol if symbol.endswith("-EQ") else f"{symbol}-EQ"
    return f"NSE:{symbol}-EQ"

def get_history(symbol: str, resolution: str, days_back: int) -> pd.DataFrame:
    if fyers is None:
        return pd.DataFrame()
    now = datetime.now()
    start = now - timedelta(days=days_back)
    payload = {"symbol": symbol, "resolution": resolution, "date_format": "1", "range_from": start.strftime("%Y-%m-%d"), "range_to": now.strftime("%Y-%m-%d"), "cont_flag": "1"}
    try:
        res = fyers.history(data=payload)
    except Exception:
        return pd.DataFrame()
    candles = (res or {}).get("candles", [])
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

def compute_obv(df: pd.DataFrame) -> float:
    if df is None or df.empty or len(df) < 2:
        return np.nan
    close = pd.to_numeric(df["close"], errors="coerce")
    vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    obv = 0.0
    for i in range(1, len(df)):
        if pd.isna(close.iloc[i]) or pd.isna(close.iloc[i - 1]):
            continue
        if close.iloc[i] > close.iloc[i - 1]:
            obv += vol.iloc[i]
        elif close.iloc[i] < close.iloc[i - 1]:
            obv -= vol.iloc[i]
    return round(obv, 2)

def compute_today_obv(df: pd.DataFrame) -> float:
    if df is None or df.empty or len(df) < 2:
        return np.nan
    d = df.copy().sort_values("timestamp")
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    today = pd.Timestamp.now(tz=None).date()
    d = d[d["timestamp"].dt.date == today].copy()
    if d.empty:
        return np.nan
    start_anchor = pd.Timestamp.combine(today, time(9, 15))
    d = d[d["timestamp"] >= start_anchor].copy()
    if len(d) < 2:
        return np.nan
    close = pd.to_numeric(d["close"], errors="coerce")
    vol = pd.to_numeric(d["volume"], errors="coerce").fillna(0.0)
    obv = 0.0
    for i in range(1, len(d)):
        if pd.isna(close.iloc[i]) or pd.isna(close.iloc[i - 1]):
            continue
        if close.iloc[i] > close.iloc[i - 1]:
            obv += vol.iloc[i]
        elif close.iloc[i] < close.iloc[i - 1]:
            obv -= vol.iloc[i]
    return round(obv, 2)

def compute_ivp(history_df: pd.DataFrame) -> Tuple[float, str]:
    if history_df is None or history_df.empty or len(history_df) < 30:
        return np.nan, "Neutral Vol"
    close = pd.to_numeric(history_df["close"], errors="coerce")
    high = pd.to_numeric(history_df["high"], errors="coerce")
    low = pd.to_numeric(history_df["low"], errors="coerce")
    proxy = ((high - low) / close.replace(0, np.nan) * 100.0).replace([np.inf, -np.inf], np.nan).dropna()
    if proxy.empty:
        return np.nan, "Neutral Vol"
    lookback = proxy.tail(min(IVP_LOOKBACK_DAYS, len(proxy)))
    current = float(lookback.iloc[-1])
    ivp = round((lookback.lt(current).sum() / len(lookback)) * 100, 2)
    if ivp < 30:
        return ivp, "Buyer Zone"
    if ivp > 50:
        return ivp, "Avoid Buy Premium"
    return ivp, "Neutral Vol"

def score_label(delta: float) -> str:
    if pd.isna(delta): return "Neutral"
    if delta >= 7: return "Buy++"
    if delta >= 4: return "Buy+"
    if delta >= 1: return "Buy"
    if delta <= -7: return "Sell++"
    if delta <= -4: return "Sell+"
    if delta <= -1: return "Sell"
    return "Neutral"

def long_short_label(raw_label: str, side: str) -> str:
    if side == "long":
        return {"Buy": "LONG", "Buy+": "LONG+", "Buy++": "LONG++"}.get(raw_label, "Neutral")
    if side == "short":
        return {"Sell": "SHORT", "Sell+": "SHORT+", "Sell++": "SHORT++"}.get(raw_label, "Neutral")
    return raw_label

def apply_display_labels(df: pd.DataFrame, side: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in ["5m_Signal", "15m_Signal", "30m_Signal", "60m_Signal", "Bull_Signal", "Bear_Signal", "Overall_Signal"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: long_short_label(str(x), side))
    return out

def intraday_window_score(df: pd.DataFrame, window_minutes: int = 5) -> float:
    if df is None or df.empty or len(df) < 2:
        return np.nan
    d = df.copy().sort_values("timestamp")
    end_ts = pd.to_datetime(d["timestamp"].iloc[-1])
    start_ts = end_ts - timedelta(minutes=window_minutes)
    cur = d[(d["timestamp"] >= start_ts) & (d["timestamp"] <= end_ts)]
    if cur.empty or len(cur) < 2:
        return np.nan
    first_close = safe_float(cur["close"].iloc[0])
    last_close = safe_float(cur["close"].iloc[-1])
    if pd.isna(first_close) or first_close == 0:
        return np.nan
    return round(((last_close - first_close) / first_close) * 100.0, 2)

def previous_trading_day_same_time_score(df: pd.DataFrame, window_minutes: int = 5) -> float:
    if df is None or df.empty or len(df) < 4:
        return np.nan
    d = df.copy().sort_values("timestamp").reset_index(drop=True)
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    end_ts = pd.to_datetime(d["timestamp"].iloc[-1])
    target_time = end_ts.time()
    trading_days = sorted(d["timestamp"].dt.date.unique())
    current_day = end_ts.date()
    prev_days = [day for day in trading_days if day < current_day]
    if not prev_days:
        return np.nan
    prev_day = prev_days[-1]
    prev_day_data = d[d["timestamp"].dt.date == prev_day].copy()
    if prev_day_data.empty:
        return np.nan
    same_time_rows = prev_day_data[(prev_day_data["timestamp"].dt.hour == target_time.hour) & (prev_day_data["timestamp"].dt.minute == target_time.minute)]
    if same_time_rows.empty:
        candidate_times = prev_day_data["timestamp"].sort_values()
        if candidate_times.empty:
            return np.nan
        prev_end = candidate_times.iloc[-1]
    else:
        prev_end = pd.to_datetime(same_time_rows.iloc[-1]["timestamp"])
    prev_start = prev_end - timedelta(minutes=window_minutes)
    prev_window = prev_day_data[(prev_day_data["timestamp"] >= prev_start) & (prev_day_data["timestamp"] <= prev_end)]
    if prev_window.empty or len(prev_window) < 2:
        return np.nan
    first_close = safe_float(prev_window["close"].iloc[0])
    last_close = safe_float(prev_window["close"].iloc[-1])
    if pd.isna(first_close) or first_close == 0:
        return np.nan
    return round(((last_close - first_close) / first_close) * 100.0, 2)

def compare_window_signal(current_score: float, previous_score: float) -> Tuple[float, str]:
    if pd.isna(current_score) or pd.isna(previous_score):
        return np.nan, "Neutral"
    delta = round(current_score - previous_score, 2)
    if delta >= 0.50: return delta, "Buy++"
    if delta >= 0.20: return delta, "Buy+"
    if delta >= 0.05: return delta, "Buy"
    if delta <= -0.50: return delta, "Sell++"
    if delta <= -0.20: return delta, "Sell+"
    if delta <= -0.05: return delta, "Sell"
    return delta, "Neutral"

def build_iteration_history(df: pd.DataFrame, window_minutes: int = 5, iterations: int = ITERATIONS_TO_KEEP) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy().sort_values("timestamp").reset_index(drop=True)
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    last_day = d["timestamp"].dt.date.max()
    d = d[d["timestamp"].dt.date == last_day].copy().reset_index(drop=True)
    if d.empty:
        return pd.DataFrame()
    start_anchor = pd.Timestamp.combine(pd.Timestamp(last_day).date(), time(9, 15))
    end_anchor = pd.Timestamp.combine(pd.Timestamp(last_day).date(), time(15, 30))
    d = d[(d["timestamp"] >= start_anchor) & (d["timestamp"] <= end_anchor)].copy().reset_index(drop=True)
    if d.empty:
        return pd.DataFrame()
    rows = []
    for i in range(len(d)):
        end_ts = pd.to_datetime(d.loc[i, "timestamp"])
        if end_ts < start_anchor:
            continue
        start_ts = end_ts - timedelta(minutes=window_minutes)
        cur = d[(d["timestamp"] >= start_ts) & (d["timestamp"] <= end_ts)]
        if cur.empty or len(cur) < 2:
            continue
        first_close = safe_float(cur["close"].iloc[0])
        last_close = safe_float(cur["close"].iloc[-1])
        if pd.isna(first_close) or first_close == 0:
            continue
        current_score = round(((last_close - first_close) / first_close) * 100.0, 2)
        prev_score = previous_trading_day_same_time_score(d[d["timestamp"] <= end_ts].copy(), window_minutes)
        delta, signal = compare_window_signal(current_score, prev_score)
        rows.append({"iteration": len(rows) + 1, "timestamp": end_ts.strftime("%H:%M"), "window_minutes": window_minutes, "window_start": start_ts.strftime("%H:%M"), "window_end": end_ts.strftime("%H:%M"), "current_window_score": current_score, "previous_trading_day_same_time_score": prev_score, "window_delta": delta, "window_signal": signal, "close": last_close})
        if len(rows) >= iterations:
            break
    return pd.DataFrame(rows)

def summarize_intraday(intra_df: pd.DataFrame, reference_df: pd.DataFrame) -> Dict[str, object]:
    if intra_df is None or intra_df.empty:
        return {}
    df = intra_df.copy().sort_values("timestamp").reset_index(drop=True)
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    delta = close.diff().fillna(0.0)
    avg_gain = delta.clip(lower=0.0).rolling(14, min_periods=14).mean()
    avg_loss = (-delta).clip(lower=0.0).rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).fillna(50)
    tr = pd.concat([(high - low).abs(), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    up_move = high.diff(); down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = pd.Series(tr).rolling(14, min_periods=14).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(14, min_periods=14).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm).rolling(14, min_periods=14).mean() / atr.replace(0, np.nan)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    adx = dx.rolling(14, min_periods=14).mean().fillna(0)
    typical = (high + low + close) / 3.0
    cum_vol = volume.cumsum().replace(0, np.nan)
    vwap = ((typical * volume).cumsum() / cum_vol).ffill().fillna(close)
    vwap_std = (((typical - vwap) ** 2 * volume).cumsum() / cum_vol).pow(0.5).replace(0, np.nan)
    vwap_z = ((close - vwap) / vwap_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    range_now = (high - low).clip(lower=0)
    avg_range5 = range_now.rolling(5, min_periods=3).mean()
    avg_vol5 = volume.rolling(5, min_periods=3).mean()
    price_lead_flag = ((range_now / avg_range5.replace(0, np.nan)) >= 1.5) & ((volume / avg_vol5.replace(0, np.nan)) <= 1.0)
    streak = []
    run = 0
    for flag in price_lead_flag.fillna(False).astype(bool):
        run = run + 1 if flag else 0
        streak.append(run)
    streak = pd.Series(streak)
    lead_status = np.select([price_lead_flag & (streak >= 3), price_lead_flag & (streak >= 2), price_lead_flag], ["STRONG_PRICE_LEAD_FADE", "PRICE_LEADING_FADE_RISK", "EARLY_PRICE_LEAD"], default="NORMAL")
    prev_close = safe_float(reference_df["close"].iloc[-2]) if reference_df is not None and len(reference_df) >= 2 else np.nan
    ltp = safe_float(close.iloc[-1])
    pct_change = ((ltp - prev_close) / prev_close * 100.0) if pd.notna(prev_close) and prev_close != 0 else 0.0
    current_win = intraday_window_score(df)
    previous_win = previous_trading_day_same_time_score(df)
    win_delta, win_signal = compare_window_signal(current_win, previous_win)
    iteration_history = build_iteration_history(df)
    bull = 0
    bear = 0
    if pct_change > 0: bull += 1
    if pct_change < 0: bear += 1
    if safe_float(vwap_z.iloc[-1], 0) >= 0.30: bull += 1
    if safe_float(vwap_z.iloc[-1], 0) <= -0.30: bear += 1
    if safe_float(plus_di.iloc[-1], 0) > safe_float(minus_di.iloc[-1], 0): bull += 1
    if safe_float(minus_di.iloc[-1], 0) > safe_float(plus_di.iloc[-1], 0): bear += 1
    if safe_float(adx.iloc[-1], 0) >= 20:
        bull += 1
        bear += 1
    if safe_float(rsi.iloc[-1], 50) >= 55: bull += 1
    if safe_float(rsi.iloc[-1], 50) <= 45: bear += 1
    if win_signal.startswith("Buy"): bull += 2
    elif win_signal.startswith("Sell"): bear += 2
    rank_delta = bull - bear
    ivp, vol_state = compute_ivp(reference_df)
    last_ts = pd.to_datetime(df["timestamp"].iloc[-1])
    return {"LTP": round(ltp, 2), "% Change": round(pct_change, 2), "5m_Signal": score_label(rank_delta), "15m_Signal": win_signal, "30m_Signal": score_label(rank_delta * 0.8), "60m_Signal": score_label(rank_delta * 0.7), "Bull_Signal": score_label(bull), "Bear_Signal": score_label(-bear), "Overall_Signal": win_signal, "Price_Lead_Status": str(lead_status[-1]), "IVP": ivp, "Volatility State": vol_state, "Last Iteration Time": last_ts.strftime("%H:%M"), "Bull Rank": bull, "Bear Rank": bear, "Rank Delta": rank_delta, "Cumulative ADX": round(safe_float(adx.iloc[-1], np.nan), 2), "Iteration History": iteration_history}

def choose_top_candidates(summary_df: pd.DataFrame, top_n: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if summary_df is None or summary_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    long_df = summary_df[pd.to_numeric(summary_df["Rank Delta"], errors="coerce") > 0].copy()
    short_df = summary_df[pd.to_numeric(summary_df["Rank Delta"], errors="coerce") < 0].copy()
    long_df = long_df.sort_values(["Rank Delta", "Cumulative ADX", "% Change"], ascending=[False, False, False]).head(top_n)
    short_df = short_df.sort_values(["Rank Delta", "Cumulative ADX", "% Change"], ascending=[True, False, True]).head(top_n)
    return long_df.reset_index(drop=True), short_df.reset_index(drop=True)

def fetch_raw_option_chain(symbol: str) -> pd.DataFrame:
    if fyers is None:
        return pd.DataFrame()
    symu = str(symbol).strip().upper()
    if symu in {"NIFTY", "NIFTY50", "NIFTY 50", "NSE:NIFTY", "NSE:NIFTY50-INDEX"}:
        eq_symbol = "NSE:NIFTY50-INDEX"
    elif symu in {"BANKNIFTY", "NIFTY BANK", "BANK NIFTY", "NSE:BANKNIFTY", "NSE:NIFTYBANK-INDEX"}:
        eq_symbol = "NSE:NIFTYBANK-INDEX"
    elif symu.startswith("NSE:") and symu.endswith("-INDEX"):
        eq_symbol = symu
    else:
        eq_symbol = format_eq_symbol(symbol)
    try:
        quote = fyers.quotes({"symbols": eq_symbol})
        ltp = safe_float(quote.get("d", [{}])[0].get("v", {}).get("lp"), np.nan)
        chain_res = fyers.optionchain(data={"symbol": eq_symbol, "strikecount": 50})
    except Exception:
        return pd.DataFrame()
    chain = ((chain_res or {}).get("data") or {}).get("optionsChain", [])
    rows = []
    for item in chain or []:
        rows.append({
            "Underlying": symu,
            "EqSymbol": eq_symbol,
            "UnderlyingLTP": ltp,
            "Strike": safe_float(item.get("strike_price") or item.get("strike"), np.nan),
            "Option Type": str(item.get("option_type") or item.get("type") or "").upper(),
            "Option Symbol": str(item.get("symbol", "")),
            "LTP": safe_float(item.get("ltp") or item.get("lp"), np.nan),
            "OI": safe_float(item.get("oi") or item.get("open_interest"), np.nan),
            "Volume": safe_float(item.get("volume"), np.nan),
        })
    return pd.DataFrame(rows)

def nearest_step(value: float) -> int:
    val = abs(safe_float(value, 0))
    if val >= 20000: return 100
    if val >= 10000: return 50
    if val >= 2000: return 20
    if val >= 500: return 10
    if val >= 100: return 5
    return 1

def fetch_option_pairs(symbol: str, pair_count: int = OPTION_PAIRS_TO_KEEP) -> pd.DataFrame:
    if fyers is None:
        return pd.DataFrame()
    symu = str(symbol).strip().upper()
    if symu in {"NIFTY", "NIFTY50", "NIFTY 50", "NSE:NIFTY", "NSE:NIFTY50-INDEX"}:
        eq_symbol = "NSE:NIFTY50-INDEX"
    elif symu in {"BANKNIFTY", "NIFTY BANK", "BANK NIFTY", "NSE:BANKNIFTY", "NSE:NIFTYBANK-INDEX"}:
        eq_symbol = "NSE:NIFTYBANK-INDEX"
    elif symu.startswith("NSE:") and symu.endswith("-INDEX"):
        eq_symbol = symu
    else:
        eq_symbol = format_eq_symbol(symbol)
    try:
        quote = fyers.quotes({"symbols": eq_symbol})
        ltp = safe_float(quote.get("d", [{}])[0].get("v", {}).get("lp"), np.nan)
        chain_res = fyers.optionchain(data={"symbol": eq_symbol, "strikecount": 50})
    except Exception:
        return pd.DataFrame()
    chain = ((chain_res or {}).get("data") or {}).get("optionsChain", [])
    if not chain:
        return pd.DataFrame()
    rows = []
    for item in chain:
        strike = safe_float(item.get("strike_price") or item.get("strike"), np.nan)
        typ = str(item.get("option_type") or item.get("type") or "").upper()
        if pd.isna(strike) or typ not in {"CE", "PE"}:
            continue
        rows.append({"Strike": strike, "Type": typ, "OptionSymbol": str(item.get("symbol", "")), "OptionLTP": safe_float(item.get("ltp") or item.get("lp"), 0.0), "OI": safe_float(item.get("oi") or item.get("open_interest"), np.nan), "Volume": safe_float(item.get("volume"), np.nan)})
    if not rows:
        return pd.DataFrame()
    oc = pd.DataFrame(rows)
    step = nearest_step(ltp if pd.notna(ltp) else oc["Strike"].median())
    atm = round(ltp / step) * step if pd.notna(ltp) else oc["Strike"].median()
    strikes = sorted(oc["Strike"].dropna().unique(), key=lambda x: abs(x - atm))[:pair_count]
    final_rows = []
    for strike in sorted(strikes):
        sub = oc[oc["Strike"] == strike]
        ce, pe = sub[sub["Type"] == "CE"], sub[sub["Type"] == "PE"]
        final_rows.append({"Strike": strike, "CE Symbol": ce["OptionSymbol"].iloc[0] if not ce.empty else "", "PE Symbol": pe["OptionSymbol"].iloc[0] if not pe.empty else "", "CE OI": ce["OI"].iloc[0] if not ce.empty else 0, "CE Volume": ce["Volume"].iloc[0] if not ce.empty else 0, "PE OI": pe["OI"].iloc[0] if not pe.empty else 0, "PE Volume": pe["Volume"].iloc[0] if not pe.empty else 0})
    return pd.DataFrame(final_rows)

def get_today_stats(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"OI": 0, "Volume": 0, "OBV": 0}
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    today = pd.Timestamp.now(tz=None).date()
    d = d[d["timestamp"].dt.date == today].copy()
    if d.empty:
        return {"OI": 0, "Volume": 0, "OBV": 0}
    start_anchor = pd.Timestamp.combine(today, time(9, 15))
    d = d[d["timestamp"] >= start_anchor].copy()
    if d.empty:
        return {"OI": 0, "Volume": 0, "OBV": 0}
    return {"OI": d["oi"].iloc[-1] if "oi" in d.columns else 0, "Volume": d["volume"].sum(), "OBV": compute_today_obv(d)}

def get_prev_day_stats(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"OI": 0, "Volume": 0, "OBV": 0}
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    days = sorted(d["timestamp"].dt.date.unique())
    if len(days) < 2:
        return {"OI": 0, "Volume": 0, "OBV": 0}
    prev_day = days[-2]
    d = d[d["timestamp"].dt.date == prev_day].copy()
    return {"OI": d["oi"].iloc[-1] if "oi" in d.columns else 0, "Volume": d["volume"].sum(), "OBV": compute_obv(d)}

def scan_single_option(option_symbol: str, option_type: str, strike: float, underlying: str) -> Optional[Dict]:
    hist_symbol = option_symbol if option_symbol.startswith("NSE:") else f"NSE:{option_symbol}"
    daily_df = get_history(hist_symbol, "D", max(DAILY_LOOKBACK_DAYS, IVP_LOOKBACK_DAYS))
    intra_df = get_history(hist_symbol, "5", INTRADAY_LOOKBACK_DAYS)
    if daily_df.empty or intra_df.empty:
        return None
    today_stats = get_today_stats(intra_df)
    prev_stats = get_prev_day_stats(intra_df)
    summary = summarize_intraday(intra_df, daily_df)
    if not summary:
        return None
    summary.update({"Underlying": underlying, "Option Type": option_type, "Option Symbol": option_symbol, "Strike": strike, "OBV": today_stats["OBV"], "OI": today_stats["OI"], "Volume": today_stats["Volume"], "OI_Delta": today_stats["OI"] - prev_stats["OI"], "Vol_Delta": today_stats["Volume"] - prev_stats["Volume"], "OBV_Delta": today_stats["OBV"] - prev_stats["OBV"]})
    return summary

def option_liquidity_score(oi, volume, obv) -> float:
    return round((np.log1p(max(safe_float(oi, 0), 0)) * 0.45) + (np.log1p(max(safe_float(volume, 0), 0)) * 0.35) + (np.log1p(max(abs(safe_float(obv, 0)), 0)) * 0.20), 4)

def build_option_candidates(candidates_df: pd.DataFrame, side: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if candidates_df.empty or "Symbol" not in candidates_df.columns:
        return pd.DataFrame(), pd.DataFrame()
    rows, iter_rows = [], []
    for underlying in candidates_df["Symbol"].dropna().astype(str):
        pair_df = fetch_option_pairs(underlying)
        if pair_df.empty:
            continue
        for _, row in pair_df.iterrows():
            strike = row.get("Strike", np.nan)
            for opt_type in ["CE", "PE"]:
                sym = row.get(f"{opt_type} Symbol", "")
                if not sym:
                    continue
                scanned = scan_single_option(sym, opt_type, strike, underlying)
                if not scanned:
                    continue
                scanned["OI"] = row.get(f"{opt_type} OI", 0)
                scanned["Volume"] = row.get(f"{opt_type} Volume", 0)
                scanned["OI+Volume+OBV Score"] = option_liquidity_score(scanned["OI"], scanned["Volume"], scanned["OBV"])
                if safe_float(scanned.get("LTP"), 0.0) < MIN_OPTION_LTP:
                    continue
                rows.append(scanned)
                hist = scanned.get("Iteration History")
                if isinstance(hist, pd.DataFrame) and not hist.empty:
                    tmp = hist.copy()
                    tmp.insert(0, "Option Symbol", sym)
                    tmp.insert(1, "Underlying", underlying)
                    tmp.insert(2, "Strike", strike)
                    tmp.insert(3, "Strike Name", str(strike))
                    iter_rows.append(tmp)
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    out = pd.DataFrame(rows)
    rd = pd.to_numeric(out["Rank Delta"], errors="coerce")
    pct = pd.to_numeric(out["% Change"], errors="coerce")
    if side == "long":
        out = out[(rd > 0) & (pct > 0)].copy()
        out = out.sort_values(["OI+Volume+OBV Score", "Rank Delta", "Cumulative ADX", "% Change"], ascending=[False, False, False, False])
    else:
        out = out[(rd < 0) & (pct < 0)].copy()
        out = out.sort_values(["OI+Volume+OBV Score", "Rank Delta", "Cumulative ADX", "% Change"], ascending=[False, True, False, True])
    final_out = out[[c for c in OPTION_EMAIL_COLS if c in out.columns]].reset_index(drop=True)
    iter_df = pd.DataFrame()
    if iter_rows and not final_out.empty:
        all_iters = pd.concat(iter_rows, ignore_index=True)
        all_iters = all_iters[all_iters["Option Symbol"].isin(final_out["Option Symbol"])].copy()
        all_iters = all_iters.sort_values(["Underlying", "Strike", "Option Symbol", "iteration"]).reset_index(drop=True)
        if not all_iters.empty:
            all_iters = all_iters.groupby(["Underlying", "Strike", "Option Symbol"], as_index=False, group_keys=False).apply(lambda x: x.assign(iteration=range(1, min(len(x), ITERATIONS_TO_KEEP) + 1)))
            all_iters["iteration"] = pd.to_numeric(all_iters["iteration"], errors="coerce").astype("Int64")
            all_iters = all_iters[all_iters["iteration"].between(1, ITERATIONS_TO_KEEP)]
            iter_df = all_iters.reset_index(drop=True)
    return final_out, iter_df

def format_cell(col: str, val) -> str:
    if pd.isna(val):
        return ""
    if col == "% Change":
        return f"{float(val):.2f}%"
    if isinstance(val, (int, float, np.integer, np.floating)):
        return f"{float(val):.2f}"
    return str(val)

def _cell_bg(col: str, value: str) -> str:
    v = str(value).strip().upper()
    if col == "% Change":
        try:
            num = float(str(value).replace('%', '').strip())
        except Exception:
            return '#2d3651'
        if num > 0:
            return '#2e7d32'
        if num < 0:
            return '#c62828'
        return '#546e7a'
    if col in {"5m_Signal", "15m_Signal", "30m_Signal", "60m_Signal", "Bull_Signal", "Bear_Signal", "Overall_Signal"}:
        if "LONG++" in v or "BUY++" in v:
            return '#2e7d32'
        if "LONG+" in v or "BUY+" in v:
            return '#388e3c'
        if v in {"LONG", "BUY"}:
            return '#43a047'
        if "SHORT++" in v or "SELL++" in v:
            return '#c62828'
        if "SHORT+" in v or "SELL+" in v:
            return '#d32f2f'
        if v in {"SHORT", "SELL"}:
            return '#e53935'
        if "NEUTRAL" in v:
            return '#6b7280'
    if col == "Volatility State":
        if "AVOID BUY PREMIUM" in v:
            return '#fbc02d'
        if "BUYER ZONE" in v:
            return '#9ccc65'
        return '#546e7a'
    if col == "Price_Lead_Status":
        if "EARLY_PRICE_LEAD" in v:
            return '#00897b'
        if "FADE" in v:
            return '#8e24aa'
        return '#2d3651'
    return '#2d3651'

def _text_color(bg: str) -> str:
    return '#111111' if bg.lower() in {'#fbc02d', '#9ccc65'} else '#ffffff'

def colored_table_html(df: pd.DataFrame, columns: List[str], title: str) -> str:
    html = ["<div style='margin:0 0 12px 0;'>", f"<div style='margin:10px 0 4px 0; font-family:Arial,Helvetica,sans-serif; font-size:13px; color:#000;'><b>{title}</b></div>"]
    if df is None or df.empty:
        html.append("<div style='font-family:Arial,Helvetica,sans-serif; font-size:12px; color:#000;'>No data found.</div></div>")
        return "".join(html)
    view = df[[c for c in columns if c in df.columns]].copy()
    html.append("<table cellpadding='0' cellspacing='1' style='border-collapse:separate; border-spacing:1px; background:#ffffff; font-family:Arial,Helvetica,sans-serif; font-size:10px;'>")
    html.append("<tr>")
    for c in view.columns:
        html.append(f"<th style='background:#2f3b59; color:#ffffff; text-align:center; font-weight:bold; padding:5px 6px; white-space:nowrap; border:none;'>{c}</th>")
    html.append("</tr>")
    for _, row in view.iterrows():
        html.append("<tr>")
        for c in view.columns:
            cell_val = format_cell(c, row[c])
            bg = _cell_bg(c, cell_val)
            fg = _text_color(bg)
            html.append(f"<td style='background:{bg}; color:{fg}; text-align:center; padding:4px 6px; white-space:nowrap; border:none;'>{cell_val}</td>")
        html.append("</tr>")
    html.append("</table></div>")
    return "".join(html)

def prepare_option_email_view(df: pd.DataFrame, side: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=OPTION_EMAIL_COLS)
    out = df.copy()
    if "LTP" in out.columns:
        out = out[pd.to_numeric(out["LTP"], errors="coerce") >= MIN_OPTION_LTP].copy()
    if "OBV" in out.columns:
        obv = pd.to_numeric(out["OBV"], errors="coerce")
        out = out[obv > 0].copy() if side == "long" else out[obv < 0].copy()
    out = apply_display_labels(out, side)
    timing_cols = [c for c in ["5m_Signal", "15m_Signal", "30m_Signal", "60m_Signal"] if c in out.columns]
    if timing_cols:
        neutral_like = {"", "-", "NEUTRAL", "NAN", "NONE"}
        mask = ~out[timing_cols].apply(lambda row: any(str(v).strip().upper() in neutral_like for v in row), axis=1)
        out = out[mask].copy()
    if "Overall_Signal" in out.columns:
        out["Overall_Signal"] = "LONG++" if side == "long" else "SHORT++"
    final_cols = [c for c in OPTION_EMAIL_COLS if c in out.columns]
    return out[final_cols].reset_index(drop=True)

def send_email(long_df, short_df, ce_df, pe_df, attachments) -> bool:
    sender_email = os.environ.get("SENDER_EMAIL")
    recipient_email = os.environ.get("RECIPIENT_EMAIL")
    sender_password = os.environ.get("SENDER_PASSWORD")
    if not sender_email or not recipient_email or not sender_password:
        logger.error("Missing email credentials.")
        return False

    subject_time = datetime.now().strftime("%d %b %H:%M")
    scan_time = datetime.now().strftime("%d %b %Y, %H:%M")
    buy_view = prepare_option_email_view(ce_df, "long")
    short_view = prepare_option_email_view(pe_df, "short")

    html = "".join([
        "<html>",
        '<body style="margin:0; padding:8px; font-family:Arial,Helvetica,sans-serif; font-size:13px; color:#000; background:#ffffff;">',
        '<div style="margin:0 0 6px 0; font-family:Arial,Helvetica,sans-serif; font-size:13px; color:#000;"><b>Intraday Vol Iteration Alert</b></div>',
        f'<div style="margin:0 0 10px 0; font-family:Arial,Helvetica,sans-serif; font-size:12px; color:#000;">Scan completed at {scan_time}.</div>',
        colored_table_html(buy_view, OPTION_EMAIL_COLS, "Buy Candidates"),
        colored_table_html(short_view, OPTION_EMAIL_COLS, "Short Candidates"),
        "</body>",
        "</html>",
    ])

    text = chr(10).join([
        "Intraday Vol Iteration Alert",
        "Scan completed at " + scan_time + ".",
        "",
        "Buy Candidates: " + str(len(buy_view) if buy_view is not None else 0),
        "Short Candidates: " + str(len(short_view) if short_view is not None else 0),
        "",
    ])

    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = f"Intraday Vol Iteration Alert - {subject_time}"
    msg.set_content(text)
    msg.add_alternative(html, subtype="html")

    for path in attachments or []:
        if not path or not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            data = f.read()
        msg.add_attachment(data, maintype="application", subtype="octet-stream", filename=os.path.basename(path))

    try:
        smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
        with smtplib.SMTP_SSL(smtp_host, 465, timeout=40) as s:
            s.login(sender_email, sender_password)
            s.send_message(msg)
        return True
    except Exception as e:
        logger.exception("Email send failed: %s", e)
        return False

def scan_single_option(option_symbol: str, option_type: str, strike: float, underlying: str) -> Optional[Dict]:
    hist_symbol = option_symbol if option_symbol.startswith("NSE:") else f"NSE:{option_symbol}"
    daily_df = get_history(hist_symbol, "D", max(DAILY_LOOKBACK_DAYS, IVP_LOOKBACK_DAYS))
    intra_df = get_history(hist_symbol, "5", INTRADAY_LOOKBACK_DAYS)
    if daily_df.empty or intra_df.empty:
        return None
    today_stats = get_today_stats(intra_df)
    prev_stats = get_prev_day_stats(intra_df)
    summary = summarize_intraday(intra_df, daily_df)
    if not summary:
        return None
    summary.update({"Underlying": underlying, "Option Type": option_type, "Option Symbol": option_symbol, "Strike": strike, "OBV": today_stats["OBV"], "OI": today_stats["OI"], "Volume": today_stats["Volume"], "OI_Delta": today_stats["OI"] - prev_stats["OI"], "Vol_Delta": today_stats["Volume"] - prev_stats["Volume"], "OBV_Delta": today_stats["OBV"] - prev_stats["OBV"]})
    return summary

def scan_symbol(symbol: str) -> Optional[Dict]:
    hist_symbol = symbol if (symbol.startswith("NSE:") and symbol.endswith("-INDEX")) else format_eq_symbol(symbol)
    daily_df = get_history(hist_symbol, "D", max(DAILY_LOOKBACK_DAYS, IVP_LOOKBACK_DAYS))
    intra_df = get_history(hist_symbol, "5", INTRADAY_LOOKBACK_DAYS)
    if daily_df.empty or intra_df.empty:
        return None
    summary = summarize_intraday(intra_df, daily_df)
    if not summary:
        return None
    summary["Symbol"] = symbol
    return summary

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    init_fyers()
    symbols = load_scan_universe(SECTORS_DIR)
    rows = []
    for i, symbol in enumerate(symbols, start=1):
        logger.info("[%s/%s] Scanning %s", i, len(symbols), symbol)
        row = scan_symbol(symbol)
        if row:
            rows.append(row)
    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        raise RuntimeError("No symbols returned usable market data.")
    summary_df = summary_df.sort_values(["Rank Delta", "% Change"], ascending=[False, False]).reset_index(drop=True)
    long_df, short_df = choose_top_candidates(summary_df, top_n=60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    nifty_option_csv = os.path.join(OUTPUT_DIR, f"nifty_option_chain_{timestamp}.csv")
    banknifty_option_csv = os.path.join(OUTPUT_DIR, f"banknifty_option_chain_{timestamp}.csv")

    raw_nifty = fetch_raw_option_chain("NIFTY50")
    raw_banknifty = fetch_raw_option_chain("BANKNIFTY")
    raw_nifty.to_csv(nifty_option_csv, index=False)
    raw_banknifty.to_csv(banknifty_option_csv, index=False)

    nifty_df = pd.DataFrame({"Symbol": ["NIFTY50-INDEX"]})
    banknifty_df = pd.DataFrame({"Symbol": ["BANKNIFTY-INDEX"]})

    ce_df_a, ce_iter_df_a = build_option_candidates(nifty_df, side="long")
    pe_df_a, pe_iter_df_a = build_option_candidates(nifty_df, side="short")
    ce_df_b, ce_iter_df_b = build_option_candidates(banknifty_df, side="long")
    pe_df_b, pe_iter_df_b = build_option_candidates(banknifty_df, side="short")

    ce_df = pd.concat([ce_df_a, ce_df_b], ignore_index=True) if not ce_df_a.empty or not ce_df_b.empty else pd.DataFrame()
    pe_df = pd.concat([pe_df_a, pe_df_b], ignore_index=True) if not pe_df_a.empty or not pe_df_b.empty else pd.DataFrame()
    ce_iter_df = pd.concat([ce_iter_df_a, ce_iter_df_b], ignore_index=True) if not ce_iter_df_a.empty or not ce_iter_df_b.empty else pd.DataFrame()
    pe_iter_df = pd.concat([pe_iter_df_a, pe_iter_df_b], ignore_index=True) if not pe_iter_df_a.empty or not pe_iter_df_b.empty else pd.DataFrame()

    iteration_df = pd.concat([ce_iter_df, pe_iter_df], ignore_index=True) if not ce_iter_df.empty or not pe_iter_df.empty else pd.DataFrame()
    summary_csv = os.path.join(OUTPUT_DIR, f"fo_summary_{timestamp}.csv")
    ce_csv = os.path.join(OUTPUT_DIR, f"fo_ce_candidates_{timestamp}.csv")
    pe_csv = os.path.join(OUTPUT_DIR, f"fo_pe_candidates_{timestamp}.csv")
    iter_csv = os.path.join(OUTPUT_DIR, f"fo_iteration_history_{timestamp}.csv")
    summary_df.to_csv(summary_csv, index=False)
    ce_df.to_csv(ce_csv, index=False)
    pe_df.to_csv(pe_csv, index=False)
    if iteration_df.empty:
        iteration_df = pd.DataFrame(columns=["iteration", "Underlying", "Strike", "Strike Name", "Option Symbol", "timestamp", "window_minutes", "window_start", "window_end", "current_window_score", "previous_trading_day_same_time_score", "window_delta", "window_signal", "close"])
    iteration_df.to_csv(iter_csv, index=False)
    send_email(long_df, short_df, ce_df, pe_df, [summary_csv, ce_csv, pe_csv, iter_csv, nifty_option_csv, banknifty_option_csv])
    logger.info("Completed.")
    logger.info("Iteration rows: %s", len(iteration_df))
    logger.info("Iteration CSV: %s", iter_csv)

if __name__ == "__main__":
    main()
