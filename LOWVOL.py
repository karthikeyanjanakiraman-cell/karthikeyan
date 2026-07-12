#!/usr/bin/env python3
"""
FO_FNO_FYERS_CONFLUENCE_EMAIL_FIXED_OPTIONS_FINAL.py

Strict FYERS confluence screener - Production Version with Exact Interpolated Volume Speed
Modified for Index Options (NIFTY, BANKNIFTY, SENSEX) - Nearest Expiry
Sorted by Avg Volume Speed Ascending (Fastest Tapes First)
** Fully Multithreaded & Weekend/API Failsafe Enabled **

Design rules:
- Anchor_915 is ONLY the 09:15 candle open.
- Hard filter: Automatically removes candidates with < 20 blocks.
- Backward volume momentum: Calculates the EXACT time it took for each 10k block to pass.
- Sorting: Sorted by Average Volume Speed ascending (fastest average block time first).
"""

import os
import sys
import time
import json
import logging
import warnings
import smtplib
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from fyers_apiv3 import fyersModel

IST = "Asia/Kolkata"
FYERS_MASTER_URLS = [
    "https://public.fyers.in/sym_details/NSE_FO.csv",
    "https://public.fyers.in/sym_details/BSE_FO.csv"
]
MARKET_OPEN = "09:15:00"
MARKET_CLOSE = "15:30:00"
VERSION = "2026-07-12-prod-options-v4-final"
STATE_FILE = Path("fyers_state_options.json")

class Config:
    def __init__(self):
        self.client_id = os.environ.get("CLIENT_ID") or os.environ.get("CLIENTID")
        self.access_token = os.environ.get("ACCESS_TOKEN") or os.environ.get("ACCESSTOKEN")

        self.smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        self.sender_email = os.environ.get("SENDER_EMAIL", "you@example.com")
        self.sender_password = os.environ.get("SENDER_PASSWORD", "password")
        self.recipient_email = os.environ.get("RECIPIENT_EMAIL", "you@example.com")

        self.lookback_days = int(os.environ.get("LOOKBACK_DAYS", "365")) 
        self.max_retries = int(os.environ.get("MAX_RETRIES", "3"))
        self.symbol_limit = int(os.environ.get("STOCK_LIMIT", "0"))
        
        # FYERS limits apps to ~10 req/sec. Keep at 8-10 for threaded safety.
        self.max_requests_per_sec = float(os.environ.get("MAX_REQ_PER_SEC", "8"))
        self.max_threads = int(os.environ.get("MAX_THREADS", "10")) 
        
        self.disable_index_scan = os.environ.get("DISABLE_INDEX_SCAN", "0") == "1"
        self.always_send_watchlist = os.environ.get("ALWAYS_SEND_WATCHLIST", "1") == "1"
        self.include_late_anchor_in_ranked = os.environ.get("INCLUDE_LATE_ANCHOR", "0") == "1"

        self.index_symbols = [
            "NSE:NIFTY50-INDEX",
            "NSE:NIFTYBANK-INDEX",
            "BSE:SENSEX-INDEX",
        ]

cfg = Config()

EMAIL_DISPLAY_COLS = [
    "Symbol", "Anchor_Source", "% Change", "Conf_Below-3",
    "Conf_Below-2", "Conf_Below-1", "Anchor_915", "LTP",
    "Conf_Above-1", "Conf_Above-2", "Conf_Above-3", "Breach_Time", "Breach_Type", "Vol_Speed_10k"
]

RESULT_COLS = [
    "Symbol", "Anchor_915", "Anchor_Source", "Calc_Anchor",
    "LTP", "Current_Close", "% Change", "Signal",
    "Conf_Below-3", "Conf_Below-2", "Conf_Below-1",
    "Conf_Above-1", "Conf_Above-2", "Conf_Above-3",
    "Support", "Resistance", "Support_Gap_Pct", "Resistance_Gap_Pct", "Vol_Speed_10k"
]

logger = logging.getLogger("fyers_confluence_options")
logger.setLevel(logging.INFO)
logger.handlers.clear()
stream = logging.StreamHandler(sys.stdout)
stream.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
logger.addHandler(stream)
warnings.filterwarnings("ignore")


# --- THREAD-SAFE RATE LIMITER ---
class RateLimiter:
    def __init__(self, max_per_sec):
        self.min_interval = 1.0 / max_per_sec if max_per_sec > 0 else 0.0
        self._last = 0.0
        self.lock = threading.Lock()

    def wait(self):
        if self.min_interval <= 0:
            return
        with self.lock:
            now = time.monotonic()
            elapsed = now - self._last
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last = time.monotonic()

rate_limiter = RateLimiter(cfg.max_requests_per_sec)


def load_state(session_date):
    date_str = str(session_date)
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
                if data.get("date") == date_str:
                    return set(data.get("seen", []))
        except Exception as e:
            logger.error(f"Error loading state: {e}")
    return set()

def save_state(session_date, seen_set):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump({"date": str(session_date), "seen": list(seen_set)}, f)
    except Exception as e:
        logger.error(f"Error saving state: {e}")

def safe_float(val):
    try:
        if val is None or pd.isna(val):
            return np.nan
        return float(val)
    except Exception:
        return np.nan

def format_value(val):
    if pd.isna(val) or val in [float("inf"), float("-inf")]:
        return "-"
    if isinstance(val, (int, float, np.integer, np.floating)):
        return f"{float(val):.2f}"
    return str(val)

def format_change(val):
    try:
        if pd.isna(val):
            return "-"
        return f"{float(val):.2f}%"
    except Exception:
        return "-"

def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

def now_ist():
    return pd.Timestamp.now(tz=IST)

def current_session_date():
    now = now_ist()
    d = now.date()
    if now.time() < pd.Timestamp(MARKET_OPEN).time():
        d = d - timedelta(days=1)
    while pd.Timestamp(d).weekday() >= 5:
        d = d - timedelta(days=1)
    return d

def init_fyers():
    try:
        if not cfg.client_id or not cfg.access_token:
            logger.error("Missing CLIENT_ID/ACCESS_TOKEN.")
            return None
        return fyersModel.FyersModel(client_id=cfg.client_id, is_async=False, token=cfg.access_token, log_path="")
    except Exception as e:
        logger.error(f"FYERS init failed: {e}")
        return None

def call_with_retries(func, *args, **kwargs):
    last_err = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            rate_limiter.wait()
            out = func(*args, **kwargs)
            if isinstance(out, dict) and out.get("s") == "error":
                raise RuntimeError(out.get("message") or out.get("code") or "FYERS API error")
            return out
        except Exception as e:
            last_err = e
            wait = min(2 ** (attempt - 1), 5)
            if attempt < cfg.max_retries:
                time.sleep(wait)
    raise RuntimeError(last_err)

def normalize_history_df(candles):
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(IST).dt.tz_localize(None)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

def get_history(fyers, symbol, resolution, days=None, range_from=None, range_to=None, date_format="1"):
    try:
        if date_format == "1":
            now = pd.Timestamp.now(tz=IST)
            start = (now - timedelta(days=days or cfg.lookback_days)).strftime("%Y-%m-%d")
            end = now.strftime("%Y-%m-%d")
        else:
            start, end = str(range_from), str(range_to)

        payload = {
            "symbol": symbol, "resolution": str(resolution), "date_format": str(date_format),
            "range_from": start, "range_to": end, "cont_flag": "1"
        }
        res = call_with_retries(fyers.history, data=payload)
        candles = res.get("candles", []) if isinstance(res, dict) else []
        if not candles:
            return None
        return normalize_history_df(candles)
    except Exception as e:
        return None

def get_opening_anchor(fyers, symbol, session_date):
    try:
        session_start = int(pd.Timestamp(f"{session_date} {MARKET_OPEN}", tz=IST).timestamp())
        session_end = int(pd.Timestamp(f"{session_date} {MARKET_CLOSE}", tz=IST).timestamp())
        intraday = get_history(fyers, symbol, resolution="5", range_from=session_start, range_to=session_end, date_format="0")
        if intraday is None or intraday.empty:
            return np.nan, False

        intraday = intraday[intraday["timestamp"].dt.date == session_date].copy()
        if intraday.empty:
            return np.nan, False

        target = pd.Timestamp(f"{session_date} {MARKET_OPEN}")
        row = intraday.loc[intraday["timestamp"] == target]
        if not row.empty:
            return safe_float(row.iloc[0]["open"]), True

        market_rows = intraday.sort_values("timestamp")
        if not market_rows.empty:
            return safe_float(market_rows.iloc[0]["open"]), False

        return np.nan, False
    except Exception:
        return np.nan, False

def extract_quote_data(quote_item):
    if not isinstance(quote_item, dict):
        return {"ltp": np.nan, "open": np.nan}
    nested = quote_item.get("v") if isinstance(quote_item.get("v"), dict) else quote_item
    return {
        "ltp": safe_float(nested.get("lp") or nested.get("last_price")),
        "open": safe_float(nested.get("open_price"))
    }

def get_quotes_map(fyers, symbols):
    out = {}
    for batch in chunked(symbols, 50):
        try:
            res = call_with_retries(fyers.quotes, data={"symbols": ",".join(batch)})
            items = res.get("d", []) if isinstance(res, dict) else []
            for item in items:
                symbol = item.get("n") or item.get("symbol")
                if symbol:
                    out[symbol] = extract_quote_data(item)
        except Exception:
            for sym in batch:
                out.setdefault(sym, {"ltp": np.nan, "open": np.nan})
    return out

def extract_volume_weighted_nodes(df, bins=120, peak_quantile=0.60):
    if df is None or df.empty:
        return []
    work = df[["high", "low", "close", "volume"]].copy()
    work["volume"] = work["volume"].fillna(0)
    work = work.dropna()
    if work.empty:
        return []

    typical_price = (work["high"] + work["low"] + work["close"]) / 3.0
    min_p, max_p = safe_float(typical_price.min()), safe_float(typical_price.max())
    if pd.isna(min_p) or pd.isna(max_p) or min_p >= max_p:
        return []

    edges = np.linspace(min_p, max_p, bins)
    weights = work["volume"] if work["volume"].sum() > 0 else None
    hist, bin_edges = np.histogram(typical_price, bins=edges, weights=weights)

    threshold = np.nanquantile(hist, peak_quantile)
    nodes = []
    for i in range(1, len(hist) - 1):
        if hist[i] >= hist[i - 1] and hist[i] >= hist[i + 1] and hist[i] >= threshold and hist[i] > 0:
            nodes.append(round((bin_edges[i] + bin_edges[i + 1]) / 2.0, 2))

    if not nodes and peak_quantile > 0.30:
        return extract_volume_weighted_nodes(df, bins, 0.30)
    return sorted(set(nodes))

def nearest_levels(levels, ref_price, count=3):
    levels = sorted(set([x for x in levels if not pd.isna(x)]))
    below = [x for x in levels if x <= ref_price]
    above = [x for x in levels if x >= ref_price]
    below_vals = [np.nan] * max(0, count - len(below[-count:])) + below[-count:]
    above_vals = above[:count] + [np.nan] * max(0, count - len(above[:count]))
    return below_vals, above_vals

def nearest_support_resistance(levels, ref_price):
    levels = sorted(set([x for x in levels if not pd.isna(x)]))
    support = max([x for x in levels if x <= ref_price], default=np.nan)
    resistance = min([x for x in levels if x >= ref_price], default=np.nan)
    return support, resistance

def build_row(symbol, anchor_915, anchor_source, calc_anchor, ltp_price, levels):
    below_vals, above_vals = nearest_levels(levels, calc_anchor, count=3)
    support, resistance = nearest_support_resistance(levels, ltp_price)

    pct_change = ((ltp_price - calc_anchor) / calc_anchor * 100.0) if calc_anchor and not pd.isna(ltp_price) else np.nan
    signal = "-" if pd.isna(pct_change) else ("Long" if pct_change >= 0 else "Short")

    support_gap_pct = ((ltp_price - support) / ltp_price * 100.0) if not pd.isna(support) and ltp_price else np.nan
    res_gap_pct = ((resistance - ltp_price) / ltp_price * 100.0) if not pd.isna(resistance) and ltp_price else np.nan

    return {
        "Symbol": symbol,
        "Anchor_915": round(anchor_915, 2) if not pd.isna(anchor_915) else np.nan,
        "Anchor_Source": anchor_source,
        "Calc_Anchor": round(calc_anchor, 2) if not pd.isna(calc_anchor) else np.nan,
        "LTP": round(ltp_price, 2) if not pd.isna(ltp_price) else np.nan,
        "Current_Close": round(ltp_price, 2) if not pd.isna(ltp_price) else np.nan,
        "% Change": round(pct_change, 2) if not pd.isna(pct_change) else np.nan,
        "Signal": signal,
        "Conf_Below-3": below_vals[0], "Conf_Below-2": below_vals[1], "Conf_Below-1": below_vals[2],
        "Conf_Above-1": above_vals[0], "Conf_Above-2": above_vals[1], "Conf_Above-3": above_vals[2],
        "Support": support, "Resistance": resistance,
        "Support_Gap_Pct": round(support_gap_pct, 2) if not pd.isna(support_gap_pct) else np.nan,
        "Resistance_Gap_Pct": round(res_gap_pct, 2) if not pd.isna(res_gap_pct) else np.nan,
        "Vol_Speed_10k": "-"
    }

def get_recent_expiry_options(fyers):
    symbols = []
    pattern = re.compile(r"^(NSE:NIFTY|NSE:BANKNIFTY|BSE:SENSEX)\d+")
    
    for url in FYERS_MASTER_URLS:
        try:
            logger.info(f"Fetching option chain from {url}...")
            text = requests.get(url, timeout=30).text
            raw = pd.read_csv(StringIO(text), header=None, low_memory=False)
            
            ticker_col, expiry_col, strike_col, underlying_col = None, None, None, None
            for col in raw.columns:
                sample = raw[col].dropna().astype(str).head(100)
                if ticker_col is None and (sample.str.startswith("NSE:").any() or sample.str.startswith("BSE:").any()):
                    ticker_col = col
                elif pd.api.types.is_numeric_dtype(raw[col]):
                    num_sample = raw[col].dropna()
                    if not num_sample.empty:
                        if expiry_col is None and (num_sample > 1500000000).any() and (num_sample < 2200000000).any():
                            expiry_col = col
                        elif strike_col is None and (num_sample > 1000).any() and (num_sample < 200000).any():
                            strike_col = col
                elif underlying_col is None and sample.isin(["NIFTY", "BANKNIFTY", "SENSEX"]).any():
                    underlying_col = col

            if ticker_col is None: ticker_col = 9
            if expiry_col is None: expiry_col = 8
            if strike_col is None: strike_col = 16
            if underlying_col is None: underlying_col = 13
                
            tickers = raw[ticker_col].astype(str).str.strip().str.upper()
            expiries = raw[expiry_col]
            strikes = pd.to_numeric(raw[strike_col], errors='coerce')
            underlyings = raw[underlying_col].astype(str).str.strip().str.upper()
            
            mask = tickers.str.match(pattern) & tickers.str.endswith(('CE', 'PE'))
            opt_df = raw[mask].copy()
            if opt_df.empty: continue
                
            opt_df['Ticker'], opt_df['Expiry'] = tickers[mask], expiries[mask]
            opt_df['Strike'], opt_df['Underlying'] = strikes[mask], underlyings[mask]
            
            for prefix, u_key in [("NSE:NIFTY", "NIFTY"), ("NSE:BANKNIFTY", "BANKNIFTY"), ("BSE:SENSEX", "SENSEX")]:
                u_df = opt_df[opt_df['Underlying'] == u_key]
                if u_df.empty: u_df = opt_df[opt_df['Ticker'].str.startswith(prefix)]
                if u_df.empty: continue
                
                min_expiry = u_df['Expiry'].min()
                nearest_df = u_df[u_df['Expiry'] == min_expiry]
                
                # --- Offline ATM Math Fix ---
                # Finding the median strike reliably targets the center ATM price dynamically.
                center_strike = nearest_df['Strike'].median()
                if not pd.isna(center_strike) and center_strike > 0:
                    distance_pct = 0.025 
                    lower_bound, upper_bound = center_strike * (1 - distance_pct), center_strike * (1 + distance_pct)
                    nearest_df = nearest_df[(nearest_df['Strike'] >= lower_bound) & (nearest_df['Strike'] <= upper_bound)]
                
                symbols.extend(nearest_df['Ticker'].tolist())
                
        except Exception as e:
            logger.warning(f"Failed to fetch master data: {e}")
            
    out = sorted(list(set(symbols)))
    logger.info(f"Mathematical Filter: Found {len(out)} near-ATM strikes.")
    return out[:cfg.symbol_limit] if cfg.symbol_limit > 0 else out


# --- SINGLE THREAD PROCESSOR FOR SCAN UNIVERSE ---
def process_single_scan(fyers, sym, q_data, session_date, is_index):
    try:
        live_quote, open_quote = q_data.get("ltp"), q_data.get("open")
        
        history_tf = "D" if is_index else "15"
        history_days = cfg.lookback_days if is_index else 7

        df_hist = get_history(fyers, sym, history_tf, days=history_days)
        if df_hist is None or df_hist.empty: return None

        hist_past = df_hist[df_hist["timestamp"].dt.date < session_date].copy()
        if hist_past.empty:
            hist_past = df_hist.copy()

        levels = extract_volume_weighted_nodes(hist_past)
        if not levels: return None

        prev_close = safe_float(hist_past["close"].iloc[-1]) if not hist_past.empty else np.nan

        if is_index:
            anchor_915, anchor_source = np.nan, "QUOTE_INDEX"
            calc_anchor = open_quote if not pd.isna(open_quote) else prev_close
        else:
            # Preserving EXACT strict design rule requirement: 09:15 open anchor via 5-min candle.
            anchor_915, is_exact = get_opening_anchor(fyers, sym, session_date)
            if pd.isna(anchor_915):
                anchor_source = "QUOTE_FALLBACK"
                calc_anchor = open_quote if not pd.isna(open_quote) else prev_close
            elif is_exact:
                anchor_source, calc_anchor = "OPEN_915", anchor_915
            else:
                anchor_source, calc_anchor = "OPEN_915_LATE", anchor_915

        ltp = live_quote if not pd.isna(live_quote) else prev_close
        if pd.isna(ltp): return None

        return build_row(sym, anchor_915, anchor_source, calc_anchor, ltp, levels)
    except:
        return None


# --- MULTITHREADED SCANNER ---
def scan_universe(fyers, symbol_list, session_date, is_index=False):
    if not symbol_list:
        return pd.DataFrame(columns=RESULT_COLS)
        
    quotes_map = get_quotes_map(fyers, symbol_list)
    
    active_symbols = []
    MINIMUM_PREMIUM = 5.0 

    for sym in symbol_list:
        q_data = quotes_map.get(sym, {})
        ltp = q_data.get("ltp")
        
        # --- Weekend / API Failure Failsafe ---
        if is_index or pd.isna(ltp) or ltp >= MINIMUM_PREMIUM:
            active_symbols.append(sym)
            
    if not is_index:
        logger.info(f"Initiating Multithreaded Scan for {len(active_symbols)} active symbols...")

    rows = []
    with ThreadPoolExecutor(max_workers=cfg.max_threads) as executor:
        futures = {executor.submit(process_single_scan, fyers, sym, quotes_map.get(sym, {}), session_date, is_index): sym for sym in active_symbols}
        for future in as_completed(futures):
            res = future.result()
            if res:
                rows.append(res)

    out = pd.DataFrame(rows) if rows else pd.DataFrame(columns=RESULT_COLS)
    for col in RESULT_COLS:
        if col not in out.columns: out[col] = np.nan
    return out[RESULT_COLS]


# --- SINGLE THREAD PROCESSOR FOR BREACH METRICS ---
def process_single_breach(fyers, row, session_date, direction):
    level = row["Conf_Above-1"] if direction == "above" else row["Conf_Below-1"]
    if pd.isna(level):
        return row["Symbol"], (pd.NaT, "None", "-", 0, float('inf'))

    try:
        session_start = int(pd.Timestamp(f"{session_date} {MARKET_OPEN}", tz=IST).timestamp())
        session_end = int(pd.Timestamp(f"{session_date} {MARKET_CLOSE}", tz=IST).timestamp())
        
        intraday = get_history(fyers, row["Symbol"], resolution="5", range_from=session_start, range_to=session_end, date_format="0")
        if intraday is None or intraday.empty:
            return row["Symbol"], (pd.NaT, "None", "-", 0, float('inf'))
            
        intraday = intraday[intraday["timestamp"].dt.date == session_date].sort_values("timestamp").reset_index(drop=True)
        if intraday.empty:
            return row["Symbol"], (pd.NaT, "None", "-", 0, float('inf'))

        prev_close = intraday["close"].shift(1)
        curr_close = intraday["close"]

        breach_time, breach_type = pd.NaT, "None"

        hits = intraday[(curr_close > level) & (prev_close <= level) & prev_close.notna()] if direction == "above" else intraday[(curr_close < level) & (prev_close >= level) & prev_close.notna()]

        if not hits.empty:
            breach_time, breach_type = hits.iloc[-1]["timestamp"], "Intraday"
        else:
            first_close = safe_float(intraday.iloc[0]["close"])
            if not pd.isna(first_close):
                if (direction == "above" and first_close > level) or (direction == "below" and first_close < level):
                    breach_time, breach_type = intraday.iloc[0]["timestamp"], "GapOpen"

        vol_speed_10k, total_blocks, avg_rate = "-", 0, float('inf')
        
        if not pd.isna(breach_time):
            post_breach = intraday[intraday["timestamp"] >= breach_time].sort_values("timestamp", ascending=False)
            
            speeds, bucket_vol, bucket_time = [], 0, 0
            overall_vol, overall_time = 0, 0
            
            for _, r in post_breach.iterrows():
                vol = safe_float(r["volume"])
                overall_vol += vol
                overall_time += 5.0
                
                if bucket_vol + vol >= 10000:
                    t_vol, t_time = bucket_vol + vol, bucket_time + 5.0
                    rate_mins = t_time / (t_vol / 10000.0)
                    full_blocks = int(t_vol // 10000)
                    speeds.extend([rate_mins] * full_blocks)
                    bucket_vol = t_vol - (full_blocks * 10000)
                    bucket_time = rate_mins * (bucket_vol / 10000.0)
                else:
                    bucket_vol += vol
                    bucket_time += 5.0
                    
            total_blocks = int(overall_vol // 10000)
            
            def fmt_rate(r):
                if r < 1.0: return f"{max(1, int(r * 60))}s"
                elif r >= 10: return f"{int(r)}m"
                else: return f"{r:.1f}m"
            
            if total_blocks == 0:
                vol_speed_10k = "Slow (<10k total)"
            else:
                avg_rate = overall_time / (overall_vol / 10000.0)
                vol_speed_10k = ", ".join([fmt_rate(s) for s in speeds[:4]]) + f" | Avg: {fmt_rate(avg_rate)}"
                
        return row["Symbol"], (breach_time, breach_type, vol_speed_10k, total_blocks, avg_rate)
    except:
        return row["Symbol"], (pd.NaT, "None", "-", 0, float('inf'))

# --- MULTITHREADED BREACH RUNNER ---
def add_breach_metrics(fyers, df, session_date, direction):
    results_map = {}
    with ThreadPoolExecutor(max_workers=cfg.max_threads) as executor:
        futures = {executor.submit(process_single_breach, fyers, row, session_date, direction): row["Symbol"] for _, row in df.iterrows()}
        for future in as_completed(futures):
            sym, res = future.result()
            results_map[sym] = res
            
    df["Breach_Time"] = df["Symbol"].map(lambda x: results_map.get(x, (pd.NaT, "None", "-", 0, float('inf')))[0])
    df["Breach_Type"] = df["Symbol"].map(lambda x: results_map.get(x, (pd.NaT, "None", "-", 0, float('inf')))[1])
    df["Vol_Speed_10k"] = df["Symbol"].map(lambda x: results_map.get(x, (pd.NaT, "None", "-", 0, float('inf')))[2])
    df["Total_Blocks"] = df["Symbol"].map(lambda x: results_map.get(x, (pd.NaT, "None", "-", 0, float('inf')))[3])
    df["Avg_Rate"] = df["Symbol"].map(lambda x: results_map.get(x, (pd.NaT, "None", "-", 0, float('inf')))[4])
    return df


def build_html_table(df, title, cols):
    if df is None or df.empty:
        return f"<h3 style='color:#fbbf24; font-family:sans-serif;'>{title}</h3><p style='color:#94a3b8;'>No candidates found.</p>"

    html = f"<h3 style='color:#fbbf24; font-family:sans-serif; margin-top:25px;'>{title}</h3><table style='border-collapse:collapse; width:100%; font-family:sans-serif; font-size:13px; text-align:left; background-color:#0f172a;'>"
    html += "<tr style='background-color:#1e293b; color:#f1f5f9;'>" + "".join([f"<th style='padding:10px; border:1px solid #334155;'>{c}</th>" for c in cols]) + "</tr>"

    for i, (_, row) in enumerate(df.iterrows()):
        bg = "#0f172a" if i % 2 == 0 else "#1e293b"
        html += f"<tr style='background-color:{bg}; color:#e2e8f0;'>"
        for c in cols:
            val = row.get(c, "-")
            style = "padding:8px; border:1px solid #334155;"
            if c == "% Change":
                val_num, val_str = safe_float(val), format_change(val)
                if not pd.isna(val_num):
                    style += " color:#4ade80; font-weight:bold;" if val_num > 0 else " color:#f87171; font-weight:bold;"
                html += f"<td style='{style}'>{val_str}</td>"
            elif c == "Breach_Time" and (pd.isna(val) or val == "-"):
                html += f"<td style='{style}'>-</td>"
            elif c == "Breach_Time":
                html += f"<td style='{style}'>{val.strftime('%H:%M')}</td>"
            elif c == "Breach_Type" and (pd.isna(val) or val == "-" or val == "None"):
                html += f"<td style='{style}'>-</td>"
            elif c == "Breach_Type":
                tag_color = "#facc15" if val == "GapOpen" else "#38bdf8"
                html += f"<td style='{style} color:{tag_color}; font-weight:600;'>{val}</td>"
            elif c == "Vol_Speed_10k":
                val_str = str(val)
                if "Slow" in val_str or val_str == "-":
                    speed_color = "#94a3b8"
                elif "s" in val_str.split("|")[0]:
                    speed_color = "#4ade80"  
                else:
                    speed_color = "#38bdf8"  
                html += f"<td style='{style} color:{speed_color}; font-weight:600;'>{val_str}</td>"
            else:
                html += f"<td style='{style}'>{format_value(val)}</td>"
        html += "</tr>"
    return html + "</table>"


def send_email(index_df, long_df, short_df, fallback_df, session_date):
    have_ranked = not (long_df.empty and short_df.empty)
    have_watchlist = not fallback_df.empty

    if not have_ranked and not (cfg.always_send_watchlist and have_watchlist):
        logger.info("No new actionable candidates and no watchlist to report. Skipping email.")
        return

    recipients = [x.strip() for x in cfg.recipient_email.split(",") if x.strip()]
    if not recipients:
        return

    msg = MIMEMultipart("alternative")
    msg["From"] = cfg.sender_email
    msg["To"] = ", ".join(recipients)
    total_candidates = len(long_df) + len(short_df)
    subject_suffix = f"{total_candidates} New Option Candidates" if total_candidates > 0 else "Options Watchlist Update"
    msg["Subject"] = f"FYERS Alert: {subject_suffix} - {datetime.now().strftime('%H:%M')}"

    html = f"<html><body style='background-color:#030712; padding:20px; font-family:sans-serif;'><h2 style='color:#e2e8f0;'>FYERS Options Confluence</h2>"
    html += f"{build_html_table(index_df, 'Market Index Snapshot', EMAIL_DISPLAY_COLS)}"
    html += f"{build_html_table(long_df, 'New Long Option Candidates', EMAIL_DISPLAY_COLS)}"
    html += f"{build_html_table(short_df, 'New Short Option Candidates', EMAIL_DISPLAY_COLS)}"
    html += f"{build_html_table(fallback_df, 'Watchlist (Missing/Late Open)', EMAIL_DISPLAY_COLS)}</body></html>"

    msg.attach(MIMEText(html, "html"))
    try:
        with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port) as s:
            s.starttls()
            s.login(cfg.sender_email, cfg.sender_password)
            s.sendmail(cfg.sender_email, recipients, msg.as_string())
        logger.info("Email sent.")
    except Exception as e:
        logger.error(f"Email failed: {e}")


def main():
    start_time = time.time()
    fyers = init_fyers()
    if not fyers:
        return

    session_date = current_session_date()
    seen_candidates = load_state(session_date)

    index_df = pd.DataFrame(columns=RESULT_COLS) if cfg.disable_index_scan else scan_universe(fyers, cfg.index_symbols, session_date, is_index=True)
    live_option_symbols = get_recent_expiry_options(fyers)
    option_df = scan_universe(fyers, live_option_symbols, session_date)

    long_options, short_options = pd.DataFrame(columns=RESULT_COLS), pd.DataFrame(columns=RESULT_COLS)
    fallback_df = pd.DataFrame(columns=RESULT_COLS)
    new_seen = set()

    if not option_df.empty:
        eligible_sources = {"OPEN_915"}
        if cfg.include_late_anchor_in_ranked:
            eligible_sources.add("OPEN_915_LATE")

        valid_df = option_df[option_df["Anchor_Source"].isin(eligible_sources)].copy()
        fallback_df = option_df[~option_df["Anchor_Source"].isin(eligible_sources)].copy()

        if not valid_df.empty:
            c_a1, c_a2 = valid_df["Conf_Above-1"], valid_df["Conf_Above-3"]
            c_b1, c_b2 = valid_df["Conf_Below-1"], valid_df["Conf_Below-3"]
            ltp = valid_df["LTP"]

            long_mask = (valid_df["Signal"] == "Long") & c_a1.notna() & c_a2.notna() & (ltp > c_a1) & (ltp < c_a2)
            short_mask = (valid_df["Signal"] == "Short") & c_b1.notna() & c_b2.notna() & (ltp < c_b1) & (ltp > c_b2)

            long_candidates = valid_df[long_mask].copy()
            short_candidates = valid_df[short_mask].copy()

            long_candidates = long_candidates[~long_candidates["Symbol"].isin(seen_candidates)]
            short_candidates = short_candidates[~short_candidates["Symbol"].isin(seen_candidates)]

            if not long_candidates.empty:
                logger.info(f"Calculating breach metrics for {len(long_candidates)} long candidates in parallel...")
                long_candidates = add_breach_metrics(fyers, long_candidates, session_date, "above")
                long_candidates = long_candidates[long_candidates["Total_Blocks"] >= 20]
                
                if not long_candidates.empty:
                    long_options = long_candidates.sort_values(
                        by=["Avg_Rate", "% Change"], ascending=[True, False], na_position="last"
                    )
                    new_seen.update(long_options["Symbol"].tolist())

            if not short_candidates.empty:
                logger.info(f"Calculating breach metrics for {len(short_candidates)} short candidates in parallel...")
                short_candidates = add_breach_metrics(fyers, short_candidates, session_date, "below")
                short_candidates = short_candidates[short_candidates["Total_Blocks"] >= 20]
                
                if not short_candidates.empty:
                    short_options = short_candidates.sort_values(
                        by=["Avg_Rate", "% Change"], ascending=[True, True], na_position="last"
                    )
                    new_seen.update(short_options["Symbol"].tolist())

        if not fallback_df.empty:
            fallback_df = fallback_df.sort_values(by=["% Change"], ascending=[False])

    if not index_df.empty:
        index_df = index_df.sort_values(by=["% Change"], ascending=[False])

    send_email(index_df, long_options, short_options, fallback_df, session_date)

    seen_candidates.update(new_seen)
    save_state(session_date, seen_candidates)
    
    logger.info(f"Execution finished in {round(time.time() - start_time, 2)} seconds.")

if __name__ == "__main__":
    main()
