"""
System5.py - Institutional Multi-Timeframe Trading Engine
Features: Live Rolling Windows, Fractal 45-Degree Renko, True BB-RSI
"""

import concurrent.futures
import datetime
from datetime import datetime, timedelta
import json
import os
import random
import re
import sys
import time
import urllib.parse
import warnings
import threading
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

try:
    from fyers_apiv3.FyersWebsocket import data_ws
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False

warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

# ==============================================================================
# 0. ENGINE CONSTANTS & TERMINAL COLORS
# ==============================================================================
COLOR_GREEN, COLOR_RED, COLOR_CYAN = "\033[92m", "\033[91m", "\033[96m"
COLOR_YELLOW, COLOR_RESET, COLOR_BOLD = "\033[93m", "\033[0m", "\033[1m"

BACKTRACE_DAYS = 25
LIQUIDITY_CACHE_FILE = "liquidity_cache.json"
LIQUIDITY_CACHE_RETENTION_DAYS = 30

EXCLUDED_INDICES = {
    "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX", "BANKEX", "NIFTY50", "NIFTYBANK",
    "LIQUIDBEES", "NIFTYBEES", "BANKBEES", "GOLDBEES"
}

_FYERS_ERROR_LOG_CAP = 5
_fyers_error_log_count = 0

def _log_fyers_error(context, status_code=None, body=None):
    global _fyers_error_log_count
    if _fyers_error_log_count >= _FYERS_ERROR_LOG_CAP: return
    _fyers_error_log_count += 1
    print(f"{COLOR_YELLOW}[API Error #{_fyers_error_log_count}] {context} | HTTP {status_code}{COLOR_RESET}")

# ==============================================================================
# 🎛️ TIER 0: TRADING MODE & DATA FEED SWITCH
# ==============================================================================
DATA_FEED_MODE = "REST"       
TRADING_MODE = "STOCK_FNO"       
ENABLE_STAGE1_STOCK_FILTER = False  

MIN_STOCK_PRICE = 100.0
MAX_STOCK_PRICE = 400.0
MIN_STOCK_VOLUME = 500000

# ==============================================================================
# 🎛️ GLOBAL CONFIGURATION
# ==============================================================================
# v23 FIX - replaces the raw `.rolling(window=tf_mins)` "Live Rolling Window"
# (previous build) with ACTIVE-MINUTE lookback gates.
#
# Why the raw rolling window had to go: `.rolling(window=tf_mins)` operates on
# ROW COUNT, not real elapsed active trading time, and it does not know where
# one trading day ends and the next begins. With MACRO_TIMEFRAMES = ["420min"]
# (420 > the 375-row session), every single day's first ~45 minutes silently
# pulled rows from YESTERDAY's tail into today's High/Low/Volume/ATR/RSI
# window - so a plain overnight gap-up/gap-down got counted as if it were
# real intraday range every single morning. That's a fresh instance of the
# same "overnight contamination" failure mode from the original clock-bar
# bug, just moved from "bar boundaries" to "every window's cold start".
#
# MICRO_LOOKBACKS / MACRO_LOOKBACKS keep the same "how far back does this gate
# look" values (60, 420) but now mean N ACTIVE (Volume > 0) 1-minute candles,
# built by build_rolling_lookback_gate:
#   - Execution/pricing is always the raw 1-minute tape - no resampling, no
#     forward-shifted Eval_Time. Every genuinely-traded candle updates a
#     gate's Renko/ADX/score the instant it prints.
#   - The window slides over ACTIVE candles only, so an illiquid symbol's
#     window reaches further back in wall-clock time instead of freezing,
#     and a gate is fully mature (N real samples) from 09:15 even mid-session
#     (it reaches into the already-loaded BACKTRACE_DAYS history).
#   - GAP_EXCLUDE_OVERNIGHT blanks out the "previous close"/"previous bar"
#     reference at each day's FIRST active candle before True Range / RSI
#     diff are computed, so a window that reaches back across the day
#     boundary to stay mature never mistakes the overnight gap for intraday
#     movement - this is the actual fix for the bug above.
MICRO_LOOKBACKS = [60]
MACRO_LOOKBACKS = [420]   # active 1-min candles - "whole session so far" macro context
GAP_EXCLUDE_OVERNIGHT = True

ATR_PERIOD = 14
RSI_PERIOD = 14
BB_SMA_PERIOD = 20
BB_STD_DEV = 2.0
ADX_PERIOD = 14
ADX_THRESHOLD = 20
STOCH_PERIOD = 14

MICRO_RENKO_CONFIRM_BRICKS = 0
MACRO_RENKO_CONFIRM_BRICKS = 0
RENKO_MIN_BRICK = 0.05
RENKO_DEFAULT_PCT = 0.005
GLOBAL_MACRO_STRATEGY_2D = "BOTH"

# --- Tier 1: Macro Context ---
MACRO_CONFIRMATION_MODE = "MAJORITY"
MACRO_MANDATORY_LIVE_PERCENTILE = 0.0     
MACRO_MANDATORY_PRICE_RENKO    = True    
MACRO_MANDATORY_VOL_RENKO      = True
MACRO_MANDATORY_RENKO_VELOCITY = False
MACRO_MANDATORY_RSI_BB         = False
MACRO_MANDATORY_ADX_DMI        = True
MACRO_MANDATORY_EMA_SPREAD     = False
MACRO_MANDATORY_STOCHASTIC     = False
MACRO_MANDATORY_ATR_BB         = False   
MACRO_MANDATORY_RENKO_BB       = False   
MACRO_MINIMUM_SCORE            = 3       

# --- Tier 2: Micro Execution ---
MICRO_CONFIRMATION_MODE = "MAJORITY"
SYNC_MICRO_WITH_MACRO          = False
MICRO_MANDATORY_LIVE_PERCENTILE = 25.0   
MICRO_MANDATORY_PRICE_RENKO    = True    
MICRO_MANDATORY_VOL_RENKO      = True    
MICRO_MANDATORY_RENKO_VELOCITY = True    
MICRO_MANDATORY_RSI_BB         = False
MICRO_MANDATORY_ADX_DMI        = False   
MICRO_MANDATORY_EMA_SPREAD     = False
MICRO_MANDATORY_STOCHASTIC     = False
MICRO_MANDATORY_ATR_BB         = False   
MICRO_MANDATORY_RENKO_BB       = False   
MICRO_MINIMUM_SCORE            = 3       

# --- Tier 3 & 4: Exits and Options ---
MICRO_EXIT_PRICE_BRICKS = 2              
MICRO_EXIT_VOL_BRICKS   = 2
MACRO_EXIT_PRICE_BRICKS = 99 
MACRO_EXIT_VOL_BRICKS   = 99 
MACRO_EXIT_CONFIRMATION_MODE = "MAJORITY"
MICRO_EXIT_CONFIRMATION_MODE = "MAJORITY"
RENKO_VELOCITY_MAX_BARS = 3  
ENTRY_CUTOFF_TIME = "15:15"              
MAX_DAILY_TRADES_PER_SYMBOL = 2

OPTIONS_TARGET_EXPIRY = "CURRENT"   
STRIKE_RANGE_OFFSET = 2             
MIN_OPT_PREMIUM = 15.0              
MIN_OPT_VOLUME = 50000             
OPTIONS_STRATEGY_2D = "BULLISH"     

TARGET_INDICES = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX", "BANKEX"]
INDEX_SPOT_KEY_MAP = {
    "NIFTY": "NSE:NIFTY50-INDEX", "BANKNIFTY": "NSE:NIFTYBANK-INDEX",
    "FINNIFTY": "NSE:FINNIFTY-INDEX", "MIDCPNIFTY": "NSE:MIDCPNIFTY-INDEX",
    "SENSEX": "BSE:SENSEX-INDEX", "BANKEX": "BSE:BANKEX-INDEX",
}

_WEEKLY_MONTH_CHAR = {'1':'Jan','2':'Feb','3':'Mar','4':'Apr','5':'May','6':'Jun','7':'Jul','8':'Aug','9':'Sep','O':'Oct','N':'Nov','D':'Dec'}

def lb_tag(lookback_n):
    """Column-naming tag for a rolling active-minute lookback, e.g. 420 -> 'L420'."""
    return f"L{lookback_n}"


# ==============================================================================
# 1. LIVE INGESTION & DATA FETCHING
# ==============================================================================
def get_fyers_auth_headers():
    client_id = os.environ.get('FYERS_CLIENT_ID', '').strip()
    token = os.environ.get('FYERS_ACCESS_TOKEN', '').strip()
    return {"Authorization": f"{client_id}:{token}"}

def validate_fyers_token():
    client_id = os.environ.get("FYERS_CLIENT_ID")
    token = os.environ.get("FYERS_ACCESS_TOKEN")
    
    if not client_id or not token:
        print(f"\n{COLOR_RED}❌ CRITICAL ERROR: FYERS API Credentials Missing!{COLOR_RESET}")
        print(f"{COLOR_YELLOW}Please set 'FYERS_CLIENT_ID' and 'FYERS_ACCESS_TOKEN' in your environment variables, or hardcode them directly into the script.{COLOR_RESET}\n")
        return False
        
    try:
        res = requests.get("https://api-t1.fyers.in/api/v3/profile", headers=get_fyers_auth_headers(), timeout=10)
        body = res.json() if res.status_code == 200 else {}
        if res.status_code == 200 and body.get("s") == "ok":
            print(f"{COLOR_GREEN}✅ Fyers Token Validated (Account: {body.get('data', {}).get('name', 'Unknown')}){COLOR_RESET}")
            return True
        else:
            print(f"\n{COLOR_RED}❌ CRITICAL ERROR: Fyers API rejected your token!{COLOR_RESET}")
            print(f"{COLOR_YELLOW}HTTP {res.status_code} | Response: {body}{COLOR_RESET}\n")
            return False
    except Exception as e:
        print(f"\n{COLOR_RED}❌ CRITICAL ERROR: Could not connect to Fyers servers. (Network issue){COLOR_RESET}\n{str(e)}\n")
        return False

def get_cash_equity_universe():
    print("📡 Fetching Cash Equity Universe via FYERS (NSE_CM.csv)...")
    spot_inst = []
    try:
        res = requests.get("https://public.fyers.in/sym_details/NSE_CM.csv", headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if res.status_code == 200:
            for line in res.text.strip().split("\n"):
                cols = [c.strip() for c in line.split(",")]
                for c in cols:
                    if c.startswith("NSE:") and c.endswith("-EQ"):
                        base = c.replace("NSE:", "").replace("-EQ", "")
                        if base not in EXCLUDED_INDICES and not base.isdigit():
                            spot_inst.append({"symbol": base, "key": c, "underlying": base})
    except Exception: pass
    return spot_inst

def get_fno_universe_and_options():
    print("📡 Fetching Master Instrument Matrix via FYERS...")
    spot_inst, opt_inst = [], []
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res_cm = requests.get("https://public.fyers.in/sym_details/NSE_CM.csv", headers=headers, timeout=15)
        spot_map = {}
        if res_cm.status_code == 200:
            for line in res_cm.text.strip().split("\n"):
                for c in [x.strip() for x in line.split(",")]:
                    if c.startswith("NSE:") and c.endswith("-EQ"):
                        spot_map[c.replace("NSE:", "").replace("-EQ", "")] = c

        res_fo = requests.get("https://public.fyers.in/sym_details/NSE_FO.csv", headers=headers, timeout=15)
        valid_und = set()
        if res_fo.status_code == 200:
            for line in res_fo.text.strip().split("\n"):
                cols = [c.strip() for c in line.split(",")]
                o_type, t_idx = None, -1
                for i in range(len(cols)-1, -1, -1):
                    if cols[i] in ("CE", "PE"):
                        o_type, t_idx = cols[i], i
                        break
                if not o_type or t_idx < 3: continue
                try:
                    strike = float(cols[t_idx-1])
                    base_sym = cols[t_idx-3].strip()
                    if base_sym in EXCLUDED_INDICES or base_sym.isdigit(): continue
                    ticker = next((c for c in cols if c.startswith("NSE:") and o_type in c), None)
                    if not ticker: continue
                    exp_date = next((datetime.fromtimestamp(int(float(c))).strftime("%Y-%m-%d") for c in cols if c.replace('.','',1).isdigit() and 1.5e9 < float(c) < 3e9), None)
                    if not exp_date: continue
                    opt_inst.append({"symbol": ticker, "key": ticker, "underlying": base_sym, "type": o_type, "strike": strike, "expiry": exp_date})
                    if base_sym not in valid_und and base_sym in spot_map:
                        valid_und.add(base_sym)
                        spot_inst.append({"symbol": base_sym, "key": spot_map[base_sym], "underlying": base_sym})
                except: pass
    except Exception as e:
        print(f"⚠️ Failed to map instruments: {e}")
    
    opts_by_und = {}
    for o in opt_inst: opts_by_und.setdefault(o["underlying"], []).append(o)
    return spot_inst, opts_by_und

def get_past_trading_days(target_str, num_days=20):
    try:
        target_dt = datetime.strptime(target_str, "%Y-%m-%d")
        days, curr = [], target_dt
        while len(days) < num_days:
            if curr.weekday() < 5: days.append(curr.strftime("%Y-%m-%d"))
            curr -= timedelta(days=1)
        days.reverse()
        return days
    except Exception: return []

def truncate_to_cutoff(df, target_str, cutoff_dt):
    if df is None or df.empty: return df
    target_date = pd.to_datetime(target_str).date()
    is_target_day = df["Datetime"].dt.date == target_date
    is_after_cutoff = df["Datetime"] > cutoff_dt
    return df[~(is_target_day & is_after_cutoff)].reset_index(drop=True)

def regularize_intraday_tape(df, freq="1min"):
    if df is None or df.empty: return df
    df = df.drop_duplicates(subset=["Datetime"], keep="last").sort_values("Datetime").set_index("Datetime")
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    full_idx = full_idx[(full_idx.time >= pd.Timestamp("09:15").time()) & (full_idx.time <= pd.Timestamp("15:30").time())]
    df = df.reindex(full_idx)
    df["Close"] = df["Close"].ffill()
    df["Open"] = df["Open"].fillna(df["Close"])
    df["High"] = df["High"].fillna(df["Close"])
    df["Low"] = df["Low"].fillna(df["Close"])
    df["Volume"] = df["Volume"].fillna(0)
    df["Symbol"] = df["Symbol"].ffill().bfill()
    return df.dropna(subset=["Close"]).reset_index().rename(columns={"index": "Datetime"})

def fetch_fyers_candles(key, start_dt, end_dt, resolution="1"):
    headers = get_fyers_auth_headers()
    for attempt in range(3):
        try:
            time.sleep(0.15)
            url = f"https://api-t1.fyers.in/data/history?symbol={urllib.parse.quote(key, safe=':')}&resolution={resolution}&date_format=1&range_from={start_dt}&range_to={end_dt}"
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code == 200:
                data = res.json()
                if data.get("s") == "ok" and data.get("candles"):
                    df = pd.DataFrame(data["candles"], columns=["Epoch", "Open", "High", "Low", "Close", "Volume"])
                    df["Datetime"] = pd.to_datetime(df["Epoch"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
                    return df
                break
            elif res.status_code in (429, 500, 502, 503): time.sleep(1.0 * (attempt + 1))
            else: break
        except Exception: time.sleep(1)
    return None

def fetch_stock_bars_worker(task):
    item, start_date, end_date = task
    df = fetch_fyers_candles(item["key"], start_date, end_date, resolution="1")
    if df is None or df.empty: return None
    df = df.drop_duplicates(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)
    df["Symbol"] = item["symbol"]
    df = regularize_intraday_tape(df, freq="1min")
    df['Wick_Spread'] = (df['High'] - df['Low']).replace(0, 1e-9)
    df['Net_Delta_1m'] = df['Volume'] * ((df['Close'] - df['Open']) / df['Wick_Spread'])
    return df

def fetch_all_spot_reference_prices(spot_universe, target_date_str):
    target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
    prev_dt = target_dt - timedelta(days=1)
    while prev_dt.weekday() >= 5: prev_dt -= timedelta(days=1)
    prev_day = prev_dt.strftime("%Y-%m-%d")
    lookback = (prev_dt - timedelta(days=7)).strftime("%Y-%m-%d")
    
    spot_ref = {}
    def worker(item):
        df = fetch_fyers_candles(item["key"], lookback, prev_day, resolution="D")
        return (item["symbol"], float(df.sort_values("Datetime").iloc[-1]["Close"])) if df is not None and not df.empty else (item["symbol"], None)
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as ex:
        for sym, px in ex.map(worker, spot_universe):
            if px is not None: spot_ref[sym] = px
    return spot_ref

def filter_liquid_contracts(contracts, target_date_str):
    if not contracts: return []
    target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
    prev_dt = target_dt - timedelta(days=1)
    while prev_dt.weekday() >= 5: prev_dt -= timedelta(days=1)
    lookback = (prev_dt - timedelta(days=7)).strftime("%Y-%m-%d")
    
    valid_contracts = []
    def worker(c):
        df = fetch_fyers_candles(c["key"], lookback, prev_dt.strftime("%Y-%m-%d"), resolution="D")
        if df is None or df.empty: return None
        last = df.sort_values("Datetime").iloc[-1]
        return c if last["Close"] >= MIN_OPT_PREMIUM and last["Volume"] >= MIN_OPT_VOLUME else None

    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as ex:
        for res in ex.map(worker, contracts):
            if res: valid_contracts.append(res)
    return valid_contracts

def build_strike_range(symbol, spot_price, opts_by_und, target_str, offset):
    opts = opts_by_und.get(symbol, [])
    if not opts: return []
    target_dt = pd.to_datetime(target_str)
    valid_exp = sorted(set(pd.to_datetime(o["expiry"]) for o in opts if pd.to_datetime(o["expiry"]) >= target_dt))
    if not valid_exp: return []
    chosen = valid_exp[0] if OPTIONS_TARGET_EXPIRY == "CURRENT" else (valid_exp[1] if len(valid_exp) > 1 else valid_exp[0])
    same_exp = [o for o in opts if pd.to_datetime(o["expiry"]) == chosen]
    strikes = sorted(set(o["strike"] for o in same_exp))
    if not strikes: return []
    idx = strikes.index(min(strikes, key=lambda x: abs(x - spot_price)))
    selected = set(strikes[max(0, idx - offset):min(len(strikes), idx + offset + 1)])
    return [o for o in same_exp if o["strike"] in selected]


# ==============================================================================
# 2. CORE MATHEMATICAL & TECHNICAL ENGINES
# ==============================================================================
def calculate_core_technicals(frame_tf, period=None, gap_exclude=None):
    """
    period: overrides ATR_PERIOD/RSI_PERIOD/ADX_PERIOD with one characteristic
    window (used by build_rolling_lookback_gate, where the lookback itself
    IS the window). BB_SMA_PERIOD(20) and EMA 8/21 scale proportionally so a
    lookback=60 gate stays fast and a lookback=420 gate stays slow, instead
    of every gate sharing fixed 8/21/20-sample windows. None keeps the
    original fixed constants (used for anything not lookback-scaled).
    gap_exclude: blanks the "previous close/high/low" reference at each
    Symbol's FIRST row of a new calendar day before TR/RSI-diff are computed,
    so True Range and RSI momentum never count an overnight gap as intraday
    movement. Defaults to GAP_EXCLUDE_OVERNIGHT. Requires "Date" (added here
    if missing).
    """
    if gap_exclude is None: gap_exclude = GAP_EXCLUDE_OVERNIGHT
    atr_p = period if period else ATR_PERIOD
    rsi_p = period if period else RSI_PERIOD
    adx_p = period if period else ADX_PERIOD
    bb_p = max(2, round(period * BB_SMA_PERIOD / ATR_PERIOD)) if period else BB_SMA_PERIOD
    ema_fast = max(2, round(period * 8 / ATR_PERIOD)) if period else 8
    ema_slow = max(ema_fast + 1, round(period * 21 / ATR_PERIOD)) if period else 21

    if "Date" not in frame_tf.columns:
        frame_tf["Date"] = frame_tf["Datetime"].dt.date
    # True only at each Symbol's first row of a new calendar day - where a
    # naive shift(1)/diff() would otherwise reach back across the overnight
    # gap into the previous session's last print.
    is_new_day = gap_exclude & (frame_tf["Date"] != frame_tf.groupby("Symbol")["Date"].shift(1))

    prev_close = frame_tf.groupby("Symbol")["Close"].shift(1)
    prev_high = frame_tf.groupby("Symbol")["High"].shift(1)
    prev_low = frame_tf.groupby("Symbol")["Low"].shift(1)
    if gap_exclude:
        prev_close = prev_close.where(~is_new_day, np.nan)
        prev_high = prev_high.where(~is_new_day, np.nan)
        prev_low = prev_low.where(~is_new_day, np.nan)

    frame_tf["H-L"] = frame_tf["High"] - frame_tf["Low"]
    frame_tf["H-PC"] = (frame_tf["High"] - prev_close).abs()
    frame_tf["L-PC"] = (frame_tf["Low"] - prev_close).abs()
    # NaN prev_close (gap-excluded first-of-day row) collapses TR to plain
    # H-L for that one row instead of pulling in the overnight jump.
    frame_tf["TR"] = frame_tf[["H-L", "H-PC", "L-PC"]].max(axis=1, skipna=True).fillna(frame_tf["H-L"])
    frame_tf["ATR"] = frame_tf.groupby("Symbol")["TR"].transform(lambda x: x.rolling(atr_p, min_periods=1).mean()).fillna(frame_tf["Close"] * RENKO_DEFAULT_PCT)

    delta = frame_tf["Close"] - prev_close  # NaN (gap-excluded) instead of overnight jump on day's first row
    gain, loss = delta.where(delta > 0, 0), -delta.where(delta < 0, 0)
    avg_gain = gain.groupby(frame_tf["Symbol"]).transform(lambda x: x.rolling(rsi_p, min_periods=1).mean())
    avg_loss = loss.groupby(frame_tf["Symbol"]).transform(lambda x: x.rolling(rsi_p, min_periods=1).mean())

    frame_tf["RSI"] = 100 - (100 / (1 + (avg_gain / (avg_loss + 1e-8))))

    # 🔴 TRUE BB-RSI Breakout
    frame_tf["RSI_SMA"] = frame_tf.groupby("Symbol")["RSI"].transform(lambda x: x.rolling(bb_p, min_periods=1).mean())
    frame_tf["RSI_STD"] = frame_tf.groupby("Symbol")["RSI"].transform(lambda x: x.rolling(bb_p, min_periods=1).std()).fillna(0)
    frame_tf["RSI_Upper_BB"] = frame_tf["RSI_SMA"] + (BB_STD_DEV * frame_tf["RSI_STD"])
    frame_tf["RSI_Lower_BB"] = frame_tf["RSI_SMA"] - (BB_STD_DEV * frame_tf["RSI_STD"])

    h_diff = frame_tf["High"] - prev_high
    l_diff = prev_low - frame_tf["Low"]
    frame_tf["+DM"] = np.where((h_diff > l_diff) & (h_diff > 0), h_diff, 0)
    frame_tf["-DM"] = np.where((l_diff > h_diff) & (l_diff > 0), l_diff, 0)
    frame_tf["+DI"] = 100 * (frame_tf.groupby("Symbol")["+DM"].transform(lambda x: x.rolling(adx_p, min_periods=1).mean()) / (frame_tf["ATR"] + 1e-8))
    frame_tf["-DI"] = 100 * (frame_tf.groupby("Symbol")["-DM"].transform(lambda x: x.rolling(adx_p, min_periods=1).mean()) / (frame_tf["ATR"] + 1e-8))
    frame_tf["DX"] = 100 * abs(frame_tf["+DI"] - frame_tf["-DI"]) / (frame_tf["+DI"] + frame_tf["-DI"] + 1e-8)
    frame_tf["ADX"] = frame_tf.groupby("Symbol")["DX"].transform(lambda x: x.rolling(adx_p, min_periods=1).mean())

    frame_tf["EMA_8"] = frame_tf.groupby("Symbol")["Close"].transform(lambda x: x.rolling(ema_fast, min_periods=1).mean())
    frame_tf["EMA_21"] = frame_tf.groupby("Symbol")["Close"].transform(lambda x: x.rolling(ema_slow, min_periods=1).mean())
    frame_tf["EMA_Spread"] = abs(frame_tf["EMA_8"] - frame_tf["EMA_21"])
    spread_limit = frame_tf.groupby("Symbol")["EMA_Spread"].transform(lambda x: x.rolling(bb_p, min_periods=1).mean()) * 0.20
    frame_tf["EMA_Bull_Expanded"] = (frame_tf["EMA_8"] > frame_tf["EMA_21"]) & (frame_tf["EMA_Spread"] >= spread_limit)
    frame_tf["EMA_Bear_Expanded"] = (frame_tf["EMA_8"] < frame_tf["EMA_21"]) & (frame_tf["EMA_Spread"] >= spread_limit)
    return frame_tf

def construct_45deg_renko_matrix(df, tf_name, confirm_bricks):
    # 🔴 TRUE FRACTAL BREAKOUT ENGINE (Structural Swing Tracking)
    r_counts, f_bull, f_bear = np.zeros(len(df)), np.zeros(len(df), dtype=bool), np.zeros(len(df), dtype=bool)
    
    for sym, indices in df.groupby("Symbol").indices.items():
        px_sub = df["Close"].values[indices]
        atr_sub = df["ATR"].values[indices]
        if len(px_sub) > 0:
            counts = np.zeros(len(px_sub))
            b_breaks, br_breaks = np.zeros(len(px_sub), dtype=bool), np.zeros(len(px_sub), dtype=bool)
            c_trend, c_count, c_px = 0, 0, px_sub[0]
            last_peak, last_valley, ext_px = np.nan, np.nan, px_sub[0]
            is_bull, is_bear = False, False
            
            for i in range(1, len(px_sub)):
                bs = max(atr_sub[i], RENKO_MIN_BRICK)
                move = px_sub[i] - c_px
                
                if c_trend == 0:
                    if move >= bs: c_trend, c_count, c_px, ext_px = 1, int(move//bs), c_px + int(move//bs)*bs, c_px
                    elif move <= -bs: c_trend, c_count, c_px, ext_px = -1, -int(abs(move)//bs), c_px - int(abs(move)//bs)*bs, c_px
                elif c_trend > 0:
                    if move >= bs:
                        c_count, c_px = c_count + int(move//bs), c_px + int(move//bs)*bs
                        ext_px = max(ext_px, c_px)
                        if not np.isnan(last_peak) and c_px > last_peak: is_bull = True
                    elif move <= -(2*bs):
                        last_peak, is_bull = ext_px, False
                        c_trend, c_count, c_px = -1, -int(abs(move)//bs), c_px - int(abs(move)//bs)*bs
                        ext_px = c_px
                else:
                    if move <= -bs:
                        c_count, c_px = c_count - int(abs(move)//bs), c_px - int(abs(move)//bs)*bs
                        ext_px = min(ext_px, c_px)
                        if not np.isnan(last_valley) and c_px < last_valley: is_bear = True
                    elif move >= (2*bs):
                        last_valley, is_bear = ext_px, False
                        c_trend, c_count, c_px = 1, int(move//bs), c_px + int(move//bs)*bs
                        ext_px = c_px
                        
                counts[i], b_breaks[i], br_breaks[i] = c_count, is_bull, is_bear
            r_counts[indices], f_bull[indices], f_bear[indices] = counts, b_breaks, br_breaks

    df[f"Renko_Count_{tf_name}"] = r_counts
    if confirm_bricks > 0:
        df[f"Renko_Bull_{tf_name}"] = (r_counts >= confirm_bricks) & f_bull
        df[f"Renko_Bear_{tf_name}"] = (r_counts <= -confirm_bricks) & f_bear
    else:
        df[f"Renko_Bull_{tf_name}"] = (r_counts > 0) & f_bull
        df[f"Renko_Bear_{tf_name}"] = (r_counts < 0) & f_bear
    return df

def construct_volume_delta_renko_matrix(df, tf_name, confirm_bricks, vol_sma_period=None):
    vol_p = vol_sma_period if vol_sma_period else 20
    df['Cum_Delta'] = df.groupby('Symbol')['Net_Delta_1m'].cumsum()
    df['Vol_SMA_20'] = df.groupby('Symbol')['Volume'].transform(lambda x: x.rolling(vol_p, min_periods=1).mean()).fillna(1000)
    v_counts = np.zeros(len(df))
    
    for sym, idxs in df.groupby("Symbol").indices.items():
        sub_d, sub_bs = df["Cum_Delta"].values[idxs], df["Vol_SMA_20"].values[idxs]
        if len(sub_d) > 0:
            counts = np.zeros(len(sub_d))
            c_trend, c_count, c_d = 0, 0, sub_d[0]
            for i in range(1, len(sub_d)):
                bs, move = max(sub_bs[i], 1.0), sub_d[i] - c_d
                if c_trend == 0:
                    if move >= bs: b = int(move//bs); c_trend, c_count, c_d = 1, b, c_d + b*bs
                    elif move <= -bs: b = int(abs(move)//bs); c_trend, c_count, c_d = -1, -b, c_d - b*bs
                elif c_trend > 0:
                    if move >= bs: b = int(move//bs); c_count, c_d = c_count + b, c_d + b*bs
                    elif move <= -(2*bs): b = int(abs(move)//bs); c_trend, c_count, c_d = -1, -b, c_d - b*bs
                else:
                    if move <= -bs: b = int(abs(move)//bs); c_count, c_d = c_count - b, c_d - b*bs
                    elif move >= (2*bs): b = int(move//bs); c_trend, c_count, c_d = 1, b, c_d + b*bs
                counts[i] = c_count
            v_counts[idxs] = counts
            
    df[f"Vol_Renko_Count_{tf_name}"] = v_counts
    if confirm_bricks > 0:
        df[f"Vol_Renko_Bull_{tf_name}"] = v_counts >= confirm_bricks
        df[f"Vol_Renko_Bear_{tf_name}"] = v_counts <= -confirm_bricks
    else:
        df[f"Vol_Renko_Bull_{tf_name}"] = v_counts > 0
        df[f"Vol_Renko_Bear_{tf_name}"] = v_counts < 0
    return df

def construct_renko_velocity_engine(df, tf_name, lookback_n):
    """lookback_n: the active-minute lookback this gate represents (stands in
    for what a bar's clock-minute width used to mean), so a macro gate is
    still allowed proportionally longer to go quiet than a fast micro gate."""
    changed = df.groupby("Symbol")[f"Renko_Count_{tf_name}"].diff().fillna(1) != 0
    df["Last_Brick_Time"] = df["Datetime"].where(changed).groupby(df["Symbol"]).ffill()
    df[f"Mins_Since_{tf_name}"] = (df["Datetime"] - df["Last_Brick_Time"]).dt.total_seconds() / 60
    active = df[f"Mins_Since_{tf_name}"] <= (RENKO_VELOCITY_MAX_BARS * lookback_n)
    df[f"Velocity_Bull_{tf_name}"] = (df[f"Renko_Count_{tf_name}"] > 0) & active
    df[f"Velocity_Bear_{tf_name}"] = (df[f"Renko_Count_{tf_name}"] < 0) & active
    return df

def construct_bb_meta_pillars(df, tf_name, bb_period=None):
    bb_p = bb_period if bb_period else BB_SMA_PERIOD
    # 🔴 CORRECTED Bollinger Expansions (> <)
    atr_m = df.groupby("Symbol")["ATR"].transform(lambda x: x.rolling(bb_p, min_periods=1).mean())
    atr_s = df.groupby("Symbol")["ATR"].transform(lambda x: x.rolling(bb_p, min_periods=1).std()).fillna(0)
    df[f"ATR_BB_Bull_{tf_name}"] = df["ATR"] > (atr_m + BB_STD_DEV * atr_s)
    df[f"ATR_BB_Bear_{tf_name}"] = df["ATR"] < (atr_m - BB_STD_DEV * atr_s)
    
    r_col = f"Renko_Count_{tf_name}"
    r_m = df.groupby("Symbol")[r_col].transform(lambda x: x.rolling(bb_p, min_periods=1).mean())
    r_s = df.groupby("Symbol")[r_col].transform(lambda x: x.rolling(bb_p, min_periods=1).std()).fillna(0)
    df[f"Renko_BB_Bull_{tf_name}"] = df[r_col] > (r_m + (BB_STD_DEV * r_s))
    df[f"Renko_BB_Bear_{tf_name}"] = df[r_col] < (r_m - (BB_STD_DEV * r_s))
    return df

def apply_dual_tier_scorecard(df, tf_str, tier_type):
    min_score = globals()[f"{tier_type}_MINIMUM_SCORE"]
    
    c_p_b, c_p_br = df[f"Renko_Bull_{tf_str}"].astype(int), df[f"Renko_Bear_{tf_str}"].astype(int)
    c_v_b, c_v_br = df[f"Vol_Renko_Bull_{tf_str}"].astype(int), df[f"Vol_Renko_Bear_{tf_str}"].astype(int)
    c_vel_b, c_vel_br = df[f"Velocity_Bull_{tf_str}"].astype(int), df[f"Velocity_Bear_{tf_str}"].astype(int)
    
    c_rsi_b, c_rsi_br = (df["RSI"] > df["RSI_Upper_BB"]).astype(int), (df["RSI"] < df["RSI_Lower_BB"]).astype(int)
    c_adx_b, c_adx_br = ((df["ADX"] >= ADX_THRESHOLD) & (df["+DI"] > df["-DI"])).astype(int), ((df["ADX"] >= ADX_THRESHOLD) & (df["-DI"] > df["+DI"])).astype(int)
    c_ema_b, c_ema_br = df["EMA_Bull_Expanded"].astype(int), df["EMA_Bear_Expanded"].astype(int)
    c_atr_b, c_atr_br = df[f"ATR_BB_Bull_{tf_str}"].astype(int), df[f"ATR_BB_Bear_{tf_str}"].astype(int)
    c_rbb_b, c_rbb_br = df[f"Renko_BB_Bull_{tf_str}"].astype(int), df[f"Renko_BB_Bear_{tf_str}"].astype(int)

    df[f"Score_Bull_{tf_str}"] = c_p_b + c_v_b + c_vel_b + c_rsi_b + c_adx_b + c_ema_b + c_atr_b + c_rbb_b
    df[f"Score_Bear_{tf_str}"] = c_p_br + c_v_br + c_vel_br + c_rsi_br + c_adx_br + c_ema_br + c_atr_br + c_rbb_br

    b_veto, br_veto = pd.Series(False, index=df.index), pd.Series(False, index=df.index)
    if globals()[f"{tier_type}_MANDATORY_PRICE_RENKO"]: b_veto |= (c_p_b==0); br_veto |= (c_p_br==0)
    if globals()[f"{tier_type}_MANDATORY_VOL_RENKO"]: b_veto |= (c_v_b==0); br_veto |= (c_v_br==0)
    if globals()[f"{tier_type}_MANDATORY_RENKO_VELOCITY"]: b_veto |= (c_vel_b==0); br_veto |= (c_vel_br==0)
    if globals()[f"{tier_type}_MANDATORY_RSI_BB"]: b_veto |= (c_rsi_b==0); br_veto |= (c_rsi_br==0)
    if globals()[f"{tier_type}_MANDATORY_ADX_DMI"]: b_veto |= (c_adx_b==0); br_veto |= (c_adx_br==0)
    
    pct_req = globals().get(f"{tier_type}_MANDATORY_LIVE_PERCENTILE", 0.0)
    if pct_req > 0 and "Net_Delta_Pct" in df.columns:
        b_veto |= (df["Net_Delta_Pct"] < pct_req); br_veto |= (df["Net_Delta_Pct"] > -pct_req)

    df[f"Armed_Bull_{tf_str}"] = (df[f"Score_Bull_{tf_str}"] >= min_score) & (~b_veto)
    df[f"Armed_Bear_{tf_str}"] = (df[f"Score_Bear_{tf_str}"] >= min_score) & (~br_veto)
    return df


# ==============================================================================
# 3. ROLLING ACTIVE-MINUTE LOOKBACK ENGINE
# ==============================================================================
def _scaled_bb_period(lookback_n):
    """Same scaling ratio calculate_core_technicals uses internally, reused
    here for the volume-brick SMA and the ATR/Renko Bollinger meta-pillars so
    every window on a gate scales together with its lookback."""
    return max(2, round(lookback_n * BB_SMA_PERIOD / ATR_PERIOD))

def build_rolling_lookback_gate(master_df, lookback_n, tier_type="MACRO"):
    """
    Builds Armed_Bull/Armed_Bear/Score/Renko columns from a ROLLING window of
    the trailing `lookback_n` ACTIVE (Volume > 0) 1-minute candles per symbol
    - replaces the raw `.rolling(window=tf_mins)` "Live Rolling Window"
    (previous build), which operated on row count with no day-boundary
    awareness and silently blended each day's opening minutes with the prior
    day's close (see the MICRO_LOOKBACKS/MACRO_LOOKBACKS config comment).

    Every genuinely-traded candle updates this gate's ATR/RSI/ADX/Renko/score
    the instant it prints - Eval_Time is just the candle's own timestamp, no
    forward shift. Because the window slides over ACTIVE candles only, it
    automatically reaches back through the already-loaded BACKTRACE_DAYS
    history to stay mature (full lookback_n samples) from the first minute of
    the day. GAP_EXCLUDE_OVERNIGHT (via calculate_core_technicals) keeps the
    overnight gap out of TR/RSI-diff when the window reaches across a day
    boundary to do that.
    """
    tag = lb_tag(lookback_n)
    df_active = master_df[master_df["Volume"] > 0].sort_values(["Symbol", "Datetime"]).reset_index(drop=True).copy()
    df_active["Net_Delta_Pct"] = (df_active["Net_Delta_1m"] / (df_active["Volume"] + 1e-9)) * 100

    df_active = calculate_core_technicals(df_active, period=lookback_n, gap_exclude=GAP_EXCLUDE_OVERNIGHT)
    c_bricks = MACRO_RENKO_CONFIRM_BRICKS if tier_type == "MACRO" else MICRO_RENKO_CONFIRM_BRICKS
    bb_p = _scaled_bb_period(lookback_n)
    df_active = construct_45deg_renko_matrix(df_active, tag, c_bricks)
    df_active = construct_volume_delta_renko_matrix(df_active, tag, c_bricks, vol_sma_period=bb_p)
    df_active = construct_renko_velocity_engine(df_active, tag, lookback_n)
    df_active = construct_bb_meta_pillars(df_active, tag, bb_period=bb_p)
    df_active = apply_dual_tier_scorecard(df_active, tag, tier_type)

    cols = ["Symbol", "Datetime", f"Armed_Bull_{tag}", f"Armed_Bear_{tag}",
            f"Score_Bull_{tag}", f"Score_Bear_{tag}", f"Renko_Count_{tag}",
            f"Vol_Renko_Count_{tag}", f"Mins_Since_{tag}"]
    return df_active[cols].copy().sort_values("Datetime").reset_index(drop=True)

def prepare_unified_execution_tape(master_df, micro_lookbacks, macro_lookbacks, strat_mode="BOTH"):
    exec_lb = micro_lookbacks[0]
    exec_tag = lb_tag(exec_lb)

    df_micro = master_df.copy().sort_values(["Symbol", "Datetime"]).reset_index(drop=True)
    df_micro["Net_Delta_Pct"] = (df_micro["Net_Delta_1m"] / (df_micro["Volume"] + 1e-9)) * 100

    # FRACTAL-BASED RENKO (45-DEGREE BREAKOUT) - execution tape's own gate,
    # sized to the fastest configured lookback. Runs directly on the raw
    # 1-min tape (no resampling), so a brick can form and arm a trigger on
    # the very tick it completes.
    df_micro = calculate_core_technicals(df_micro, period=exec_lb, gap_exclude=GAP_EXCLUDE_OVERNIGHT)
    exec_bb_p = _scaled_bb_period(exec_lb)
    df_micro = construct_45deg_renko_matrix(df_micro, exec_tag, MICRO_RENKO_CONFIRM_BRICKS)
    df_micro = construct_volume_delta_renko_matrix(df_micro, exec_tag, MICRO_RENKO_CONFIRM_BRICKS, vol_sma_period=exec_bb_p)
    df_micro = construct_renko_velocity_engine(df_micro, exec_tag, exec_lb)
    df_micro = construct_bb_meta_pillars(df_micro, exec_tag, bb_period=exec_bb_p)
    df_micro = apply_dual_tier_scorecard(df_micro, exec_tag, "MICRO").sort_values("Datetime").reset_index(drop=True)

    bull_gates, bear_gates = [], []
    for lb in macro_lookbacks:
        tag = lb_tag(lb)
        env_df = build_rolling_lookback_gate(master_df, lb, "MACRO")
        b_col, br_col = f"Armed_Bull_{tag}", f"Armed_Bear_{tag}"
        bull_gates.append(b_col); bear_gates.append(br_col)
        df_micro = pd.merge_asof(df_micro.sort_values("Datetime"), env_df.sort_values("Datetime"), on="Datetime", by="Symbol", direction="backward")
        df_micro[b_col] = df_micro[b_col].fillna(False)
        df_micro[br_col] = df_micro[br_col].fillna(False)

    n_macro = len(bull_gates)
    b_count = df_micro[bull_gates].sum(axis=1)
    br_count = df_micro[bear_gates].sum(axis=1)
    req = n_macro if MACRO_CONFIRMATION_MODE == "ALL" else ((n_macro // 2) + 1 if MACRO_CONFIRMATION_MODE == "MAJORITY" else 1)
        
    df_micro["Master_Armed_Bull"] = b_count >= req
    df_micro["Master_Armed_Bear"] = br_count >= req
    if strat_mode == "BULLISH": df_micro["Master_Armed_Bear"] = False
    elif strat_mode == "BEARISH": df_micro["Master_Armed_Bull"] = False

    mi_bull_gates, mi_bear_gates = [f"Armed_Bull_{exec_tag}"], [f"Armed_Bear_{exec_tag}"]
    for lb in micro_lookbacks[1:]:
        tag = lb_tag(lb)
        env_df = build_rolling_lookback_gate(master_df, lb, "MICRO")
        b_col, br_col = f"Armed_Bull_{tag}", f"Armed_Bear_{tag}"
        mi_bull_gates.append(b_col); mi_bear_gates.append(br_col)
        df_micro = pd.merge_asof(df_micro.sort_values("Datetime"), env_df.sort_values("Datetime"), on="Datetime", by="Symbol", direction="backward")
        df_micro[b_col] = df_micro[b_col].fillna(False)
        df_micro[br_col] = df_micro[br_col].fillna(False)

    n_micro = len(mi_bull_gates)
    mi_b_count = df_micro[mi_bull_gates].sum(axis=1)
    mi_br_count = df_micro[mi_bear_gates].sum(axis=1)
    mi_req = n_micro if MICRO_CONFIRMATION_MODE == "ALL" else ((n_micro // 2) + 1 if MICRO_CONFIRMATION_MODE == "MAJORITY" else 1)
        
    df_micro["Master_Armed_Micro_Bull"] = mi_b_count >= mi_req
    df_micro["Master_Armed_Micro_Bear"] = mi_br_count >= mi_req

    df_micro["Trigger_Bull"] = df_micro["Master_Armed_Bull"] & df_micro["Master_Armed_Micro_Bull"]
    df_micro["Trigger_Bear"] = df_micro["Master_Armed_Bear"] & df_micro["Master_Armed_Micro_Bear"]
    df_micro["Trigger_Bull_Prev"] = df_micro.groupby("Symbol")["Trigger_Bull"].shift(1).fillna(False)
    df_micro["Trigger_Bear_Prev"] = df_micro.groupby("Symbol")["Trigger_Bear"].shift(1).fillna(False)
    df_micro["New_Bull"] = df_micro["Trigger_Bull"] & ~df_micro["Trigger_Bull_Prev"]
    df_micro["New_Bear"] = df_micro["Trigger_Bear"] & ~df_micro["Trigger_Bear_Prev"]
    df_micro["Direction"] = np.where(df_micro["New_Bull"], 1, np.where(df_micro["New_Bear"], -1, 0))

    return df_micro.sort_values("Datetime").reset_index(drop=True)


# ==============================================================================
# 4. TRADE MANAGEMENT & DISPLAY ENGINE
# ==============================================================================
def _run_dual_layer_trade_management(tape, micro_lookbacks, macro_lookbacks, cutoff_time):
    exec_lb = micro_lookbacks[0]
    exec_tag = lb_tag(exec_lb)
    anomalies = tape[tape["Direction"] != 0].copy()
    anom_by_time = anomalies.groupby("Datetime")
    closes = tape.set_index(["Datetime", "Symbol"])["Close"].to_dict()
    
    mi_p_r = {lb: tape.set_index(["Datetime", "Symbol"])[f"Renko_Count_{lb_tag(lb)}"].to_dict() for lb in micro_lookbacks}
    mi_v_r = {lb: tape.set_index(["Datetime", "Symbol"])[f"Vol_Renko_Count_{lb_tag(lb)}"].to_dict() for lb in micro_lookbacks}
    ma_p_r = {lb: tape.set_index(["Datetime", "Symbol"])[f"Renko_Count_{lb_tag(lb)}"].to_dict() for lb in macro_lookbacks}
    ma_v_r = {lb: tape.set_index(["Datetime", "Symbol"])[f"Vol_Renko_Count_{lb_tag(lb)}"].to_dict() for lb in macro_lookbacks}

    bank, last_px, last_dir, daily_cnt = {}, {}, {}, defaultdict(int)
    max_stall = RENKO_VELOCITY_MAX_BARS * exec_lb

    def _req_votes(mode, n): return n if mode == "ALL" else ((n//2)+1 if mode == "MAJORITY" else 1)

    for t in np.sort(tape["Datetime"].unique()):
        t_dt = pd.to_datetime(t)

        if t_dt.time() == pd.Timestamp("09:15").time():
            for sym, eps in bank.items():
                if eps and eps[-1]["state"] == "ACTIVE":
                    eps[-1].update({"state": "EXITED", "exit_time": t_dt.strftime("%Y-%m-%d %H:%M"), "exit_price": closes.get((t_dt, sym), eps[-1]["origin"]), "exit_reason": "Overnight Flush"})
            last_px.clear(); last_dir.clear(); daily_cnt.clear()

        for sym, eps in bank.items():
            if eps and eps[-1]["state"] == "ACTIVE":
                st = eps[-1]
                if (ltp := closes.get((t_dt, sym))) is not None:
                    reason = None
                    mins_in = (t_dt - pd.to_datetime(f"{st['date']} {st['time']}")).total_seconds() / 60
                    cur_p = mi_p_r[exec_lb].get((t_dt, sym), 0)
                    if cur_p != st["curr_r"]: st["last_brick_dt"], st["curr_r"] = t_dt, cur_p
                    if mins_in >= max_stall and (t_dt - st["last_brick_dt"]).total_seconds()/60 >= max_stall:
                        reason = f"Velocity Stall ({max_stall}m without new brick)"

                    if not reason:
                        rev_mi = []
                        for lb in micro_lookbacks:
                            ent_p, ent_v = st["ent_mi_p"].get(lb), st["ent_mi_v"].get(lb)
                            if ent_p is None: continue
                            cur_mi_p, cur_mi_v = mi_p_r[lb].get((t_dt, sym), ent_p), mi_v_r[lb].get((t_dt, sym), ent_v)
                            if st["dir"] == 1:
                                if cur_mi_p <= (ent_p - MICRO_EXIT_PRICE_BRICKS) or cur_mi_v <= (ent_v - MICRO_EXIT_VOL_BRICKS): rev_mi.append(lb)
                            else:
                                if cur_mi_p >= (ent_p + MICRO_EXIT_PRICE_BRICKS) or cur_mi_v >= (ent_v + MICRO_EXIT_VOL_BRICKS): rev_mi.append(lb)
                        if len(rev_mi) >= _req_votes(MICRO_EXIT_CONFIRMATION_MODE, len(micro_lookbacks)):
                            reason = f"Micro Reversal ({','.join(lb_tag(l) for l in rev_mi)})"

                    if not reason:
                        trig_ma = st.get("trig_ma", [])
                        rev_ma = []
                        for lb in trig_ma:
                            ent_p, ent_v = st["ent_ma_p"].get(lb), st["ent_ma_v"].get(lb)
                            if ent_p is None: continue
                            cur_ma_p, cur_ma_v = ma_p_r[lb].get((t_dt, sym), ent_p), ma_v_r[lb].get((t_dt, sym), ent_v)
                            if st["dir"] == 1:
                                if cur_ma_p <= (ent_p - MACRO_EXIT_PRICE_BRICKS) or cur_ma_v <= (ent_v - MACRO_EXIT_VOL_BRICKS): rev_ma.append(lb)
                            else:
                                if cur_ma_p >= (ent_p + MACRO_EXIT_PRICE_BRICKS) or cur_ma_v >= (ent_v + MACRO_EXIT_VOL_BRICKS): rev_ma.append(lb)
                        if trig_ma and len(rev_ma) >= _req_votes(MACRO_EXIT_CONFIRMATION_MODE, len(trig_ma)):
                            reason = f"Macro Reversal ({','.join(lb_tag(l) for l in rev_ma)})"

                    if reason:
                        st.update({"state": "EXITED", "exit_time": t_dt.strftime("%Y-%m-%d %H:%M"), "exit_price": ltp, "exit_reason": reason})
                        last_px[sym], last_dir[sym] = ltp, st["dir"]

        if t_dt in anom_by_time.groups and t_dt.time() < cutoff_time:
            for _, row in anom_by_time.get_group(t_dt).iterrows():
                sym, d = row["Symbol"], row["Direction"]
                if bank.get(sym) and bank[sym][-1]["state"] == "ACTIVE": continue
                if daily_cnt[sym] >= MAX_DAILY_TRADES_PER_SYMBOL: continue
                if pd.isna(row.get("Last_Brick_Time")) or row["Last_Brick_Time"].date() != t_dt.date(): continue
                if sym in last_px and last_dir.get(sym) == d:
                    if d == 1 and row["Close"] <= last_px[sym]: continue
                    if d == -1 and row["Close"] >= last_px[sym]: continue

                trig_ma = [lb for lb in macro_lookbacks if row.get(f"Armed_Bull_{lb_tag(lb)}" if d==1 else f"Armed_Bear_{lb_tag(lb)}", False)]
                bank.setdefault(sym, []).append({
                    "state": "ACTIVE", "origin": row["Close"], "date": t_dt.strftime("%Y-%m-%d"), "time": t_dt.strftime("%H:%M"), "dir": d,
                    "curr_r": row.get(f"Renko_Count_{exec_tag}", 0), "last_brick_dt": t_dt,
                    "exit_time": None, "exit_price": None, "exit_reason": None,
                    "trig_ma": trig_ma,
                    "ent_ma_p": {lb: ma_p_r[lb].get((t_dt, sym), 0) for lb in trig_ma},
                    "ent_ma_v": {lb: ma_v_r[lb].get((t_dt, sym), 0) for lb in trig_ma},
                    "ent_mi_p": {lb: mi_p_r[lb].get((t_dt, sym), 0) for lb in micro_lookbacks},
                    "ent_mi_v": {lb: mi_v_r[lb].get((t_dt, sym), 0) for lb in micro_lookbacks},
                })
                daily_cnt[sym] += 1

        if t_dt.hour == 15 and t_dt.minute >= 25:
            for sym, eps in bank.items():
                if eps and eps[-1]["state"] == "ACTIVE":
                    eps[-1].update({"state": "EXITED", "exit_time": f"{t_dt.strftime('%Y-%m-%d %H:%M')} (EOD)", "exit_price": closes.get((t_dt, sym), eps[-1]["origin"]), "exit_reason": "Market Close"})

    return bank

def _parse_option_symbol(sym):
    s = sym.replace("NSE:", "").replace("BSE:", "")
    m = re.match(r"^(?P<underlying>.+?)(?P<yy>\d{2})(?:(?P<mon3>[A-Z]{3})|(?P<mchar>[1-9OND])(?P<dd>\d{2}))(?P<strike>\d+)(?P<type>CE|PE)$", s)
    if not m: return s, None, None, None
    underlying, strike, opt_type = m.group("underlying"), m.group("strike"), m.group("type")
    expiry_label = f"{m.group('mon3')}'{m.group('yy')} Monthly" if m.group("mon3") else f"{m.group('dd')}-{_WEEKLY_MONTH_CHAR.get(m.group('mchar'), m.group('mchar'))}-{m.group('yy')} Weekly"
    return underlying, strike, opt_type, expiry_label

def display_final_results(tape, bank, target_dt, target_str):
    today = tape[tape["Datetime"].dt.date == target_dt.date()]
    ltp_dict = today.groupby("Symbol")["Close"].last().to_dict() if not today.empty else {}

    act, clo = [], []
    for sym, eps in bank.items():
        for st in eps:
            if st["state"] == "ACTIVE": act.append({**st, "sym": sym})
            elif st["state"] == "EXITED" and st["exit_time"].startswith(target_str): clo.append({**st, "sym": sym})

    def _pnl(st, exit_px):
        return (((exit_px - st["origin"]) / st["origin"]) * 100) if st["dir"] == 1 else (((st["origin"] - exit_px) / st["origin"]) * 100)

    def _print_table(title, icon, rows, headers, col_fn, sort_key=None):
        print(f"\n{COLOR_BOLD}{icon} {title}{COLOR_RESET}")
        if not rows:
            print("  (empty)")
            return
        if sort_key: rows = sorted(rows, key=sort_key)
        rendered = [col_fn(r) for r in rows]
        widths = [max(len(h), *(len(str(c[i])) for c in rendered)) + 2 for i, h in enumerate(headers)]
        print("  " + "".join(f"{h:<{w}}" for h, w in zip(headers, widths)))
        print("  " + "-" * (sum(widths)))
        for r, cols in zip(rows, rendered):
            pnl_val = r.get("_pnl", 0.0)
            c = COLOR_GREEN if pnl_val >= 0 else COLOR_RED
            print(f"  {c}" + "".join(f"{str(v):<{w}}" for v, w in zip(cols, widths)) + COLOR_RESET)

    print(f"\n{COLOR_CYAN}================================================================================================{COLOR_RESET}")
    print(f"{COLOR_BOLD}ENGINE RESULTS [{TRADING_MODE}]{COLOR_RESET}")
    print(f"{COLOR_CYAN}================================================================================================{COLOR_RESET}")

    # --- BASKET 1: ACTIVE RUNNERS ---
    act_rows = [{**st, "_pnl": _pnl(st, ltp_dict[st["sym"]])} for st in act if st["sym"] in ltp_dict]
    _print_table(
        "BASKET 1: ACTIVE RUNNERS (Riding the Trend)", "🟢", act_rows,
        ["Symbol", "Dir", "Entry", "LTP", "P&L%", "Entry Time"],
        lambda st: (st["sym"], "BUY" if st["dir"] == 1 else "SELL", f"₹{st['origin']:.2f}", f"₹{ltp_dict[st['sym']]:.2f}", f"{st['_pnl']:+.2f}%", st["time"]),
        sort_key=lambda st: -st["_pnl"],
    )

    # --- BASKET 2: CLOSED TRADES ---
    clo_rows = [{**st, "_pnl": _pnl(st, st["exit_price"])} for st in clo]
    _print_table(
        "BASKET 2: CLOSED TRADES (Today)", "🛑", clo_rows,
        ["Symbol", "Dir", "Entry", "Exit", "P&L%", "Entry Time", "Exit Time", "Reason"],
        lambda st: (st["sym"], "BUY" if st["dir"] == 1 else "SELL", f"₹{st['origin']:.2f}", f"₹{st['exit_price']:.2f}", f"{st['_pnl']:+.2f}%", st["time"], st["exit_time"].split(" ")[-1] if " " in st["exit_time"] else st["exit_time"], st["exit_reason"]),
        sort_key=lambda st: -st["_pnl"],
    )

    # =========================================================
    # BASKET 3: CONSOLIDATED SUMMARY (by underlying/strike)
    # =========================================================
    def _row_label(sym):
        und, strike, opt_type, expiry_label = _parse_option_symbol(sym)
        return f"{und} {strike} {opt_type} [{expiry_label}]" if strike else und

    active_counts, closed_counts, closed_pnl = defaultdict(int), defaultdict(int), defaultdict(float)
    for st in act_rows: active_counts[_row_label(st["sym"])] += 1
    for st in clo_rows:
        lbl = _row_label(st["sym"])
        closed_counts[lbl] += 1
        closed_pnl[lbl] += st["_pnl"]

    all_rows = sorted(
        set(active_counts) | set(closed_counts),
        key=lambda k: (-active_counts.get(k, 0), -closed_counts.get(k, 0), k)
    )

    print(f"\n{COLOR_CYAN}------------------------------------------------------------------------------------------------{COLOR_RESET}")
    print(f"{COLOR_BOLD}📊 BASKET 3: CONSOLIDATED SUMMARY — Trade Count & P&L by Strike{COLOR_RESET}")
    if all_rows:
        label_width = max(len("Underlying"), len("TOTAL"), *(len(k) for k in all_rows)) + 2
        print(f"  {'Underlying':<{label_width}}{'Active':<8}{'Closed':<8}{'Closed P&L%':<14}")
        for k in all_rows:
            print(f"  {k:<{label_width}}{active_counts.get(k, 0):<8}{closed_counts.get(k, 0):<8}{closed_pnl.get(k, 0.0):<+14.2f}")
        print(f"  {'-' * (label_width + 30)}")
        print(f"  {'TOTAL':<{label_width}}{sum(active_counts.values()):<8}{sum(closed_counts.values()):<8}{sum(closed_pnl.values()):<+14.2f}")
    else:
        print("  No trades in either basket for this run.")
    print(f"{COLOR_CYAN}================================================================================================{COLOR_RESET}\n")

    
# ==============================================================================
# 5. WEBSOCKET & PIPELINE ROUTER
# ==============================================================================
class LiveWebsocketEngine:
    def __init__(self, hist_df, target_str, cutoff_str):
        self.hist_df = hist_df
        self.target_str = target_str
        self.cutoff_obj = pd.to_datetime(f"{target_str} {cutoff_str}").time()
        self.live_c, self.last_px, self.last_v = {}, {}, {}
        self.lock = threading.Lock()
        self.ws = None
        self.token = f"{os.environ.get('FYERS_CLIENT_ID')}:{os.environ.get('FYERS_ACCESS_TOKEN')}"
        self.syms = list(hist_df["Symbol"].unique())
        self.ws_syms = [f"NSE:{s}-EQ" for s in self.syms] if TRADING_MODE == "CASH_EQUITY" else list(self.syms)
        self.strip = (lambda s: s.replace("NSE:", "").replace("-EQ", "")) if TRADING_MODE == "CASH_EQUITY" else (lambda s: s)

    def onmessage(self, msg):
        with self.lock:
            if 'symbol' not in msg or 'ltp' not in msg: return
            s = self.strip(msg['symbol'])
            if s not in self.syms: return
            
            ltp, v = float(msg['ltp']), float(msg.get('vol_traded_today', 0))
            if ltp <= 0: return

            p_ltp, p_v = self.last_px.get(s, ltp), self.last_v.get(s, v)
            tick_v = v - p_v if v >= p_v else 0
            tick_d = tick_v if ltp > p_ltp else (-tick_v if ltp < p_ltp else 0)
            self.last_px[s], self.last_v[s] = ltp, v

            if s not in self.live_c: self.live_c[s] = {"Open": ltp, "High": ltp, "Low": ltp, "Close": ltp, "Volume": tick_v, "Net_Delta_1m": tick_d}
            else:
                c = self.live_c[s]
                c["High"], c["Low"], c["Close"], c["Volume"], c["Net_Delta_1m"] = max(c["High"], ltp), min(c["Low"], ltp), ltp, c["Volume"] + tick_v, c["Net_Delta_1m"] + tick_d

    def start_socket(self):
        self.ws = data_ws.FyersDataSocket(access_token=self.token, log_path="", litemode=False, write_to_file=False, reconnect=True, 
                                          on_connect=lambda: self.ws.subscribe(data_type="SymbolUpdate", symbols=self.ws_syms),
                                          on_close=lambda m: print(f"{COLOR_YELLOW}[WS] Reconnecting...{COLOR_RESET}"),
                                          on_error=lambda m: print(f"{COLOR_RED}[WS Error] {m}{COLOR_RESET}"), on_message=self.onmessage)
        self.ws.connect()

    def run(self):
        threading.Thread(target=self.start_socket, daemon=True).start()
        curr_min = datetime.utcnow().minute
        while True:
            now = datetime.utcnow() + timedelta(hours=5, minutes=30)
            if now.time() >= datetime.strptime("15:30:00", "%H:%M:%S").time(): break
            time.sleep(1)
            
            if now.minute != curr_min:
                curr_min = now.minute
                rnd_dt = now.replace(second=0, microsecond=0) - timedelta(minutes=1)
                with self.lock:
                    new_rows = [{"Datetime": rnd_dt, "Symbol": s, **c} for s, c in self.live_c.items()]
                    for s in self.syms:
                        if s not in self.live_c and s in self.last_px:
                            new_rows.append({"Datetime": rnd_dt, "Symbol": s, "Open": self.last_px[s], "High": self.last_px[s], "Low": self.last_px[s], "Close": self.last_px[s], "Volume": 0, "Net_Delta_1m": 0})
                    self.live_c.clear()
                
                if new_rows:
                    self.hist_df = pd.concat([self.hist_df, pd.DataFrame(new_rows)], ignore_index=True)
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print(f"📡 Update: {now.strftime('%H:%M:%S')} | Mode: WEBSOCKET LIVE TICK")
                    strat = GLOBAL_MACRO_STRATEGY_2D if TRADING_MODE == "CASH_EQUITY" else OPTIONS_STRATEGY_2D
                    tape = prepare_unified_execution_tape(self.hist_df, MICRO_LOOKBACKS, MACRO_LOOKBACKS, strat)
                    bank = _run_dual_layer_trade_management(tape, MICRO_LOOKBACKS, MACRO_LOOKBACKS, self.cutoff_obj)
                    display_final_results(tape, bank, now, self.target_str)

def run_production_sweep():
    print(f"\n{COLOR_CYAN}🚀 Initializing System Engine...{COLOR_RESET}")
    if not validate_fyers_token(): return
    
    target_dt = datetime.utcnow() + timedelta(hours=5, minutes=30)
    if target_dt.weekday() == 5: target_dt -= timedelta(days=1)
    elif target_dt.weekday() == 6: target_dt -= timedelta(days=2)
    
    date_str = target_dt.strftime("%Y-%m-%d")
    print(f"📡 Initiating Pipeline [{TRADING_MODE}] for {date_str}...")
    
    master_df = pd.DataFrame()
    if TRADING_MODE == "CASH_EQUITY":
        univ = filter_cash_equities_by_price_range(get_cash_equity_universe(), date_str)
        if not univ: 
            print(f"{COLOR_RED}❌ No equities passed the price/volume filter.{COLOR_RESET}")
            return
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as ex:
            dfs = [df for df in ex.map(fetch_stock_bars_worker, [(u, get_past_trading_days(date_str, BACKTRACE_DAYS)[0], date_str) for u in univ]) if df is not None]
        if dfs: master_df = pd.concat(dfs, ignore_index=True)
        
    elif TRADING_MODE == "STOCK_FNO":
        univ, opts = get_fno_universe_and_options()
        if univ:
            spot_ref = fetch_all_spot_reference_prices(univ, date_str)
            candidates = []
            for u in univ:
                if sp := spot_ref.get(u["symbol"]): candidates.extend(build_strike_range(u["symbol"], sp, opts, date_str, STRIKE_RANGE_OFFSET))
            liquid = filter_liquid_contracts(candidates, date_str)
            with concurrent.futures.ThreadPoolExecutor(max_workers=15) as ex:
                dfs = [df for df in ex.map(lambda c: fetch_stock_bars_worker((c, get_past_trading_days(date_str, BACKTRACE_DAYS)[0], date_str)), liquid) if df is not None]
            if dfs: master_df = pd.concat(dfs, ignore_index=True)
        else:
            print(f"{COLOR_RED}❌ Failed to fetch FNO universe.{COLOR_RESET}")
            return

    if master_df.empty: 
        print(f"\n{COLOR_RED}❌ CRITICAL: Tape generation failed. No execution data built.{COLOR_RESET}")
        print(f"{COLOR_YELLOW}Possible reasons: The market is currently closed, Fyers returned empty data, or no contracts passed the liquidity filters.{COLOR_RESET}\n")
        return

    if DATA_FEED_MODE == "WEBSOCKET" and WS_AVAILABLE:
        LiveWebsocketEngine(master_df, date_str, ENTRY_CUTOFF_TIME).run()
    else:
        cutoff = pd.to_datetime(f"{date_str} 15:30:00")
        master_df = truncate_to_cutoff(master_df, date_str, cutoff)
        strat = GLOBAL_MACRO_STRATEGY_2D if TRADING_MODE == "CASH_EQUITY" else OPTIONS_STRATEGY_2D
        tape = prepare_unified_execution_tape(master_df, MICRO_LOOKBACKS, MACRO_LOOKBACKS, strat)
        if not tape.empty:
            bank = _run_dual_layer_trade_management(tape, MICRO_LOOKBACKS, MACRO_LOOKBACKS, pd.to_datetime(f"{date_str} {ENTRY_CUTOFF_TIME}").time())
            display_final_results(tape, bank, pd.to_datetime(date_str), date_str)

if __name__ == "__main__":
    run_production_sweep()
                                
