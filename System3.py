"""system3.py - Asit Baran Pati Multi-Timeframe Trading System Implementation

Production-Grade Universal N-Timeframe & Dual-Tier 45-Degree Renko Engine (v21)
- True Live WebSocket Integration (Tick-by-Tick Institutional Delta)
- Configurable Micro Execution Timeframe (e.g., "5min", "15min")
- Configurable Macro Hierarchy Array (e.g., ["60min", "240min"])
- Dual-Tier Scorecard (9 Pillars) & Global Mandatory Veto Switches
- Cumulative Volume Delta 45-Degree Renko Matrix
- True Post-Entry Renko Velocity Stall Engine (No instant stop-outs)
- Internal Delta Percentage (Absolute Order Flow Conviction)
- STALE MOMENTUM, PRICE PROGRESSION & CHOP GUARDS
- STRICT CUTOFFS (09:15 Overnight Flush & Absolute 14:15 Cutoff)
"""

import argparse
import bisect
import concurrent.futures
import datetime
from datetime import datetime, timedelta
import gzip
import io
import json
import os
import random
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

print("🔖 SYSTEM3 BUILD: v21-RESTORED-LIVE-WS (2026-08-29)")

# ==============================================================================
# 0. ENGINE CONSTANTS & TERMINAL COLORS
# ==============================================================================
COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_CYAN = "\033[96m"
COLOR_YELLOW = "\033[93m"
COLOR_MAGENTA = "\033[95m"
COLOR_DIM = "\033[2m"
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"

BACKTRACE_DAYS = 5
LIQUIDITY_CACHE_FILE = "liquidity_cache.json"

EXCLUDED_INDICES = {
    "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX", "BANKEX", "NIFTY50", "NIFTYBANK",
    "HDFCGOLD", "GOLDBEES", "SILVERBEES", "LIQUIDBEES", "NIFTYBEES", "BANKBEES",
    "LIQUIDCASE", "LIQUIDETF", "SETFGOLD", "GOLDIETF", "MON100", "MAFANG"
}



_FYERS_ERROR_LOG_CAP = 5
_fyers_error_log_count = 0

def _log_fyers_error(context, status_code=None, body=None):
    global _fyers_error_log_count
    if _fyers_error_log_count >= _FYERS_ERROR_LOG_CAP: return
    _fyers_error_log_count += 1
    snippet = str(body)[:300] if body is not None else ""
    print(f"{COLOR_YELLOW}  [Fyers Diagnostic #{_fyers_error_log_count}] {context}"
          f"{' | HTTP ' + str(status_code) if status_code else ''} {snippet}{COLOR_RESET}")

# ==============================================================================
# 🎛️ TIER 0: TRADING MODE, PIPELINE ROUTING & DATA FEED SWITCH
# ==============================================================================
DATA_FEED_MODE = "REST"           
TRADING_MODE = "CASH_EQUITY"      
ENABLE_STAGE1_STOCK_FILTER = False  

MIN_STOCK_PRICE = 100.0
MAX_STOCK_PRICE = 400.0
MIN_STOCK_VOLUME = 500000

# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================
MICRO_TIMEFRAME = "5min"
MACRO_TIMEFRAMES = ["60min"]

ATR_PERIOD = 14
RSI_PERIOD = 14
BB_SMA_PERIOD = 20
BB_STD_DEV = 2.0
ADX_PERIOD = 14
ADX_THRESHOLD = 20
STOCH_PERIOD = 14

MICRO_RENKO_CONFIRM_BRICKS = 1
MACRO_RENKO_CONFIRM_BRICKS = 0
RENKO_MIN_BRICK = 0.05
RENKO_DEFAULT_PCT = 0.005

GLOBAL_MACRO_STRATEGY_2D = "BOTH"

# ==============================================================================
# TIER 1: MACRO CONTEXT SWITCHBOARD (THE GENERAL) - 9 PILLARS
# ==============================================================================
MACRO_MANDATORY_LIVE_PERCENTILE = 0.0     
MACRO_MANDATORY_PRICE_RENKO    = True    
MACRO_MANDATORY_VOL_RENKO      = False
MACRO_MANDATORY_RENKO_VELOCITY = False
MACRO_MANDATORY_RSI_BB         = False
MACRO_MANDATORY_ADX_DMI        = True
MACRO_MANDATORY_EMA_SPREAD     = False
MACRO_MANDATORY_STOCHASTIC     = False
MACRO_MANDATORY_ATR_BB         = False   
MACRO_MANDATORY_RENKO_BB       = False   
MACRO_MINIMUM_SCORE            = 2       

# ==============================================================================
# TIER 2: MICRO EXECUTION SWITCHBOARD (THE SNIPER) - 9 PILLARS
# ==============================================================================
SYNC_MICRO_WITH_MACRO          = False
MICRO_MANDATORY_LIVE_PERCENTILE = 0.0    
MICRO_MANDATORY_PRICE_RENKO    = True    
MICRO_MANDATORY_VOL_RENKO      = True    
MICRO_MANDATORY_RENKO_VELOCITY = False
MICRO_MANDATORY_RSI_BB         = False
MICRO_MANDATORY_ADX_DMI        = False
MICRO_MANDATORY_EMA_SPREAD     = False
MICRO_MANDATORY_STOCHASTIC     = False
MICRO_MANDATORY_ATR_BB         = False   
MICRO_MANDATORY_RENKO_BB       = False   
MICRO_MINIMUM_SCORE            = 2       

# ==============================================================================
# TIER 3: TRADE MANAGEMENT & TEMPORAL GATES (EXIT & TIMING)
# ==============================================================================
MICRO_EXIT_PRICE_BRICKS = 5              
MICRO_EXIT_VOL_BRICKS   = 30
MACRO_EXIT_PRICE_BRICKS = 2              
MACRO_EXIT_VOL_BRICKS   = 20
RENKO_VELOCITY_MAX_BARS = 8              
ENTRY_CUTOFF_TIME = "15:15"              
MAX_DAILY_TRADES_PER_SYMBOL = 2

# ==============================================================================
# TIER 4: OPTIONS STAGE 2 CONFIG
# ==============================================================================
OPTIONS_TARGET_EXPIRY = "CURRENT"   
STRIKE_RANGE_OFFSET = 2             
MIN_OPT_PREMIUM = 15.0              
MIN_OPT_VOLUME = 50000             
OPTIONS_STRATEGY_2D = "BULLISH"     

# ==============================================================================
# HELPER: TIMEFRAME PARSER
# ==============================================================================
def _parse_tf_to_minutes(tf_str):
    if "min" in tf_str: return int(tf_str.replace("min", ""))
    if "D" in tf_str: return int(tf_str.replace("D", "")) * 1440
    return int(tf_str)


# ==============================================================================
# 1. LIVE INGESTION (REST FYERS)
# ==============================================================================
def get_fyers_auth_headers():
    return {"Authorization": f"{os.environ.get('FYERS_CLIENT_ID', '')}:{os.environ.get('FYERS_ACCESS_TOKEN', '')}"}

def validate_fyers_token():
    if not os.environ.get("FYERS_CLIENT_ID") or not os.environ.get("FYERS_ACCESS_TOKEN"):
        print(f"❌ {COLOR_RED}Error: FYERS_CLIENT_ID or FYERS_ACCESS_TOKEN environment variables not found.{COLOR_RESET}")
        return False
    try:
        headers = get_fyers_auth_headers()
        res = requests.get("https://api-t1.fyers.in/api/v3/profile", headers=headers, timeout=10)
        body = res.json() if res.status_code == 200 else {}
        if res.status_code != 200 or body.get("s") != "ok":
            print(f"{COLOR_RED}❌ Fyers token validation FAILED.{COLOR_RESET}")
            return False
        fy_name = body.get("data", {}).get("name", "Unknown")
        print(f"{COLOR_GREEN}✅ Fyers token validated OK (Account: {fy_name}){COLOR_RESET}")
        return True
    except Exception: return False

def get_cash_equity_universe():
    print("📡 Fetching Cash Equity Universe via FYERS (NSE_CM.csv)...")
    spot_inst = []
    try:
        res_cm = requests.get("https://public.fyers.in/sym_details/NSE_CM.csv", headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if res_cm.status_code == 200:
            for line in res_cm.text.strip().split("\n"):
                cols = [c.strip() for c in line.split(",")]
                for c in cols:
                    if c.startswith("NSE:") and c.endswith("-EQ"):
                        base = c.replace("NSE:", "").replace("-EQ", "")
                        if base not in EXCLUDED_INDICES and not base.isdigit() and base:
                            spot_inst.append({"symbol": base, "key": c, "underlying": base})
                        break
    except Exception: pass
    print(f"  ├─ Mapped {len(spot_inst)} total cash equities.")
    return spot_inst

def get_fno_universe_and_options():
    print("📡 Fetching Master Instrument Matrix via FYERS...")
    spot_inst, opt_inst = [], []
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res_cm = requests.get("https://public.fyers.in/sym_details/NSE_CM.csv", headers=headers, timeout=15)
        spot_key_map = {}
        if res_cm.status_code == 200:
            for line in res_cm.text.strip().split("\n"):
                cols = [c.strip() for c in line.split(",")]
                for c in cols:
                    if c.startswith("NSE:") and c.endswith("-EQ"):
                        base = c.replace("NSE:", "").replace("-EQ", "")
                        spot_key_map[base] = c
                        break

        res_fo = requests.get("https://public.fyers.in/sym_details/NSE_FO.csv", headers=headers, timeout=15)
        valid_underlyings = set()
        for line in res_fo.text.strip().split("\n"):
            cols = [c.strip() for c in line.split(",")]
            opt_type, type_idx = None, -1
            for i in range(len(cols) - 1, -1, -1):
                if cols[i] in ("CE", "PE"):
                    opt_type, type_idx = cols[i], i
                    break
            if not opt_type or type_idx < 3: continue
            try:
                strike_val = float(cols[type_idx - 1])
                base_symbol = cols[type_idx - 3].strip()
                if base_symbol in EXCLUDED_INDICES or base_symbol.isdigit() or not base_symbol: continue
                sym_ticker = next((c for c in cols if c.startswith("NSE:") and opt_type in c), None)
                if not sym_ticker: continue
                expiry_date = None
                for c in cols:
                    try:
                        num = int(float(c))
                        if 1.5e9 < num < 3e9:
                            expiry_date = datetime.fromtimestamp(num).strftime("%Y-%m-%d")
                            break
                    except Exception: pass
                if not expiry_date: continue
                opt_inst.append({"symbol": sym_ticker, "key": sym_ticker, "underlying": base_symbol, "type": opt_type, "strike": strike_val, "expiry": expiry_date})
                if base_symbol not in valid_underlyings:
                    valid_underlyings.add(base_symbol)
                    if spot_ticker := spot_key_map.get(base_symbol):
                        spot_inst.append({"symbol": base_symbol, "key": spot_ticker, "underlying": base_symbol})
            except Exception: pass
    except Exception: return [], {}

    options_by_underlying = {}
    for o in opt_inst: options_by_underlying.setdefault(o["underlying"], []).append(o)
    print(f"  ├─ Mapped {len(spot_inst)} Spot Instruments & {len(opt_inst)} Options Contracts.")
    return spot_inst, options_by_underlying

def get_past_trading_days(target_date_str, num_days=20):
    try:
        target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
        days, curr = [], target_dt
        while len(days) < num_days:
            if curr.weekday() < 5: days.append(curr.strftime("%Y-%m-%d"))
            curr -= timedelta(days=1)
        days.reverse()
        return days
    except Exception: return []

def truncate_to_cutoff(df, target_date_str, cutoff_dt):
    if df is None or df.empty: return df
    target_date = pd.to_datetime(target_date_str).date()
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
    df = df.dropna(subset=["Close"]) 
    return df.reset_index().rename(columns={"index": "Datetime"})

def compute_base_net_delta(df):
    if 'Net_Delta_1m' not in df.columns:
        df['Wick_Spread'] = df['High'] - df['Low']
        df['Wick_Spread'] = df['Wick_Spread'].replace(0, 1e-9)
        df['Net_Delta_1m'] = df['Volume'] * ((df['Close'] - df['Open']) / df['Wick_Spread'])
    return df

def fetch_stock_bars_worker(task):
    item, start_date, end_date = task
    df = fetch_fyers_candles(item["key"], start_date, end_date, resolution="1")
    if df is None or df.empty: return None
    df = df.drop_duplicates(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)
    df["Symbol"] = item["symbol"]
    df = regularize_intraday_tape(df, freq="1min")
    if DATA_FEED_MODE == "REST":
        df = compute_base_net_delta(df)
    return df

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
                    df["Datetime"] = pd.to_datetime(df["Epoch"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None).astype("datetime64[ns]")
                    return df
            elif res.status_code in (429, 500, 502, 503):
                time.sleep(random.uniform(1.0, 2.5) * (attempt + 1))
        except Exception: time.sleep(1)
    return None

def filter_cash_equities_by_price_range(universe, target_date_str):
    target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
    prev_dt = target_dt - timedelta(days=1)
    while prev_dt.weekday() >= 5: prev_dt -= timedelta(days=1)
    prev_day = prev_dt.strftime("%Y-%m-%d")
    lookback_start = (prev_dt - timedelta(days=7)).strftime("%Y-%m-%d")

    print(f"🧹 Applying Price Range Filter (₹{MIN_STOCK_PRICE} - ₹{MAX_STOCK_PRICE}) & Volume Filter...")
    def worker(item):
        df = fetch_fyers_candles(item["key"], lookback_start, prev_day, resolution="D")
        if df is not None and not df.empty:
            last = df.sort_values("Datetime").iloc[-1]
            if MIN_STOCK_PRICE <= last["Close"] <= MAX_STOCK_PRICE and last["Volume"] >= MIN_STOCK_VOLUME:
                return item
        return None
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        filtered = [r for r in executor.map(worker, universe) if r is not None]
    print(f"  ├─ ✅ {len(filtered)}/{len(universe)} cash equities passed price & volume filters.")
    return filtered

def filter_liquid_contracts(contracts, target_date_str):
    if not contracts: return []
    target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
    prev_dt = target_dt - timedelta(days=1)
    while prev_dt.weekday() >= 5: prev_dt -= timedelta(days=1)
    prev_day = prev_dt.strftime("%Y-%m-%d")
    lookback_start = (prev_dt - timedelta(days=7)).strftime("%Y-%m-%d")

    cache = {}
    if os.path.exists(LIQUIDITY_CACHE_FILE):
        try:
            with open(LIQUIDITY_CACHE_FILE, "r") as f: cache = json.load(f)
        except Exception: pass

    def worker(c):
        cache_key = f"{c['symbol']}_{prev_day}"
        if cache_key in cache and "volume" in cache[cache_key]:
            return (c, cache_key, cache[cache_key]["close"], cache[cache_key]["volume"], False)
        df = fetch_fyers_candles(c["key"], lookback_start, prev_day, resolution="D")
        if df is None or df.empty: return (c, cache_key, None, None, True)
        last = df.sort_values("Datetime").iloc[-1]
        return (c, cache_key, float(last["Close"]), float(last["Volume"]), True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        results = list(executor.map(worker, contracts))

    valid_contracts, cache_needs_update = [], False
    for c, cache_key, close_price, volume, hit_api in results:
        if hit_api and volume is not None:
            cache[cache_key] = {"close": close_price, "volume": volume}
            cache_needs_update = True
        if close_price is not None and close_price >= MIN_OPT_PREMIUM and volume >= MIN_OPT_VOLUME:
            valid_contracts.append(c)

    if cache_needs_update:
        try:
            with open(LIQUIDITY_CACHE_FILE, "w") as f: json.dump(cache, f)
        except Exception: pass
    return valid_contracts

def build_strike_range(symbol, spot_price, options_by_underlying, target_date_str, offset):
    opts = options_by_underlying.get(symbol, [])
    if not opts: return []
    target_dt = pd.to_datetime(target_date_str)
    valid_expiries = sorted(set(pd.to_datetime(o["expiry"]) for o in opts if pd.to_datetime(o["expiry"]) >= target_dt)) or sorted(set(pd.to_datetime(o["expiry"]) for o in opts))
    if not valid_expiries: return []
    chosen_expiry = valid_expiries[0] if OPTIONS_TARGET_EXPIRY == "CURRENT" else (valid_expiries[1] if len(valid_expiries) > 1 else valid_expiries[0])
    same_expiry = [o for o in opts if pd.to_datetime(o["expiry"]) == chosen_expiry]
    strikes_sorted = sorted(set(o["strike"] for o in same_expiry))
    if not strikes_sorted: return []
    idx = strikes_sorted.index(min(strikes_sorted, key=lambda x: abs(x - spot_price)))
    selected_strikes = set(strikes_sorted[max(0, idx - offset):min(len(strikes_sorted), idx + offset + 1)])
    return [o for o in same_expiry if o["strike"] in selected_strikes]

def fetch_all_spot_reference_prices(spot_universe, target_date_str):
    target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
    prev_dt = target_dt - timedelta(days=1)
    while prev_dt.weekday() >= 5: prev_dt -= timedelta(days=1)
    prev_day = prev_dt.strftime("%Y-%m-%d")
    lookback_start = (prev_dt - timedelta(days=7)).strftime("%Y-%m-%d")
    spot_ref = {}
    def worker(item):
        df = fetch_fyers_candles(item["key"], lookback_start, prev_day, resolution="D")
        return (item["symbol"], float(df.sort_values("Datetime").iloc[-1]["Close"])) if df is not None and not df.empty else (item["symbol"], None)
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        for sym, price in executor.map(worker, spot_universe):
            if price is not None: spot_ref[sym] = price
    return spot_ref


# ==============================================================================
# 2. CORE TECHNICAL & 45-DEGREE RENKO ENGINES (SMA BASED)
# ==============================================================================
def calculate_core_technicals(df_tf):
    df_tf["H-L"] = df_tf["High"] - df_tf["Low"]
    df_tf["H-PC"] = (df_tf["High"] - df_tf.groupby("Symbol")["Close"].shift(1)).abs()
    df_tf["L-PC"] = (df_tf["Low"] - df_tf.groupby("Symbol")["Close"].shift(1)).abs()
    df_tf["TR"] = df_tf[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df_tf["ATR"] = df_tf.groupby("Symbol")["TR"].transform(lambda x: x.rolling(window=ATR_PERIOD, min_periods=1).mean()).fillna(df_tf["Close"] * RENKO_DEFAULT_PCT)

    delta = df_tf.groupby("Symbol")["Close"].diff()
    gain, loss = delta.where(delta > 0, 0), -delta.where(delta < 0, 0)
    avg_gain = gain.groupby(df_tf["Symbol"]).transform(lambda x: x.rolling(window=RSI_PERIOD, min_periods=1).mean())
    avg_loss = loss.groupby(df_tf["Symbol"]).transform(lambda x: x.rolling(window=RSI_PERIOD, min_periods=1).mean())
    df_tf["RSI"] = 100 - (100 / (1 + (avg_gain / (avg_loss + 1e-8))))
    df_tf["RSI_SMA"] = df_tf.groupby("Symbol")["RSI"].transform(lambda x: x.rolling(BB_SMA_PERIOD, min_periods=1).mean())

    high_d = df_tf["High"] - df_tf.groupby("Symbol")["High"].shift(1)
    low_d = df_tf.groupby("Symbol")["Low"].shift(1) - df_tf["Low"]
    df_tf["+DM"] = np.where((high_d > low_d) & (high_d > 0), high_d, 0)
    df_tf["-DM"] = np.where((low_d > high_d) & (low_d > 0), low_d, 0)
    df_tf["+DI"] = (100 * (df_tf.groupby("Symbol")["+DM"].transform(lambda x: x.rolling(ADX_PERIOD, min_periods=1).mean()) / (df_tf["ATR"] + 1e-8)))
    df_tf["-DI"] = (100 * (df_tf.groupby("Symbol")["-DM"].transform(lambda x: x.rolling(ADX_PERIOD, min_periods=1).mean()) / (df_tf["ATR"] + 1e-8)))
    df_tf["DX"] = (100 * abs(df_tf["+DI"] - df_tf["-DI"]) / (df_tf["+DI"] + df_tf["-DI"] + 1e-8))
    df_tf["ADX"] = df_tf.groupby("Symbol")["DX"].transform(lambda x: x.rolling(ADX_PERIOD, min_periods=1).mean())

    df_tf["EMA_8"] = df_tf.groupby("Symbol")["Close"].transform(lambda x: x.rolling(8, min_periods=1).mean())
    df_tf["EMA_21"] = df_tf.groupby("Symbol")["Close"].transform(lambda x: x.rolling(21, min_periods=1).mean())
    df_tf["EMA_Spread"] = abs(df_tf["EMA_8"] - df_tf["EMA_21"])
    spread_thresh = df_tf.groupby("Symbol")["EMA_Spread"].transform(lambda x: x.rolling(20, min_periods=1).mean()) * 0.20
    df_tf["EMA_Bull_Expanded"] = (df_tf["EMA_8"] > df_tf["EMA_21"]) & (df_tf["EMA_Spread"] >= spread_thresh)
    df_tf["EMA_Bear_Expanded"] = (df_tf["EMA_8"] < df_tf["EMA_21"]) & (df_tf["EMA_Spread"] >= spread_thresh)

    lowest_low = df_tf.groupby("Symbol")["Low"].transform(lambda x: x.rolling(STOCH_PERIOD, min_periods=1).min())
    highest_high = df_tf.groupby("Symbol")["High"].transform(lambda x: x.rolling(STOCH_PERIOD, min_periods=1).max())
    df_tf["Stoch_K"] = ((df_tf["Close"] - lowest_low) / (highest_high - lowest_low + 1e-9)) * 100
    df_tf["Vol_Pass"] = df_tf["ATR"] >= (df_tf.groupby("Symbol")["ATR"].transform(lambda x: x.rolling(50, min_periods=1).median()) * 0.75)
    df_tf["Stoch_Bull_Pass"] = (df_tf["Stoch_K"] >= 50) & df_tf["Vol_Pass"]
    df_tf["Stoch_Bear_Pass"] = (df_tf["Stoch_K"] <= 50) & df_tf["Vol_Pass"]
    return df_tf

def construct_45deg_renko_matrix(df, tf_name, confirm_bricks):
    renko_counts = np.zeros(len(df))
    for sym, indices in df.groupby("Symbol").indices.items():
        sub_closes = df["Close"].values[indices]
        sub_atrs = df["ATR"].values[indices]
        if len(sub_closes) > 0:
            counts = np.zeros(len(sub_closes))
            curr_trend, curr_count, curr_price = 0, 0, sub_closes[0]
            for i in range(1, len(sub_closes)):
                bs = max(sub_atrs[i], RENKO_MIN_BRICK)
                move = sub_closes[i] - curr_price
                if curr_trend >= 0:
                    if move >= bs:
                        bricks = int(move // bs); curr_trend = 1; curr_count = curr_count + bricks if curr_count > 0 else bricks; curr_price += bricks * bs
                    elif move <= -(2 * bs):
                        bricks = int(abs(move) // bs); curr_trend = -1; curr_count = -bricks; curr_price -= bricks * bs
                else:
                    if move <= -bs:
                        bricks = int(abs(move) // bs); curr_trend = -1; curr_count = curr_count - bricks if curr_count < 0 else -bricks; curr_price -= bricks * bs
                    elif move >= (2 * bs):
                        bricks = int(move // bs); curr_trend = 1; curr_count = bricks; curr_price += bricks * bs
                counts[i] = curr_count
            renko_counts[indices] = counts
    df[f"Renko_Count_{tf_name}"] = renko_counts
    df[f"Renko_Bull_{tf_name}"] = renko_counts >= confirm_bricks
    df[f"Renko_Bear_{tf_name}"] = renko_counts <= -confirm_bricks
    return df

def construct_volume_delta_renko_matrix(df, tf_name, confirm_bricks):
    df['Wick_Spread'] = df['High'] - df['Low']
    df['Wick_Spread'] = df['Wick_Spread'].replace(0, 1e-9)
    if 'Net_Delta_1m' in df.columns:
        df['Cum_Delta'] = df.groupby('Symbol')['Net_Delta_1m'].cumsum()
    else:
        df['Cum_Delta'] = (df['Volume'] * ((df['Close'] - df['Open']) / df['Wick_Spread'])).groupby(df['Symbol']).cumsum()
    
    df['Vol_SMA_20'] = df.groupby('Symbol')['Volume'].transform(lambda x: x.rolling(20, min_periods=1).mean()).fillna(1000)
    vol_renko_counts = np.zeros(len(df))
    for sym, indices in df.groupby("Symbol").indices.items():
        sub_delta = df["Cum_Delta"].values[indices]
        sub_bs = df["Vol_SMA_20"].values[indices]
        if len(sub_delta) > 0:
            counts = np.zeros(len(sub_delta))
            curr_trend, curr_count, curr_delta = 0, 0, sub_delta[0]
            for i in range(1, len(sub_delta)):
                bs = max(sub_bs[i], 1.0)
                move = sub_delta[i] - curr_delta
                if curr_trend >= 0:
                    if move >= bs:
                        b = int(move // bs); curr_trend = 1; curr_count = curr_count + b if curr_count > 0 else b; curr_delta += b * bs
                    elif move <= -(2 * bs):
                        b = int(abs(move) // bs); curr_trend = -1; curr_count = -b; curr_delta -= b * bs
                else:
                    if move <= -bs:
                        b = int(abs(move) // bs); curr_trend = -1; curr_count = curr_count - b if curr_count < 0 else -b; curr_delta -= b * bs
                    elif move >= (2 * bs):
                        b = int(move // bs); curr_trend = 1; curr_count = b; curr_delta += b * bs
                counts[i] = curr_count
            vol_renko_counts[indices] = counts
    df[f"Vol_Renko_Count_{tf_name}"] = vol_renko_counts
    df[f"Vol_Renko_Bull_{tf_name}"] = vol_renko_counts >= confirm_bricks
    df[f"Vol_Renko_Bear_{tf_name}"] = vol_renko_counts <= -confirm_bricks
    return df

def construct_renko_velocity_engine(df, tf_name):
    brick_diff = df.groupby("Symbol")[f"Renko_Count_{tf_name}"].diff().fillna(1)
    brick_changed = (brick_diff != 0)
    df["Last_Brick_Time"] = df["Datetime"].where(brick_changed).groupby(df["Symbol"]).ffill()
    df[f"Minutes_Since_Brick_{tf_name}"] = (df["Datetime"] - df["Last_Brick_Time"]).dt.total_seconds() / 60
    has_velocity = df[f"Minutes_Since_Brick_{tf_name}"] <= (RENKO_VELOCITY_MAX_BARS * _parse_tf_to_minutes(tf_name))
    df[f"Velocity_Bull_{tf_name}"] = (df[f"Renko_Count_{tf_name}"] > 0) & has_velocity
    df[f"Velocity_Bear_{tf_name}"] = (df[f"Renko_Count_{tf_name}"] < 0) & has_velocity
    return df

def construct_bb_meta_pillars(df, tf_name):
    atr_mean = df.groupby("Symbol")["ATR"].transform(lambda x: x.rolling(BB_SMA_PERIOD, min_periods=1).mean())
    atr_std = df.groupby("Symbol")["ATR"].transform(lambda x: x.rolling(BB_SMA_PERIOD, min_periods=1).std()).fillna(0)
    df[f"ATR_BB_Bull_{tf_name}"] = df["ATR"] > (atr_mean + BB_STD_DEV * atr_std)
    df[f"ATR_BB_Bear_{tf_name}"] = df[f"ATR_BB_Bull_{tf_name}"]
    r_col = f"Renko_Count_{tf_name}"
    r_m = df.groupby("Symbol")[r_col].transform(lambda x: x.rolling(BB_SMA_PERIOD, min_periods=1).mean())
    r_s = df.groupby("Symbol")[r_col].transform(lambda x: x.rolling(BB_SMA_PERIOD, min_periods=1).std()).fillna(0)
    df[f"Renko_BB_Bull_{tf_name}"] = df[r_col] <= (r_m + BB_STD_DEV * r_s)
    df[f"Renko_BB_Bear_{tf_name}"] = df[r_col] >= (r_m - BB_STD_DEV * r_s)
    return df

def apply_dual_tier_scorecard(df, tf_str, tier_type):
    req_price = globals()[f"{tier_type}_MANDATORY_PRICE_RENKO"]
    req_vol = globals()[f"{tier_type}_MANDATORY_VOL_RENKO"]
    req_vel = globals()[f"{tier_type}_MANDATORY_RENKO_VELOCITY"]
    req_rsi = globals()[f"{tier_type}_MANDATORY_RSI_BB"]
    req_adx = globals()[f"{tier_type}_MANDATORY_ADX_DMI"]
    req_ema = globals()[f"{tier_type}_MANDATORY_EMA_SPREAD"]
    req_stoch = globals()[f"{tier_type}_MANDATORY_STOCHASTIC"]
    req_atr_bb = globals()[f"{tier_type}_MANDATORY_ATR_BB"]
    req_renko_bb = globals()[f"{tier_type}_MANDATORY_RENKO_BB"]
    min_score = globals()[f"{tier_type}_MINIMUM_SCORE"]

    c_price_b, c_price_br = df[f"Renko_Bull_{tf_str}"].astype(int), df[f"Renko_Bear_{tf_str}"].astype(int)
    c_vol_b, c_vol_br = df[f"Vol_Renko_Bull_{tf_str}"].astype(int), df[f"Vol_Renko_Bear_{tf_str}"].astype(int)
    c_vel_b, c_vel_br = df[f"Velocity_Bull_{tf_str}"].astype(int), df[f"Velocity_Bear_{tf_str}"].astype(int)
    c_rsi_b, c_rsi_br = (df["RSI"] >= df["RSI_SMA"]).astype(int), (df["RSI"] <= df["RSI_SMA"]).astype(int)
    c_adx_b, c_adx_br = ((df["ADX"] >= ADX_THRESHOLD) & (df["+DI"] > df["-DI"])).astype(int), ((df["ADX"] >= ADX_THRESHOLD) & (df["-DI"] > df["+DI"])).astype(int)
    c_ema_b, c_ema_br = df["EMA_Bull_Expanded"].astype(int), df["EMA_Bear_Expanded"].astype(int)
    c_stoch_b, c_stoch_br = df["Stoch_Bull_Pass"].astype(int), df["Stoch_Bear_Pass"].astype(int)
    c_atr_bb_b, c_atr_bb_br = df[f"ATR_BB_Bull_{tf_str}"].astype(int), df[f"ATR_BB_Bear_{tf_str}"].astype(int)
    c_renko_bb_b, c_renko_bb_br = df[f"Renko_BB_Bull_{tf_str}"].astype(int), df[f"Renko_BB_Bear_{tf_str}"].astype(int)

    df[f"Score_Bull_{tf_str}"] = c_price_b + c_vol_b + c_vel_b + c_rsi_b + c_adx_b + c_ema_b + c_stoch_b + c_atr_bb_b + c_renko_bb_b
    df[f"Score_Bear_{tf_str}"] = c_price_br + c_vol_br + c_vel_br + c_rsi_br + c_adx_br + c_ema_br + c_stoch_br + c_atr_bb_br + c_renko_bb_br

    bull_veto, bear_veto = pd.Series(False, index=df.index), pd.Series(False, index=df.index)
    if req_price: bull_veto |= (c_price_b == 0); bear_veto |= (c_price_br == 0)
    if req_vol: bull_veto |= (c_vol_b == 0); bear_veto |= (c_vol_br == 0)
    if req_vel: bull_veto |= (c_vel_b == 0); bear_veto |= (c_vel_br == 0)
    if req_rsi: bull_veto |= (c_rsi_b == 0); bear_veto |= (c_rsi_br == 0)
    if req_adx: bull_veto |= (c_adx_b == 0); bear_veto |= (c_adx_br == 0)
    if req_ema: bull_veto |= (c_ema_b == 0); bear_veto |= (c_ema_br == 0)
    if req_stoch: bull_veto |= (c_stoch_b == 0); bear_veto |= (c_stoch_br == 0)
    if req_atr_bb: bull_veto |= (c_atr_bb_b == 0); bear_veto |= (c_atr_bb_br == 0)
    if req_renko_bb: bull_veto |= (c_renko_bb_b == 0); bear_veto |= (c_renko_bb_br == 0)

    percentile_req = globals().get(f"{tier_type}_MANDATORY_LIVE_PERCENTILE", 0.0)
    if percentile_req > 0.0 and "Net_Delta_Pct" in df.columns:
        bull_veto |= (df["Net_Delta_Pct"] < percentile_req)
        bear_veto |= (df["Net_Delta_Pct"] > -percentile_req)

    df[f"Armed_Bull_{tf_str}"] = (df[f"Score_Bull_{tf_str}"] >= min_score) & (~bull_veto)
    df[f"Armed_Bear_{tf_str}"] = (df[f"Score_Bear_{tf_str}"] >= min_score) & (~bear_veto)
    return df

def evaluate_single_timeframe_gates(df_base, tf_str):
    df_tf = df_base.groupby(["Symbol", pd.Grouper(key="Datetime", freq=tf_str, closed="left", label="left")]).agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum", "Net_Delta_1m": "sum"}
    ).reset_index().dropna(subset=["Close"]).sort_values(["Symbol", "Datetime"])
    df_tf.rename(columns={"Net_Delta_1m": "Timeframe_Net_Delta"}, inplace=True)
    df_tf["Net_Delta_Pct"] = (df_tf["Timeframe_Net_Delta"] / (df_tf["Volume"] + 1e-9)) * 100
    df_tf = calculate_core_technicals(df_tf)
    df_tf = construct_45deg_renko_matrix(df_tf, tf_str, MACRO_RENKO_CONFIRM_BRICKS)
    df_tf = construct_volume_delta_renko_matrix(df_tf, tf_str, MACRO_RENKO_CONFIRM_BRICKS)
    df_tf = construct_renko_velocity_engine(df_tf, tf_str)
    df_tf = construct_bb_meta_pillars(df_tf, tf_str)
    df_tf = apply_dual_tier_scorecard(df_tf, tf_str, "MACRO")
    df_tf["Eval_Time"] = (df_tf["Datetime"] + pd.to_timedelta(tf_str)).astype("datetime64[ns]")
    cols = ["Symbol", "Eval_Time", f"Armed_Bull_{tf_str}", f"Armed_Bear_{tf_str}", f"Score_Bull_{tf_str}", f"Score_Bear_{tf_str}", f"Renko_Count_{tf_str}", f"Vol_Renko_Count_{tf_str}", f"Minutes_Since_Brick_{tf_str}"]
    return df_tf[cols].copy().rename(columns={"Eval_Time": "Datetime"}).sort_values("Datetime").reset_index(drop=True)

def prepare_unified_execution_tape(rolling_master_df, micro_tf, macro_timeframes, strategy_mode="BOTH"):
    if micro_tf != "1min":
        df_micro = rolling_master_df.groupby(["Symbol", pd.Grouper(key="Datetime", freq=micro_tf, closed="left", label="left")]).agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum", "Net_Delta_1m": "sum"}
        ).reset_index().dropna(subset=["Close"]).sort_values(["Symbol", "Datetime"])
        df_micro.rename(columns={"Net_Delta_1m": "Timeframe_Net_Delta"}, inplace=True)
    else:
        df_micro = rolling_master_df.sort_values(["Symbol", "Datetime"]).copy()
        df_micro["Timeframe_Net_Delta"] = df_micro["Net_Delta_1m"]

    df_micro["Net_Delta_Pct"] = (df_micro["Timeframe_Net_Delta"] / (df_micro["Volume"] + 1e-9)) * 100
    df_micro = calculate_core_technicals(df_micro)
    df_micro = construct_45deg_renko_matrix(df_micro, micro_tf, MICRO_RENKO_CONFIRM_BRICKS)
    df_micro = construct_volume_delta_renko_matrix(df_micro, micro_tf, MICRO_RENKO_CONFIRM_BRICKS)
    df_micro = construct_renko_velocity_engine(df_micro, micro_tf)
    df_micro = construct_bb_meta_pillars(df_micro, micro_tf)
    df_micro = apply_dual_tier_scorecard(df_micro, micro_tf, "MICRO").sort_values("Datetime").reset_index(drop=True)

    bull_gates, bear_gates = [], []
    for tf in macro_timeframes:
        env_df = evaluate_single_timeframe_gates(rolling_master_df, tf)
        b_col, br_col = f"Armed_Bull_{tf}", f"Armed_Bear_{tf}"
        bull_gates.append(b_col); bear_gates.append(br_col)
        df_micro["Datetime"] = df_micro["Datetime"].astype("datetime64[ns]")
        env_df["Datetime"] = env_df["Datetime"].astype("datetime64[ns]")
        df_micro = pd.merge_asof(df_micro.sort_values("Datetime"), env_df.sort_values("Datetime"), on="Datetime", by="Symbol", direction="backward")
        df_micro[b_col] = df_micro[b_col].fillna(False)
        df_micro[br_col] = df_micro[br_col].fillna(False)

    df_micro["Master_Armed_Bull"] = df_micro[bull_gates].any(axis=1)
    df_micro["Master_Armed_Bear"] = df_micro[bear_gates].any(axis=1)
    if strategy_mode == "BULLISH": df_micro["Master_Armed_Bear"] = False
    elif strategy_mode == "BEARISH": df_micro["Master_Armed_Bull"] = False

    df_micro["Trigger_Bull"] = df_micro["Master_Armed_Bull"] & df_micro[f"Armed_Bull_{micro_tf}"]
    df_micro["Trigger_Bear"] = df_micro["Master_Armed_Bear"] & df_micro[f"Armed_Bear_{micro_tf}"]
    df_micro["Trigger_Bull_Prev"] = df_micro.groupby("Symbol")["Trigger_Bull"].shift(1).fillna(False)
    df_micro["Trigger_Bear_Prev"] = df_micro.groupby("Symbol")["Trigger_Bear"].shift(1).fillna(False)
    df_micro["New_Bull"] = df_micro["Trigger_Bull"] & ~df_micro["Trigger_Bull_Prev"]
    df_micro["New_Bear"] = df_micro["Trigger_Bear"] & ~df_micro["Trigger_Bear_Prev"]
    df_micro["Direction"] = np.where(df_micro["New_Bull"], 1, np.where(df_micro["New_Bear"], -1, 0))

    return df_micro.sort_values("Datetime").reset_index(drop=True)

# ==============================================================================
# TRADE MANAGEMENT ENGINE
# ==============================================================================
def _run_dual_layer_trade_management(tape_exec, micro_timeframe, macro_timeframes, cutoff_time_obj):
    all_anomalies = tape_exec[tape_exec["Direction"] != 0].copy()
    anomalies_by_time = all_anomalies.groupby("Datetime")
    closes_dict = tape_exec.set_index(["Datetime", "Symbol"])["Close"].to_dict()
    micro_p_renko = tape_exec.set_index(["Datetime", "Symbol"])[f"Renko_Count_{micro_timeframe}"].to_dict()
    micro_v_renko = tape_exec.set_index(["Datetime", "Symbol"])[f"Vol_Renko_Count_{micro_timeframe}"].to_dict()
    mac_p_renkos = {tf: tape_exec.set_index(["Datetime", "Symbol"])[f"Renko_Count_{tf}"].to_dict() for tf in macro_timeframes}
    mac_v_renkos = {tf: tape_exec.set_index(["Datetime", "Symbol"])[f"Vol_Renko_Count_{tf}"].to_dict() for tf in macro_timeframes}
    
    memory_bank, last_exit_price, last_exit_dir = {}, {}, {}
    micro_tf_mins = _parse_tf_to_minutes(micro_timeframe)
    max_stall_mins = RENKO_VELOCITY_MAX_BARS * micro_tf_mins

    for t in np.sort(tape_exec["Datetime"].unique()):
        t_dt = pd.to_datetime(t)

        if t_dt.time() == pd.Timestamp("09:15").time():
            for sym, episodes in memory_bank.items():
                if episodes and episodes[-1]["state"] == "ACTIVE":
                    episodes[-1]["state"], episodes[-1]["exit_time"], episodes[-1]["exit_price"], episodes[-1]["exit_reason"] = "EXITED", t_dt.strftime("%Y-%m-%d %H:%M"), closes_dict.get((t_dt, sym), episodes[-1]["origin"]), "Overnight Gap Flush"

        for sym, episodes in memory_bank.items():
            if episodes and episodes[-1]["state"] == "ACTIVE":
                st = episodes[-1]
                if (ltp := closes_dict.get((t_dt, sym))) is not None:
                    exit_reason = None
                    mins_in_trade = (t_dt - pd.to_datetime(f"{st['date']} {st['time']}")).total_seconds() / 60
                    mi_p_count = micro_p_renko.get((t_dt, sym), 0)
                    mi_v_count = micro_v_renko.get((t_dt, sym), 0)
                    if mi_p_count != st["current_renko_count"]: st["last_brick_formed_dt"], st["current_renko_count"] = t_dt, mi_p_count
                    if mins_in_trade >= max_stall_mins and (t_dt - st["last_brick_formed_dt"]).total_seconds()/60 >= max_stall_mins:
                        exit_reason = f"Velocity Stall (No new brick in {max_stall_mins}m post-entry)"
                    if not exit_reason:
                        if st["dir"] == 1:
                            if mi_p_count <= (st["entry_renko_count"] - MICRO_EXIT_PRICE_BRICKS): exit_reason = "Micro Price Reversal"
                            elif mi_v_count <= (st["entry_vol_renko_count"] - MICRO_EXIT_VOL_BRICKS): exit_reason = "Micro Volume Reversal"
                        elif st["dir"] == -1:
                            if mi_p_count >= (st["entry_renko_count"] + MICRO_EXIT_PRICE_BRICKS): exit_reason = "Micro Price Reversal"
                            elif mi_v_count >= (st["entry_vol_renko_count"] + MICRO_EXIT_VOL_BRICKS): exit_reason = "Micro Volume Reversal"

                    if exit_reason:
                        st["state"], st["exit_time"], st["exit_price"], st["exit_reason"] = "EXITED", t_dt.strftime("%Y-%m-%d %H:%M"), ltp, exit_reason
                        last_exit_price[sym], last_exit_dir[sym] = ltp, st["dir"]

        if t_dt in anomalies_by_time.groups and t_dt.time() < cutoff_time_obj:
            for _, row in anomalies_by_time.get_group(t_dt).iterrows():
                sym, direction = row["Symbol"], row["Direction"]
                existing = memory_bank.get(sym, [])
                if existing and existing[-1]["state"] == "ACTIVE": continue
                if len(existing) >= MAX_DAILY_TRADES_PER_SYMBOL: continue 

                last_brick = row.get("Last_Brick_Time")
                if pd.isna(last_brick) or last_brick.date() != t_dt.date(): continue

                if sym in last_exit_price and last_exit_dir.get(sym) == direction:
                    if direction == 1 and row["Close"] <= last_exit_price[sym]: continue
                    if direction == -1 and row["Close"] >= last_exit_price[sym]: continue

                memory_bank.setdefault(sym, []).append({
                    "state": "ACTIVE", "origin": row["Close"], "date": t_dt.strftime("%Y-%m-%d"), "time": t_dt.strftime("%H:%M"), "dir": direction,
                    "entry_renko_count": row.get(f"Renko_Count_{micro_timeframe}", 0), "entry_vol_renko_count": row.get(f"Vol_Renko_Count_{micro_timeframe}", 0),
                    "current_renko_count": row.get(f"Renko_Count_{micro_timeframe}", 0), "last_brick_formed_dt": t_dt,
                    "exit_time": None, "exit_price": None, "exit_reason": None,
                    "triggering_macro_tfs": [tf for tf in macro_timeframes if row.get(f"Armed_Bull_{tf}" if direction == 1 else f"Armed_Bear_{tf}", False)],
                    "micro_score": row.get(f"Score_Bull_{micro_timeframe}" if direction == 1 else f"Score_Bear_{micro_timeframe}", 0)
                })

        if t_dt.hour == 15 and t_dt.minute >= 25:
            for sym, episodes in memory_bank.items():
                if episodes and episodes[-1]["state"] == "ACTIVE":
                    episodes[-1]["state"], episodes[-1]["exit_time"], episodes[-1]["exit_price"], episodes[-1]["exit_reason"] = "EXITED", t_dt.strftime("%Y-%m-%d %H:%M") + " (EOD)", closes_dict.get((t_dt, sym), episodes[-1]["origin"]), "End of Day Market Close"

    return memory_bank

def display_final_results(tape_exec, memory_bank, target_dt, target_date_str):
    today_master = tape_exec[tape_exec["Datetime"].dt.date == target_dt.date()]
    final_ltp_dict = today_master.groupby("Symbol")["Close"].last().to_dict() if not today_master.empty else {}
    
    active_runners, closed_trades = [], []
    for sym, episodes in memory_bank.items():
        for st in episodes:
            if st["state"] == "ACTIVE": active_runners.append({**st, "sym": sym})
            elif st["state"] == "EXITED" and st["exit_time"].startswith(target_date_str): closed_trades.append({**st, "sym": sym})
    closed_trades.sort(key=lambda x: (x["sym"], x["time"]))

    tf_display_str = " | ".join(MACRO_TIMEFRAMES)
    print(f"\n{COLOR_CYAN}================================================================================================{COLOR_RESET}")
    print(f"{COLOR_BOLD}9-PILLAR ENGINE [{MICRO_TIMEFRAME} Micro ⚡ Macro: {tf_display_str}] — RESULTS [{TRADING_MODE}]{COLOR_RESET}")
    print(f"{COLOR_CYAN}================================================================================================{COLOR_RESET}\n")

    if active_runners:
        print(f"{COLOR_BOLD}🟢 BASKET 1: ACTIVE RUNNERS (Riding the Trend){COLOR_RESET}")
        for st in active_runners:
            ltp = final_ltp_dict.get(st["sym"], st["origin"])
            pnl_pct = (((ltp - st["origin"]) / st["origin"]) * 100) if st["dir"] == 1 else (((st["origin"] - ltp) / st["origin"]) * 100)
            color = COLOR_GREEN if pnl_pct >= 0 else COLOR_RED
            print(f"  {color}⚡ {st['sym']:<26} Open P&L: {pnl_pct:+.2f}% ({'BULLISH' if st['dir']==1 else 'BEARISH'}){COLOR_RESET}")
            print(f"      └─ 🎯 Anchor: {st['time']} | Price: ₹{st['origin']:.2f} | Latest: ₹{ltp:.2f}\n")

    if closed_trades:
        print(f"{COLOR_BOLD}🛑 BASKET 2: CLOSED TRADES{COLOR_RESET}")
        for st in closed_trades:
            pnl_pct = (((st["exit_price"] - st["origin"]) / st["origin"]) * 100) if st["dir"] == 1 else (((st["origin"] - st["exit_price"]) / st["origin"]) * 100)
            color = COLOR_GREEN if pnl_pct >= 0 else COLOR_RED
            print(f"  {color}🛑 {st['sym']:<26} Final P&L: {pnl_pct:+.2f}% ({'BULLISH' if st['dir']==1 else 'BEARISH'}){COLOR_RESET}")
            print(f"      └─ 🎯 Anchor: {st['time']} | Exit: {st['exit_time']} | Price: ₹{st['exit_price']:.2f} | Reason: {st['exit_reason']}\n")

# ==============================================================================
# LIVE WEBSOCKET STREAMING ENGINE
# ==============================================================================
class LiveWebsocketEngine:
    def __init__(self, historical_df, target_date_str, cutoff_time_str):
        self.historical_df = historical_df
        self.target_date_str = target_date_str
        self.cutoff_time_str = cutoff_time_str
        self.cutoff_obj = pd.to_datetime(f"{target_date_str} {cutoff_time_str}").time()
        self.live_candles, self.last_tick_ltp, self.last_tick_vol = {}, {}, {}
        self.lock = threading.Lock()
        self.fyers_ws = None
        self.access_token = f"{os.environ.get('FYERS_CLIENT_ID')}:{os.environ.get('FYERS_ACCESS_TOKEN')}"
        self.symbols = list(historical_df["Symbol"].unique())
        self.ws_symbols = [f"NSE:{s}-EQ" for s in self.symbols]

    def onmessage(self, messages):
        with self.lock:
            for msg in messages:
                if 'symbol' not in msg or 'ltp' not in msg: continue
                sym_raw = msg['symbol'].replace("NSE:", "").replace("-EQ", "")
                if sym_raw not in self.symbols: continue
                
                ltp, vol_today = float(msg['ltp']), float(msg.get('vol_traded_today', 0))
                prev_ltp, prev_vol = self.last_tick_ltp.get(sym_raw, ltp), self.last_tick_vol.get(sym_raw, vol_today)
                tick_vol = vol_today - prev_vol if vol_today >= prev_vol else 0
                tick_delta = tick_vol if ltp > prev_ltp else (-tick_vol if ltp < prev_ltp else 0)
                
                self.last_tick_ltp[sym_raw], self.last_tick_vol[sym_raw] = ltp, vol_today
                
                if sym_raw not in self.live_candles:
                    self.live_candles[sym_raw] = {"Open": ltp, "High": ltp, "Low": ltp, "Close": ltp, "Volume": tick_vol, "Net_Delta_1m": tick_delta}
                else:
                    c = self.live_candles[sym_raw]
                    c["High"], c["Low"], c["Close"], c["Volume"], c["Net_Delta_1m"] = max(c["High"], ltp), min(c["Low"], ltp), ltp, c["Volume"] + tick_vol, c["Net_Delta_1m"] + tick_delta

    def onerror(self, message): print(f"{COLOR_RED}[WS Error] {message}{COLOR_RESET}")
    def onclose(self, message): print(f"{COLOR_YELLOW}[WS Closed] Reconnecting...{COLOR_RESET}")
    def onopen(self): 
        print(f"{COLOR_GREEN}[WS Connected] Subscribing to {len(self.ws_symbols)} instruments.{COLOR_RESET}")
        self.fyers_ws.subscribe(data_type="SymbolUpdate", symbol=self.ws_symbols)

    def start_socket(self):
        self.fyers_ws = data_ws.FyersDataSocket(
            access_token=self.access_token, log_path="", litemode=False, write_to_file=False,
            reconnect=True, on_connect=self.onopen, on_close=self.onclose, on_error=self.onerror, on_message=self.onmessage
        )
        self.fyers_ws.connect()

    def run_event_loop(self):
        print(f"\n{COLOR_CYAN}⚡ LIVE ENGINE ARMED. Awaiting candle closes...{COLOR_RESET}")
        threading.Thread(target=self.start_socket, daemon=True).start()
        current_minute = datetime.now().minute
        while True:
            time.sleep(1)
            now = datetime.now()
            if now.minute != current_minute:
                current_minute = now.minute
                rounded_dt = now.replace(second=0, microsecond=0) - timedelta(minutes=1)
                with self.lock:
                    new_rows = [{"Datetime": rounded_dt, "Symbol": sym, **c} for sym, c in self.live_candles.items()]
                    self.live_candles.clear()
                if new_rows:
                    self.historical_df = pd.concat([self.historical_df, pd.DataFrame(new_rows)], ignore_index=True)
                    tape_exec = prepare_unified_execution_tape(self.historical_df, MICRO_TIMEFRAME, MACRO_TIMEFRAMES, GLOBAL_MACRO_STRATEGY_2D)
                    memory_bank = _run_dual_layer_trade_management(tape_exec, MICRO_TIMEFRAME, MACRO_TIMEFRAMES, self.cutoff_obj)
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print(f"📡 Last Update: {now.strftime('%H:%M:%S')} | Mode: WEBSOCKET LIVE TICK")
                    display_final_results(tape_exec, memory_bank, now, self.target_date_str)

# ==============================================================================
# PIPELINE ROUTER
# ==============================================================================
def scan_institutional_tape(target_date_str, entry_cutoff_time_str=ENTRY_CUTOFF_TIME):
    print(f"\n📡 Initiating Pipeline [{TRADING_MODE}] for {target_date_str}...")
    trading_days = get_past_trading_days(target_date_str, num_days=BACKTRACE_DAYS)
    cutoff_time_obj = pd.to_datetime(f"{target_date_str} {entry_cutoff_time_str}").time()
    cutoff_dt = pd.to_datetime(f"{target_date_str} 15:30:00")
    
    master_df = pd.DataFrame()
    
    if TRADING_MODE == "CASH_EQUITY":
        universe = filter_cash_equities_by_price_range(get_cash_equity_universe(), target_date_str)
        if not universe: return
        fetch_tasks = [(item, trading_days[0], target_date_str) for item in universe]
        stock_dfs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            for res in executor.map(fetch_stock_bars_worker, fetch_tasks):
                if res is not None: stock_dfs.append(res)
        if stock_dfs: master_df = pd.concat(stock_dfs, ignore_index=True)
            
    else:  # OPTIONS MODE
        universe, options_by_underlying = get_fno_universe_and_options()
        if not universe or not options_by_underlying: return
        spot_ref = fetch_all_spot_reference_prices(universe, target_date_str)
        qualifying_symbols = [item["symbol"] for item in universe if item["symbol"] in spot_ref]
        candidate_contracts = []
        for sym in qualifying_symbols:
            if sp := spot_ref.get(sym): candidate_contracts.extend(build_strike_range(sym, sp, options_by_underlying, target_date_str, STRIKE_RANGE_OFFSET))
        liquid_contracts = filter_liquid_contracts(candidate_contracts, target_date_str)
        option_dfs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            for res in executor.map(lambda c: fetch_stock_bars_worker((c, trading_days[0], target_date_str)), liquid_contracts):
                if res is not None: option_dfs.append(res)
        if option_dfs: master_df = pd.concat(option_dfs, ignore_index=True)

    if master_df.empty:
        print(f"{COLOR_YELLOW}⚠️ No execution data built. Exiting.{COLOR_RESET}")
        return

    if DATA_FEED_MODE == "WEBSOCKET":
        if not WS_AVAILABLE:
            print(f"{COLOR_RED}❌ fyers_apiv3 not installed. Please run: pip install fyers-apiv3{COLOR_RESET}")
        else:
            engine = LiveWebsocketEngine(master_df, target_date_str, entry_cutoff_time_str)
            engine.run_event_loop()
            return 

    # REST HISTORICAL EXECUTION
    master_df = truncate_to_cutoff(master_df, target_date_str, cutoff_dt)
    strat = GLOBAL_MACRO_STRATEGY_2D if TRADING_MODE == "CASH_EQUITY" else OPTIONS_STRATEGY_2D
    tape_exec = prepare_unified_execution_tape(master_df, MICRO_TIMEFRAME, MACRO_TIMEFRAMES, strat)
    
    if not tape_exec.empty:
        memory_bank = _run_dual_layer_trade_management(tape_exec, MICRO_TIMEFRAME, MACRO_TIMEFRAMES, cutoff_time_obj)
        display_final_results(tape_exec, memory_bank, pd.to_datetime(target_date_str), target_date_str)

def run_production_sweep():
    if not validate_fyers_token(): return
    target_dt = datetime.utcnow() + timedelta(hours=5, minutes=30)
    if target_dt.weekday() == 5: target_dt -= timedelta(days=1)
    elif target_dt.weekday() == 6: target_dt -= timedelta(days=2)
    scan_institutional_tape(target_dt.strftime("%Y-%m-%d"), ENTRY_CUTOFF_TIME)

if __name__ == "__main__":
    run_production_sweep()
