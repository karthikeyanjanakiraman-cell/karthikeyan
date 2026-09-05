"""system3.py - Asit Baran Pati Multi-Timeframe Trading System Implementation

Production-Grade Universal N-Timeframe & Dual-Tier 45-Degree Renko Engine (v21)
- True Live WebSocket Integration (Tick-by-Tick Institutional Delta)
- Configurable Micro Execution Timeframe (e.g., "5min", "15min")
- Configurable Macro Hierarchy Array (e.g., ["60min", "240min"])
- Dual-Tier Scorecard (9 Pillars) & Global Mandatory Veto Switches
- Cumulative Volume Delta 45-Degree Renko Matrix
- True Post-Entry Renko Velocity Stall Engine (No instant stop-outs)
- Internal Delta Percentage (Absolute Order Flow Conviction)
- 09:15 Overnight Flush & Configurable Entry Cutoff (see ENTRY_CUTOFF_TIME)

NOTE: A "stale momentum / chop guard" is referenced in earlier design notes but is
NOT implemented in this file. The only stall protection is the post-entry Renko
Velocity Stall exit inside _run_dual_layer_trade_management. If a chop/stale-momentum
entry filter is required, it needs to be added explicitly.
"""

import concurrent.futures
import datetime
from datetime import datetime, timedelta
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
LIQUIDITY_CACHE_RETENTION_DAYS = 30   # fix #14: prune entries older than this

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
MACRO_TIMEFRAMES = ["15min"]

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
MACRO_MANDATORY_PRICE_RENKO    = False    
MACRO_MANDATORY_VOL_RENKO      = False
MACRO_MANDATORY_RENKO_VELOCITY = False
MACRO_MANDATORY_RSI_BB         = False
MACRO_MANDATORY_ADX_DMI        = True
MACRO_MANDATORY_EMA_SPREAD     = False
MACRO_MANDATORY_STOCHASTIC     = False
MACRO_MANDATORY_ATR_BB         = False   
MACRO_MANDATORY_RENKO_BB       = False   
MACRO_MINIMUM_SCORE            = 1       

# ==============================================================================
# TIER 2: MICRO EXECUTION SWITCHBOARD (THE SNIPER) - 9 PILLARS
# ==============================================================================
SYNC_MICRO_WITH_MACRO          = False
MICRO_MANDATORY_LIVE_PERCENTILE = 0.0    
MICRO_MANDATORY_PRICE_RENKO    = False    
MICRO_MANDATORY_VOL_RENKO      = False    
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
    # Always compute the historical delta proxy here (compute_base_net_delta is a
    # no-op if Net_Delta_1m already exists). Previously this only ran in REST mode,
    # so WEBSOCKET mode's warm-up history had no Net_Delta_1m while live ticks did -
    # causing the CVD Renko series to jump between two different calculation
    # methods right at the point live data starts.
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

    # Fix #14: nothing previously purged old entries, so liquidity_cache.json grew
    # unbounded across months of daily runs and was fully read/written every run.
    # Keep only entries whose embedded date (last 10 chars of the key, "YYYY-MM-DD")
    # falls within a rolling retention window.
    retention_cutoff = (prev_dt - timedelta(days=LIQUIDITY_CACHE_RETENTION_DAYS)).strftime("%Y-%m-%d")
    pruned_cache = {}
    for k, v in cache.items():
        date_part = k[-10:]
        try:
            datetime.strptime(date_part, "%Y-%m-%d")
        except ValueError:
            continue  # malformed/legacy key - drop it
        if date_part >= retention_cutoff:
            pruned_cache[k] = v
    if len(pruned_cache) != len(cache):
        cache = pruned_cache
        try:
            with open(LIQUIDITY_CACHE_FILE, "w") as f: json.dump(cache, f)
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
                # Hysteresis (2x brick) only applies when reversing an ESTABLISHED
                # trend. A neutral (curr_trend == 0) start requires just 1x brick
                # in either direction, so the first move of the day isn't biased bullish.
                if curr_trend == 0:
                    if move >= bs:
                        bricks = int(move // bs); curr_trend = 1; curr_count = bricks; curr_price += bricks * bs
                    elif move <= -bs:
                        bricks = int(abs(move) // bs); curr_trend = -1; curr_count = -bricks; curr_price -= bricks * bs
                elif curr_trend > 0:
                    if move >= bs:
                        bricks = int(move // bs); curr_count = curr_count + bricks; curr_price += bricks * bs
                    elif move <= -(2 * bs):
                        bricks = int(abs(move) // bs); curr_trend = -1; curr_count = -bricks; curr_price -= bricks * bs
                else:
                    if move <= -bs:
                        bricks = int(abs(move) // bs); curr_count = curr_count - bricks; curr_price -= bricks * bs
                    elif move >= (2 * bs):
                        bricks = int(move // bs); curr_trend = 1; curr_count = bricks; curr_price += bricks * bs
                counts[i] = curr_count
            renko_counts[indices] = counts
    df[f"Renko_Count_{tf_name}"] = renko_counts
    if confirm_bricks > 0:
        df[f"Renko_Bull_{tf_name}"] = renko_counts >= confirm_bricks
        df[f"Renko_Bear_{tf_name}"] = renko_counts <= -confirm_bricks
    else:
        # confirm_bricks == 0: use strict inequality so a flat count of exactly 0
        # cannot satisfy both Bull and Bear simultaneously.
        df[f"Renko_Bull_{tf_name}"] = renko_counts > 0
        df[f"Renko_Bear_{tf_name}"] = renko_counts < 0
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
                if curr_trend == 0:
                    if move >= bs:
                        b = int(move // bs); curr_trend = 1; curr_count = b; curr_delta += b * bs
                    elif move <= -bs:
                        b = int(abs(move) // bs); curr_trend = -1; curr_count = -b; curr_delta -= b * bs
                elif curr_trend > 0:
                    if move >= bs:
                        b = int(move // bs); curr_count = curr_count + b; curr_delta += b * bs
                    elif move <= -(2 * bs):
                        b = int(abs(move) // bs); curr_trend = -1; curr_count = -b; curr_delta -= b * bs
                else:
                    if move <= -bs:
                        b = int(abs(move) // bs); curr_count = curr_count - b; curr_delta -= b * bs
                    elif move >= (2 * bs):
                        b = int(move // bs); curr_trend = 1; curr_count = b; curr_delta += b * bs
                counts[i] = curr_count
            vol_renko_counts[indices] = counts
    df[f"Vol_Renko_Count_{tf_name}"] = vol_renko_counts
    if confirm_bricks > 0:
        df[f"Vol_Renko_Bull_{tf_name}"] = vol_renko_counts >= confirm_bricks
        df[f"Vol_Renko_Bear_{tf_name}"] = vol_renko_counts <= -confirm_bricks
    else:
        df[f"Vol_Renko_Bull_{tf_name}"] = vol_renko_counts > 0
        df[f"Vol_Renko_Bear_{tf_name}"] = vol_renko_counts < 0
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
    df[f"ATR_BB_Bear_{tf_name}"] = df["ATR"] < (atr_mean - BB_STD_DEV * atr_std)
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

def _session_grouper_origin(df_base):
    """
    Anchors pd.Grouper bin edges to the 09:15 session open instead of midnight.
    Without this, timeframes that don't evenly divide into 09:15 (e.g. "60min",
    "240min") produce a truncated/misaligned first bar of the day (bins land at
    00:00, 04:00, 08:00, 12:00... instead of 09:15, 13:15...), which distorts
    Open/ATR/Renko for that first bar every single day.
    """
    first_day = pd.to_datetime(df_base["Datetime"]).dt.normalize().min()
    return first_day + pd.Timedelta(hours=9, minutes=15)

def evaluate_single_timeframe_gates(df_base, tf_str):
    origin = _session_grouper_origin(df_base)
    df_tf = df_base.groupby(["Symbol", pd.Grouper(key="Datetime", freq=tf_str, closed="left", label="left", origin=origin)]).agg(
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
        origin = _session_grouper_origin(rolling_master_df)
        df_micro = rolling_master_df.groupby(["Symbol", pd.Grouper(key="Datetime", freq=micro_tf, closed="left", label="left", origin=origin)]).agg(
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
    daily_trade_count = defaultdict(int)  # per-symbol count, reset every trading day (fix #9)
    micro_tf_mins = _parse_tf_to_minutes(micro_timeframe)
    max_stall_mins = RENKO_VELOCITY_MAX_BARS * micro_tf_mins

    for t in np.sort(tape_exec["Datetime"].unique()):
        t_dt = pd.to_datetime(t)

        if t_dt.time() == pd.Timestamp("09:15").time():
            for sym, episodes in memory_bank.items():
                if episodes and episodes[-1]["state"] == "ACTIVE":
                    episodes[-1]["state"], episodes[-1]["exit_time"], episodes[-1]["exit_price"], episodes[-1]["exit_reason"] = "EXITED", t_dt.strftime("%Y-%m-%d %H:%M"), closes_dict.get((t_dt, sym), episodes[-1]["origin"]), "Overnight Gap Flush"
            # New trading day: MAX_DAILY_TRADES_PER_SYMBOL and the "don't re-enter
            # same direction after a stop" memory must reset per day. Previously
            # these persisted across the whole multi-day backtrace window, so a
            # stop-out or trade cap hit on day 1 could silently veto legitimate
            # signals on the actual target day (fix #9 / #10).
            last_exit_price.clear()
            last_exit_dir.clear()
            daily_trade_count.clear()

        for sym, episodes in memory_bank.items():
            if episodes and episodes[-1]["state"] == "ACTIVE":
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            