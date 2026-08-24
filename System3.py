"""system3.py - Asit Baran Pati Multi-Timeframe Trading System Implementation

Production-Grade Universal N-Timeframe & Dual-Tier 45-Degree Renko Engine:
- Configurable Micro Execution Timeframe (e.g., "1min", "3min", "5min")
- Configurable Macro Hierarchy Array (e.g., ["15min", "60min", "1D"])
- Phase 1 Blueprint: Dual-Tier Scorecard (9 Pillars) & Global Mandatory Veto Switches
- Phase 1 Blueprint: Order Flow / Cumulative Volume Delta 45-Degree Renko
- Phase 1 Blueprint: Renko-Velocity Engine (Time-Distance Momentum Tracking)
- EXIT STRATEGY: Dual-Layered (Triggering Macro + Micro) + Velocity Stall Cutoff
- CLI & ENV ARGS: Ultra-robust tokenized date and time parsing.
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

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

print("🔖 SYSTEM3 BUILD: v8-PROVE-IT-UPDATED (2026-08-24)")

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

BACKTRACE_DAYS = 60
LIQUIDITY_CACHE_FILE = "liquidity_cache.json"
EXCLUDED_INDICES = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX", "BANKEX", "NIFTY50", "NIFTYBANK"}

_FYERS_ERROR_LOG_CAP = 5
_fyers_error_log_count = 0


def _log_fyers_error(context, status_code=None, body=None):
    global _fyers_error_log_count
    if _fyers_error_log_count >= _FYERS_ERROR_LOG_CAP:
        return
    _fyers_error_log_count += 1
    snippet = str(body)[:300] if body is not None else ""
    print(f"{COLOR_YELLOW}  [Fyers Diagnostic #{_fyers_error_log_count}] {context}"
          f"{' | HTTP ' + str(status_code) if status_code else ''} {snippet}{COLOR_RESET}")


def get_fyers_auth_headers():
    return {"Authorization": f"{os.environ.get('FYERS_CLIENT_ID', '')}:{os.environ.get('FYERS_ACCESS_TOKEN', '')}"}


def validate_fyers_token():
    if not os.environ.get("FYERS_CLIENT_ID") or not os.environ.get("FYERS_ACCESS_TOKEN"):
        print(f"❌ {COLOR_RED}Error: FYERS_CLIENT_ID or FYERS_ACCESS_TOKEN environment variables not found.{COLOR_RESET}")
        return False

    try:
        headers = get_fyers_auth_headers()
        res = requests.get("https://api-t1.fyers.in/api/v3/profile", headers=headers, timeout=10)
        body = {}
        try:
            body = res.json()
        except Exception:
            pass

        if res.status_code != 200 or body.get("s") != "ok":
            print(f"{COLOR_RED}❌ Fyers token validation FAILED before starting the sweep.{COLOR_RESET}")
            print(f"{COLOR_RED}   HTTP {res.status_code} | Response: {str(body)[:300] or res.text[:300]}{COLOR_RESET}")
            print(f"{COLOR_YELLOW}   -> Your FYERS_ACCESS_TOKEN is expired/invalid. Please regenerate it for today.{COLOR_RESET}")
            return False

        fy_name = body.get("data", {}).get("name", "Unknown")
        print(f"{COLOR_GREEN}✅ Fyers token validated OK (Account: {fy_name}){COLOR_RESET}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"{COLOR_RED}❌ Could not reach Fyers to validate token: {e}{COLOR_RESET}")
        return False


# ==============================================================================
# 🎛️ TIER 0: TRADING MODE & PIPELINE ROUTING SWITCHES CASH_EQUITY F_AND_O_OPTIONS
# ==============================================================================
TRADING_MODE = "F_AND_O_OPTIONS"  
ENABLE_STAGE1_STOCK_FILTER = False  

MIN_STOCK_PRICE = 100.0
MAX_STOCK_PRICE = 500.0
MIN_STOCK_VOLUME = 500000

# ==============================================================================
# GLOBAL CONFIGURATION: DYNAMIC TIMEFRAMES & INDICATORS
# ==============================================================================
MICRO_TIMEFRAME = "240min"
MACRO_TIMEFRAMES = ["2400min"]

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

GLOBAL_MACRO_STRATEGY_2D = "BOTH"  # "BULLISH", "BEARISH", or "BOTH"

# ==============================================================================
# TIER 1: MACRO CONTEXT SWITCHBOARD (THE GENERAL) - 9 PILLARS
# ==============================================================================
MACRO_MANDATORY_PRICE_RENKO    = False
MACRO_MANDATORY_VOL_RENKO      = False
MACRO_MANDATORY_RENKO_VELOCITY = False
MACRO_MANDATORY_RSI_BB         = False
MACRO_MANDATORY_ADX_DMI        = False
MACRO_MANDATORY_EMA_SPREAD     = False
MACRO_MANDATORY_STOCHASTIC     = False
MACRO_MANDATORY_ATR_BB         = True   
MACRO_MANDATORY_RENKO_BB       = True   
MACRO_MINIMUM_SCORE            = 2      

# ==============================================================================
# TIER 2: MICRO EXECUTION SWITCHBOARD (THE SNIPER) - 9 PILLARS
# ==============================================================================
SYNC_MICRO_WITH_MACRO          = False

MICRO_MANDATORY_PRICE_RENKO    = False
MICRO_MANDATORY_VOL_RENKO      = False
MICRO_MANDATORY_RENKO_VELOCITY = False
MICRO_MANDATORY_RSI_BB         = False
MICRO_MANDATORY_ADX_DMI        = False
MICRO_MANDATORY_EMA_SPREAD     = False
MICRO_MANDATORY_STOCHASTIC     = False
MICRO_MANDATORY_ATR_BB         = True  
MICRO_MANDATORY_RENKO_BB       = True   

MICRO_MINIMUM_SCORE            = 2      

# ==============================================================================
# TIER 3: TRADE MANAGEMENT & TEMPORAL GATES (EXIT & TIMING)
# ==============================================================================
MICRO_EXIT_PRICE_BRICKS = 5
MICRO_EXIT_VOL_BRICKS   = 50
MACRO_EXIT_PRICE_BRICKS = 1
MACRO_EXIT_VOL_BRICKS   = 10

RENKO_VELOCITY_MAX_BARS = 6
ENTRY_CUTOFF_TIME = "15:00"

# ==============================================================================
# TIER 4: OPTIONS STAGE 2 CONFIG (Ignored if TRADING_MODE == "CASH_EQUITY")
# ==============================================================================
OPTIONS_TARGET_EXPIRY = "CURRENT"   
STRIKE_RANGE_OFFSET = 2             
MIN_OPT_PREMIUM = 15.0              
MIN_OPT_VOLUME = 250000             
OPTIONS_STRATEGY_2D = "BULLISH"     


# ==============================================================================
# 1. LIVE INGESTION (FYERS)
# ==============================================================================
def get_cash_equity_universe():
    print("📡 Fetching Cash Equity Universe via FYERS (NSE_CM.csv)...")
    spot_inst = []
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res_cm = requests.get("https://public.fyers.in/sym_details/NSE_CM.csv", headers=headers, timeout=15)
        if res_cm.status_code == 200:
            for line in res_cm.text.strip().split("\n"):
                cols = [c.strip() for c in line.split(",")]
                for c in cols:
                    if c.startswith("NSE:") and c.endswith("-EQ"):
                        base = c.replace("NSE:", "").replace("-EQ", "")
                        if base in EXCLUDED_INDICES or base.isdigit() or not base:
                            continue
                        spot_inst.append({"symbol": base, "key": c, "underlying": base})
                        break
    except Exception as e:
        print(f"{COLOR_RED}[Error] NSE_CM.csv fetch failed: {e}{COLOR_RESET}")
        return []

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
            opt_type = None
            type_idx = -1
            for i in range(len(cols) - 1, -1, -1):
                if cols[i] in ("CE", "PE"):
                    opt_type = cols[i]
                    type_idx = i
                    break
            if not opt_type or type_idx < 3:
                continue
            try:
                strike_val = float(cols[type_idx - 1])
                base_symbol = cols[type_idx - 3].strip()
                if base_symbol in EXCLUDED_INDICES or base_symbol.isdigit() or not base_symbol:
                    continue
                sym_ticker = None
                for c in cols:
                    if c.startswith("NSE:") and opt_type in c:
                        sym_ticker = c
                        break
                if not sym_ticker:
                    continue
                expiry_date = None
                for c in cols:
                    try:
                        num = int(float(c))
                        if 1.5e9 < num < 3e9:
                            expiry_date = datetime.fromtimestamp(num).strftime("%Y-%m-%d")
                            break
                    except Exception:
                        pass
                if not expiry_date:
                    continue
                opt_inst.append({
                    "symbol": sym_ticker, "key": sym_ticker, "underlying": base_symbol,
                    "type": opt_type, "strike": strike_val, "expiry": expiry_date
                })
                if base_symbol not in valid_underlyings:
                    valid_underlyings.add(base_symbol)
                    spot_ticker = spot_key_map.get(base_symbol)
                    if spot_ticker:
                        spot_inst.append({"symbol": base_symbol, "key": spot_ticker, "underlying": base_symbol})
            except Exception:
                pass
    except Exception as e:
        print(f"{COLOR_RED}[Error] FYERS CSV fetch failed: {e}{COLOR_RESET}")
        return [], {}

    options_by_underlying = {}
    for o in opt_inst:
        options_by_underlying.setdefault(o["underlying"], []).append(o)

    print(f"  ├─ Mapped {len(spot_inst)} Spot Instruments & {len(opt_inst)} Options Contracts "
          f"across {len(options_by_underlying)} underlyings.")
    return spot_inst, options_by_underlying


def get_past_trading_days(target_date_str, num_days=20):
    try:
        target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
        trading_days = []
        current_dt = target_dt
        while len(trading_days) < num_days:
            if current_dt.weekday() < 5:
                trading_days.append(current_dt.strftime("%Y-%m-%d"))
            current_dt -= timedelta(days=1)
        trading_days.reverse()
        return trading_days
    except Exception: return []


# ==============================================================================
# 1B. FYERS CANDLE FETCHER & FILTERS
# ==============================================================================
def fetch_fyers_candles(key, start_dt, end_dt, resolution="1"):
    headers = get_fyers_auth_headers()
    for attempt in range(3):
        try:
            time.sleep(0.2)
            encoded_symbol = urllib.parse.quote(key, safe=":")
            url = (f"https://api-t1.fyers.in/data/history?symbol={encoded_symbol}"
                   f"&resolution={resolution}&date_format=1&range_from={start_dt}&range_to={end_dt}")
            res = requests.get(url, headers=headers, timeout=10)

            if res.status_code == 200:
                try:
                    data = res.json()
                except ValueError:
                    _log_fyers_error(f"Non-JSON response for {key}", res.status_code, res.text[:300])
                    return None
                if not data:
                    return None

                status = data.get("s")
                if status == "ok":
                    candles = data.get("candles")
                    if not candles:
                        return None
                    df = pd.DataFrame(candles, columns=["Epoch", "Open", "High", "Low", "Close", "Volume"])
                    df["Datetime"] = pd.to_datetime(df["Epoch"], unit="s", utc=True) \
                        .dt.tz_convert("Asia/Kolkata").dt.tz_localize(None).astype("datetime64[ns]")
                    return df
                if status == "no_data":
                    return None

                _log_fyers_error(f"API error for {key} (code={data.get('code')}, msg={data.get('message')})", res.status_code)
                if data.get("code") == -16:
                    return None

            elif res.status_code in (429, 500, 502, 503):
                time.sleep(random.uniform(1.0, 3.0) * (attempt + 1))
                continue
            else:
                _log_fyers_error(f"HTTP failure for {key}", res.status_code, res.text[:300])
                return None
        except requests.exceptions.RequestException as e:
            _log_fyers_error(f"Network exception on attempt {attempt + 1}: {e}")
            time.sleep(1)
    return None


def filter_cash_equities_by_price_range(universe, target_date_str):
    target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
    prev_dt = target_dt - timedelta(days=1)
    while prev_dt.weekday() >= 5:
        prev_dt -= timedelta(days=1)
    prev_day = prev_dt.strftime("%Y-%m-%d")
    lookback_start = (prev_dt - timedelta(days=7)).strftime("%Y-%m-%d")

    print(f"🧹 Applying Price Range Filter (₹{MIN_STOCK_PRICE} - ₹{MAX_STOCK_PRICE}) & Volume Filter...")

    def worker(item):
        df = fetch_fyers_candles(item["key"], lookback_start, prev_day, resolution="D")
        if df is None or df.empty:
            return None
        last = df.sort_values("Datetime").iloc[-1]
        close_price = last["Close"]
        volume = last["Volume"]
        if MIN_STOCK_PRICE <= close_price <= MAX_STOCK_PRICE and volume >= MIN_STOCK_VOLUME:
            return item
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(worker, universe))

    filtered = [r for r in results if r is not None]
    print(f"  ├─ ✅ {len(filtered)}/{len(universe)} cash equities passed price & volume filters.")
    return filtered


def filter_liquid_contracts(contracts, target_date_str):
    if not contracts:
        return []

    target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
    prev_dt = target_dt - timedelta(days=1)
    while prev_dt.weekday() >= 5:
        prev_dt -= timedelta(days=1)
    prev_day = prev_dt.strftime("%Y-%m-%d")
    lookback_start = (prev_dt - timedelta(days=7)).strftime("%Y-%m-%d")

    cache = {}
    if os.path.exists(LIQUIDITY_CACHE_FILE):
        try:
            with open(LIQUIDITY_CACHE_FILE, "r") as f:
                cache = json.load(f)
        except Exception:
            pass

    def worker(c):
        cache_key = f"{c['symbol']}_{prev_day}"
        if cache_key in cache:
            return (c, cache_key, cache[cache_key], False)

        df = fetch_fyers_candles(c["key"], lookback_start, prev_day, resolution="D")
        if df is None or df.empty:
            return (c, cache_key, False, True)
        
        last = df.sort_values("Datetime").iloc[-1]
        passed = bool(last["Close"] >= MIN_OPT_PREMIUM and last["Volume"] >= MIN_OPT_VOLUME)
        return (c, cache_key, passed, True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(worker, contracts))

    valid_contracts = []
    cache_needs_update = False

    for c, cache_key, passed, hit_api in results:
        if hit_api:
            cache[cache_key] = passed
            cache_needs_update = True
        if passed:
            valid_contracts.append(c)

    if cache_needs_update:
        try:
            with open(LIQUIDITY_CACHE_FILE, "w") as f:
                json.dump(cache, f)
        except Exception as e:
            print(f"{COLOR_YELLOW}  [Warning] Failed to save liquidity cache: {e}{COLOR_RESET}")

    return valid_contracts


def build_strike_range(symbol, spot_price, options_by_underlying, target_date_str, offset):
    opts = options_by_underlying.get(symbol, [])
    if not opts:
        return []

    target_dt = pd.to_datetime(target_date_str)
    valid_expiries = sorted(set(pd.to_datetime(o["expiry"]) for o in opts if pd.to_datetime(o["expiry"]) >= target_dt))
    if not valid_expiries:
        valid_expiries = sorted(set(pd.to_datetime(o["expiry"]) for o in opts))
    if not valid_expiries:
        return []

    chosen_expiry = valid_expiries[0] if OPTIONS_TARGET_EXPIRY == "CURRENT" else \
        (valid_expiries[1] if len(valid_expiries) > 1 else valid_expiries[0])
    same_expiry = [o for o in opts if pd.to_datetime(o["expiry"]) == chosen_expiry]

    strikes_sorted = sorted(set(o["strike"] for o in same_expiry))
    if not strikes_sorted:
        return []

    closest = min(strikes_sorted, key=lambda x: abs(x - spot_price))
    idx = strikes_sorted.index(closest)
    start_idx = max(0, idx - offset)
    end_idx = min(len(strikes_sorted), idx + offset + 1)
    selected_strikes = set(strikes_sorted[start_idx:end_idx])

    return [o for o in same_expiry if o["strike"] in selected_strikes]


def fetch_all_spot_reference_prices(spot_universe, target_date_str):
    target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
    lookback_start = (target_dt - timedelta(days=5)).strftime("%Y-%m-%d")
    spot_ref = {}

    def worker(item):
        df = fetch_fyers_candles(item["key"], lookback_start, target_date_str, resolution="D")
        if df is not None and not df.empty:
            last_close = df.sort_values("Datetime").iloc[-1]["Close"]
            return item["symbol"], float(last_close)
        return item["symbol"], None

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(worker, spot_universe))

    for sym, price in results:
        if price is not None:
            spot_ref[sym] = price
    return spot_ref


# ==============================================================================
# 2. CORE TECHNICAL & 45-DEGREE RENKO ENGINES (SMA BASED)
# ==============================================================================
def calculate_core_technicals(df_tf):
    df_tf["H-L"] = df_tf["High"] - df_tf["Low"]
    df_tf["H-PC"] = (df_tf["High"] - df_tf.groupby("Symbol")["Close"].shift(1)).abs()
    df_tf["L-PC"] = (df_tf["Low"] - df_tf.groupby("Symbol")["Close"].shift(1)).abs()
    df_tf["TR"] = df_tf[["H-L", "H-PC", "L-PC"]].max(axis=1)
    
    df_tf["ATR"] = df_tf.groupby("Symbol")["TR"].transform(lambda x: x.rolling(window=ATR_PERIOD, min_periods=1).mean())
    df_tf["ATR"] = df_tf["ATR"].fillna(df_tf["Close"] * RENKO_DEFAULT_PCT)

    delta = df_tf.groupby("Symbol")["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.groupby(df_tf["Symbol"]).transform(lambda x: x.rolling(window=RSI_PERIOD, min_periods=1).mean())
    avg_loss = loss.groupby(df_tf["Symbol"]).transform(lambda x: x.rolling(window=RSI_PERIOD, min_periods=1).mean())
    df_tf["RSI"] = 100 - (100 / (1 + (avg_gain / (avg_loss + 1e-8))))
    df_tf["RSI_SMA"] = df_tf.groupby("Symbol")["RSI"].transform(lambda x: x.rolling(BB_SMA_PERIOD, min_periods=1).mean())

    high_d = df_tf["High"] - df_tf.groupby("Symbol")["High"].shift(1)
    low_d = df_tf.groupby("Symbol")["Low"].shift(1) - df_tf["Low"]
    df_tf["+DM"] = np.where((high_d > low_d) & (high_d > 0), high_d, 0)
    df_tf["-DM"] = np.where((low_d > high_d) & (low_d > 0), low_d, 0)

    df_tf["+DI"] = (100 * (df_tf.groupby("Symbol")["+DM"].transform(lambda x: x.rolling(window=ADX_PERIOD, min_periods=1).mean()) / (df_tf["ATR"] + 1e-8)))
    df_tf["-DI"] = (100 * (df_tf.groupby("Symbol")["-DM"].transform(lambda x: x.rolling(window=ADX_PERIOD, min_periods=1).mean()) / (df_tf["ATR"] + 1e-8)))
    df_tf["DX"] = (100 * abs(df_tf["+DI"] - df_tf["-DI"]) / (df_tf["+DI"] + df_tf["-DI"] + 1e-8))
    df_tf["ADX"] = df_tf.groupby("Symbol")["DX"].transform(lambda x: x.rolling(window=ADX_PERIOD, min_periods=1).mean())

    df_tf["EMA_8"] = df_tf.groupby("Symbol")["Close"].transform(lambda x: x.rolling(window=8, min_periods=1).mean())
    df_tf["EMA_21"] = df_tf.groupby("Symbol")["Close"].transform(lambda x: x.rolling(window=21, min_periods=1).mean())
    df_tf["EMA_Spread"] = abs(df_tf["EMA_8"] - df_tf["EMA_21"])
    
    spread_thresh = df_tf.groupby("Symbol")["EMA_Spread"].transform(lambda x: x.rolling(window=20, min_periods=1).mean()) * 0.20
    df_tf["EMA_Bull_Expanded"] = (df_tf["EMA_8"] > df_tf["EMA_21"]) & (df_tf["EMA_Spread"] >= spread_thresh)
    df_tf["EMA_Bear_Expanded"] = (df_tf["EMA_8"] < df_tf["EMA_21"]) & (df_tf["EMA_Spread"] >= spread_thresh)

    lowest_low = df_tf.groupby("Symbol")["Low"].transform(lambda x: x.rolling(window=STOCH_PERIOD, min_periods=1).min())
    highest_high = df_tf.groupby("Symbol")["High"].transform(lambda x: x.rolling(window=STOCH_PERIOD, min_periods=1).max())
    df_tf["Stoch_K"] = ((df_tf["Close"] - lowest_low) / (highest_high - lowest_low + 1e-9)) * 100
    
    atr_median = df_tf.groupby("Symbol")["ATR"].transform(lambda x: x.rolling(window=50, min_periods=1).median())
    df_tf["Vol_Pass"] = df_tf["ATR"] >= (atr_median * 0.75)
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
                        bricks = int(move // bs)
                        curr_trend = 1
                        curr_count = curr_count + bricks if curr_count > 0 else bricks
                        curr_price += bricks * bs
                    elif move <= -(2 * bs):
                        bricks = int(abs(move) // bs)
                        curr_trend = -1
                        curr_count = -bricks
                        curr_price -= bricks * bs
                else:
                    if move <= -bs:
                        bricks = int(abs(move) // bs)
                        curr_trend = -1
                        curr_count = curr_count - bricks if curr_count < 0 else -bricks
                        curr_price -= bricks * bs
                    elif move >= (2 * bs):
                        bricks = int(move // bs)
                        curr_trend = 1
                        curr_count = bricks
                        curr_price += bricks * bs
                counts[i] = curr_count
            renko_counts[indices] = counts
    df[f"Renko_Count_{tf_name}"] = renko_counts
    df[f"Renko_Bull_{tf_name}"] = renko_counts >= confirm_bricks
    df[f"Renko_Bear_{tf_name}"] = renko_counts <= -confirm_bricks
    return df


def construct_volume_delta_renko_matrix(df, tf_name, confirm_bricks):
    df['Wick_Spread'] = df['High'] - df['Low']
    df['Wick_Spread'] = df['Wick_Spread'].replace(0, 1e-9)
    df['Delta_Vol'] = df['Volume'] * ((df['Close'] - df['Open']) / df['Wick_Spread'])
    df['Cum_Delta'] = df.groupby('Symbol')['Delta_Vol'].cumsum()
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
                        bricks = int(move // bs)
                        curr_trend = 1
                        curr_count = curr_count + bricks if curr_count > 0 else bricks
                        curr_delta += bricks * bs
                    elif move <= -(2 * bs):
                        bricks = int(abs(move) // bs)
                        curr_trend = -1
                        curr_count = -bricks
                        curr_delta -= bricks * bs
                else:
                    if move <= -bs:
                        bricks = int(abs(move) // bs)
                        curr_trend = -1
                        curr_count = curr_count - bricks if curr_count < 0 else -bricks
                        curr_delta -= bricks * bs
                    elif move >= (2 * bs):
                        bricks = int(move // bs)
                        curr_trend = 1
                        curr_count = bricks
                        curr_delta += bricks * bs
                counts[i] = curr_count
            vol_renko_counts[indices] = counts
    df[f"Vol_Renko_Count_{tf_name}"] = vol_renko_counts
    df[f"Vol_Renko_Bull_{tf_name}"] = vol_renko_counts >= confirm_bricks
    df[f"Vol_Renko_Bear_{tf_name}"] = vol_renko_counts <= -confirm_bricks
    return df


def construct_renko_velocity_engine(df, tf_name):
    brick_diff = df.groupby("Symbol")[f"Renko_Count_{tf_name}"].diff().fillna(1)
    brick_changed = (brick_diff != 0)
    df["Brick_ID"] = brick_changed.cumsum()
    df[f"Bars_Since_Brick_{tf_name}"] = df.groupby(["Symbol", "Brick_ID"]).cumcount()
    df.drop("Brick_ID", axis=1, inplace=True)

    is_trending_bull = df[f"Renko_Count_{tf_name}"] > 0
    is_trending_bear = df[f"Renko_Count_{tf_name}"] < 0
    has_velocity = df[f"Bars_Since_Brick_{tf_name}"] <= RENKO_VELOCITY_MAX_BARS

    df[f"Velocity_Bull_{tf_name}"] = is_trending_bull & has_velocity
    df[f"Velocity_Bear_{tf_name}"] = is_trending_bear & has_velocity
    return df


def construct_bb_meta_pillars(df, tf_name):
    atr_mean = df.groupby("Symbol")["ATR"].transform(lambda x: x.rolling(BB_SMA_PERIOD, min_periods=1).mean())
    atr_std = df.groupby("Symbol")["ATR"].transform(lambda x: x.rolling(BB_SMA_PERIOD, min_periods=1).std()).fillna(0)
    atr_upper = atr_mean + BB_STD_DEV * atr_std
    df[f"ATR_BB_Upper_{tf_name}"] = atr_upper
    is_expanding = df["ATR"] > atr_upper
    df[f"ATR_BB_Bull_{tf_name}"] = is_expanding
    df[f"ATR_BB_Bear_{tf_name}"] = is_expanding  

    renko_col = f"Renko_Count_{tf_name}"
    renko_mean = df.groupby("Symbol")[renko_col].transform(lambda x: x.rolling(BB_SMA_PERIOD, min_periods=1).mean())
    renko_std = df.groupby("Symbol")[renko_col].transform(lambda x: x.rolling(BB_SMA_PERIOD, min_periods=1).std()).fillna(0)
    renko_upper = renko_mean + BB_STD_DEV * renko_std
    renko_lower = renko_mean - BB_STD_DEV * renko_std
    df[f"Renko_BB_Upper_{tf_name}"] = renko_upper
    df[f"Renko_BB_Lower_{tf_name}"] = renko_lower
    df[f"Renko_BB_Bull_{tf_name}"] = df[renko_col] <= renko_upper
    df[f"Renko_BB_Bear_{tf_name}"] = df[renko_col] >= renko_lower
    return df


# ==============================================================================
# 3. DUAL-TIER SCORECARD SYSTEM (OUT OF 9 PILLARS)
# ==============================================================================
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

    c_price_bull, c_price_bear = df[f"Renko_Bull_{tf_str}"].astype(int), df[f"Renko_Bear_{tf_str}"].astype(int)
    c_vol_bull, c_vol_bear = df[f"Vol_Renko_Bull_{tf_str}"].astype(int), df[f"Vol_Renko_Bear_{tf_str}"].astype(int)
    c_vel_bull, c_vel_bear = df[f"Velocity_Bull_{tf_str}"].astype(int), df[f"Velocity_Bear_{tf_str}"].astype(int)
    c_rsi_bull, c_rsi_bear = (df["RSI"] >= df["RSI_SMA"]).astype(int), (df["RSI"] <= df["RSI_SMA"]).astype(int)
    c_adx_bull, c_adx_bear = ((df["ADX"] >= ADX_THRESHOLD) & (df["+DI"] > df["-DI"])).astype(int), ((df["ADX"] >= ADX_THRESHOLD) & (df["-DI"] > df["+DI"])).astype(int)
    c_ema_bull, c_ema_bear = df["EMA_Bull_Expanded"].astype(int), df["EMA_Bear_Expanded"].astype(int)
    c_stoch_bull, c_stoch_bear = df["Stoch_Bull_Pass"].astype(int), df["Stoch_Bear_Pass"].astype(int)
    c_atr_bb_bull, c_atr_bb_bear = df[f"ATR_BB_Bull_{tf_str}"].astype(int), df[f"ATR_BB_Bear_{tf_str}"].astype(int)
    c_renko_bb_bull, c_renko_bb_bear = df[f"Renko_BB_Bull_{tf_str}"].astype(int), df[f"Renko_BB_Bear_{tf_str}"].astype(int)

    df[f"Score_Bull_{tf_str}"] = c_price_bull + c_vol_bull + c_vel_bull + c_rsi_bull + c_adx_bull + c_ema_bull + c_stoch_bull + c_atr_bb_bull + c_renko_bb_bull
    df[f"Score_Bear_{tf_str}"] = c_price_bear + c_vol_bear + c_vel_bear + c_rsi_bear + c_adx_bear + c_ema_bear + c_stoch_bear + c_atr_bb_bear + c_renko_bb_bear

    bull_veto, bear_veto = pd.Series(False, index=df.index), pd.Series(False, index=df.index)
    if req_price: bull_veto, bear_veto = bull_veto | (c_price_bull == 0), bear_veto | (c_price_bear == 0)
    if req_vol: bull_veto, bear_veto = bull_veto | (c_vol_bull == 0), bear_veto | (c_vol_bear == 0)
    if req_vel: bull_veto, bear_veto = bull_veto | (c_vel_bull == 0), bear_veto | (c_vel_bear == 0)
    if req_rsi: bull_veto, bear_veto = bull_veto | (c_rsi_bull == 0), bear_veto | (c_rsi_bear == 0)
    if req_adx: bull_veto, bear_veto = bull_veto | (c_adx_bull == 0), bear_veto | (c_adx_bear == 0)
    if req_ema: bull_veto, bear_veto = bull_veto | (c_ema_bull == 0), bear_veto | (c_ema_bear == 0)
    if req_stoch: bull_veto, bear_veto = bull_veto | (c_stoch_bull == 0), bear_veto | (c_stoch_bear == 0)
    if req_atr_bb: bull_veto, bear_veto = bull_veto | (c_atr_bb_bull == 0), bear_veto | (c_atr_bb_bear == 0)
    if req_renko_bb: bull_veto, bear_veto = bull_veto | (c_renko_bb_bull == 0), bear_veto | (c_renko_bb_bear == 0)

    df[f"Armed_Bull_{tf_str}"] = (df[f"Score_Bull_{tf_str}"] >= min_score) & (~bull_veto)
    df[f"Armed_Bear_{tf_str}"] = (df[f"Score_Bear_{tf_str}"] >= min_score) & (~bear_veto)
    return df


def evaluate_single_timeframe_gates(df_base, tf_str):
    df_tf = (
        df_base.groupby(["Symbol", pd.Grouper(key="Datetime", freq=tf_str, closed="left", label="left")])
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .reset_index()
    )
    df_tf = df_tf.dropna(subset=["Close"]).sort_values(["Symbol", "Datetime"])
    df_tf = calculate_core_technicals(df_tf)

    df_tf = construct_45deg_renko_matrix(df_tf, tf_str, MACRO_RENKO_CONFIRM_BRICKS)
    df_tf = construct_volume_delta_renko_matrix(df_tf, tf_str, MACRO_RENKO_CONFIRM_BRICKS)
    df_tf = construct_renko_velocity_engine(df_tf, tf_str)
    df_tf = construct_bb_meta_pillars(df_tf, tf_str)
    df_tf = apply_dual_tier_scorecard(df_tf, tf_str, "MACRO")

    df_tf["Eval_Time"] = (df_tf["Datetime"] + pd.to_timedelta(tf_str)).astype("datetime64[ns]")

    export_cols = [
        "Symbol", "Eval_Time",
        f"Armed_Bull_{tf_str}", f"Armed_Bear_{tf_str}",
        f"Score_Bull_{tf_str}", f"Score_Bear_{tf_str}",
        f"Renko_Count_{tf_str}", f"Vol_Renko_Count_{tf_str}",
        f"Bars_Since_Brick_{tf_str}", "ATR", "ADX"
    ]
    env_df = df_tf[export_cols].copy().rename(columns={"Eval_Time": "Datetime", "ATR": f"ATR_{tf_str}", "ADX": f"ADX_{tf_str}"})
    return env_df.sort_values("Datetime").reset_index(drop=True)


# ==============================================================================
# 4. MICRO EXECUTION TAPE & CONFLUENCE MATCHER
# ==============================================================================
def prepare_unified_execution_tape(rolling_master_df, micro_tf, macro_timeframes, strategy_mode="BOTH"):
    if micro_tf != "1min":
        df_micro = (
            rolling_master_df.groupby(["Symbol", pd.Grouper(key="Datetime", freq=micro_tf, closed="left", label="left")])
            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
            .reset_index()
        )
        df_micro = df_micro.dropna(subset=["Close"]).sort_values(["Symbol", "Datetime"])
    else:
        df_micro = rolling_master_df.sort_values(["Symbol", "Datetime"]).copy()

    df_micro = calculate_core_technicals(df_micro)
    df_micro = construct_45deg_renko_matrix(df_micro, micro_tf, MICRO_RENKO_CONFIRM_BRICKS)
    df_micro = construct_volume_delta_renko_matrix(df_micro, micro_tf, MICRO_RENKO_CONFIRM_BRICKS)
    df_micro = construct_renko_velocity_engine(df_micro, micro_tf)
    df_micro = construct_bb_meta_pillars(df_micro, micro_tf)
    df_micro = apply_dual_tier_scorecard(df_micro, micro_tf, "MICRO")
    df_micro = df_micro.sort_values("Datetime").reset_index(drop=True)

    bull_gate_cols, bear_gate_cols = [], []
    for tf in macro_timeframes:
        print(f"   ├─ Evaluating Macro Context Gates + Price/Vol/Vel Renko for [{tf}]...")
        env_df = evaluate_single_timeframe_gates(rolling_master_df, tf)
        bull_col, bear_col = f"Armed_Bull_{tf}", f"Armed_Bear_{tf}"
        bull_gate_cols.append(bull_col)
        bear_gate_cols.append(bear_col)

        df_micro["Datetime"] = df_micro["Datetime"].astype("datetime64[ns]")
        env_df["Datetime"] = env_df["Datetime"].astype("datetime64[ns]")
        df_micro = pd.merge_asof(df_micro, env_df, on="Datetime", by="Symbol", direction="backward")
        df_micro[bull_col] = df_micro[bull_col].fillna(False)
        df_micro[bear_col] = df_micro[bear_col].fillna(False)
        df_micro[f"Score_Bull_{tf}"] = df_micro[f"Score_Bull_{tf}"].fillna(0).astype(int)
        df_micro[f"Score_Bear_{tf}"] = df_micro[f"Score_Bear_{tf}"].fillna(0).astype(int)
        df_micro[f"Renko_Count_{tf}"] = df_micro[f"Renko_Count_{tf}"].fillna(0).astype(int)
        df_micro[f"Vol_Renko_Count_{tf}"] = df_micro[f"Vol_Renko_Count_{tf}"].fillna(0).astype(int)

    df_micro["Master_Armed_Bull"] = df_micro[bull_gate_cols].any(axis=1)
    df_micro["Master_Armed_Bear"] = df_micro[bear_gate_cols].any(axis=1)

    if strategy_mode == "BULLISH":
        df_micro["Master_Armed_Bear"] = False
    elif strategy_mode == "BEARISH":
        df_micro["Master_Armed_Bull"] = False

    df_micro = df_micro.sort_values(["Symbol", "Datetime"]).reset_index(drop=True)

    df_micro["Trigger_Bull"] = df_micro["Master_Armed_Bull"] & df_micro[f"Armed_Bull_{micro_tf}"]
    df_micro["Trigger_Bear"] = df_micro["Master_Armed_Bear"] & df_micro[f"Armed_Bear_{micro_tf}"]

    df_micro["Trigger_Bull_Prev"] = df_micro.groupby("Symbol")["Trigger_Bull"].shift(1).fillna(False)
    df_micro["Trigger_Bear_Prev"] = df_micro.groupby("Symbol")["Trigger_Bear"].shift(1).fillna(False)

    df_micro["New_Bull"] = df_micro["Trigger_Bull"] & ~df_micro["Trigger_Bull_Prev"]
    df_micro["New_Bear"] = df_micro["Trigger_Bear"] & ~df_micro["Trigger_Bear_Prev"]
    df_micro["Direction"] = np.where(df_micro["New_Bull"], 1, np.where(df_micro["New_Bear"], -1, 0))

    return df_micro.sort_values("Datetime").reset_index(drop=True)


# ==============================================================================
# 5. TRADE MANAGEMENT & EXECUTION ENGINE
# ==============================================================================
def scan_institutional_tape(target_date_str, entry_cutoff_time_str=ENTRY_CUTOFF_TIME):
    print(f"\n📡 Initiating Pipeline Engine [{TRADING_MODE}] for {target_date_str} (Cutoff: {entry_cutoff_time_str})...")
    trading_days = get_past_trading_days(target_date_str, num_days=BACKTRACE_DAYS)
    if not trading_days: return

    target_dt = pd.to_datetime(target_date_str)
    cutoff_time_obj = pd.to_datetime(entry_cutoff_time_str).time()

    # ==========================================================================
    # MODE ROUTING: CASH EQUITIES VS F&O OPTIONS TRANSLATION
    # ==========================================================================
    if TRADING_MODE == "CASH_EQUITY":
        raw_universe = get_cash_equity_universe()
        if not raw_universe:
            print(f"{COLOR_RED}[Error] No cash equity instruments mapped.{COLOR_RESET}")
            return
        
        universe = filter_cash_equities_by_price_range(raw_universe, target_date_str)
        if not universe:
            print(f"{COLOR_YELLOW}[Terminal Silent] No equities matched the price range (₹{MIN_STOCK_PRICE} - ₹{MAX_STOCK_PRICE}).{COLOR_RESET}\n")
            return

        print(f"\n{COLOR_BOLD}── EXECUTING DIRECT CASH EQUITY SCAN ({len(universe)} stocks) ──{COLOR_RESET}")
        print(f"🚀 Multithreading Bulk Ingestion for {len(universe)} symbols...")
        fetch_tasks = [(item, trading_days[0], target_date_str) for item in universe]
        stock_dfs = []

        def stock_fetch_worker(task):
            item, start_date, end_date = task
            df = fetch_fyers_candles(item["key"], start_date, end_date, resolution="1")
            if df is None or df.empty:
                return None
            df = df.drop_duplicates(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)
            df["Symbol"] = item["symbol"]
            return df

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(stock_fetch_worker, task): task for task in fetch_tasks}
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                completed += 1
                sys.stdout.write(f"\r📡 Fetching Stock Data... {completed}/{len(fetch_tasks)} symbols processed")
                sys.stdout.flush()
                res = future.result()
                if res is not None: stock_dfs.append(res)
        print()

        if not stock_dfs:
            print(f"{COLOR_RED}No historical stock data retrieved.{COLOR_RESET}")
            return

        stock_master_df = pd.concat(stock_dfs, ignore_index=True)
        print("⚙️ Computing 9-Pillar Scorecards & Velocity Matrices on CASH EQUITIES...")
        tape_exec = prepare_unified_execution_tape(stock_master_df, MICRO_TIMEFRAME, MACRO_TIMEFRAMES, strategy_mode=GLOBAL_MACRO_STRATEGY_2D)

    else:
        # F&O Options Translation Mode
        universe, options_by_underlying = get_fno_universe_and_options()
        if not universe or not options_by_underlying:
            print(f"{COLOR_RED}[Error] F&O universe or options chain data unavailable.{COLOR_RESET}")
            return

        qualifying_symbols = []
        spot_ref = {}

        if ENABLE_STAGE1_STOCK_FILTER:
            print(f"\n{COLOR_BOLD}── STAGE 1: STOCK-LEVEL CONFLUENCE SCAN ({len(universe)} F&O stocks) ──{COLOR_RESET}")
            print(f"🚀 Multithreading Bulk Ingestion for {len(universe)} symbols...")
            fetch_tasks = [(item, trading_days[0], target_date_str) for item in universe]
            stock_dfs = []

            def stock_fetch_worker(task):
                item, start_date, end_date = task
                df = fetch_fyers_candles(item["key"], start_date, end_date, resolution="1")
                if df is None or df.empty:
                    return None
                df = df.drop_duplicates(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)
                df["Symbol"] = item["symbol"]
                return df

            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(stock_fetch_worker, task): task for task in fetch_tasks}
                completed = 0
                for future in concurrent.futures.as_completed(futures):
                    completed += 1
                    sys.stdout.write(f"\r📡 Fetching Stock Data... {completed}/{len(fetch_tasks)} symbols processed")
                    sys.stdout.flush()
                    res = future.result()
                    if res is not None: stock_dfs.append(res)
            print()

            if not stock_dfs:
                print(f"{COLOR_RED}No historical stock data retrieved.{COLOR_RESET}")
                return

            stock_master_df = pd.concat(stock_dfs, ignore_index=True)
            print("⚙️ Computing 9-Pillar Scorecards & Velocity Matrices on STOCK charts...")
            tape_exec_stock = prepare_unified_execution_tape(stock_master_df, MICRO_TIMEFRAME, MACRO_TIMEFRAMES, strategy_mode=GLOBAL_MACRO_STRATEGY_2D)

            stock_anomalies = tape_exec_stock[
                (tape_exec_stock["Direction"] != 0) & (tape_exec_stock["Datetime"].dt.time <= cutoff_time_obj)
            ].copy()

            if stock_anomalies.empty:
                print(f"\n{COLOR_YELLOW}[Terminal Silent] STAGE 1: No stocks qualified today under the strict confluence filter.{COLOR_RESET}\n")
                return

            qualifying_symbols = sorted(stock_anomalies["Symbol"].unique().tolist())
            spot_ref = stock_anomalies.sort_values("Datetime").groupby("Symbol")["Close"].last().to_dict()

            print(f"{COLOR_GREEN}✅ STAGE 1 complete: {len(qualifying_symbols)}/{len(universe)} stocks qualified "
                  f"(Basket eligible): {', '.join(qualifying_symbols)}{COLOR_RESET}")
        else:
            print(f"\n{COLOR_BOLD}── STAGE 1 BYPASSED: Direct Option Scan for ALL {len(universe)} F&O Stocks ──{COLOR_RESET}")
            print("🚀 Fetching spot reference prices across entire F&O Universe to center ATM strikes...")
            spot_ref = fetch_all_spot_reference_prices(universe, target_date_str)
            qualifying_symbols = [item["symbol"] for item in universe if item["symbol"] in spot_ref]
            print(f"  ├─ Successfully locked ATM centers for {len(qualifying_symbols)} symbols.")

        print(f"\n{COLOR_BOLD}── STAGE 2: OPTION-LEVEL SCAN (ATM ±{STRIKE_RANGE_OFFSET} strikes) ──{COLOR_RESET}")
        candidate_contracts = []
        for sym in qualifying_symbols:
            sp = spot_ref.get(sym)
            if sp: candidate_contracts.extend(build_strike_range(sym, sp, options_by_underlying, target_date_str, STRIKE_RANGE_OFFSET))

        if not candidate_contracts:
            print(f"{COLOR_YELLOW}[Terminal Silent] No option chain coverage found.{COLOR_RESET}\n")
            return

        liquid_contracts = filter_liquid_contracts(candidate_contracts, target_date_str)
        if not liquid_contracts:
            print(f"{COLOR_YELLOW}[Terminal Silent] No option contracts passed liquidity filter.{COLOR_RESET}\n")
            return

        option_dfs = []
        for c in liquid_contracts:
            df = fetch_fyers_candles(c["key"], trading_days[0], target_date_str, resolution="1")
            if df is not None and not df.empty:
                df["Symbol"] = c["symbol"]
                option_dfs.append(df)

        option_master_df = pd.concat(option_dfs, ignore_index=True)
        tape_exec = prepare_unified_execution_tape(option_master_df, MICRO_TIMEFRAME, MACRO_TIMEFRAMES, strategy_mode=OPTIONS_STRATEGY_2D)

    memory_bank = _run_dual_layer_trade_management(tape_exec, MICRO_TIMEFRAME, MACRO_TIMEFRAMES, cutoff_time_obj)

    today_master = tape_exec[tape_exec["Datetime"].dt.date == target_dt.date()]
    if today_master.empty:
        print(f"\n{COLOR_YELLOW}[Terminal Standby] Market data for {target_date_str} is empty.{COLOR_RESET}\n")
        return
    final_ltp_dict = today_master.groupby("Symbol")["Close"].last().to_dict()

    # ==========================================================================
    # FINAL OUTPUT: BASKET 1 & BASKET 2 
    # ==========================================================================
    active_runners = []
    closed_trades = []
    for sym, episodes in memory_bank.items():
        for st in episodes:
            if st["state"] == "ACTIVE":
                active_runners.append({**st, "sym": sym})
            elif st["state"] == "EXITED" and st["exit_time"] and st["exit_time"].startswith(target_date_str):
                closed_trades.append({**st, "sym": sym})
    closed_trades.sort(key=lambda x: (x["sym"], x["time"]))

    tf_display_str = " | ".join(MACRO_TIMEFRAMES)
    print(f"\n{COLOR_CYAN}================================================================================================{COLOR_RESET}")
    print(f"{COLOR_BOLD}9-PILLAR ENGINE [{MICRO_TIMEFRAME} Micro ⚡ Macro: {tf_display_str}] — RESULTS [{TRADING_MODE}]{COLOR_RESET}")
    print(f"{COLOR_CYAN}================================================================================================{COLOR_RESET}\n")

    if active_runners:
        print(f"{COLOR_BOLD}🟢 BASKET 1: ACTIVE RUNNERS (Riding the Trend){COLOR_RESET}")
        for st in active_runners:
            ltp = final_ltp_dict.get(st["sym"], st["origin"])
            pnl_pct = ((ltp - st["origin"]) / st["origin"]) * 100
            color = COLOR_GREEN if pnl_pct >= 0 else COLOR_RED
            d_str = "BULLISH" if st["dir"] == 1 else "BEARISH"

            print(f"  {color}⚡ {st['sym']:<26} Open P&L: {pnl_pct:+.2f}% ({d_str}){COLOR_RESET}")
            print(f"      └─ ⚓ Qualifying Macro TFs        : {', '.join(st['triggering_macro_tfs'])}")
            print(f"      └─ 🔫 Micro Execution [{MICRO_TIMEFRAME}] : Score >= {MICRO_MINIMUM_SCORE}/9 (Score={st['micro_score']})")
            print(f"      └─ ⚓ True Birth Anchor           : {st['date']} @ {st['time']} | Price: ₹{st['origin']:.2f}")
            print(f"      └─ 🎯 Latest Price               : {target_date_str} @ EOD   | Price: ₹{ltp:.2f}\n")

    if closed_trades:
        print(f"{COLOR_BOLD}🛑 BASKET 2: CLOSED TRADES (Renko Structure Broken / Stagnation){COLOR_RESET}")
        for st in closed_trades:
            pnl_pct = ((st["exit_price"] - st["origin"]) / st["origin"]) * 100
            color = COLOR_GREEN if pnl_pct >= 0 else COLOR_RED
            d_str = "BULLISH" if st["dir"] == 1 else "BEARISH"

            print(f"  {color}🛑 {st['sym']:<26} Final P&L: {pnl_pct:+.2f}% ({d_str}){COLOR_RESET}")
            print(f"      └─ ⚓ Qualifying Macro TFs        : {', '.join(st['triggering_macro_tfs'])}")
            print(f"      └─ 🔫 Micro Execution [{MICRO_TIMEFRAME}] : Score >= {MICRO_MINIMUM_SCORE}/9 (Score={st['micro_score']})")
            print(f"      └─ ⚓ True Birth Anchor           : {st['date']} @ {st['time']} | Price: ₹{st['origin']:.2f}")
            print(f"      └─ 🎯 Exit Time & Price           : {st['exit_time']} | Price: ₹{st['exit_price']:.2f}")
            print(f"      └─ 📉 Reason                      : {st['exit_reason']}\n")

    if not active_runners and not closed_trades:
        print(f"{COLOR_DIM}[Terminal Silent] No instruments triggered micro+macro conditions today.{COLOR_RESET}\n")


def _run_dual_layer_trade_management(tape_exec, micro_timeframe, macro_timeframes, cutoff_time_obj):
    all_anomalies = tape_exec[tape_exec["Direction"] != 0].copy()
    anomalies_by_time = all_anomalies.groupby("Datetime")

    closes_dict = tape_exec.set_index(["Datetime", "Symbol"])["Close"].to_dict()
    micro_price_renko = tape_exec.set_index(["Datetime", "Symbol"])[f"Renko_Count_{micro_timeframe}"].to_dict()
    micro_vol_renko = tape_exec.set_index(["Datetime", "Symbol"])[f"Vol_Renko_Count_{micro_timeframe}"].to_dict()
    micro_vel_bars = tape_exec.set_index(["Datetime", "Symbol"])[f"Bars_Since_Brick_{micro_timeframe}"].to_dict()
    macro_price_renkos = {tf: tape_exec.set_index(["Datetime", "Symbol"])[f"Renko_Count_{tf}"].to_dict() for tf in macro_timeframes}
    macro_vol_renkos = {tf: tape_exec.set_index(["Datetime", "Symbol"])[f"Vol_Renko_Count_{tf}"].to_dict() for tf in macro_timeframes}

    all_times = np.sort(tape_exec["Datetime"].unique())
    memory_bank = {}

    for t in all_times:
        t_dt = pd.to_datetime(t)

        for sym, episodes in memory_bank.items():
            if not episodes:
                continue
            st = episodes[-1]
            if st["state"] == "ACTIVE":
                ltp = closes_dict.get((t_dt, sym))
                mi_p_count = micro_price_renko.get((t_dt, sym), 0)
                mi_v_count = micro_vol_renko.get((t_dt, sym), 0)
                mi_bars_stalled = micro_vel_bars.get((t_dt, sym), 0)

                if ltp is not None:
                    exit_reason = None
                    if mi_bars_stalled > RENKO_VELOCITY_MAX_BARS:
                        exit_reason = f"Velocity Stall (No brick in {RENKO_VELOCITY_MAX_BARS} bars)"

                    if not exit_reason:
                        if st["dir"] == 1:
                            if mi_p_count <= -MICRO_EXIT_PRICE_BRICKS: exit_reason = "Micro Price Reversal"
                            elif mi_v_count <= -MICRO_EXIT_VOL_BRICKS: exit_reason = "Micro Volume Reversal"
                            else:
                                for tf in st["triggering_macro_tfs"]:
                                    ma_p = macro_price_renkos[tf].get((t_dt, sym), 0)
                                    ma_v = macro_vol_renkos[tf].get((t_dt, sym), 0)
                                    if ma_p <= -MACRO_EXIT_PRICE_BRICKS:
                                        exit_reason = f"Macro [{tf}] Price Break"; break
                                    if ma_v <= -MACRO_EXIT_VOL_BRICKS:
                                        exit_reason = f"Macro [{tf}] Volume Break"; break
                        elif st["dir"] == -1:
                            if mi_p_count >= MICRO_EXIT_PRICE_BRICKS: exit_reason = "Micro Price Reversal"
                            elif mi_v_count >= MICRO_EXIT_VOL_BRICKS: exit_reason = "Micro Volume Reversal"
                            else:
                                for tf in st["triggering_macro_tfs"]:
                                    ma_p = macro_price_renkos[tf].get((t_dt, sym), 0)
                                    ma_v = macro_vol_renkos[tf].get((t_dt, sym), 0)
                                    if ma_p >= MACRO_EXIT_PRICE_BRICKS:
                                        exit_reason = f"Macro [{tf}] Price Break"; break
                                    if ma_v >= MACRO_EXIT_VOL_BRICKS:
                                        exit_reason = f"Macro [{tf}] Volume Break"; break

                    if exit_reason:
                        st["state"] = "EXITED"
                        st["exit_time"] = t_dt.strftime("%Y-%m-%d %H:%M")
                        st["exit_price"] = ltp
                        st["exit_reason"] = exit_reason

        if t_dt in anomalies_by_time.groups and t_dt.time() <= cutoff_time_obj:
            for _, row in anomalies_by_time.get_group(t_dt).iterrows():
                sym = row["Symbol"]
                direction = row["Direction"]

                existing = memory_bank.get(sym, [])
                if existing and existing[-1]["state"] == "ACTIVE":
                    continue

                triggered_m_tfs = []
                for tf in macro_timeframes:
                    armed_col = f"Armed_Bull_{tf}" if direction == 1 else f"Armed_Bear_{tf}"
                    if row.get(armed_col, False):
                        triggered_m_tfs.append(tf)

                new_episode = {
                    "state": "ACTIVE",
                    "origin": row["Close"],
                    "date": t_dt.strftime("%Y-%m-%d"),
                    "time": t_dt.strftime("%H:%M"),
                    "dir": direction,
                    "exit_time": None,
                    "exit_price": None,
                    "exit_reason": None,
                    "triggering_macro_tfs": triggered_m_tfs,
                    "macro_scores": {tf: row.get(f"Score_Bull_{tf}" if direction == 1 else f"Score_Bear_{tf}", 0) for tf in macro_timeframes},
                    "micro_score": row.get(f"Score_Bull_{micro_timeframe}" if direction == 1 else f"Score_Bear_{micro_timeframe}", 0)
                }
                memory_bank.setdefault(sym, []).append(new_episode)

        if t_dt.hour == 15 and t_dt.minute >= 15:
            for sym, episodes in memory_bank.items():
                if episodes and episodes[-1]["state"] == "ACTIVE":
                    st = episodes[-1]
                    st["state"] = "EXITED"
                    st["exit_time"] = t_dt.strftime("%Y-%m-%d %H:%M") + " (EOD)"
                    st["exit_price"] = closes_dict.get((t_dt, sym), st["origin"])
                    st["exit_reason"] = "End of Day Market Close"

    return memory_bank


def run_production_sweep():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--date", type=str, default="")
    parser.add_argument("-t", "--time", type=str, default="")
    args, _ = parser.parse_known_args()
    
    raw_date = args.date or os.environ.get("PARAM_BACKTEST_DATE", "").strip()
    raw_time = args.time or os.environ.get("PARAM_BACKTEST_TIME", "").strip()

    print(f"\n⚙️ Raw Input Detected -> Date: '{raw_date}', Time: '{raw_time}'")

    # ==========================================
    # ABSOLUTE BULLETPROOF DATE/TIME CLEANING
    # ==========================================
    if raw_date:
        # Standardize any separators to spaces
        raw_date = raw_date.replace("T", " ")
        # If there's a space, aggressively split the time out of the date string
        if " " in raw_date:
            parts = raw_date.split()
            raw_date = parts[0]
            # If no time was explicitly provided, steal it from the date string
            if not raw_time and len(parts) > 1:
                raw_time = parts[1]
        
        # Hard lock the date string to exactly 10 characters (YYYY-MM-DD)
        raw_date = raw_date[:10]  

    if raw_time:
        raw_time = raw_time.replace(".", ":").strip()
        raw_time = raw_time[:5]  # Hard lock to 5 chars (HH:MM)

    print(f"⚙️ Cleaned Input      -> Date: '{raw_date}', Time: '{raw_time}'")
    # ==========================================

    if not raw_date:
        target_dt = datetime.utcnow() + timedelta(hours=5, minutes=30)
        if target_dt.weekday() == 5: target_dt -= timedelta(days=1)
        elif target_dt.weekday() == 6: target_dt -= timedelta(days=2)
        target_date_str = target_dt.strftime("%Y-%m-%d")
    else:
        # This will NEVER crash now because raw_date is guaranteed to be "YYYY-MM-DD"
        target_date_str = datetime.strptime(raw_date, "%Y-%m-%d").strftime("%Y-%m-%d")

    cutoff_time_str = raw_time if raw_time else ENTRY_CUTOFF_TIME

    if not validate_fyers_token():
        return

    scan_institutional_tape(target_date_str, cutoff_time_str)


if __name__ == "__main__":
    run_production_sweep()

