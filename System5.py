"""system3.py - Asit Baran Pati Multi-Timeframe Trading System Implementation (Options Edition)

Production-Grade Universal N-Timeframe & Dual-Tier 45-Degree Renko Engine
- Target: F&O Stocks Only (Indices Excluded)
- Execution: Direct Options Premium Charting (Method A)
- Configurable Expiry & Strike Offset Matrices
- TWO-STAGE INGESTION: Ultra-Fast Previous Day Volume & Premium Pre-Filtering
- Aggressive Multithreading with HTTP 429 (Rate Limit) Evasion
"""

import argparse
import concurrent.futures
import gzip
import io
import json
import os
import random
import sys
import time
import urllib.parse
import warnings
from datetime import datetime as dt, timedelta

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. ENGINE CONSTANTS & TERMINAL COLORS
# ==============================================================================
COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_CYAN = "\033[96m"
COLOR_YELLOW = "\033[93m"
COLOR_DIM = "\033[2m"
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"

# ==============================================================================
# ★ GLOBAL CONFIGURATION: OPTIONS, TIMEFRAMES & INDICATORS ★
# ==============================================================================
STRIKE_RANGE_OFFSET = 1      
TARGET_EXPIRY = "CURRENT"    
BACKTRACE_DAYS = 2          

MIN_OPT_PREMIUM = 30.0        
MIN_PREV_DAY_VOLUME = 800000  

MAX_API_WORKERS = 25         # Safely optimized for Upstox limits

MICRO_TIMEFRAME = "1min"
MACRO_TIMEFRAMES = ["20min"]

ATR_PERIOD = 14
RSI_PERIOD = 14
BB_SMA_PERIOD = 20
BB_STD_DEV = 2.0
ADX_PERIOD = 14
ADX_THRESHOLD = 20
STOCH_PERIOD = 14

MICRO_RENKO_CONFIRM_BRICKS = 1
MACRO_RENKO_CONFIRM_BRICKS = 1
RENKO_MIN_BRICK = 0.05
RENKO_DEFAULT_PCT = 0.05     

GLOBAL_MACRO_STRATEGY_2D = "BOTH" 

# ==============================================================================
# 🎛️ TIER 1 & 2: DUAL-TIER SWITCHBOARDS - 7 PILLARS
# ==============================================================================
MACRO_MANDATORY_PRICE_RENKO    = True
MACRO_MANDATORY_VOL_RENKO      = True
MACRO_MANDATORY_RENKO_VELOCITY = False
MACRO_MANDATORY_RSI_BB         = False
MACRO_MANDATORY_ADX_DMI        = False
MACRO_MANDATORY_EMA_SPREAD     = False
MACRO_MANDATORY_STOCHASTIC     = False
MACRO_MINIMUM_SCORE            = 2      

SYNC_MICRO_WITH_MACRO          = False  

MICRO_MANDATORY_PRICE_RENKO    = True
MICRO_MANDATORY_VOL_RENKO      = True
MICRO_MANDATORY_RENKO_VELOCITY = False
MICRO_MANDATORY_RSI_BB         = True
MICRO_MANDATORY_ADX_DMI        = False
MICRO_MANDATORY_EMA_SPREAD     = False
MICRO_MANDATORY_STOCHASTIC     = False
MICRO_MINIMUM_SCORE            = 2      

# ==============================================================================
# 🎛️ TIER 3: TRADE MANAGEMENT & TEMPORAL GATES
# ==============================================================================
MICRO_EXIT_PRICE_BRICKS = 5  
MICRO_EXIT_VOL_BRICKS   = 5  
MACRO_EXIT_PRICE_BRICKS = 1
MACRO_EXIT_VOL_BRICKS   = 1  

RENKO_VELOCITY_MAX_BARS = 12 
ENTRY_CUTOFF_TIME = "15:00"

EXCLUDED_INDICES = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX", "BANKEX", "NIFTYNXT50"}


# ==============================================================================
# 1. LIVE INGESTION: OPTIONS MATRIX & FAST PRE-FILTERING
# ==============================================================================
def fetch_json_gz(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Accept': 'application/json, text/plain, */*',
    }
    try:
        print(f"  ├─ ⏳ Downloading massive instrument file (this may take 10-20 seconds)...")
        response = requests.get(url, headers=headers, timeout=45) 
        response.raise_for_status()
        
        print(f"  ├─ ✅ Download successful! Extracting & parsing JSON into memory...")
        
        # Safely check if the response is actually GZIP compressed via Magic Bytes (0x1F, 0x8B)
        if response.content[:2] == b'\x1f\x8b':
            data = json.load(gzip.GzipFile(fileobj=io.BytesIO(response.content)))
        else:
            # If requests already auto-decompressed it in the background
            data = response.json()
            
        print(f"  ├─ ✅ Parsing complete! Total instruments loaded: {len(data)}")
        return data
    except requests.exceptions.Timeout:
        print(f"{COLOR_RED}[Error] Connection timed out! The file is too large for the current network speed.{COLOR_RESET}")
    except Exception as e: 
        print(f"{COLOR_RED}[Error] Exception fetching {url}: {e}{COLOR_RESET}")
    return []

def get_options_universe():
    print(f"📡 Fetching Master Instrument Matrix (Upstox Complete File)...")
    master_data = fetch_json_gz("https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz")

    if not master_data:
        print(f"{COLOR_RED}[Error] Failed to fetch master instruments.{COLOR_RESET}")
        return [], []

    # 1. Map all Cash (Equity) stocks. Indices (like NIFTY) don't have an NSE_EQ entry.
    eq_stocks = {}
    for item in master_data:
        # Check both 'exchange' and 'segment' to counter Upstox API variations
        if item.get("exchange") == "NSE_EQ" or item.get("segment") == "NSE_EQ":
            # Check both 'tradingsymbol' and 'trading_symbol'
            sym = item.get("tradingsymbol") or item.get("trading_symbol")
            if sym and sym not in EXCLUDED_INDICES:
                eq_stocks[sym] = item

    # 2. Map F&O contracts to their verified NSE_EQ cash equivalents
    options_instruments = []
    valid_fo_names = set()
    
    for item in master_data:
        inst_type = item.get("instrument_type")
        
        # Capture CE/PE and OPTSTK formats
        if inst_type in ("OPTSTK", "CE", "PE"):
            # Upstox stores the underlying ticker in 'name' or 'underlying_symbol' for F&O
            underlying = item.get("name") or item.get("underlying_symbol")
            
            # If the underlying exists in our Cash market map, it's a valid F&O stock
            if underlying in eq_stocks:
                options_instruments.append(item)
                valid_fo_names.add(underlying)
                
    # 3. Retrieve the final Spot mappings based ONLY on valid options found
    spot_instruments = [eq_stocks[name] for name in valid_fo_names]

    print(f"  ├─ 🔍 Found {len(valid_fo_names)} Pure F&O Underlying Stocks (Indices completely isolated).")
    print(f"  ├─ 🎯 Mapped {len(spot_instruments)} Spot Instruments & {len(options_instruments)} Options Contracts.")

    return spot_instruments, options_instruments



def safe_api_request(url, headers, is_live=False):
    for attempt in range(5):
        try:
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code == 200:
                return res.json().get("data", {}).get("candles", [])
            elif res.status_code == 429:
                time.sleep(random.uniform(0.5, 2.0) * (attempt + 1))  
            else:
                break
        except Exception:
            time.sleep(1)
    return []

def get_latest_close_price(key, headers):
    # Try intraday first
    data = safe_api_request(f"https://api.upstox.com/v2/historical-candle/intraday/{key}/1minute", headers, is_live=True)
    if data: return data[0][4]
    
    # Fallback to historical (weekend/holiday safety)
    today = dt.utcnow().strftime("%Y-%m-%d")
    week_ago = (dt.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
    data = safe_api_request(f"https://api.upstox.com/v2/historical-candle/{key}/day/{today}/{week_ago}", headers)
    if data: return data[0][4]
    return None

def fetch_latest_spot_prices(spot_instruments, headers):
    print(f"🚀 Fetching Spot Prices for {len(spot_instruments)} Underlyings...")
    spot_prices = {}
    
    def worker(inst):
        # Safely capture the symbol accounting for both Upstox spelling variations
        sym = inst.get("tradingsymbol") or inst.get("trading_symbol", "UNKNOWN")
        try:
            raw_key = inst.get("instrument_key")
            if not raw_key:
                return sym, None
                
            key = urllib.parse.quote(raw_key)
            price = get_latest_close_price(key, headers)
            return sym, price
        except Exception:
            # Safely return the parsed symbol instead of strictly calling inst["tradingsymbol"]
            return sym, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_API_WORKERS) as executor:
        futures = {executor.submit(worker, inst): inst for inst in spot_instruments}
        for future in concurrent.futures.as_completed(futures):
            sym, price = future.result()
            # Only map the spot price if we successfully fetched it and recognized the symbol
            if price and sym != "UNKNOWN": 
                spot_prices[sym] = price
            
    return spot_prices



def build_options_matrix(spot_prices, options_instruments):
    print(f"⚙️ Building Options Contract Matrix (Offset: ±{STRIKE_RANGE_OFFSET}, Expiry: {TARGET_EXPIRY})...")
    grouped_options = {}
    for opt in options_instruments:
        grouped_options.setdefault(opt.get("name"), []).append(opt)

    target_contracts = []
    today = dt.utcnow().date()
    
    for symbol, spot_price in spot_prices.items():
        opts = grouped_options.get(symbol, [])
        if not opts: continue
        
        for o in opts:
            raw_strike = o.get("strike_price") if "strike_price" in o else o.get("strike")
            try:
                o["_normalized_strike"] = float(raw_strike)
            except (ValueError, TypeError):
                o["_normalized_strike"] = None

        # Ignore mathematically expired contracts if backtesting heavily
        valid_opts = [o for o in opts if o.get("expiry") and o.get("_normalized_strike") is not None]
        if not valid_opts: continue
        
        expiries = sorted(list(set(pd.to_datetime(o["expiry"]).date() for o in valid_opts if pd.to_datetime(o["expiry"]).date() >= today)))
        if not expiries: continue
        
        chosen_expiry = expiries[0] if TARGET_EXPIRY == "CURRENT" else (expiries[1] if len(expiries) > 1 else expiries[0])
        expiry_opts = [o for o in valid_opts if pd.to_datetime(o["expiry"]).date() == chosen_expiry]
        
        unique_strikes = sorted(list(set(o["_normalized_strike"] for o in expiry_opts)))
        if not unique_strikes: continue
        
        closest_strike = min(unique_strikes, key=lambda x: abs(x - spot_price))
        atm_idx = unique_strikes.index(closest_strike)
        
        start_idx = max(0, atm_idx - STRIKE_RANGE_OFFSET)
        end_idx = min(len(unique_strikes), atm_idx + STRIKE_RANGE_OFFSET + 1)
        selected_strikes = unique_strikes[start_idx:end_idx]
        
        final_opts = [o for o in expiry_opts if o["_normalized_strike"] in selected_strikes]
        for opt in final_opts:
            target_contracts.append({
                "symbol": opt.get("tradingsymbol", opt.get("trading_symbol", "UNKNOWN")),
                "key": opt["instrument_key"],
                "underlying": symbol,
                "type": opt.get("instrument_type")
            })
            
    return target_contracts

def filter_liquid_options(target_contracts, headers, target_date_str):
    print(f"\n🧹 STAGE 1 INGESTION: Pre-Filtering {len(target_contracts)} contracts...")
    print(f"  ├─ Rules: Prev. Day Close >= ₹{MIN_OPT_PREMIUM} | Prev. Day Vol >= {MIN_PREV_DAY_VOLUME}")
    
    target_dt = dt.strptime(target_date_str, "%Y-%m-%d")
    prev_day = (target_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    fifteen_days_ago = (target_dt - timedelta(days=15)).strftime("%Y-%m-%d")
    
    filtered_contracts = []
    
    def worker(contract):
        try:
            key = urllib.parse.quote(contract["key"])
            url = f"https://api.upstox.com/v2/historical-candle/{key}/day/{prev_day}/{fifteen_days_ago}"
            data = safe_api_request(url, headers)
            
            if data and len(data) > 0:
                latest_candle = data[0]
                close_price = latest_candle[4]
                volume = latest_candle[5]
                
                if close_price >= MIN_OPT_PREMIUM and volume >= MIN_PREV_DAY_VOLUME:
                    return contract
        except Exception:
            pass
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_API_WORKERS) as executor:
        futures = {executor.submit(worker, c): c for c in target_contracts}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            sys.stdout.write(f"\r  ├─ Checking Liquidity... {completed}/{len(target_contracts)} processed")
            sys.stdout.flush()
            res = future.result()
            if res:
                filtered_contracts.append(res)
                
    print(f"\n  ├─ ✅ Pre-Filter Complete: {len(filtered_contracts)} highly liquid contracts passed.")
    return filtered_contracts

def get_past_trading_days(target_date_str, num_days=20):
    target_dt = dt.strptime(target_date_str, "%Y-%m-%d")
    trading_days = []
    current_dt = target_dt
    while len(trading_days) < num_days:
        if current_dt.weekday() < 5:
            trading_days.append(current_dt.strftime("%Y-%m-%d"))
        current_dt -= timedelta(days=1)
    trading_days.reverse()
    return trading_days


# ==============================================================================
# 2. CORE TECHNICAL & 45-DEGREE RENKO ENGINES (DIRECT ON PREMIUMS)
# ==============================================================================
def calculate_core_technicals(df_tf):
    df_tf["H-L"] = df_tf["High"] - df_tf["Low"]
    df_tf["H-PC"] = (df_tf["High"] - df_tf.groupby("Symbol")["Close"].shift(1)).abs()
    df_tf["L-PC"] = (df_tf["Low"] - df_tf.groupby("Symbol")["Close"].shift(1)).abs()
    df_tf["TR"] = df_tf[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df_tf["ATR"] = df_tf.groupby("Symbol")["TR"].transform(lambda x: x.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean())
    df_tf["ATR"] = df_tf["ATR"].fillna(df_tf["Close"] * RENKO_DEFAULT_PCT)

    delta = df_tf.groupby("Symbol")["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.groupby(df_tf["Symbol"]).transform(lambda x: x.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean())
    avg_loss = loss.groupby(df_tf["Symbol"]).transform(lambda x: x.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean())
    df_tf["RSI"] = 100 - (100 / (1 + (avg_gain / (avg_loss + 1e-8))))
    df_tf["RSI_SMA"] = df_tf.groupby("Symbol")["RSI"].transform(lambda x: x.rolling(BB_SMA_PERIOD, min_periods=1).mean())

    high_d = df_tf["High"] - df_tf.groupby("Symbol")["High"].shift(1)
    low_d = df_tf.groupby("Symbol")["Low"].shift(1) - df_tf["Low"]
    df_tf["+DM"] = np.where((high_d > low_d) & (high_d > 0), high_d, 0)
    df_tf["-DM"] = np.where((low_d > high_d) & (low_d > 0), low_d, 0)

    df_tf["+DI"] = (100 * (df_tf.groupby("Symbol")["+DM"].transform(lambda x: x.ewm(alpha=1 / ADX_PERIOD, adjust=False).mean()) / (df_tf["ATR"] + 1e-8)))
    df_tf["-DI"] = (100 * (df_tf.groupby("Symbol")["-DM"].transform(lambda x: x.ewm(alpha=1 / ADX_PERIOD, adjust=False).mean()) / (df_tf["ATR"] + 1e-8)))
    df_tf["DX"] = (100 * abs(df_tf["+DI"] - df_tf["-DI"]) / (df_tf["+DI"] + df_tf["-DI"] + 1e-8))
    df_tf["ADX"] = df_tf.groupby("Symbol")["DX"].transform(lambda x: x.ewm(alpha=1 / ADX_PERIOD, adjust=False).mean())

    df_tf["EMA_8"] = df_tf.groupby("Symbol")["Close"].transform(lambda x: x.ewm(span=8, adjust=False).mean())
    df_tf["EMA_21"] = df_tf.groupby("Symbol")["Close"].transform(lambda x: x.ewm(span=21, adjust=False).mean())
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
    df['Vol_SMA_20'] = df.groupby('Symbol')['Volume'].transform(lambda x: x.rolling(20, min_periods=1).mean()).fillna(100)
    
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


# ==============================================================================
# 3. DUAL-TIER SCORECARD SYSTEM (7 PILLARS)
# ==============================================================================
def apply_dual_tier_scorecard(df, tf_str, tier_type):
    if SYNC_MICRO_WITH_MACRO and tier_type == "MICRO":
        req_price, req_vol = MACRO_MANDATORY_PRICE_RENKO, MACRO_MANDATORY_VOL_RENKO
        req_vel = MACRO_MANDATORY_RENKO_VELOCITY
        req_rsi, req_adx = MACRO_MANDATORY_RSI_BB, MACRO_MANDATORY_ADX_DMI
        req_ema, req_stoch = MACRO_MANDATORY_EMA_SPREAD, MACRO_MANDATORY_STOCHASTIC
        min_score = MACRO_MINIMUM_SCORE
    else:
        req_price = globals()[f"{tier_type}_MANDATORY_PRICE_RENKO"]
        req_vol = globals()[f"{tier_type}_MANDATORY_VOL_RENKO"]
        req_vel = globals()[f"{tier_type}_MANDATORY_RENKO_VELOCITY"]
        req_rsi = globals()[f"{tier_type}_MANDATORY_RSI_BB"]
        req_adx = globals()[f"{tier_type}_MANDATORY_ADX_DMI"]
        req_ema = globals()[f"{tier_type}_MANDATORY_EMA_SPREAD"]
        req_stoch = globals()[f"{tier_type}_MANDATORY_STOCHASTIC"]
        min_score = globals()[f"{tier_type}_MINIMUM_SCORE"]

    c_price_bull, c_price_bear = df[f"Renko_Bull_{tf_str}"].astype(int), df[f"Renko_Bear_{tf_str}"].astype(int)
    c_vol_bull, c_vol_bear = df[f"Vol_Renko_Bull_{tf_str}"].astype(int), df[f"Vol_Renko_Bear_{tf_str}"].astype(int)
    c_vel_bull, c_vel_bear = df[f"Velocity_Bull_{tf_str}"].astype(int), df[f"Velocity_Bear_{tf_str}"].astype(int)
    c_rsi_bull, c_rsi_bear = (df["RSI"] >= df["RSI_SMA"]).astype(int), (df["RSI"] <= df["RSI_SMA"]).astype(int)
    c_adx_bull, c_adx_bear = ((df["ADX"] >= ADX_THRESHOLD) & (df["+DI"] > df["-DI"])).astype(int), ((df["ADX"] >= ADX_THRESHOLD) & (df["-DI"] > df["+DI"])).astype(int)
    c_ema_bull, c_ema_bear = df["EMA_Bull_Expanded"].astype(int), df["EMA_Bear_Expanded"].astype(int)
    c_stoch_bull, c_stoch_bear = df["Stoch_Bull_Pass"].astype(int), df["Stoch_Bear_Pass"].astype(int)

    df[f"Score_Bull_{tf_str}"] = c_price_bull + c_vol_bull + c_vel_bull + c_rsi_bull + c_adx_bull + c_ema_bull + c_stoch_bull
    df[f"Score_Bear_{tf_str}"] = c_price_bear + c_vol_bear + c_vel_bear + c_rsi_bear + c_adx_bear + c_ema_bear + c_stoch_bear

    bull_veto, bear_veto = pd.Series(False, index=df.index), pd.Series(False, index=df.index)
    if req_price: bull_veto, bear_veto = bull_veto | (c_price_bull == 0), bear_veto | (c_price_bear == 0)
    if req_vol: bull_veto, bear_veto = bull_veto | (c_vol_bull == 0), bear_veto | (c_vol_bear == 0)
    if req_vel: bull_veto, bear_veto = bull_veto | (c_vel_bull == 0), bear_veto | (c_vel_bear == 0)
    if req_rsi: bull_veto, bear_veto = bull_veto | (c_rsi_bull == 0), bear_veto | (c_rsi_bear == 0)
    if req_adx: bull_veto, bear_veto = bull_veto | (c_adx_bull == 0), bear_veto | (c_adx_bear == 0)
    if req_ema: bull_veto, bear_veto = bull_veto | (c_ema_bull == 0), bear_veto | (c_ema_bear == 0)
    if req_stoch: bull_veto, bear_veto = bull_veto | (c_stoch_bull == 0), bear_veto | (c_stoch_bear == 0)

    df[f"Armed_Bull_{tf_str}"] = (df[f"Score_Bull_{tf_str}"] >= min_score) & (~bull_veto)
    df[f"Armed_Bear_{tf_str}"] = (df[f"Score_Bear_{tf_str}"] >= min_score) & (~bear_veto)
    return df

def evaluate_single_timeframe_gates(df_base, tf_str):
    # Alignment fix: Anchors grouping at 09:15 NSE open
    df_tf = (
        df_base.groupby(["Symbol", pd.Grouper(key="Datetime", freq=tf_str, closed="left", label="left", origin=pd.Timestamp('2000-01-01 09:15:00'))])
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .reset_index()
    )
    df_tf = df_tf.dropna(subset=["Close"]).sort_values(["Symbol", "Datetime"])
    df_tf = calculate_core_technicals(df_tf)
    
    df_tf = construct_45deg_renko_matrix(df_tf, tf_str, MACRO_RENKO_CONFIRM_BRICKS)
    df_tf = construct_volume_delta_renko_matrix(df_tf, tf_str, MACRO_RENKO_CONFIRM_BRICKS)
    df_tf = construct_renko_velocity_engine(df_tf, tf_str)
    
    df_tf = apply_dual_tier_scorecard(df_tf, tf_str, "MACRO")

    # Shift time cleanly to avoid lookahead bias in `merge_asof`
    df_tf["Eval_Time"] = df_tf["Datetime"] + pd.to_timedelta(tf_str)
    
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
def prepare_unified_execution_tape(rolling_master_df, micro_tf, macro_timeframes):
    if micro_tf != "1min":
        df_micro = (
            rolling_master_df.groupby(["Symbol", pd.Grouper(key="Datetime", freq=micro_tf, closed="left", label="left", origin=pd.Timestamp('2000-01-01 09:15:00'))])
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
    df_micro = apply_dual_tier_scorecard(df_micro, micro_tf, "MICRO")
    
    # Critical pd.merge_asof condition: Both DataFrames must be strictly sorted by 'on' before merge
    df_micro = df_micro.sort_values("Datetime").reset_index(drop=True)

    bull_gate_cols, bear_gate_cols = [], []
    for tf in macro_timeframes:
        print(f"   ├─ Evaluating Macro Context Gates + Price/Vol/Vel Renko for [{tf}]...")
        env_df = evaluate_single_timeframe_gates(rolling_master_df, tf)
        env_df = env_df.sort_values("Datetime").reset_index(drop=True)  # Ensuring sorted alignment
        
        bull_col, bear_col = f"Armed_Bull_{tf}", f"Armed_Bear_{tf}"
        bull_gate_cols.append(bull_col)
        bear_gate_cols.append(bear_col)
        
        df_micro = pd.merge_asof(df_micro, env_df, on="Datetime", by="Symbol", direction="backward")
        df_micro[bull_col] = df_micro[bull_col].fillna(False)
        df_micro[bear_col] = df_micro[bear_col].fillna(False)
        df_micro[f"Score_Bull_{tf}"] = df_micro[f"Score_Bull_{tf}"].fillna(0).astype(int)
        df_micro[f"Score_Bear_{tf}"] = df_micro[f"Score_Bear_{tf}"].fillna(0).astype(int)
        df_micro[f"Renko_Count_{tf}"] = df_micro[f"Renko_Count_{tf}"].fillna(0).astype(int)
        df_micro[f"Vol_Renko_Count_{tf}"] = df_micro[f"Vol_Renko_Count_{tf}"].fillna(0).astype(int)

    df_micro["Master_Armed_Bull"] = df_micro[bull_gate_cols].any(axis=1)
    df_micro["Master_Armed_Bear"] = df_micro[bear_gate_cols].any(axis=1)
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
def scan_institutional_tape(target_date_str):
    print(f"\n📡 Initiating Options Engine for {target_date_str}...")
    access_token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not access_token:
        print(f"❌ {COLOR_RED}Error: UPSTOX_ACCESS_TOKEN missing.{COLOR_RESET}")
        return
    headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}

    spot_inst, options_inst = get_options_universe()
    if not spot_inst: return

    spot_prices = fetch_latest_spot_prices(spot_inst, headers)
    target_contracts = build_options_matrix(spot_prices, options_inst)
    
    if not target_contracts:
        print(f"{COLOR_RED}[Error] No options contracts mapped.{COLOR_RESET}")
        return
        
    print(f"✅ Mapped {len(target_contracts)} total option contracts for analysis.")

    # 🌟 APPLY STAGE 1 FAST PRE-FILTER (Liquidity + Premium)
    target_contracts = filter_liquid_options(target_contracts, headers, target_date_str)
    
    if not target_contracts:
        print(f"{COLOR_YELLOW}⚠️ All contracts failed the Liquidity (Vol >= {MIN_PREV_DAY_VOLUME}) or Premium (Price >= ₹{MIN_OPT_PREMIUM}) checks.{COLOR_RESET}")
        return

    trading_days = get_past_trading_days(target_date_str, num_days=BACKTRACE_DAYS)
    if not trading_days: return

    target_dt = pd.to_datetime(target_date_str)
    current_now = dt.utcnow() + timedelta(hours=5, minutes=30)
    is_live_today = target_date_str == current_now.strftime("%Y-%m-%d")

    print(f"\n🚀 STAGE 2 INGESTION: Multithreading Bulk 1-Min Data for {len(target_contracts)} Liquid Contracts...")
    fetch_tasks = [(item, trading_days[0], target_date_str, is_live_today) for item in target_contracts]
    historical_dfs = []

    def fetch_worker(task):
        item, start_date, end_date, live = task
        try:
            key = urllib.parse.quote(item["key"])
            dfs = []
            hist_end = end_date if not live else (current_now - timedelta(days=1)).strftime("%Y-%m-%d")

            data = safe_api_request(f"https://api.upstox.com/v2/historical-candle/{key}/1minute/{hist_end}/{start_date}", headers)
            if data: dfs.append(pd.DataFrame(data, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume", "OI"]))

            if live:
                data = safe_api_request(f"https://api.upstox.com/v2/historical-candle/intraday/{key}/1minute", headers, is_live=True)
                if data: dfs.append(pd.DataFrame(data, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume", "OI"]))

            if dfs:
                df = pd.concat(dfs, ignore_index=True)
                df["Datetime"] = pd.to_datetime(df["Timestamp"]).dt.tz_localize(None)
                df = df.drop_duplicates(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)
                df["Symbol"] = item["symbol"]
                return df
        except Exception:
            pass
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_API_WORKERS) as executor:
        futures = {executor.submit(fetch_worker, task): task for task in fetch_tasks}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            sys.stdout.write(f"\r  ├─ Fetching 1-Min Data... {completed}/{len(fetch_tasks)} processed")
            sys.stdout.flush()
            res = future.result()
            if res is not None: historical_dfs.append(res)
    print()

    if not historical_dfs:
        print(f"❌ {COLOR_RED}No historical data retrieved.{COLOR_RESET}")
        return

    rolling_master_df = pd.concat(historical_dfs, ignore_index=True)
    if rolling_master_df.empty:
        print(f"❌ {COLOR_RED}Concat generated empty dataframe.{COLOR_RESET}")
        return

    print("\n⚙️ Computing 7-Pillar Scorecards & Velocity Matrices on Premium Data...")

    tape_exec = prepare_unified_execution_tape(rolling_master_df, MICRO_TIMEFRAME, MACRO_TIMEFRAMES)
    if GLOBAL_MACRO_STRATEGY_2D == "BULLISH": tape_exec["Master_Armed_Bear"] = False
    elif GLOBAL_MACRO_STRATEGY_2D == "BEARISH": tape_exec["Master_Armed_Bull"] = False

    all_anomalies = tape_exec[tape_exec["Direction"] != 0].copy()
    anomalies_by_time = all_anomalies.groupby("Datetime")

    closes_dict = tape_exec.set_index(["Datetime", "Symbol"])["Close"].to_dict()
    micro_price_renko = tape_exec.set_index(["Datetime", "Symbol"])[f"Renko_Count_{MICRO_TIMEFRAME}"].to_dict()
    micro_vol_renko = tape_exec.set_index(["Datetime", "Symbol"])[f"Vol_Renko_Count_{MICRO_TIMEFRAME}"].to_dict()
    micro_vel_bars = tape_exec.set_index(["Datetime", "Symbol"])[f"Bars_Since_Brick_{MICRO_TIMEFRAME}"].to_dict()
    
    macro_price_renkos = {tf: tape_exec.set_index(["Datetime", "Symbol"])[f"Renko_Count_{tf}"].to_dict() for tf in MACRO_TIMEFRAMES}
    macro_vol_renkos = {tf: tape_exec.set_index(["Datetime", "Symbol"])[f"Vol_Renko_Count_{tf}"].to_dict() for tf in MACRO_TIMEFRAMES}
    
    all_times = np.sort(tape_exec["Datetime"].unique())
    memory_bank = {}
    cutoff_time_obj = pd.to_datetime(ENTRY_CUTOFF_TIME).time()

    for t in all_times:
        t_dt = pd.to_datetime(t)
        
        for sym, st in memory_bank.items():
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
                                        exit_reason = f"Macro [{tf}] Price Break"
                                        break
                                    if ma_v <= -MACRO_EXIT_VOL_BRICKS:
                                        exit_reason = f"Macro [{tf}] Volume Break"
                                        break
                        elif st["dir"] == -1:
                            if mi_p_count >= MICRO_EXIT_PRICE_BRICKS: exit_reason = "Micro Price Reversal"
                            elif mi_v_count >= MICRO_EXIT_VOL_BRICKS: exit_reason = "Micro Volume Reversal"
                            else:
                                for tf in st["triggering_macro_tfs"]:
                                    ma_p = macro_price_renkos[tf].get((t_dt, sym), 0)
                                    ma_v = macro_vol_renkos[tf].get((t_dt, sym), 0)
                                    if ma_p >= MACRO_EXIT_PRICE_BRICKS:
                                        exit_reason = f"Macro [{tf}] Price Break"
                                        break
                                    if ma_v >= MACRO_EXIT_VOL_BRICKS:
                                        exit_reason = f"Macro [{tf}] Volume Break"
                                        break
                    
                    if exit_reason:
                        st["state"] = "EXITED"
                        st["exit_time"] = t_dt.strftime("%Y-%m-%d %H:%M")
                        st["exit_price"] = ltp
                        st["exit_reason"] = exit_reason

        if t_dt in anomalies_by_time.groups and t_dt.time() <= cutoff_time_obj:
            for _, row in anomalies_by_time.get_group(t_dt).iterrows():
                sym = row["Symbol"]
                direction = row["Direction"]
                
                triggered_m_tfs = []
                for tf in MACRO_TIMEFRAMES:
                    armed_col = f"Armed_Bull_{tf}" if direction == 1 else f"Armed_Bear_{tf}"
                    if row.get(armed_col, False):
                        triggered_m_tfs.append(tf)

                if sym not in memory_bank or memory_bank[sym]["state"] == "EXITED":
                    memory_bank[sym] = {
                        "state": "ACTIVE",
                        "origin": row["Close"],              
                        "date": t_dt.strftime("%Y-%m-%d"),
                        "time": t_dt.strftime("%H:%M"),      
                        "dir": direction,
                        "exit_time": None,
                        "exit_price": None,
                        "exit_reason": None,
                        "triggering_macro_tfs": triggered_m_tfs, 
                        "macro_scores": {tf: row.get(f"Score_Bull_{tf}" if direction == 1 else f"Score_Bear_{tf}", 0) for tf in MACRO_TIMEFRAMES},
                        "micro_score": row.get(f"Score_Bull_{MICRO_TIMEFRAME}" if direction == 1 else f"Score_Bear_{MICRO_TIMEFRAME}", 0)
                    }

        if t_dt.hour == 15 and t_dt.minute >= 15:
            for sym, st in memory_bank.items():
                if st["state"] == "ACTIVE":
                    st["state"] = "EXITED"
                    st["exit_time"] = t_dt.strftime("%Y-%m-%d %H:%M") + " (EOD)"
                    st["exit_price"] = closes_dict.get((t_dt, sym), st["origin"])
                    st["exit_reason"] = "End of Day Market Close"

    today_master = tape_exec[tape_exec["Datetime"].dt.date == target_dt.date()]
    if today_master.empty:
        print(f"\n{COLOR_YELLOW}[Terminal Standby] Market data for {target_date_str} is empty.{COLOR_RESET}\n")
        return
        
    final_ltp_dict = today_master.groupby("Symbol")["Close"].last().to_dict()

    # ==============================================================================
    # 6. TERMINAL OUTPUT
    # ==============================================================================
    active_runners = {sym: st for sym, st in memory_bank.items() if st["state"] == "ACTIVE"}
    closed_trades = [{**st, "sym": sym} for sym, st in memory_bank.items() if st["state"] == "EXITED" and st["date"] == target_date_str]

    tf_display_str = " | ".join(MACRO_TIMEFRAMES)
    print(f"\n{COLOR_CYAN}================================================================================================{COLOR_RESET}")
    print(f"{COLOR_BOLD}7-PILLAR QUALIFYING-TF OPTIONS ENGINE [{MICRO_TIMEFRAME} Micro ⚡ Macro: {tf_display_str}]{COLOR_RESET}")
    print(f"{COLOR_CYAN}================================================================================================{COLOR_RESET}\n")

    if active_runners:
        print(f"{COLOR_BOLD}🟢 BASKET 1: ACTIVE RUNNERS (Riding the Trend){COLOR_RESET}")
        for sym, st in active_runners.items():
            ltp = final_ltp_dict.get(sym, st["origin"])
            pnl_pct = ((ltp - st["origin"]) / st["origin"]) * 100 if st["dir"] == 1 else ((st["origin"] - ltp) / st["origin"]) * 100
            color = COLOR_GREEN if pnl_pct >= 0 else COLOR_RED
            d_str = "BULLISH" if st["dir"] == 1 else "BEARISH"
            
            print(f"  {color}⚡ {sym:<20} Open P&L: {pnl_pct:+.2f}% ({d_str}){COLOR_RESET}")
            print(f"      └─ ⚓ Qualifying Macro TFs        : {', '.join(st['triggering_macro_tfs'])}")
            print(f"      └─ 🔫 Micro Execution [{MICRO_TIMEFRAME}] : Score >= {MICRO_MINIMUM_SCORE}/7 (Score={st['micro_score']})")
            print(f"      └─ ⚓ True Birth Anchor            : {st['date']} @ {st['time']} | Price: ₹{st['origin']:.2f}")
            print(f"      └─ 🎯 Latest LTP                  : {target_date_str} @ EOD   | Price: ₹{ltp:.2f}\n")

    if closed_trades:
        print(f"{COLOR_BOLD}🛑 BASKET 2: CLOSED TRADES (Renko Structure Broken / Stagnation){COLOR_RESET}")
        for st in closed_trades:
            pnl_pct = ((st["exit_price"] - st["origin"]) / st["origin"]) * 100 if st["dir"] == 1 else ((st["origin"] - st["exit_price"]) / st["origin"]) * 100
            color = COLOR_GREEN if pnl_pct >= 0 else COLOR_RED
            d_str = "BULLISH" if st["dir"] == 1 else "BEARISH"

            print(f"  {color}🛑 {st['sym']:<20} Final P&L: {pnl_pct:+.2f}% ({d_str}){COLOR_RESET}")
            print(f"      └─ ⚓ Qualifying Macro TFs        : {', '.join(st['triggering_macro_tfs'])}")
            print(f"      └─ 🔫 Micro Execution [{MICRO_TIMEFRAME}] : Score >= {MICRO_MINIMUM_SCORE}/7 (Score={st['micro_score']})")
            print(f"      └─ ⚓ True Birth Anchor            : {st['date']} @ {st['time']} | Price: ₹{st['origin']:.2f}")
            print(f"      └─ 🎯 Exit Time & Price            : {st['exit_time']} | Price: ₹{st['exit_price']:.2f}")
            print(f"      └─ 📉 Reason                      : {st['exit_reason']}\n")

    if not active_runners and not closed_trades:
        print(f"{COLOR_DIM}[Terminal Silent] No trades triggered today.{COLOR_RESET}\n")

# ==============================================================================
# 7. RUN EXECUTOR
# ==============================================================================
def run_production_sweep():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--date", type=str, default="")
    args, _ = parser.parse_known_args()
    raw_date_str = args.date or os.environ.get("PARAM_BACKTEST_DATE", "").strip()

    if not raw_date_str:
        target_dt = dt.utcnow() + timedelta(hours=5, minutes=30)
        if target_dt.weekday() == 5: target_dt -= timedelta(days=1)
        elif target_dt.weekday() == 6: target_dt -= timedelta(days=2)
        target_date_str = target_dt.strftime("%Y-%m-%d")
    else:
        target_date_str = dt.strptime(raw_date_str, "%Y-%m-%d").strftime("%Y-%m-%d")

    if not os.environ.get("UPSTOX_ACCESS_TOKEN"):
        print(f"❌ {COLOR_RED}Error: UPSTOX_ACCESS_TOKEN environment variable not found.{COLOR_RESET}")
        return

    scan_institutional_tape(target_date_str)

if __name__ == "__main__":
    run_production_sweep()

