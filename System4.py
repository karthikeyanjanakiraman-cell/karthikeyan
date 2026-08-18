"""
system3.py - NIFTY & SENSEX Options Multi-Timeframe Renko Execution Engine
Fyers API Implementation (Production-Grade & CI/CD Resilient)

Key Features:
- FYERS NIFTY 50 & SENSEX CE / PE Options Dynamic Extraction
- Indestructible Fuzzy Parser (100% Immune to Fyers Data Schema Shifts)
- Symbol-Based Filtering (Bypasses underlying column naming inconsistencies)
- Datetime Resolution Normalization (Fixes Pandas MergeErrors)
- 09:15 AM Opening Strike Freeze (Locked for the Whole Session)
- Dual-Tier 7-Pillar Scorecard & 45-Degree Price/Volume/Velocity Renko Engine
"""

import argparse
import concurrent.futures
import datetime
from datetime import datetime, timedelta
import os
import sys
import time
import warnings
import traceback

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

BACKTRACE_DAYS = 1

# ==============================================================================
# ★ GLOBAL CONFIGURATION: INDEX OPTIONS & STRIKE PROCESSING ★
# ==============================================================================
NUM_STRIKES_PER_SIDE = 4

INDEX_CONFIG = {
    "NIFTY": {
        "segment_file": "NSE_FO",
        "spot_key": "NSE:NIFTY50-INDEX",
        "strike_step": 50,
        "underlying_symbol": "NIFTY"
    },
    "SENSEX": {
        "segment_file": "BSE_FO",
        "spot_key": "BSE:SENSEX-INDEX",
        "strike_step": 100,
        "underlying_symbol": "SENSEX"
    }
}

# ==============================================================================
# ★ GLOBAL CONFIGURATION: DYNAMIC TIMEFRAMES & INDICATORS ★
# ==============================================================================
MICRO_TIMEFRAME = "3min"
MACRO_TIMEFRAMES = ["15min"]

ATR_PERIOD = 14
RSI_PERIOD = 14
BB_SMA_PERIOD = 20
ADX_PERIOD = 14
ADX_THRESHOLD = 20
STOCH_PERIOD = 14

MICRO_RENKO_CONFIRM_BRICKS = 1
MACRO_RENKO_CONFIRM_BRICKS = 1
RENKO_MIN_BRICK = 0.50
RENKO_DEFAULT_PCT = 0.01

GLOBAL_MACRO_STRATEGY_2D = "BOTH"  

# ==============================================================================
# 🎛️ TIER 1 & 2: MACRO & MICRO CONTEXT SWITCHBOARDS
# ==============================================================================
MACRO_MANDATORY_PRICE_RENKO    = True
MACRO_MANDATORY_VOL_RENKO      = True
MACRO_MANDATORY_RENKO_VELOCITY = True
MACRO_MANDATORY_RSI_BB         = False
MACRO_MANDATORY_ADX_DMI        = False
MACRO_MANDATORY_EMA_SPREAD     = False
MACRO_MANDATORY_STOCHASTIC     = False
MACRO_MINIMUM_SCORE            = 3

SYNC_MICRO_WITH_MACRO          = False
MICRO_MANDATORY_PRICE_RENKO    = True
MICRO_MANDATORY_VOL_RENKO      = True
MICRO_MANDATORY_RENKO_VELOCITY = True
MICRO_MANDATORY_RSI_BB         = False
MICRO_MANDATORY_ADX_DMI        = False
MICRO_MANDATORY_EMA_SPREAD     = True
MICRO_MANDATORY_STOCHASTIC     = False
MICRO_MINIMUM_SCORE            = 4

# ==============================================================================
# 🎛️ TIER 3: TRADE MANAGEMENT & TEMPORAL GATES
# ==============================================================================
MICRO_EXIT_PRICE_BRICKS = 5  
MICRO_EXIT_VOL_BRICKS   = 5  
MACRO_EXIT_PRICE_BRICKS = 1
MACRO_EXIT_VOL_BRICKS   = 1  
RENKO_VELOCITY_MAX_BARS = 12
ENTRY_CUTOFF_TIME = "15:00"


# ==============================================================================
# 1. FYERS SPECIFIC INGESTION & 09:15 STRIKE SELECTION
# ==============================================================================
def fetch_fyers_instruments(segment):
    """
    Indestructible Fuzzy Parser for Fyers CSV. 
    Bypasses rigid column numbers and identifies data by its literal value structure.
    """
    url = f"https://public.fyers.in/sym_details/{segment}.csv"
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code != 200:
            print(f"{COLOR_RED}[API Error] HTTP {res.status_code} fetching {segment}{COLOR_RESET}")
            return []
            
        text_data = res.text.strip()
        lines = text_data.split('\n')
        print(f"   ├─ {COLOR_DIM}[Data Loader] Downloaded {len(lines)} raw lines from {segment}{COLOR_RESET}")

        contracts = []
        
        for line in lines:
            parts = line.split(',')
            if len(parts) < 10:
                continue
            
            symbol = next((p for p in parts if p.startswith("NSE:") or p.startswith("BSE:")), None)
            if not symbol: continue
            
            if "NIFTY" not in symbol and "SENSEX" not in symbol:
                continue

            opt_type = next((p.strip().upper() for p in parts if p.strip().upper() in ["CE", "PE"]), None)
            if not opt_type:
                if symbol.endswith("CE"): opt_type = "CE"
                elif symbol.endswith("PE"): opt_type = "PE"
                else: continue

            valid_prefixes = tuple(str(i) for i in range(16, 22))
            expiry_epoch_str = next((p for p in parts if p.isdigit() and len(p) == 10 and p.startswith(valid_prefixes)), None)
            if not expiry_epoch_str: continue
            
            expiry_epoch = int(expiry_epoch_str)
            expiry_date = datetime.utcfromtimestamp(expiry_epoch).strftime('%Y-%m-%d')

            strike = None
            for p in reversed(parts):
                try:
                    val = float(p)
                    if 1000.0 <= val <= 200000.0 and val % 50 == 0:
                        if str(int(val)) != expiry_epoch_str:
                            strike = val
                            break
                except ValueError:
                    pass
                    
            if not strike: continue

            contracts.append({
                "symbolDetails": symbol,
                "expiry": expiry_date,
                "strikePrice": strike,
                "optionType": opt_type
            })
            
        print(f"   ├─ {COLOR_DIM}[Data Loader] Successfully extracted {len(contracts)} Index Options from {segment}{COLOR_RESET}")
        return contracts
    except Exception as e:
        print(f"{COLOR_RED}[API Error] Failed fetching/parsing {segment}: {e}{COLOR_RESET}")
        return []


def get_fyers_spot_opening_price_at_0915(spot_key, target_date_str, headers):
    url = "https://api-t1.fyers.in/data/history"
    params = {
        "symbol": spot_key,
        "resolution": "1",
        "date_format": "1",
        "range_from": target_date_str,
        "range_to": target_date_str
    }
    try:
        res = requests.get(url, headers=headers, params=params, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if data.get("s") == "ok" and data.get("candles") and len(data["candles"]) > 0:
                first_candle_open = float(data["candles"][0][1])
                return first_candle_open
    except Exception:
        pass
    return None

def build_locked_options_universe(target_date_str, headers):
    print(f"\n{COLOR_CYAN}🔍 Initializing Fyers Near-Index Option Strike Engine for {target_date_str}...{COLOR_RESET}")
    nse_master = fetch_fyers_instruments("NSE_FO")
    bse_master = fetch_fyers_instruments("BSE_FO")
    
    selected_option_universe = []

    for index_name, config in INDEX_CONFIG.items():
        master_data = nse_master if index_name == "NIFTY" else bse_master
        if not master_data:
            continue

        underlying_tag = f":{config['underlying_symbol']}"
        option_contracts = [
            item for item in master_data
            if underlying_tag in item["symbolDetails"].upper()
        ]

        if not option_contracts:
            print(f"   ├─ [{index_name}] {COLOR_RED}Failed to match any contracts for {config['underlying_symbol']}.{COLOR_RESET}")
            continue

        all_expiries = sorted(list({item.get("expiry") for item in option_contracts if item.get("expiry") >= target_date_str}))
        
        if not all_expiries:
            fallback_expiries = sorted(list({item.get("expiry") for item in option_contracts}))
            if fallback_expiries:
                target_expiry = fallback_expiries[0]
                print(f"   ├─ [{index_name}] {COLOR_YELLOW}Warning: {target_date_str} not available. Auto-correcting to Nearest Expiry: {target_expiry}{COLOR_RESET}")
            else:
                continue
        else:
            if all_expiries[0] == target_date_str:
                target_expiry = all_expiries[1] if len(all_expiries) > 1 else all_expiries[0]
                print(f"   ├─ [{index_name}] Today is Expiry Day! ➔ {COLOR_YELLOW}Rolling to Next Week: {target_expiry}{COLOR_RESET}")
            else:
                target_expiry = all_expiries[0]
                print(f"   ├─ [{index_name}] Near Expiry Selected: {COLOR_GREEN}{target_expiry}{COLOR_RESET}")

        spot_price = get_fyers_spot_opening_price_at_0915(config["spot_key"], target_date_str, headers)
        
        if spot_price is None:
            available_strikes = [float(item.get("strikePrice", 0)) for item in option_contracts if item.get("expiry") == target_expiry]
            spot_price = float(np.median(available_strikes)) if available_strikes else (24000.0 if index_name == "NIFTY" else 79000.0)
            print(f"   ├─ [{index_name}] {COLOR_YELLOW}Notice: 09:15 Spot API failed/unreached. Fallback Anchor @ ₹{spot_price:.2f}{COLOR_RESET}")
        else:
            print(f"   ├─ [{index_name}] ⚓ 09:15 AM Spot Index Price: ₹{spot_price:.2f}")

        step = config["strike_step"]
        atm_strike = round(spot_price / step) * step
        selected_strikes = [atm_strike + (i * step) for i in range(-NUM_STRIKES_PER_SIDE, NUM_STRIKES_PER_SIDE + 1)]
        
        matched_count = 0
        for item in option_contracts:
            if item.get("expiry") == target_expiry and float(item.get("strikePrice", 0)) in selected_strikes:
                selected_option_universe.append({
                    "symbol": item.get("symbolDetails"),
                    "underlying": index_name,
                    "strike": float(item.get("strikePrice", 0)),
                    "option_type": item.get("optionType"),
                    "expiry": target_expiry
                })
                matched_count += 1

        print(f"   └─ [{index_name}] Frozen Near-Index Contracts: {matched_count} (CE + PE)\n")

    return selected_option_universe

def get_past_trading_days(target_date_str, num_days=20):
    target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
    trading_days = []
    current_dt = target_dt
    while len(trading_days) < num_days:
        if current_dt.weekday() < 5:
            trading_days.append(current_dt.strftime("%Y-%m-%d"))
        current_dt -= timedelta(days=1)
    trading_days.reverse()
    return trading_days


# ==============================================================================
# 2. CORE TECHNICAL & RENKO ENGINES
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


# ==============================================================================
# 3. DUAL-TIER SCORECARD SYSTEM
# ==============================================================================
def apply_dual_tier_scorecard(df, tf_str, tier_type):
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
    
    df_tf = apply_dual_tier_scorecard(df_tf, tf_str, "MACRO")
    df_tf["Eval_Time"] = df_tf["Datetime"] + pd.to_timedelta(tf_str)
    
    export_cols = [
        "Symbol", "Eval_Time", 
        f"Armed_Bull_{tf_str}", f"Armed_Bear_{tf_str}", 
        f"Score_Bull_{tf_str}", f"Score_Bear_{tf_str}", 
        f"Renko_Count_{tf_str}", f"Vol_Renko_Count_{tf_str}",
        f"Bars_Since_Brick_{tf_str}"
    ]
    env_df = df_tf[export_cols].copy().rename(columns={"Eval_Time": "Datetime"})
    return env_df.sort_values("Datetime").reset_index(drop=True)


# ==============================================================================
# 4. MICRO EXECUTION TAPE & CONFLUENCE MATCHER
# ==============================================================================
def prepare_unified_execution_tape(rolling_master_df, micro_tf, macro_timeframes):
    df_micro = rolling_master_df.sort_values(["Symbol", "Datetime"]).copy()
    
    # 🔥 PANDAS MERGEERROR FIX: Force normalize df_micro Datetime to datetime64[ns]
    df_micro["Datetime"] = pd.to_datetime(df_micro["Datetime"]).astype("datetime64[ns]")

    df_micro = calculate_core_technicals(df_micro)
    df_micro = construct_45deg_renko_matrix(df_micro, micro_tf, MICRO_RENKO_CONFIRM_BRICKS)
    df_micro = construct_volume_delta_renko_matrix(df_micro, micro_tf, MICRO_RENKO_CONFIRM_BRICKS)
    df_micro = construct_renko_velocity_engine(df_micro, micro_tf)
    df_micro = apply_dual_tier_scorecard(df_micro, micro_tf, "MICRO")
    df_micro = df_micro.sort_values("Datetime").reset_index(drop=True)

    bull_gate_cols, bear_gate_cols = [], []
    for tf in macro_timeframes:
        print(f"   ├─ Evaluating Macro Context Gates for Options [{tf}]...")
        env_df = evaluate_single_timeframe_gates(rolling_master_df, tf)
        
        # 🔥 PANDAS MERGEERROR FIX: Force normalize env_df Datetime to match df_micro
        env_df["Datetime"] = pd.to_datetime(env_df["Datetime"]).astype("datetime64[ns]")

        bull_col, bear_col = f"Armed_Bull_{tf}", f"Armed_Bear_{tf}"
        bull_gate_cols.append(bull_col)
        bear_gate_cols.append(bear_col)
        
        df_micro = pd.merge_asof(df_micro, env_df, on="Datetime", by="Symbol", direction="backward")
        df_micro[bull_col] = df_micro[bull_col].fillna(False)
        df_micro[bear_col] = df_micro[bear_col].fillna(False)

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
# 5. FYERS SCANNING & TRADE MANAGEMENT ENGINE
# ==============================================================================
def scan_fyers_institutional_tape(target_date_str):
    try:
        app_id = os.environ.get("FYERS_CLIENT_ID") or os.environ.get("FYERS_APP_ID")
        access_token = os.environ.get("FYERS_ACCESS_TOKEN")
        
        print(f"\n{COLOR_DIM}[System Auth Check] Client ID Passed: {'Yes' if app_id else 'NO'} | Token Passed: {'Yes' if access_token else 'NO'}{COLOR_RESET}")

        if not app_id or not access_token:
            print(f"❌ {COLOR_RED}Error: FYERS_CLIENT_ID or FYERS_ACCESS_TOKEN environment variables not found.{COLOR_RESET}")
            return

        headers = {"Authorization": f"{app_id}:{access_token}"}
        universe = build_locked_options_universe(target_date_str, headers)
        
        if not universe:
            print(f"{COLOR_RED}❌ No valid option strikes generated for {target_date_str}.{COLOR_RESET}")
            return

        trading_days = get_past_trading_days(target_date_str, num_days=BACKTRACE_DAYS)
        if not trading_days:
            return

        target_dt = pd.to_datetime(target_date_str)
        print(f"🚀 Multithreading Bulk Ingestion for {len(universe)} Fyers Option Contracts...")
        fetch_tasks = [(item, trading_days[0], target_date_str) for item in universe]
        historical_dfs = []

        def fetch_fyers_worker(task):
            item, start_date, end_date = task
            symbol = item["symbol"]
            url = "https://api-t1.fyers.in/data/history"
            params = {
                "symbol": symbol,
                "resolution": "1",
                "date_format": "1",
                "range_from": start_date,
                "range_to": end_date
            }
            
            for attempt in range(3):
                try:
                    res = requests.get(url, headers=headers, params=params, timeout=15)
                    if res.status_code == 200:
                        data = res.json()
                        if data.get("s") == "ok" and data.get("candles"):
                            df = pd.DataFrame(data["candles"], columns=["Epoch", "Open", "High", "Low", "Close", "Volume"])
                            df["Datetime"] = pd.to_datetime(df["Epoch"], unit='s')
                            df["Datetime"] = df["Datetime"].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
                            
                            # 🔥 PANDAS MERGEERROR FIX: Force resolution out of the gate
                            df["Datetime"] = df["Datetime"].astype("datetime64[ns]")
                            
                            df = df.drop_duplicates(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)
                            df["Symbol"] = symbol
                            return df
                        break
                    elif res.status_code == 429: time.sleep(1.5)
                    else: break
                except Exception: time.sleep(1)
            return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_fyers_worker, task): task for task in fetch_tasks}
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                completed += 1
                sys.stdout.write(f"\r📡 Fetching Fyers History... {completed}/{len(fetch_tasks)} options contracts processed")
                sys.stdout.flush()
                res = future.result()
                if res is not None: historical_dfs.append(res)
        print()

        if not historical_dfs:
            print(f"{COLOR_YELLOW}[Warning] No candle data returned for the selected Fyers option strikes.{COLOR_RESET}")
            return

        rolling_master_df = pd.concat(historical_dfs, ignore_index=True)
        print("⚙️ Computing Technicals, Velocity Matrices & Option Micro/Macro Tape...")

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
                                        if macro_price_renkos[tf].get((t_dt, sym), 0) <= -MACRO_EXIT_PRICE_BRICKS:
                                            exit_reason = f"Macro [{tf}] Price Break"
                                            break
                                        if macro_vol_renkos[tf].get((t_dt, sym), 0) <= -MACRO_EXIT_VOL_BRICKS:
                                            exit_reason = f"Macro [{tf}] Volume Break"
                                            break
                            elif st["dir"] == -1:
                                if mi_p_count >= MICRO_EXIT_PRICE_BRICKS: exit_reason = "Micro Price Reversal"
                                elif mi_v_count >= MICRO_EXIT_VOL_BRICKS: exit_reason = "Micro Volume Reversal"
                                else:
                                    for tf in st["triggering_macro_tfs"]:
                                        if macro_price_renkos[tf].get((t_dt, sym), 0) >= MACRO_EXIT_PRICE_BRICKS:
                                            exit_reason = f"Macro [{tf}] Price Break"
                                            break
                                        if macro_vol_renkos[tf].get((t_dt, sym), 0) >= MACRO_EXIT_VOL_BRICKS:
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
                    triggered_m_tfs = [tf for tf in MACRO_TIMEFRAMES if row.get(f"Armed_Bull_{tf}" if direction == 1 else f"Armed_Bear_{tf}", False)]

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
                            "triggering_macro_tfs": triggered_m_tfs
                        }

            if t_dt.hour == 15 and t_dt.minute >= 15:
                for sym, st in memory_bank.items():
                    if st["state"] == "ACTIVE":
                        st["state"] = "EXITED"
                        st["exit_time"] = t_dt.strftime("%Y-%m-%d %H:%M") + " (EOD)"
                        st["exit_price"] = closes_dict.get((t_dt, sym), st["origin"])
                        st["exit_reason"] = "End of Day Market Close"

        today_master = tape_exec[tape_exec["Datetime"].dt.date == target_dt.date()]
        if today_master.empty: return
            
        final_ltp_dict = today_master.groupby("Symbol")["Close"].last().to_dict()

        # ==============================================================================
        # 6. TERMINAL OUTPUT DISPLAY
        # ==============================================================================
        active_runners = {sym: st for sym, st in memory_bank.items() if st["state"] == "ACTIVE"}
        closed_trades = [{**st, "sym": sym} for sym, st in memory_bank.items() if st["state"] == "EXITED" and st["date"] == target_date_str]

        tf_display_str = " | ".join(MACRO_TIMEFRAMES)
        print(f"\n{COLOR_CYAN}================================================================================================{COLOR_RESET}")
        print(f"{COLOR_BOLD}FYERS NIFTY & SENSEX DUAL-TIER ENGINE [{MICRO_TIMEFRAME} Micro ⚡ Macro: {tf_display_str}]{COLOR_RESET}")
        print(f"{COLOR_CYAN}================================================================================================{COLOR_RESET}\n")

        if active_runners:
            print(f"{COLOR_BOLD}🟢 BASKET 1: ACTIVE OPTION RUNNERS{COLOR_RESET}")
            for sym, st in active_runners.items():
                ltp = final_ltp_dict.get(sym, st["origin"])
                pnl_pct = ((ltp - st["origin"]) / st["origin"]) * 100 if st["dir"] == 1 else ((st["origin"] - ltp) / st["origin"]) * 100
                color = COLOR_GREEN if pnl_pct >= 0 else COLOR_RED
                d_str = "BUY/LONG" if st["dir"] == 1 else "SELL/SHORT"
                print(f"  {color}⚡ {sym:<25} P&L: {pnl_pct:+.2f}% ({d_str}){COLOR_RESET}")
                print(f"      └─ ⚓ Qualifying Macro TFs : {', '.join(st['triggering_macro_tfs'])}")
                print(f"      └─ 🎯 Entry / LTP          : ₹{st['origin']:.2f} ➔ ₹{ltp:.2f}\n")

        if closed_trades:
            print(f"{COLOR_BOLD}🛑 BASKET 2: CLOSED OPTION TRADES{COLOR_RESET}")
            for st in closed_trades:
                pnl_pct = ((st["exit_price"] - st["origin"]) / st["origin"]) * 100 if st["dir"] == 1 else ((st["origin"] - st["exit_price"]) / st["origin"]) * 100
                color = COLOR_GREEN if pnl_pct >= 0 else COLOR_RED
                d_str = "BUY/LONG" if st["dir"] == 1 else "SELL/SHORT"
                print(f"  {color}🛑 {st['sym']:<25} Final P&L: {pnl_pct:+.2f}% ({d_str}){COLOR_RESET}")
                print(f"      └─ 🎯 Exit Time / Price    : {st['exit_time']} | ₹{st['exit_price']:.2f}")
                print(f"      └─ 📉 Exit Reason          : {st['exit_reason']}\n")

    except Exception as e:
        print(f"\n{COLOR_RED}💥 CRITICAL ENGINE FAILURE: {e}{COLOR_RESET}")
        traceback.print_exc()


def run_production_sweep():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--date", type=str, default="")
    args, _ = parser.parse_known_args()
    raw_date_str = args.date or os.environ.get("PARAM_BACKTEST_DATE", "").strip()

    if not raw_date_str:
        target_dt = datetime.utcnow() + timedelta(hours=5, minutes=30)
        # Offset for weekends if backtest date isn't explicitly provided
        if target_dt.weekday() == 5: target_dt -= timedelta(days=1)
        elif target_dt.weekday() == 6: target_dt -= timedelta(days=2)
        target_date_str = target_dt.strftime("%Y-%m-%d")
    else:
        target_date_str = datetime.strptime(raw_date_str, "%Y-%m-%d").strftime("%Y-%m-%d")

    scan_fyers_institutional_tape(target_date_str)

if __name__ == "__main__":
    run_production_sweep()

