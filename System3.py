"""system3.py - Asit Baran Pati Multi-Timeframe Trading System Implementation

Production-Grade Universal N-Timeframe & Dual-Tier 45-Degree Renko Engine:
- Configurable Micro Execution Timeframe (e.g., "1min", "3min", "5min")
- Configurable Macro Hierarchy Array (e.g., ["15min", "60min", "1D"])
- Phase 1 Blueprint: Dual-Tier Scorecard & Global Mandatory Veto Switches
- Phase 1 Blueprint: Order Flow / Cumulative Volume Delta 45-Degree Renko
- EXIT STRATEGY: Dual-Layered (Macro + Micro) Asymmetric OR-Condition Renko Trailing Stop
- Persistent Momentum State Gates (Fixes single-bar Stochastic/EMA bottleneck)
- Zero-Lookahead Vectorized Alignment via pd.merge_asof (Strict Timestamp Sorting Fixed)
"""

import argparse
import concurrent.futures
import datetime
from datetime import datetime, timedelta
import gzip
import io
import json
import os
import sys
import time
import urllib.parse
import warnings

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
COLOR_MAGENTA = "\033[95m"
COLOR_DIM = "\033[2m"
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"

BACKTRACE_DAYS = 1
API_ERROR_LOGGED = False

# ==============================================================================
# ★ GLOBAL CONFIGURATION: DYNAMIC TIMEFRAMES & INDICATORS ★
# ==============================================================================
MICRO_TIMEFRAME = "1min"  # Micro Execution & Tactical Trigger
MACRO_TIMEFRAMES = ["5min"]  # Strategic Structural & Pilot Tiers

ATR_PERIOD = 14
RSI_PERIOD = 14
BB_SMA_PERIOD = 20
BB_STD_DEV = 2.0
ADX_PERIOD = 14
ADX_THRESHOLD = 20
STOCH_PERIOD = 14

MICRO_RENKO_CONFIRM_BRICKS = 2  # Micro Tactical Trigger (2-Brick Rule)
MACRO_RENKO_CONFIRM_BRICKS = 1  # Macro Structural Trend Confirmation
RENKO_MIN_BRICK = 0.05
RENKO_DEFAULT_PCT = 0.005

GLOBAL_MACRO_STRATEGY_2D = "BOTH"  # "BULLISH", "BEARISH", or "BOTH"

# ==============================================================================
# 🎛️ TIER 1: MACRO CONTEXT SWITCHBOARD (THE GENERAL)
# ==============================================================================
MACRO_MANDATORY_PRICE_RENKO = True
MACRO_MANDATORY_VOL_RENKO   = True
MACRO_MANDATORY_RSI_BB      = False
MACRO_MANDATORY_ADX_DMI     = False
MACRO_MANDATORY_EMA_SPREAD  = False
MACRO_MANDATORY_STOCHASTIC  = False
MACRO_MINIMUM_SCORE         = 4      # Out of 6

# ==============================================================================
# 🎛️ TIER 2: MICRO EXECUTION SWITCHBOARD (THE SNIPER)
# ==============================================================================
SYNC_MICRO_WITH_MACRO       = False  # If True, Micro overrides to match Macro

MICRO_MANDATORY_PRICE_RENKO = True
MICRO_MANDATORY_VOL_RENKO   = True
MICRO_MANDATORY_RSI_BB      = False
MICRO_MANDATORY_ADX_DMI     = False
MICRO_MANDATORY_EMA_SPREAD  = False
MICRO_MANDATORY_STOCHASTIC  = False
MICRO_MINIMUM_SCORE         = 2      # Out of 6

# ==============================================================================
# 🎛️ TIER 3: TRADE MANAGEMENT (DUAL-LAYERED EXIT STRATEGY)
# ==============================================================================
# The "OR" Gate: Trade closes if ANY of these thresholds are hit. 
# (Set to 99 to disable a specific layer)

# MICRO TAPE (Fast execution exits)
MICRO_EXIT_PRICE_BRICKS = 4  
MICRO_EXIT_VOL_BRICKS   = 4  

# MACRO TAPE (Structural hold exits)
MACRO_EXIT_PRICE_BRICKS = 2  
MACRO_EXIT_VOL_BRICKS   = 2  


# ==============================================================================
# 1. LIVE INGESTION (F&O Universe & Parallel Bulk Fetching)
# ==============================================================================
def get_dynamic_fno_universe():
    nse_url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
    try:
        response = requests.get(nse_url, timeout=15)
        if response.status_code != 200: return []
        nse_data = json.load(gzip.GzipFile(fileobj=io.BytesIO(response.content)))
        fno_underlying = {
            item.get("underlying_symbol") for item in nse_data
            if item.get("segment") == "NSE_FO" and item.get("underlying_symbol")
        }
        return [
            {"symbol": item.get("trading_symbol"), "key": item.get("instrument_key")}
            for item in nse_data
            if item.get("segment") in ("NSE_EQ", "NSE_INDEX")
            and item.get("trading_symbol") in fno_underlying
        ]
    except Exception as e:
        print(f"{COLOR_RED}[API Error] Failed to fetch F&O universe: {e}{COLOR_RESET}")
        return []

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
# 2. CORE TECHNICAL & 45-DEGREE RENKO ENGINES
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

    high_diff = df_tf["High"] - df_tf.groupby("Symbol")["High"].shift(1)
    low_diff = df_tf.groupby("Symbol")["Low"].shift(1) - df_tf["Low"]
    df_tf["+DM"] = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    df_tf["-DM"] = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
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


# ==============================================================================
# 3. DUAL-TIER SCORECARD SYSTEM
# ==============================================================================
def apply_dual_tier_scorecard(df, tf_str, tier_type):
    if SYNC_MICRO_WITH_MACRO and tier_type == "MICRO":
        req_price, req_vol = MACRO_MANDATORY_PRICE_RENKO, MACRO_MANDATORY_VOL_RENKO
        req_rsi, req_adx = MACRO_MANDATORY_RSI_BB, MACRO_MANDATORY_ADX_DMI
        req_ema, req_stoch = MACRO_MANDATORY_EMA_SPREAD, MACRO_MANDATORY_STOCHASTIC
        min_score = MACRO_MINIMUM_SCORE
    else:
        req_price, req_vol = globals()[f"{tier_type}_MANDATORY_PRICE_RENKO"], globals()[f"{tier_type}_MANDATORY_VOL_RENKO"]
        req_rsi, req_adx = globals()[f"{tier_type}_MANDATORY_RSI_BB"], globals()[f"{tier_type}_MANDATORY_ADX_DMI"]
        req_ema, req_stoch = globals()[f"{tier_type}_MANDATORY_EMA_SPREAD"], globals()[f"{tier_type}_MANDATORY_STOCHASTIC"]
        min_score = globals()[f"{tier_type}_MINIMUM_SCORE"]

    c_price_bull, c_price_bear = df[f"Renko_Bull_{tf_str}"].astype(int), df[f"Renko_Bear_{tf_str}"].astype(int)
    c_vol_bull, c_vol_bear = df[f"Vol_Renko_Bull_{tf_str}"].astype(int), df[f"Vol_Renko_Bear_{tf_str}"].astype(int)
    c_rsi_bull, c_rsi_bear = (df["RSI"] >= df["RSI_SMA"]).astype(int), (df["RSI"] <= df["RSI_SMA"]).astype(int)
    c_adx_bull, c_adx_bear = ((df["ADX"] >= ADX_THRESHOLD) & (df["+DI"] > df["-DI"])).astype(int), ((df["ADX"] >= ADX_THRESHOLD) & (df["-DI"] > df["+DI"])).astype(int)
    c_ema_bull, c_ema_bear = df["EMA_Bull_Expanded"].astype(int), df["EMA_Bear_Expanded"].astype(int)
    c_stoch_bull, c_stoch_bear = df["Stoch_Bull_Pass"].astype(int), df["Stoch_Bear_Pass"].astype(int)

    df[f"Score_Bull_{tf_str}"] = c_price_bull + c_vol_bull + c_rsi_bull + c_adx_bull + c_ema_bull + c_stoch_bull
    df[f"Score_Bear_{tf_str}"] = c_price_bear + c_vol_bear + c_rsi_bear + c_adx_bear + c_ema_bear + c_stoch_bear

    bull_veto, bear_veto = pd.Series(False, index=df.index), pd.Series(False, index=df.index)
    if req_price: bull_veto, bear_veto = bull_veto | (c_price_bull == 0), bear_veto | (c_price_bear == 0)
    if req_vol: bull_veto, bear_veto = bull_veto | (c_vol_bull == 0), bear_veto | (c_vol_bear == 0)
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
    df_tf = apply_dual_tier_scorecard(df_tf, tf_str, "MACRO")

    df_tf["Eval_Time"] = df_tf["Datetime"] + pd.to_timedelta(tf_str)
    
    # Exporting Macro Renko Counts specifically for the Exit Engine
    export_cols = [
        "Symbol", "Eval_Time", 
        f"Armed_Bull_{tf_str}", f"Armed_Bear_{tf_str}", 
        f"Score_Bull_{tf_str}", f"Score_Bear_{tf_str}", 
        f"Renko_Count_{tf_str}", f"Vol_Renko_Count_{tf_str}",
        "ATR", "ADX"
    ]
    env_df = df_tf[export_cols].copy().rename(columns={"Eval_Time": "Datetime", "ATR": f"ATR_{tf_str}", "ADX": f"ADX_{tf_str}"})
    return env_df.sort_values("Datetime").reset_index(drop=True)


# ==============================================================================
# 4. MICRO EXECUTION TAPE & CONFLUENCE MATCHER
# ==============================================================================
def prepare_unified_execution_tape(rolling_master_df, micro_tf, macro_timeframes):
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
    df_micro = apply_dual_tier_scorecard(df_micro, micro_tf, "MICRO")
    df_micro = df_micro.sort_values("Datetime").reset_index(drop=True)

    bull_gate_cols, bear_gate_cols = [], []
    for tf in macro_timeframes:
        print(f"   ├─ Evaluating Macro Context Gates + Price/Vol Renko for [{tf}]...")
        env_df = evaluate_single_timeframe_gates(rolling_master_df, tf)
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

    df_micro["Master_Armed_Bull"] = df_micro[bull_gate_cols].all(axis=1)
    df_micro["Master_Armed_Bear"] = df_micro[bear_gate_cols].all(axis=1)
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
# 5. TRADE MANAGEMENT: THE RENKO EXIT ENGINE
# ==============================================================================
def scan_institutional_tape(target_date_str):
    global API_ERROR_LOGGED

    print(f"\n📡 Initiating Dual-Tier Execution & Exit Engine for {target_date_str}...")
    universe = get_dynamic_fno_universe()
    if not universe: return

    trading_days = get_past_trading_days(target_date_str, num_days=BACKTRACE_DAYS)
    if not trading_days: return

    target_dt = pd.to_datetime(target_date_str)
    current_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
    is_live_today = target_date_str == current_now.strftime("%Y-%m-%d")

    print(f"🚀 Multithreading Bulk Ingestion for {len(universe)} symbols (20 days lookback)...")
    fetch_tasks = [(item, trading_days[0], target_date_str, is_live_today) for item in universe]
    historical_dfs = []

    def fetch_worker(task):
        global API_ERROR_LOGGED
        item, start_date, end_date, live = task
        key = urllib.parse.quote(item["key"])
        access_token = os.environ.get("UPSTOX_ACCESS_TOKEN")
        headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}
        dfs = []
        hist_end = end_date if not live else (current_now - timedelta(days=1)).strftime("%Y-%m-%d")

        for attempt in range(3):
            try:
                res = requests.get(f"https://api.upstox.com/v2/historical-candle/{key}/1minute/{hist_end}/{start_date}", headers=headers, timeout=15)
                if res.status_code == 200:
                    data = res.json().get("data", {}).get("candles")
                    if data: dfs.append(pd.DataFrame(data, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume", "OI"]))
                    break
                elif res.status_code == 429: time.sleep(1.5)
                else: break
            except Exception: time.sleep(1)

        if live:
            for attempt in range(3):
                try:
                    res = requests.get(f"https://api.upstox.com/v2/historical-candle/intraday/{key}/1minute", headers=headers, timeout=15)
                    if res.status_code == 200 and res.json().get("data", {}).get("candles"):
                        dfs.append(pd.DataFrame(res.json()["data"]["candles"], columns=["Timestamp", "Open", "High", "Low", "Close", "Volume", "OI"]))
                    break
                except Exception: time.sleep(1)

        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            df["Datetime"] = pd.to_datetime(df["Timestamp"]).dt.tz_localize(None)
            df = df.drop_duplicates(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)
            df["Symbol"] = item["symbol"]
            return df
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_worker, task): task for task in fetch_tasks}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            sys.stdout.write(f"\r📡 Fetching Data... {completed}/{len(fetch_tasks)} symbols processed")
            sys.stdout.flush()
            res = future.result()
            if res is not None: historical_dfs.append(res)
    print()

    rolling_master_df = pd.concat(historical_dfs, ignore_index=True)
    print("⚙️ Computing Scorecards & Renko Matrices...")

    tape_exec = prepare_unified_execution_tape(rolling_master_df, MICRO_TIMEFRAME, MACRO_TIMEFRAMES)
    if GLOBAL_MACRO_STRATEGY_2D == "BULLISH": tape_exec["Master_Armed_Bear"] = False
    elif GLOBAL_MACRO_STRATEGY_2D == "BEARISH": tape_exec["Master_Armed_Bull"] = False

    # Extract Data Dictionaries for O(1) State Lookups (Both Micro and Macro layers)
    all_anomalies = tape_exec[tape_exec["Direction"] != 0].copy()
    anomalies_by_time = all_anomalies.groupby("Datetime")

    closes_dict = tape_exec.set_index(["Datetime", "Symbol"])["Close"].to_dict()
    
    # Micro Renko Dicts
    micro_price_renko = tape_exec.set_index(["Datetime", "Symbol"])[f"Renko_Count_{MICRO_TIMEFRAME}"].to_dict()
    micro_vol_renko = tape_exec.set_index(["Datetime", "Symbol"])[f"Vol_Renko_Count_{MICRO_TIMEFRAME}"].to_dict()
    
    # Macro Renko Dicts (using the primary structural timeframe)
    primary_macro = MACRO_TIMEFRAMES[-1]
    macro_price_renko = tape_exec.set_index(["Datetime", "Symbol"])[f"Renko_Count_{primary_macro}"].to_dict()
    macro_vol_renko = tape_exec.set_index(["Datetime", "Symbol"])[f"Vol_Renko_Count_{primary_macro}"].to_dict()
    
    all_times = np.sort(tape_exec["Datetime"].unique())
    memory_bank = {}

    for t in all_times:
        t_dt = pd.to_datetime(t)
        
        # 1. Manage Active Trades (The Dual-Layered OR-Condition Exit Logic)
        for sym, st in memory_bank.items():
            if st["state"] == "ACTIVE":
                ltp = closes_dict.get((t_dt, sym))
                
                # Fetch counts for both layers
                mi_p_count = micro_price_renko.get((t_dt, sym), 0)
                mi_v_count = micro_vol_renko.get((t_dt, sym), 0)
                ma_p_count = macro_price_renko.get((t_dt, sym), 0)
                ma_v_count = macro_vol_renko.get((t_dt, sym), 0)

                if ltp is not None:
                    exit_reason = None
                    if st["dir"] == 1: # LONG EXIT LOGIC
                        if mi_p_count <= -MICRO_EXIT_PRICE_BRICKS: exit_reason = "Micro Price Reversal"
                        elif mi_v_count <= -MICRO_EXIT_VOL_BRICKS: exit_reason = "Micro Volume Reversal"
                        elif ma_p_count <= -MACRO_EXIT_PRICE_BRICKS: exit_reason = "Macro Price Break"
                        elif ma_v_count <= -MACRO_EXIT_VOL_BRICKS: exit_reason = "Macro Volume Break"
                        
                    elif st["dir"] == -1: # SHORT EXIT LOGIC
                        if mi_p_count >= MICRO_EXIT_PRICE_BRICKS: exit_reason = "Micro Price Reversal"
                        elif mi_v_count >= MICRO_EXIT_VOL_BRICKS: exit_reason = "Micro Volume Reversal"
                        elif ma_p_count >= MACRO_EXIT_PRICE_BRICKS: exit_reason = "Macro Price Break"
                        elif ma_v_count >= MACRO_EXIT_VOL_BRICKS: exit_reason = "Macro Volume Break"
                    
                    if exit_reason:
                        st["state"] = "EXITED"
                        st["exit_time"] = t_dt.strftime("%Y-%m-%d %H:%M")
                        st["exit_price"] = ltp
                        st["exit_reason"] = exit_reason

        # 2. Process New Entrances (Entries)
        if t_dt in anomalies_by_time.groups:
            for _, row in anomalies_by_time.get_group(t_dt).iterrows():
                sym = row["Symbol"]
                direction = row["Direction"]
                # Enter if no active trade exists (allows re-entry after exit)
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
                        "macro_score": row.get(f"Score_Bull_{primary_macro}" if direction == 1 else f"Score_Bear_{primary_macro}", 0),
                        "micro_score": row.get(f"Score_Bull_{MICRO_TIMEFRAME}" if direction == 1 else f"Score_Bear_{MICRO_TIMEFRAME}", 0)
                    }

        # 3. EOD Force Exit (15:15)
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
    # 6. TERMINAL OUTPUT (TRADE DASHBOARD)
    # ==============================================================================
    active_runners = {sym: st for sym, st in memory_bank.items() if st["state"] == "ACTIVE"}
    closed_trades = [{**st, "sym": sym} for sym, st in memory_bank.items() if st["state"] == "EXITED" and st["date"] == target_date_str]

    tf_display_str = " + ".join(MACRO_TIMEFRAMES)
    print(f"\n{COLOR_CYAN}================================================================================================{COLOR_RESET}")
    print(f"{COLOR_BOLD}DUAL-TIER SCORECARD + DUAL-LAYER RENKO TRAILING EXIT [{MICRO_TIMEFRAME} Micro ⚡ {tf_display_str} Macro]{COLOR_RESET}")
    print(f"{COLOR_CYAN}================================================================================================{COLOR_RESET}\n")

    if active_runners:
        print(f"{COLOR_BOLD}🟢 BASKET 1: ACTIVE RUNNERS (Riding the Trend){COLOR_RESET}")
        for sym, st in active_runners.items():
            ltp = final_ltp_dict.get(sym, st["origin"])
            pnl_pct = ((ltp - st["origin"]) / st["origin"]) * 100 if st["dir"] == 1 else ((st["origin"] - ltp) / st["origin"]) * 100
            color = COLOR_GREEN if pnl_pct >= 0 else COLOR_RED
            d_str = "BULLISH" if st["dir"] == 1 else "BEARISH"
            
            print(f"  {color}⚡ {sym:<12} Open P&L: {pnl_pct:+.2f}% ({d_str}){COLOR_RESET}")
            print(f"      └─ 🎯 Macro Score : {st['macro_score']}/{MACRO_MINIMUM_SCORE} | Micro Score : {st['micro_score']}/{MICRO_MINIMUM_SCORE}")
            print(f"      └─ ⚓ Entry        : {st['date']} @ {st['time']} | Price: ₹{st['origin']:.2f}")
            print(f"      └─ 📈 Current LTP  : ₹{ltp:.2f} (Trailing via Dual-Layer OR Condition)\n")

    if closed_trades:
        print(f"{COLOR_BOLD}🛑 BASKET 2: CLOSED TRADES (Renko Structure Broken){COLOR_RESET}")
        for st in closed_trades:
            pnl_pct = ((st["exit_price"] - st["origin"]) / st["origin"]) * 100 if st["dir"] == 1 else ((st["origin"] - st["exit_price"]) / st["origin"]) * 100
            color = COLOR_GREEN if pnl_pct >= 0 else COLOR_RED
            d_str = "BULLISH" if st["dir"] == 1 else "BEARISH"

            print(f"  {color}🛑 {st['sym']:<12} Final P&L: {pnl_pct:+.2f}% ({d_str}){COLOR_RESET}")
            print(f"      └─ ⚓ Entry        : {st['date']} @ {st['time']} | Price: ₹{st['origin']:.2f}")
            print(f"      └─ 🎯 Exit         : {st['exit_time']} | Price: ₹{st['exit_price']:.2f}")
            print(f"      └─ 📉 Reason       : {st['exit_reason']}\n")

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
        target_dt = datetime.utcnow() + timedelta(hours=5, minutes=30)
        if target_dt.weekday() == 5: target_dt -= timedelta(days=1)
        elif target_dt.weekday() == 6: target_dt -= timedelta(days=2)
        target_date_str = target_dt.strftime("%Y-%m-%d")
    else:
        target_date_str = datetime.strptime(raw_date_str, "%Y-%m-%d").strftime("%Y-%m-%d")

    if not os.environ.get("UPSTOX_ACCESS_TOKEN"):
        print(f"❌ {COLOR_RED}Error: UPSTOX_ACCESS_TOKEN environment variable not found.{COLOR_RESET}")
        return

    scan_institutional_tape(target_date_str)

if __name__ == "__main__":
    run_production_sweep()
