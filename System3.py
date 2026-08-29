"""system3.py - Asit Baran Pati Multi-Timeframe Trading System Implementation

Production-Grade Universal N-Timeframe & Dual-Tier 45-Degree Renko Engine (v20)
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

print("🔖 SYSTEM3 BUILD: v20-ULTIMATE-LIVE-WS (2026-08-29)")

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
    "HDFCGOLD", "GOLDBEES", "SILVERBEES", "LIQUIDBEES", "NIFTYBEES", "BANKBEES"
}

# ==============================================================================
# 🎛️ TIER 0: TRADING MODE, PIPELINE ROUTING & DATA FEED SWITCH
# ==============================================================================
# "REST" for historical backtesting, "WEBSOCKET" for true live tick execution
DATA_FEED_MODE = "WEBSOCKET"           
TRADING_MODE = "CASH_EQUITY"      
ENABLE_STAGE1_STOCK_FILTER = False  

MIN_STOCK_PRICE = 100.0
MAX_STOCK_PRICE = 600.0
MIN_STOCK_VOLUME = 1500000  # PURGES ILLIQUID SMALL CAPS

# ==============================================================================
# GLOBAL CONFIGURATION: BALANCED TIMEFRAMES & INDICATORS
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
MACRO_MANDATORY_ADX_DMI        = True    # PURGES SIDEWAYS / DEAD MARKETS
MACRO_MANDATORY_EMA_SPREAD     = False
MACRO_MANDATORY_STOCHASTIC     = False
MACRO_MANDATORY_ATR_BB         = False   
MACRO_MANDATORY_RENKO_BB       = False   
MACRO_MINIMUM_SCORE            = 5       # REQUIRES MAJORITY CONFLUENCE

# ==============================================================================
# TIER 2: MICRO EXECUTION SWITCHBOARD (THE SNIPER) - 9 PILLARS
# ==============================================================================
SYNC_MICRO_WITH_MACRO          = False
MICRO_MANDATORY_LIVE_PERCENTILE = 50.0    # REQUIRES > 50% AGGRESSIVE DELTA VOL
MICRO_MANDATORY_PRICE_RENKO    = True    
MICRO_MANDATORY_VOL_RENKO      = True    
MICRO_MANDATORY_RENKO_VELOCITY = False
MICRO_MANDATORY_RSI_BB         = False
MICRO_MANDATORY_ADX_DMI        = False
MICRO_MANDATORY_EMA_SPREAD     = False
MICRO_MANDATORY_STOCHASTIC     = False
MICRO_MANDATORY_ATR_BB         = False   
MICRO_MANDATORY_RENKO_BB       = False   
MICRO_MINIMUM_SCORE            = 5       # REQUIRES MAJORITY CONFLUENCE

# ==============================================================================
# TIER 3: TRADE MANAGEMENT & TEMPORAL GATES (EXIT & TIMING)
# ==============================================================================
MICRO_EXIT_PRICE_BRICKS = 5              
MICRO_EXIT_VOL_BRICKS   = 30
MACRO_EXIT_PRICE_BRICKS = 2              
MACRO_EXIT_VOL_BRICKS   = 20
RENKO_VELOCITY_MAX_BARS = 8              
ENTRY_CUTOFF_TIME = "14:15"              
MAX_DAILY_TRADES_PER_SYMBOL = 2          

# ==============================================================================
# AUTHENTICATION
# ==============================================================================
def get_fyers_auth_headers():
    return {"Authorization": f"{os.environ.get('FYERS_CLIENT_ID', '')}:{os.environ.get('FYERS_ACCESS_TOKEN', '')}"}

def validate_fyers_token():
    if not os.environ.get("FYERS_CLIENT_ID") or not os.environ.get("FYERS_ACCESS_TOKEN"):
        print(f"❌ {COLOR_RED}Error: FYERS_CLIENT_ID or FYERS_ACCESS_TOKEN not found.{COLOR_RESET}")
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
    except Exception:
        return False

# ==============================================================================
# DATA INGESTION (REST)
# ==============================================================================
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
    return spot_inst

def filter_cash_equities_by_price_range(universe, target_date_str):
    target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
    prev_dt = target_dt - timedelta(days=1)
    while prev_dt.weekday() >= 5: prev_dt -= timedelta(days=1)
    prev_day = prev_dt.strftime("%Y-%m-%d")
    lookback_start = (prev_dt - timedelta(days=7)).strftime("%Y-%m-%d")

    print(f"🧹 Applying Range (₹{MIN_STOCK_PRICE}-₹{MAX_STOCK_PRICE}) & Vol (>{MIN_STOCK_VOLUME}) Filter...")
    def worker(item):
        df = fetch_fyers_candles(item["key"], lookback_start, prev_day, resolution="D")
        if df is not None and not df.empty:
            last = df.iloc[-1]
            if MIN_STOCK_PRICE <= last["Close"] <= MAX_STOCK_PRICE and last["Volume"] >= MIN_STOCK_VOLUME:
                return item
        return None
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        filtered = [r for r in executor.map(worker, universe) if r is not None]
    print(f"  ├─ ✅ {len(filtered)}/{len(universe)} equities passed filters.")
    return filtered

def get_past_trading_days(target_date_str, num_days=5):
    try:
        target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
        days, curr = [], target_dt
        while len(days) < num_days:
            if curr.weekday() < 5: days.append(curr.strftime("%Y-%m-%d"))
            curr -= timedelta(days=1)
        days.reverse()
        return days
    except Exception: return []

def fetch_fyers_candles(key, start_dt, end_dt, resolution="1"):
    headers = get_fyers_auth_headers()
    for attempt in range(3):
        try:
            url = f"https://api-t1.fyers.in/data/history?symbol={urllib.parse.quote(key, safe=':')}&resolution={resolution}&date_format=1&range_from={start_dt}&range_to={end_dt}"
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code == 200:
                data = res.json()
                if data.get("s") == "ok" and data.get("candles"):
                    df = pd.DataFrame(data["candles"], columns=["Epoch", "Open", "High", "Low", "Close", "Volume"])
                    df["Datetime"] = pd.to_datetime(df["Epoch"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None).astype("datetime64[ns]")
                    return df
            elif res.status_code in (429, 500, 502, 503):
                time.sleep(random.uniform(0.5, 1.5) * (attempt + 1))
        except Exception: time.sleep(1)
    return None

def compute_base_net_delta(df):
    if 'Net_Delta_1m' not in df.columns:
        df['Wick_Spread'] = df['High'] - df['Low']
        df['Wick_Spread'] = df['Wick_Spread'].replace(0, 1e-9)
        df['Net_Delta_1m'] = df['Volume'] * ((df['Close'] - df['Open']) / df['Wick_Spread'])
    return df

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

# ==============================================================================
# TECHNICALS & RENKO MATRIX
# ==============================================================================
def calculate_core_technicals(df_tf):
    df_tf["H-L"] = df_tf["High"] - df_tf["Low"]
    df_tf["H-PC"] = (df_tf["High"] - df_tf.groupby("Symbol")["Close"].shift(1)).abs()
    df_tf["L-PC"] = (df_tf["Low"] - df_tf.groupby("Symbol")["Close"].shift(1)).abs()
    df_tf["TR"] = df_tf[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df_tf["ATR"] = df_tf.groupby("Symbol")["TR"].transform(lambda x: x.rolling(window=ATR_PERIOD, min_periods=1).mean()).fillna(df_tf["Close"] * RENKO_DEFAULT_PCT)

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

    ll = df_tf.groupby("Symbol")["Low"].transform(lambda x: x.rolling(STOCH_PERIOD, min_periods=1).min())
    hh = df_tf.groupby("Symbol")["High"].transform(lambda x: x.rolling(STOCH_PERIOD, min_periods=1).max())
    df_tf["Stoch_K"] = ((df_tf["Close"] - ll) / (hh - ll + 1e-9)) * 100
    atr_median = df_tf.groupby("Symbol")["ATR"].transform(lambda x: x.rolling(50, min_periods=1).median())
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

    # TRUE INTERNAL DELTA PERCENTAGE MATH
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
    
    # Internal Delta Percentage
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

        # 0. INTRADAY OVERNIGHT FLUSH
        if t_dt.time() == pd.Timestamp("09:15").time():
            for sym, episodes in memory_bank.items():
                if episodes and episodes[-1]["state"] == "ACTIVE":
                    episodes[-1]["state"] = "EXITED"
                    episodes[-1]["exit_time"] = t_dt.strftime("%Y-%m-%d %H:%M")
                    episodes[-1]["exit_price"] = closes_dict.get((t_dt, sym), episodes[-1]["origin"])
                    episodes[-1]["exit_reason"] = "Overnight Gap Flush"

        # 1. EXITS
        for sym, episodes in memory_bank.items():
            if episodes and episodes[-1]["state"] == "ACTIVE":
                st = episodes[-1]
                ltp = closes_dict.get((t_dt, sym))
                if ltp is not None:
                    exit_reason = None
                    mins_in_trade = (t_dt - pd.to_datetime(f"{st['date']} {st['time']}")).total_seconds() / 60
                    mi_p_count = micro_p_renko.get((t_dt, sym), 0)
                    mi_v_count = micro_v_renko.get((t_dt, sym), 0)
                    
                    if mi_p_count != st["current_renko_count"]:
                        st["last_brick_formed_dt"], st["current_renko_count"] = t_dt, mi_p_count
                    
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

        # 2. ENTRIES
        if t_dt in anomalies_by_time.groups and t_dt.time() < cutoff_time_obj:
            for _, row in anomalies_by_time.get_group(t_dt).iterrows():
                sym, direction = row["Symbol"], row["Direction"]
                existing = memory_bank.get(sym, [])
                if existing and existing[-1]["state"] == "ACTIVE": continue
                if len(existing) >= MAX_DAILY_TRADES_PER_SYMBOL: continue # HARD CAP

                last_brick = row.get("Last_Brick_Time")
                if pd.isna(last_brick) or last_brick.date() != t_dt.date(): continue # STALE MOMENTUM

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

        # 3. EOD SQUARE OFF
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
        
        # Live State Memory
        self.live_candles = {}    # sym -> {Open, High, Low, Close, Volume, Delta}
        self.last_tick_ltp = {}   
        self.last_tick_vol = {}
        
        # Thread sync
        self.lock = threading.Lock()
        self.fyers_ws = None
        self.access_token = f"{os.environ.get('FYERS_CLIENT_ID')}:{os.environ.get('FYERS_ACCESS_TOKEN')}"
        self.symbols = list(historical_df["Symbol"].unique())
        
        # Map raw symbols to Fyers WS syntax (e.g. "NSE:SBIN-EQ")
        self.ws_symbols = [f"NSE:{s}-EQ" for s in self.symbols]

    def onmessage(self, messages):
        with self.lock:
            for msg in messages:
                if 'symbol' not in msg or 'ltp' not in msg: continue
                sym_raw = msg['symbol'].replace("NSE:", "").replace("-EQ", "")
                if sym_raw not in self.symbols: continue
                
                ltp = float(msg['ltp'])
                vol_today = float(msg.get('vol_traded_today', 0))
                
                prev_ltp = self.last_tick_ltp.get(sym_raw, ltp)
                prev_vol = self.last_tick_vol.get(sym_raw, vol_today)
                tick_vol = vol_today - prev_vol if vol_today >= prev_vol else 0
                
                # TRUE INSTITUTIONAL TICK DELTA
                tick_delta = 0
                if ltp > prev_ltp: tick_delta = tick_vol
                elif ltp < prev_ltp: tick_delta = -tick_vol
                
                self.last_tick_ltp[sym_raw] = ltp
                self.last_tick_vol[sym_raw] = vol_today
                
                if sym_raw not in self.live_candles:
                    self.live_candles[sym_raw] = {"Open": ltp, "High": ltp, "Low": ltp, "Close": ltp, "Volume": tick_vol, "Net_Delta_1m": tick_delta}
                else:
                    c = self.live_candles[sym_raw]
                    c["High"] = max(c["High"], ltp)
                    c["Low"] = min(c["Low"], ltp)
                    c["Close"] = ltp
                    c["Volume"] += tick_vol
                    c["Net_Delta_1m"] += tick_delta

    def onerror(self, message): print(f"{COLOR_RED}[WS Error] {message}{COLOR_RESET}")
    def onclose(self, message): print(f"{COLOR_YELLOW}[WS Closed] Reconnecting...{COLOR_RESET}")
    def onopen(self): 
        print(f"{COLOR_GREEN}[WS Connected] Subscribing to {len(self.ws_symbols)} instruments.{COLOR_RESET}")
        self.fyers_ws.subscribe(data_type="SymbolUpdate", symbol=self.ws_symbols)

    def start_socket(self):
        self.fyers_ws = data_ws.FyersDataSocket(
            access_token=self.access_token, log_path="", litemode=False, write_to_file=False,
            reconnect=True, on_connect=self.onopen, on_close=self.onclose,
            on_error=self.onerror, on_message=self.onmessage
        )
        self.fyers_ws.connect()

    def run_event_loop(self):
        print(f"\n{COLOR_CYAN}⚡ LIVE ENGINE ARMED. Awaiting candle closes...{COLOR_RESET}")
        ws_thread = threading.Thread(target=self.start_socket, daemon=True)
        ws_thread.start()
        
        current_minute = datetime.now().minute
        while True:
            time.sleep(1)
            now = datetime.now()
            
            # Exactly on the minute rollover (e.g. 10:05:00)
            if now.minute != current_minute:
                current_minute = now.minute
                rounded_dt = now.replace(second=0, microsecond=0) - timedelta(minutes=1)
                
                with self.lock:
                    new_rows = []
                    for sym, c in self.live_candles.items():
                        new_rows.append({"Datetime": rounded_dt, "Symbol": sym, **c})
                    self.live_candles.clear()
                
                if new_rows:
                    df_new = pd.DataFrame(new_rows)
                    self.historical_df = pd.concat([self.historical_df, df_new], ignore_index=True)
                    
                    # RUN SYSTEM LOGIC ON NEW DATA
                    tape_exec = prepare_unified_execution_tape(self.historical_df, MICRO_TIMEFRAME, MACRO_TIMEFRAMES, GLOBAL_MACRO_STRATEGY_2D)
                    memory_bank = _run_dual_layer_trade_management(tape_exec, MICRO_TIMEFRAME, MACRO_TIMEFRAMES, self.cutoff_obj)
                    
                    # CLEAR TERMINAL & RE-PRINT
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print(f"📡 Last Update: {now.strftime('%H:%M:%S')} | Mode: WEBSOCKET LIVE TICK")
                    display_final_results(tape_exec, memory_bank, now, self.target_date_str)


# ==============================================================================
# PIPELINE ROUTER
# ==============================================================================
def scan_institutional_tape(target_date_str, entry_cutoff_time_str=ENTRY_CUTOFF_TIME):
    print(f"\n📡 Initiating REST Pipeline [{TRADING_MODE}] for {target_date_str}...")
    trading_days = get_past_trading_days(target_date_str, num_days=BACKTRACE_DAYS)
    
    cutoff_time_obj = pd.to_datetime(f"{target_date_str} {entry_cutoff_time_str}").time()
    universe = filter_cash_equities_by_price_range(get_cash_equity_universe(), target_date_str)
    
    if not universe: return
    
    fetch_tasks = [(item, trading_days[0], target_date_str) for item in universe]
    stock_dfs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        for res in executor.map(fetch_stock_bars_worker, fetch_tasks):
            if res is not None: stock_dfs.append(res)
            
    if not stock_dfs: return
    stock_master_df = pd.concat(stock_dfs, ignore_index=True)

    if DATA_FEED_MODE == "WEBSOCKET":
        if not WS_AVAILABLE:
            print(f"{COLOR_RED}❌ fyers_apiv3 not installed. Fallback to REST.{COLOR_RESET}")
        else:
            engine = LiveWebsocketEngine(stock_master_df, target_date_str, entry_cutoff_time_str)
            engine.run_event_loop()
            return # Blocks forever in live loop

    # REST HISTORICAL EXECUTION
    stock_master_df = truncate_to_cutoff(stock_master_df, target_date_str, pd.to_datetime(f"{target_date_str} 15:30:00"))
    tape_exec = prepare_unified_execution_tape(stock_master_df, MICRO_TIMEFRAME, MACRO_TIMEFRAMES, GLOBAL_MACRO_STRATEGY_2D)
    
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
