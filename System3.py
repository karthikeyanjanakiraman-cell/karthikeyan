"""system3.py - Asit Baran Pati Multi-Timeframe Trading System Implementation

Production-Grade Universal N-Timeframe & Dual-Tier 45-Degree Renko Engine:
- Configurable Micro Execution Timeframe (e.g., "1min", "3min", "5min")
- Configurable Macro Hierarchy Array (e.g., ["15min", "60min", "1D"])
- 45-Degree Geometric Renko across BOTH Macro Structure & Micro Execution
- Solves 09:15-10:15 AM Opening Gap via Pilot Timeframe & Prior-Day Macro Carryover
- Persistent Momentum State Gates (Fixes single-bar Stochastic/EMA bottleneck)
- Complete Indicator Gauntlet (BB-RSI, ADXBO, 8/21 EMA Expansion, ATR-Stochastic)
- Zero-Lookahead Vectorized Alignment via pd.merge_asof (Strict Timestamp Sorting Fixed)
- State-Based 4-Phase Memory Bank (Intrusions, Reloads, Breaches, Reclaims)
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
MAX_BREACH_DAYS = 0

API_ERROR_LOGGED = False

# ==============================================================================
# ★ GLOBAL CONFIGURATION: DYNAMIC TIMEFRAMES & INDICATORS ★
# ==============================================================================
# 1. Configurable Timeframe Hierarchy
MICRO_TIMEFRAME = "1min"  # Micro Execution & Tactical Trigger
MACRO_TIMEFRAMES = [
    "5min",
    # "30min",
    # "45min",
    # "60min",
]  # Strategic Structural & Pilot Tiers (N-Timeframe Array)
MACRO_STRATEGIC_WINDOW = "2D"  # Multi-Day Trend Horizon

# 2. Indicator Parameters
ATR_PERIOD = 14
RSI_PERIOD = 14
BB_SMA_PERIOD = 20
BB_STD_DEV = 2.0
ADX_PERIOD = 14
ADX_THRESHOLD = 20
STOCH_PERIOD = 14

# 3. 45-Degree Renko Parameters (Dual-Tier Calibration)
MICRO_RENKO_CONFIRM_BRICKS = 2  # Micro Tactical Trigger (2-Brick Rule)
MACRO_RENKO_CONFIRM_BRICKS = 1  # Macro Structural Trend Confirmation
RENKO_MIN_BRICK = 0.05
RENKO_DEFAULT_PCT = 0.005

# 4. Risk Management & Strategy Bias
BREACH_PURGE_PCT = 0.015
GLOBAL_MACRO_STRATEGY_2D = "BOTH"  # Options: "BULLISH", "BEARISH", or "BOTH"


# ==============================================================================
# 1. LIVE INGESTION (F&O Universe & Parallel Bulk Fetching)
# ==============================================================================
def get_dynamic_fno_universe():
  """Fetches all active underlying F&O equities and indices from NSE."""
  nse_url = (
      "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
  )
  try:
    response = requests.get(nse_url, timeout=15)
    if response.status_code != 200:
      return []
    nse_data = json.load(gzip.GzipFile(fileobj=io.BytesIO(response.content)))
    fno_underlying = {
        item.get("underlying_symbol")
        for item in nse_data
        if item.get("segment") == "NSE_FO" and item.get("underlying_symbol")
    }
    return [
        {
            "symbol": item.get("trading_symbol"),
            "key": item.get("instrument_key"),
        }
        for item in nse_data
        if item.get("segment") in ("NSE_EQ", "NSE_INDEX")
        and item.get("trading_symbol") in fno_underlying
    ]
  except Exception as e:
    print(
        f"{COLOR_RED}[API Error] Failed to fetch F&O universe:"
        f" {e}{COLOR_RESET}"
    )
    return []


def get_past_trading_days(target_date_str, num_days=20):
  """Computes historical trading calendar skipping weekends."""
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
  except Exception:
    return []


# ==============================================================================
# 2. 45-DEGREE RENKO GEOMETRIC CONSTRUCTION ENGINE
# ==============================================================================
def construct_45deg_renko_matrix(df, tf_name, confirm_bricks=2):
  """Builds dynamic, time-independent 45-degree Renko diagonal bricks.

  Enforces:
  - Unit slope (Delta Y / Delta X = 1 => 45 degrees)
  - 2 x Brick Size trend reversal hysteresis
  - Directional brick count tracking
  """
  renko_trends = np.zeros(len(df))
  renko_counts = np.zeros(len(df))

  for sym, indices in df.groupby("Symbol").indices.items():
    sub_closes = df["Close"].values[indices]
    sub_atrs = (
        df["ATR"].values[indices]
        if "ATR" in df.columns
        else (sub_closes * RENKO_DEFAULT_PCT)
    )

    if len(sub_closes) > 0:
      trends = np.zeros(len(sub_closes))
      counts = np.zeros(len(sub_closes))
      curr_trend = 0
      curr_count = 0
      curr_price = sub_closes[0]

      for i in range(1, len(sub_closes)):
        bs = max(sub_atrs[i], RENKO_MIN_BRICK)
        move = sub_closes[i] - curr_price

        if curr_trend >= 0:
          # Continuation Bullish
          if move >= bs:
            bricks = int(move // bs)
            curr_trend = 1
            curr_count = curr_count + bricks if curr_count > 0 else bricks
            curr_price += bricks * bs
          # Reversal Bearish (Requires 2 * BS displacement)
          elif move <= -(2 * bs):
            bricks = int(abs(move) // bs)
            curr_trend = -1
            curr_count = -bricks
            curr_price -= bricks * bs
        else:
          # Continuation Bearish
          if move <= -bs:
            bricks = int(abs(move) // bs)
            curr_trend = -1
            curr_count = curr_count - bricks if curr_count < 0 else -bricks
            curr_price -= bricks * bs
          # Reversal Bullish (Requires 2 * BS displacement)
          elif move >= (2 * bs):
            bricks = int(move // bs)
            curr_trend = 1
            curr_count = bricks
            curr_price += bricks * bs

        trends[i] = curr_trend
        counts[i] = curr_count

      renko_trends[indices] = trends
      renko_counts[indices] = counts

  df[f"Renko_Trend_{tf_name}"] = renko_trends
  df[f"Renko_Brick_Count_{tf_name}"] = renko_counts
  df[f"Renko_Bull_{tf_name}"] = renko_counts >= confirm_bricks
  df[f"Renko_Bear_{tf_name}"] = renko_counts <= -confirm_bricks
  return df


# ==============================================================================
# 3. UNIVERSAL MODULAR GATE EVALUATOR (Applied to ANY Timeframe)
# ==============================================================================
def evaluate_single_timeframe_gates(df_base, tf_str):
  """Resamples atomic data to tf_str, computes all technical gates + 45-deg Renko,

  and forward-shifts timestamps to eliminate lookahead bias.
  """
  # 1. Resample to Target Timeframe
  df_tf = (
      df_base.groupby([
          "Symbol",
          pd.Grouper(
              key="Datetime", freq=tf_str, closed="left", label="left"
          ),
      ])
      .agg({
          "Open": "first",
          "High": "max",
          "Low": "min",
          "Close": "last",
          "Volume": "sum",
      })
      .reset_index()
  )

  df_tf = df_tf.dropna(subset=["Close"]).sort_values(["Symbol", "Datetime"])

  # 2. True Range & ATR-14 Baseline
  df_tf["H-L"] = df_tf["High"] - df_tf["Low"]
  df_tf["H-PC"] = (
      df_tf["High"] - df_tf.groupby("Symbol")["Close"].shift(1)
  ).abs()
  df_tf["L-PC"] = (
      df_tf["Low"] - df_tf.groupby("Symbol")["Close"].shift(1)
  ).abs()
  df_tf["TR"] = df_tf[["H-L", "H-PC", "L-PC"]].max(axis=1)
  atr_series = df_tf.groupby("Symbol")["TR"].transform(
      lambda x: x.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()
  )
  df_tf["ATR"] = atr_series

  # 3. RSI-14 & Bollinger Bands on RSI (20, 2)
  delta = df_tf.groupby("Symbol")["Close"].diff()
  gain = delta.where(delta > 0, 0)
  loss = -delta.where(delta < 0, 0)
  avg_gain = gain.groupby(df_tf["Symbol"]).transform(
      lambda x: x.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
  )
  avg_loss = loss.groupby(df_tf["Symbol"]).transform(
      lambda x: x.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
  )
  df_tf["RSI"] = 100 - (100 / (1 + (avg_gain / (avg_loss + 1e-8))))

  df_tf["RSI_SMA"] = df_tf.groupby("Symbol")["RSI"].transform(
      lambda x: x.rolling(BB_SMA_PERIOD, min_periods=1).mean()
  )
  df_tf["RSI_STD"] = df_tf.groupby("Symbol")["RSI"].transform(
      lambda x: x.rolling(BB_SMA_PERIOD, min_periods=1).std()
  )
  df_tf["BB_Upper"] = df_tf["RSI_SMA"] + (BB_STD_DEV * df_tf["RSI_STD"])
  df_tf["BB_Lower"] = df_tf["RSI_SMA"] - (BB_STD_DEV * df_tf["RSI_STD"])

  # 4. ADX Directional Breakout (ADXBO)
  high_diff = df_tf["High"] - df_tf.groupby("Symbol")["High"].shift(1)
  low_diff = df_tf.groupby("Symbol")["Low"].shift(1) - df_tf["Low"]
  df_tf["+DM"] = np.where(
      (high_diff > low_diff) & (high_diff > 0), high_diff, 0
  )
  df_tf["-DM"] = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

  df_tf["+DI"] = (
      100
      * (
          df_tf.groupby("Symbol")["+DM"].transform(
              lambda x: x.ewm(alpha=1 / ADX_PERIOD, adjust=False).mean()
          )
          / (atr_series + 1e-8)
      )
  )
  df_tf["-DI"] = (
      100
      * (
          df_tf.groupby("Symbol")["-DM"].transform(
              lambda x: x.ewm(alpha=1 / ADX_PERIOD, adjust=False).mean()
          )
          / (atr_series + 1e-8)
      )
  )
  df_tf["DX"] = (
      100
      * abs(df_tf["+DI"] - df_tf["-DI"])
      / (df_tf["+DI"] + df_tf["-DI"] + 1e-8)
  )
  df_tf["ADX"] = df_tf.groupby("Symbol")["DX"].transform(
      lambda x: x.ewm(alpha=1 / ADX_PERIOD, adjust=False).mean()
  )

  # 5. 8/21 EMA Fast-Line Alignment & Expansion Gate
  df_tf["EMA_8"] = (
      df_tf.groupby("Symbol")["Close"]
      .transform(lambda x: x.ewm(span=8, adjust=False).mean())
  )
  df_tf["EMA_21"] = (
      df_tf.groupby("Symbol")["Close"]
      .transform(lambda x: x.ewm(span=21, adjust=False).mean())
  )
  df_tf["EMA_Spread"] = abs(df_tf["EMA_8"] - df_tf["EMA_21"])
  spread_thresh = (
      df_tf.groupby("Symbol")["EMA_Spread"].transform(
          lambda x: x.rolling(window=20, min_periods=1).mean()
      )
      * 0.20
  )

  df_tf["EMA_Bull_Expanded"] = (df_tf["EMA_8"] > df_tf["EMA_21"]) & (
      df_tf["EMA_Spread"] >= spread_thresh
  )
  df_tf["EMA_Bear_Expanded"] = (df_tf["EMA_8"] < df_tf["EMA_21"]) & (
      df_tf["EMA_Spread"] >= spread_thresh
  )

  # 6. ATR-Stochastic Volatility & Momentum State Gate
  lowest_low = df_tf.groupby("Symbol")["Low"].transform(
      lambda x: x.rolling(window=STOCH_PERIOD, min_periods=1).min()
  )
  highest_high = df_tf.groupby("Symbol")["High"].transform(
      lambda x: x.rolling(window=STOCH_PERIOD, min_periods=1).max()
  )
  df_tf["Stoch_K"] = (
      (df_tf["Close"] - lowest_low) / (highest_high - lowest_low + 1e-9)
  ) * 100

  atr_median = df_tf.groupby("Symbol")["ATR"].transform(
      lambda x: x.rolling(window=50, min_periods=1).median()
  )
  df_tf["Vol_Pass"] = df_tf["ATR"] >= (atr_median * 0.75)

  df_tf["Stoch_Bull_Pass"] = (df_tf["Stoch_K"] >= 50) & df_tf["Vol_Pass"]
  df_tf["Stoch_Bear_Pass"] = (df_tf["Stoch_K"] <= 50) & df_tf["Vol_Pass"]

  # 7. 45-Degree Macro Structural Renko Construction
  df_tf = construct_45deg_renko_matrix(
      df_tf, tf_name=tf_str, confirm_bricks=MACRO_RENKO_CONFIRM_BRICKS
  )

  # 8. Timeframe Master Permission Gauntlet
  bull_col = f"Armed_Bull_{tf_str}"
  bear_col = f"Armed_Bear_{tf_str}"

  df_tf[bull_col] = (
      (df_tf["RSI"] >= df_tf["RSI_SMA"])
      & (df_tf["ADX"] >= ADX_THRESHOLD)
      & (df_tf["+DI"] > df_tf["-DI"])
      & df_tf["EMA_Bull_Expanded"]
      & df_tf["Stoch_Bull_Pass"]
      & df_tf[f"Renko_Bull_{tf_str}"]
  )

  df_tf[bear_col] = (
      (df_tf["RSI"] <= df_tf["RSI_SMA"])
      & (df_tf["ADX"] >= ADX_THRESHOLD)
      & (df_tf["-DI"] > df_tf["+DI"])
      & df_tf["EMA_Bear_Expanded"]
      & df_tf["Stoch_Bear_Pass"]
      & df_tf[f"Renko_Bear_{tf_str}"]
  )

  # Forward-shift timestamp by timeframe interval to eliminate lookahead bias
  shift_delta = pd.to_timedelta(tf_str)
  df_tf["Eval_Time"] = df_tf["Datetime"] + shift_delta

  export_cols = ["Symbol", "Eval_Time", bull_col, bear_col, "ATR", "ADX"]
  env_df = df_tf[export_cols].copy()
  env_df = env_df.rename(
      columns={
          "Eval_Time": "Datetime",
          "ATR": f"ATR_{tf_str}",
          "ADX": f"ADX_{tf_str}",
      }
  )
  # Strict monotonic time sort required for pd.merge_asof
  env_df = env_df.sort_values("Datetime").reset_index(drop=True)
  return env_df


# ==============================================================================
# 4. MICRO EXECUTION TAPE & CONFLUENCE MATCHER
# ==============================================================================
def prepare_unified_execution_tape(
    rolling_master_df, micro_tf, macro_timeframes
):
  """Builds atomic micro tape, maps all macro gates with morning handoff logic,

  and constructs the micro 45-degree Renko tactical execution trigger.
  """
  # 1. Resample to Configured Micro Timeframe
  if micro_tf != "1min":
    df_micro = (
        rolling_master_df.groupby([
            "Symbol",
            pd.Grouper(
                key="Datetime", freq=micro_tf, closed="left", label="left"
            ),
        ])
        .agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        })
        .reset_index()
    )
    df_micro = df_micro.dropna(subset=["Close"]).sort_values(
        ["Symbol", "Datetime"]
    )
  else:
    df_micro = rolling_master_df.sort_values(["Symbol", "Datetime"]).copy()

  # Baseline micro ATR
  df_micro["H-L"] = df_micro["High"] - df_micro["Low"]
  df_micro["H-PC"] = (
      df_micro["High"] - df_micro.groupby("Symbol")["Close"].shift(1)
  ).abs()
  df_micro["L-PC"] = (
      df_micro["Low"] - df_micro.groupby("Symbol")["Close"].shift(1)
  ).abs()
  df_micro["TR"] = df_micro[["H-L", "H-PC", "L-PC"]].max(axis=1)
  df_micro["ATR"] = df_micro.groupby("Symbol")["TR"].transform(
      lambda x: x.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()
  )
  df_micro["ATR"] = df_micro["ATR"].fillna(
      df_micro["Close"] * RENKO_DEFAULT_PCT
  )

  bull_gate_cols = []
  bear_gate_cols = []

  # CRITICAL: pd.merge_asof requires DataFrame sorted purely by the 'on' time key
  df_micro = df_micro.sort_values("Datetime").reset_index(drop=True)

  # 2. Resample and Merge Each Configured Macro Timeframe
  for tf in macro_timeframes:
    print(f"   ├─ Evaluating Confluence Gates + 45° Renko for [{tf}]...")
    env_df = evaluate_single_timeframe_gates(rolling_master_df, tf)

    bull_col = f"Armed_Bull_{tf}"
    bear_col = f"Armed_Bear_{tf}"
    bull_gate_cols.append(bull_col)
    bear_gate_cols.append(bear_col)

    # Backward merge guarantees prior-day macro state carryover between 09:15-10:15 AM
    df_micro = pd.merge_asof(
        df_micro, env_df, on="Datetime", by="Symbol", direction="backward"
    )
    df_micro[bull_col] = df_micro[bull_col].fillna(False)
    df_micro[bear_col] = df_micro[bear_col].fillna(False)

  # 3. Master Multi-Timeframe Confluence Logical AND
  df_micro["Master_Armed_Bull"] = df_micro[bull_gate_cols].all(axis=1)
  df_micro["Master_Armed_Bear"] = df_micro[bear_gate_cols].all(axis=1)

  # 4. Construct Micro 45-Degree Renko Tactical Trigger
  # Re-sort by Symbol and Datetime for contiguous sequential Renko processing
  df_micro = df_micro.sort_values(["Symbol", "Datetime"]).reset_index(drop=True)

  df_micro = construct_45deg_renko_matrix(
      df_micro, tf_name=micro_tf, confirm_bricks=MICRO_RENKO_CONFIRM_BRICKS
  )

  df_micro["Trigger_Bull"] = (
      df_micro["Master_Armed_Bull"] & df_micro[f"Renko_Bull_{micro_tf}"]
  )
  df_micro["Trigger_Bear"] = (
      df_micro["Master_Armed_Bear"] & df_micro[f"Renko_Bear_{micro_tf}"]
  )

  df_micro["Trigger_Bull_Prev"] = (
      df_micro.groupby("Symbol")["Trigger_Bull"].shift(1).fillna(False)
  )
  df_micro["Trigger_Bear_Prev"] = (
      df_micro.groupby("Symbol")["Trigger_Bear"].shift(1).fillna(False)
  )

  df_micro["New_Bull"] = df_micro["Trigger_Bull"] & ~df_micro[
      "Trigger_Bull_Prev"
  ]
  df_micro["New_Bear"] = df_micro["Trigger_Bear"] & ~df_micro[
      "Trigger_Bear_Prev"
  ]

  df_micro["Direction"] = np.where(
      df_micro["New_Bull"], 1, np.where(df_micro["New_Bear"], -1, 0)
  )

  # Final chronological sort for live tape simulation
  return df_micro.sort_values("Datetime").reset_index(drop=True)


# ==============================================================================
# 5. LIGHTNING STATE-BASED MEMORY ENGINE
# ==============================================================================
def scan_institutional_tape(target_date_str):
  """Executes the complete multi-timeframe backtrace and real-time state analysis."""
  global API_ERROR_LOGGED

  print(
      f"\n📡 Initiating Multi-Timeframe Zero-Lag Engine for {target_date_str}..."
  )
  universe = get_dynamic_fno_universe()
  if not universe:
    print(f"⚠️ {COLOR_RED}No F&O universe found.{COLOR_RESET}")
    return

  trading_days = get_past_trading_days(
      target_date_str, num_days=BACKTRACE_DAYS
  )
  if not trading_days:
    return

  target_dt = pd.to_datetime(target_date_str)
  current_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
  is_live_today = target_date_str == current_now.strftime("%Y-%m-%d")

  print(
      f"🚀 Multithreading Bulk Ingestion for {len(universe)} symbols (20 days"
      " lookback)..."
  )
  fetch_tasks = [
      (item, trading_days[0], target_date_str, is_live_today)
      for item in universe
  ]
  historical_dfs = []

  def fetch_worker(task):
    global API_ERROR_LOGGED
    item, start_date, end_date, live = task
    key = urllib.parse.quote(item["key"])
    access_token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
    }

    dfs = []
    hist_end = (
        end_date
        if not live
        else (current_now - timedelta(days=1)).strftime("%Y-%m-%d")
    )

    for attempt in range(3):
      try:
        url = f"https://api.upstox.com/v2/historical-candle/{key}/1minute/{hist_end}/{start_date}"
        res = requests.get(url, headers=headers, timeout=15)

        if res.status_code == 429:
          time.sleep(1.5)
          continue
        elif res.status_code == 200:
          data = res.json().get("data", {}).get("candles")
          if data:
            dfs.append(
                pd.DataFrame(
                    data,
                    columns=[
                        "Timestamp",
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Volume",
                        "OI",
                    ],
                )
            )
          break
        else:
          if not API_ERROR_LOGGED:
            print(
                f"\n\n{COLOR_RED}❌ [UPSTOX API REJECTION] HTTP"
                f" {res.status_code}{COLOR_RESET}"
            )
            print(f"{COLOR_YELLOW}Response Message: {res.text}{COLOR_RESET}\n")
            API_ERROR_LOGGED = True
          break
      except Exception:
        time.sleep(1)

    if live:
      for attempt in range(3):
        try:
          res = requests.get(
              f"https://api.upstox.com/v2/historical-candle/intraday/{key}/1minute",
              headers=headers,
              timeout=15,
          )
          if res.status_code == 429:
            time.sleep(1.5)
            continue
          if (
              res.status_code == 200
              and res.json().get("data", {}).get("candles")
          ):
            dfs.append(
                pd.DataFrame(
                    res.json()["data"]["candles"],
                    columns=[
                        "Timestamp",
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Volume",
                        "OI",
                    ],
                )
            )
          break
        except Exception:
          time.sleep(1)

    if dfs:
      df = pd.concat(dfs, ignore_index=True)
      df["Datetime"] = pd.to_datetime(df["Timestamp"]).dt.tz_localize(None)
      df = (
          df.drop_duplicates(subset=["Datetime"])
          .sort_values("Datetime")
          .reset_index(drop=True)
      )
      df["Symbol"] = item["symbol"]
      return df
    return None

  with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = {
        executor.submit(fetch_worker, task): task for task in fetch_tasks
    }
    completed = 0
    for future in concurrent.futures.as_completed(futures):
      completed += 1
      sys.stdout.write(
          f"\r📡 Fetching Data... {completed}/{len(fetch_tasks)} symbols"
          " processed"
      )
      sys.stdout.flush()
      res = future.result()
      if res is not None:
        historical_dfs.append(res)
  print()

  if not historical_dfs:
    print(
        f"⚠️ {COLOR_RED}Fatal Error: No data retrieved. Check the API Rejection"
        f" message above.{COLOR_RESET}"
    )
    return

  rolling_master_df = pd.concat(historical_dfs, ignore_index=True)

  # Global Strategy Bias Log
  global GLOBAL_MACRO_STRATEGY_2D
  print(
      f"{COLOR_CYAN}[Macro Bias] Configured to: {GLOBAL_MACRO_STRATEGY_2D} |"
      f" Micro TF: [{MICRO_TIMEFRAME}] | Macro Hierarchy:"
      f" {MACRO_TIMEFRAMES}{COLOR_RESET}"
  )
  print(
      f"⚙️ Computing N-Timeframe Confluence Matrix & Dual-Tier 45° Renko"
      " Construction..."
  )

  tape_exec = prepare_unified_execution_tape(
      rolling_master_df, MICRO_TIMEFRAME, MACRO_TIMEFRAMES
  )

  if GLOBAL_MACRO_STRATEGY_2D == "BULLISH":
    tape_exec["Master_Armed_Bear"] = False
  elif GLOBAL_MACRO_STRATEGY_2D == "BEARISH":
    tape_exec["Master_Armed_Bull"] = False

  all_anomalies = tape_exec[tape_exec["Direction"] != 0].copy()
  anomalies_by_time = all_anomalies.groupby("Datetime")

  closes_dict = tape_exec.set_index(["Datetime", "Symbol"])["Close"].to_dict()
  all_times = np.sort(tape_exec["Datetime"].unique())

  memory_bank = {}
  historical_times = [
      t for t in all_times if pd.to_datetime(t).date() < target_dt.date()
  ]

  for t in historical_times:
    t_dt = pd.to_datetime(t)
    if t_dt.hour == 9 and t_dt.minute == 15:
      day_str = t_dt.strftime("%Y-%m-%d")
      for sym, st in memory_bank.items():
        ltp = closes_dict.get((t_dt, sym))
        if ltp and st["state"] == "ACTIVE":
          if (st["dir"] == 1 and ltp < st["origin"]) or (
              st["dir"] == -1 and ltp > st["origin"]
          ):
            st["state"], st["breach_time"] = (
                "BREACHED",
                f"{day_str} 09:15 (GAP)",
            )

    for sym, st in memory_bank.items():
      ltp = closes_dict.get((t_dt, sym))
      if ltp:
        if st["state"] == "ACTIVE" and (
            (st["dir"] == 1 and ltp < st["origin"])
            or (st["dir"] == -1 and ltp > st["origin"])
        ):
          st["state"], st["breach_time"] = "BREACHED", t_dt.strftime(
              "%Y-%m-%d %H:%M"
          )
        elif st["state"] == "BREACHED" and (
            (st["dir"] == 1 and ltp >= st["origin"])
            or (st["dir"] == -1 and ltp <= st["origin"])
        ):
          st["state"], st["breach_time"] = "ACTIVE", None

    if t_dt in anomalies_by_time.groups:
      for _, row in anomalies_by_time.get_group(t_dt).iterrows():
        sym = row["Symbol"]
        if sym not in memory_bank:
          memory_bank[sym] = {
              "state": "ACTIVE",
              "origin": row["Close"],
              "date": t_dt.strftime("%Y-%m-%d"),
              "time": t_dt.strftime("%H:%M"),
              "dir": row["Direction"],
              "breach_time": None,
          }

    if t_dt.hour == 15 and t_dt.minute == 15:
      for sym in list(memory_bank.keys()):
        ltp = closes_dict.get((t_dt, sym))
        if ltp and memory_bank[sym]["state"] == "BREACHED":
          if (
              memory_bank[sym]["dir"] == 1
              and ltp
              < memory_bank[sym]["origin"] * (1.0 - BREACH_PURGE_PCT)
          ) or (
              memory_bank[sym]["dir"] == -1
              and ltp
              > memory_bank[sym]["origin"] * (1.0 + BREACH_PURGE_PCT)
          ):
            del memory_bank[sym]

  today_times = [
      t for t in all_times if pd.to_datetime(t).date() == target_dt.date()
  ]
  if not today_times:
    print(
        f"\n{COLOR_YELLOW}[Terminal Standby] Market data for {target_date_str}"
        f" is empty or not available yet.{COLOR_RESET}\n"
    )
    return

  today_master = tape_exec[tape_exec["Datetime"].dt.date == target_dt.date()]
  morning_opens = (
      today_master[
          today_master["Datetime"].dt.time
          == pd.to_datetime("09:15").time()
      ]
      .set_index("Symbol")["Open"]
      .to_dict()
  )

  all_fresh_intrusions, all_reloads, all_reclaims = {}, {}, {}

  for t in today_times:
    t_dt = pd.to_datetime(t)
    for sym, st in memory_bank.items():
      ltp = closes_dict.get((t_dt, sym))
      if ltp:
        if st["state"] == "ACTIVE" and (
            (st["dir"] == 1 and ltp < st["origin"])
            or (st["dir"] == -1 and ltp > st["origin"])
        ):
          st["state"], st["breach_time"] = "BREACHED", t_dt.strftime(
              "%Y-%m-%d %H:%M"
          )
        elif st["state"] == "BREACHED" and (
            (st["dir"] == 1 and ltp >= st["origin"])
            or (st["dir"] == -1 and ltp <= st["origin"])
        ):
          st["state"], st["breach_time"] = "ACTIVE", None

    if t_dt in anomalies_by_time.groups:
      for _, row in anomalies_by_time.get_group(t_dt).iterrows():
        sym, price, direction = row["Symbol"], row["Close"], row["Direction"]
        if sym not in memory_bank:
          if sym not in all_fresh_intrusions:
            row["Eval_Time_Str"] = t_dt.strftime("%H:%M")
            all_fresh_intrusions[sym] = row
            memory_bank[sym] = {
                "state": "ACTIVE",
                "origin": price,
                "date": target_date_str,
                "time": row["Eval_Time_Str"],
                "dir": direction,
                "breach_time": None,
            }
        else:
          st = memory_bank[sym]
          row["Net_Drift"] = (
              ((price - st["origin"]) / st["origin"] * 100)
              if st["dir"] == 1
              else ((st["origin"] - price) / st["origin"] * 100)
          )
          if st["state"] == "ACTIVE" and direction == st["dir"]:
            row["Eval_Time_Str"] = t_dt.strftime("%H:%M")
            row["Macro_Price"], row["Macro_Date"], row["Micro_Price"] = (
                st["origin"],
                st["date"],
                price,
            )
            all_reloads[sym] = row
          elif st["state"] == "BREACHED" and direction == st["dir"]:
            st["state"], st["breach_time"] = "ACTIVE", None
            row["Eval_Time_Str"] = t_dt.strftime("%H:%M")
            row["Origin"], row["First_Date"] = st["origin"], st["date"]
            all_reclaims[sym] = row

  final_ltp_dict = today_master.groupby("Symbol")["Close"].last().to_dict()
  breached = []

  for sym, st in memory_bank.items():
    if (
        st["state"] == "BREACHED"
        and sym in final_ltp_dict
        and sym not in all_reclaims
    ):
      breached.append({
          "Symbol": sym,
          "LTP": final_ltp_dict[sym],
          "Origin": st["origin"],
          "Dir": "BULLISH" if st["dir"] == 1 else "BEARISH",
          "Time": st["breach_time"],
          "First_Date": st["date"],
          "Anchor_Time": st.get("time", "09:15"),
      })

  # ==============================================================================
  # 6. TERMINAL OUTPUT
  # ==============================================================================
  tf_display_str = " + ".join(MACRO_TIMEFRAMES)
  print(
      f"\n{COLOR_CYAN}================================================================================================{COLOR_RESET}"
  )
  print(
      f"{COLOR_BOLD}UNIVERSAL CONFLUENCE TAPE [{MICRO_TIMEFRAME} Micro ⚡"
      f" {tf_display_str} Macro] | DATE: {target_date_str}{COLOR_RESET}"
  )
  print(
      f"{COLOR_CYAN}================================================================================================{COLOR_RESET}\n"
  )

  if all_fresh_intrusions:
    print(
        f"{COLOR_BOLD}⚡ BASKET 1: FRESH INTRUSIONS (Phase 1 - Day-1"
        f" Births){COLOR_RESET}"
    )
    for sym, row in all_fresh_intrusions.items():
      ltp = row["Close"]
      pct_move = (
          (ltp - morning_opens.get(sym, ltp)) / morning_opens.get(sym, ltp)
      ) * 100
      color, d_str = (
          (COLOR_GREEN, "BULLISH")
          if row["Direction"] == 1
          else (COLOR_RED, "BEARISH")
      )
      print(
          f"  {color}🚨 {sym:<12} Day Move: {pct_move:+.2f}%"
          f" ({d_str}){COLOR_RESET}"
      )
      print(
          f"      └─ 🎯 Macro Structural Alignment   : [{tf_display_str}]"
          f" [BB-RSI + ADX + EMA + Stoch + 45° Renko >= {MACRO_RENKO_CONFIRM_BRICKS}B]"
      )
      print(
          f"      └─ 🔫 {MICRO_TIMEFRAME} Tactical 45° Renko Trigger : >="
          f" {MICRO_RENKO_CONFIRM_BRICKS} Bricks"
      )
      print(
          "      └─ ⚓ Zero-Lag Anchor              :"
          f" {target_date_str} @ {row['Eval_Time_Str']} | Price: ₹{ltp:.2f}"
      )
      print(
          "      └─ 🎯 Latest LTP                 :"
          f" {target_date_str} @ EOD   | Price:"
          f" ₹{final_ltp_dict.get(sym, ltp):.2f}\n"
      )

  if all_reloads:
    print(
        f"{COLOR_BOLD}🔄 BASKET 2: ALGORITHMIC RELOADS (Phase 2 - Institutional"
        f" Continuations){COLOR_RESET}"
    )
    for sym, row in all_reloads.items():
      ltp = row["Close"]
      pct_move = (
          (ltp - morning_opens.get(sym, ltp)) / morning_opens.get(sym, ltp)
      ) * 100
      color, d_str = (
          (COLOR_GREEN, "BULLISH")
          if row["Direction"] == 1
          else (COLOR_RED, "BEARISH")
      )
      print(
          f"  {color}🔄 {sym:<12} Day Move: {pct_move:+.2f}%"
          f" ({d_str}){COLOR_RESET}"
      )
      print(
          f"      └─ 🎯 Macro Structural Alignment   : [{tf_display_str}]"
          f" [BB-RSI + ADX + EMA + Stoch + 45° Renko >= {MACRO_RENKO_CONFIRM_BRICKS}B]"
      )
      print(
          f"      └─ 🔫 {MICRO_TIMEFRAME} Tactical 45° Renko Trigger : >="
          f" {MICRO_RENKO_CONFIRM_BRICKS} Bricks"
      )
      print(
          "      └─ ⚓ Macro Floor (Origin)       :"
          f" {row['Macro_Date']} @ {memory_bank[sym].get('time', '09:15')} |"
          f" Price: ₹{row['Macro_Price']:.2f}"
      )
      print(
          "      └─ ⚡ Micro Floor (Reload)        :"
          f" {target_date_str} @ {row['Eval_Time_Str']} | Price:"
          f" ₹{row['Micro_Price']:.2f}"
      )
      print(
          "      └─ 🎯 Latest LTP                 :"
          f" {target_date_str} @ EOD   | Price:"
          f" ₹{final_ltp_dict.get(sym, ltp):.2f} (Trend Drift:"
          f" {row['Net_Drift']:+.2f}%)\n"
      )

  if breached:
    print(
        f"{COLOR_DIM}⚠️ BASKET 3: BREACHED PIVOTS (Phase 3 - Trapped Capital /"
        f" Dead Trends){COLOR_RESET}"
    )
    for b in breached:
      print(
          f"  {COLOR_YELLOW}⚠️ {b['Symbol']:<12} {b['Dir']}"
          f" Anchor shattered!{COLOR_RESET}"
      )
      print(
          "      └─ ⚓ Anchor :"
          f" {b['First_Date']} @ {b['Anchor_Time']} | LTP: ₹{b['Origin']:.2f}"
      )
      print(
          "      └─ 🎯 Latest : Breached At"
          f" {b.get('Time', 'Pending')} | Current LTP: ₹{b['LTP']:.2f}\n"
      )

  if all_reclaims:
    print(
        f"{COLOR_BOLD}🪤 BASKET 4: INSTITUTIONAL RECLAIMS (Phase 4 - Liquidity"
        f" Traps){COLOR_RESET}"
    )
    for sym, row in all_reclaims.items():
      ltp = row["Close"]
      pct_move = (
          (ltp - morning_opens.get(sym, ltp)) / morning_opens.get(sym, ltp)
      ) * 100
      d_str = "BULLISH" if row["Direction"] == 1 else "BEARISH"
      print(
          f"  {COLOR_MAGENTA}🔥 {sym:<12} Day Move: {pct_move:+.2f}%"
          f" ({d_str}){COLOR_RESET}"
      )
      print(
          f"      └─ 🎯 Macro Structural Alignment   : [{tf_display_str}]"
          f" [BB-RSI + ADX + EMA + Stoch + 45° Renko >= {MACRO_RENKO_CONFIRM_BRICKS}B]"
      )
      print(
          f"      └─ 🔫 {MICRO_TIMEFRAME} Tactical 45° Renko Trigger : >="
          f" {MICRO_RENKO_CONFIRM_BRICKS} Bricks"
      )
      print(
          "      └─ ⚓ Anchor :"
          f" {row['First_Date']} @ {memory_bank[sym].get('time', '09:15')} | LTP:"
          f" ₹{row['Origin']:.2f}"
      )
      print(
          "      └─ 🎯 Latest : Reclaimed At"
          f" {target_date_str} @ {row['Eval_Time_Str']} | LTP: ₹{ltp:.2f}\n"
      )

  if not any([all_fresh_intrusions, all_reloads, all_reclaims, breached]):
    print(
        f"{COLOR_DIM}[Terminal Silent] No active institutional structure"
        f" passing strict filters.{COLOR_RESET}\n"
    )


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
    if target_dt.weekday() == 5:
      target_dt -= timedelta(days=1)
    elif target_dt.weekday() == 6:
      target_dt -= timedelta(days=2)
    target_date_str = target_dt.strftime("%Y-%m-%d")
  else:
    target_date_str = datetime.strptime(raw_date_str, "%Y-%m-%d").strftime(
        "%Y-%m-%d"
    )

  if not os.environ.get("UPSTOX_ACCESS_TOKEN"):
    print(
        f"❌ {COLOR_RED}Error: UPSTOX_ACCESS_TOKEN environment variable not"
        f" found.{COLOR_RESET}"
    )
    return

  scan_institutional_tape(target_date_str)


if __name__ == "__main__":
  run_production_sweep()
