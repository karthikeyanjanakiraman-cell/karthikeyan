"""system3.py - Asit Baran Pati Multi-Timeframe Trading System Implementation

Production-Grade Universal N-Timeframe & Dual-Tier 45-Degree Renko Engine:
- Configurable Micro Execution Timeframe (e.g., "1min", "3min", "5min")
- Configurable Macro Hierarchy Array (e.g., ["15min", "60min", "1D"])
- Phase 1 Blueprint: Dual-Tier Scorecard (7 Pillars) & Global Mandatory Veto Switches
- Phase 1 Blueprint: Order Flow / Cumulative Volume Delta 45-Degree Renko
- Phase 1 Blueprint: Renko-Velocity Engine (Time-Distance Momentum Tracking)
- EXIT STRATEGY: Dual-Layered (Triggering Macro + Micro) + Velocity Stall Cutoff
- TRUE BIRTH TIME TRACKING: Locks in the original structural ignition timestamp and qualifying macro TFs
- OPTIONS TRANSLATION LAYER: Stock-chart signals are translated into ATM CE/PE
  contracts (CE on bullish, PE on bearish) so Basket 1/2 report real option premium
  P&L, not stock P&L, while the signal engine itself still runs on the stock chart.
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

# 🔖 BUILD MARKER — if this line does NOT appear at the very top of your CI
# log, the workflow is not running this file. Check this FIRST before
# re-reporting any traceback.
print("🔖 SYSTEM3 BUILD: fyers-options-translation-v3 (2026-08-22) — "
      "if you don't see this line, your CI is running a different/stale file.")

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

BACKTRACE_DAYS = 15

EXCLUDED_INDICES = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX", "BANKEX", "NIFTY50", "NIFTYBANK"}

# 🌟 DIAGNOSTICS: surfaces the FIRST few real Fyers API errors (status code,
# error code/message) instead of silently swallowing them, which previously
# made "zero data" failures (expired token, bad symbol, etc.) invisible.
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
    """Real token validation via Fyers' /profile endpoint (not just an env-var
    presence check). Fyers access tokens expire daily — a stale token makes
    every fetch fail silently, so this fails fast with a clear reason instead
    of crashing downstream on an empty dataframe."""
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
            print(f"{COLOR_YELLOW}   -> Your FYERS_ACCESS_TOKEN is almost certainly expired/invalid. "
                  f"Fyers access tokens expire daily and must be regenerated each trading day.{COLOR_RESET}")
            return False

        fy_name = body.get("data", {}).get("name", "Unknown")
        print(f"{COLOR_GREEN}✅ Fyers token validated OK (Account: {fy_name}){COLOR_RESET}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"{COLOR_RED}❌ Could not reach Fyers to validate the token: {e}{COLOR_RESET}")
        return False

# ==============================================================================
# GLOBAL CONFIGURATION: DYNAMIC TIMEFRAMES & INDICATORS
# ==============================================================================
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
MACRO_RENKO_CONFIRM_BRICKS = 0
RENKO_MIN_BRICK = 0.05
RENKO_DEFAULT_PCT = 0.005

GLOBAL_MACRO_STRATEGY_2D = "BOTH"  # "BULLISH", "BEARISH", or "BOTH"

# ==============================================================================
# TIER 1: MACRO CONTEXT SWITCHBOARD (THE GENERAL) - 7 PILLARS
# ==============================================================================
MACRO_MANDATORY_PRICE_RENKO    = True
MACRO_MANDATORY_VOL_RENKO      = True
MACRO_MANDATORY_RENKO_VELOCITY = True
MACRO_MANDATORY_RSI_BB         = False
MACRO_MANDATORY_ADX_DMI        = False
MACRO_MANDATORY_EMA_SPREAD     = False
MACRO_MANDATORY_STOCHASTIC     = False
MACRO_MINIMUM_SCORE            = 3

# ==============================================================================
# TIER 2: MICRO EXECUTION SWITCHBOARD (THE SNIPER) - 7 PILLARS
# ==============================================================================
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
# TIER 3: TRADE MANAGEMENT & TEMPORAL GATES (EXIT & TIMING)
# ==============================================================================
MICRO_EXIT_PRICE_BRICKS = 5
MICRO_EXIT_VOL_BRICKS   = 5
MACRO_EXIT_PRICE_BRICKS = 1
MACRO_EXIT_VOL_BRICKS   = 1

RENKO_VELOCITY_MAX_BARS = 12
ENTRY_CUTOFF_TIME = "15:00"

# ==============================================================================
# TIER 4: OPTIONS TRANSLATION LAYER CONFIG
# ==============================================================================
OPTIONS_TARGET_EXPIRY = "CURRENT"   # "CURRENT" or "NEXT"
MIN_OPT_PREMIUM_SANITY = 0.5        # Skip a trade if the fetched premium looks broken (near-zero)


# ==============================================================================
# 1. LIVE INGESTION (FYERS): F&O Universe + Options Chain in one CSV pass
# ==============================================================================
def get_fno_universe_and_options():
    """Downloads Fyers' NSE_CM.csv (equities) and NSE_FO.csv (F&O) once, and
    returns (spot_instruments, options_by_underlying) for the signal engine
    and the ATM strike selector respectively."""
    print("📡 Fetching Master Instrument Matrix via FYERS...")
    spot_inst, opt_inst = [], []
    try:
        print("  ├─ Downloading & Parsing FYERS F&O Data...")
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
                # Symbol name sits 3 columns left of CE/PE (NOT 2 — that column
                # is a raw numeric underlying instrument token, e.g. 26037 =
                # NIFTY FIN SERVICE, which used to leak through as a bogus name).
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
                    "symbol": sym_ticker,
                    "key": sym_ticker,
                    "underlying": base_symbol,
                    "type": opt_type,
                    "strike": strike_val,
                    "expiry": expiry_date
                })

                if base_symbol not in valid_underlyings:
                    valid_underlyings.add(base_symbol)
                    # Only trust a CONFIRMED equity ticker from the CM file —
                    # never guess a fallback "-EQ" symbol (that let non-equity
                    # index names like NIFTYNXT50/NIFTYFPI leak into the universe).
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
# 1B. FYERS CANDLE FETCHER (shared by stock signal engine + option premiums)
# ==============================================================================
def fetch_fyers_candles(key, start_dt, end_dt):
    """Fetch 1-min candles for a Fyers symbol over a date range. Returns a
    DataFrame with a 'Datetime' column, or None. Retries on 429/5xx, surfaces
    real API errors, and short-circuits on auth failures (code -16)."""
    headers = get_fyers_auth_headers()
    for attempt in range(3):
        try:
            time.sleep(0.2)
            encoded_symbol = urllib.parse.quote(key, safe=":")
            url = (f"https://api-t1.fyers.in/data/history?symbol={encoded_symbol}"
                   f"&resolution=1&date_format=1&range_from={start_dt}&range_to={end_dt}")
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
                    return None  # auth error — retrying won't help

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


# ==============================================================================
# 1C. OPTIONS TRANSLATION LAYER: ATM SELECTION, PREMIUM FETCH
# ==============================================================================
def select_atm_option(symbol, spot_price, opt_type, options_by_underlying, target_date_str):
    """Pick the ATM contract of the requested type (CE/PE) on the nearest
    tradable expiry (>= target date) for a given underlying and spot price."""
    opts = options_by_underlying.get(symbol, [])
    same_type = [o for o in opts if o["type"] == opt_type]
    if not same_type:
        return None

    target_dt = pd.to_datetime(target_date_str)
    valid = [o for o in same_type if pd.to_datetime(o["expiry"]) >= target_dt]
    if not valid:
        valid = same_type

    expiries_sorted = sorted(set(pd.to_datetime(o["expiry"]) for o in valid))
    if not expiries_sorted:
        return None
    chosen_expiry = expiries_sorted[0] if OPTIONS_TARGET_EXPIRY == "CURRENT" else \
        (expiries_sorted[1] if len(expiries_sorted) > 1 else expiries_sorted[0])

    same_expiry = [o for o in valid if pd.to_datetime(o["expiry"]) == chosen_expiry]
    if not same_expiry:
        return None

    return min(same_expiry, key=lambda o: abs(o["strike"] - spot_price))


def fetch_option_day_series(option_key, target_date_str, is_live_today=None, current_now=None):
    """Fetch 1-min premium candles for a single option contract on the target
    date. Returns (sorted_timestamps_list, {timestamp: close}) — empty if no data.
    (is_live_today/current_now kept as optional args for call-site compatibility;
    Fyers' date-range API returns live-forming candles automatically when the
    range includes today, so no separate "intraday" endpoint is needed.)"""
    df = fetch_fyers_candles(option_key, target_date_str, target_date_str)
    if df is None or df.empty:
        return [], {}
    df = df.drop_duplicates(subset=["Datetime"]).sort_values("Datetime")
    price_map = dict(zip(df["Datetime"], df["Close"]))
    sorted_ts = sorted(price_map.keys())
    return sorted_ts, price_map


def lookup_price_at_or_before(sorted_ts, price_map, t):
    """Nearest available option price at-or-before timestamp t. None if t is
    before the first available candle (e.g. contract didn't trade yet)."""
    if not sorted_ts:
        return None
    idx = bisect.bisect_right(sorted_ts, t) - 1
    if idx < 0:
        return None
    return price_map[sorted_ts[idx]]


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
# 3. DUAL-TIER SCORECARD SYSTEM (OUT OF 7 PILLARS)
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

    df_tf["Eval_Time"] = (df_tf["Datetime"] + pd.to_timedelta(tf_str)).astype("datetime64[ns]")

    export_cols = [
        "Symbol", "Eval_Time",
        f"Armed_Bull_{tf_str}", f"Armed_Bear_{tf_str}",
        f"Score_Bull_{tf_str}", f"Score_Bear_{tf_str}",
        f"Renko_Count_{tf_str}", f"Vol_Renko_Count_{tf_str}",
        f"Bars_Since_Brick_{tf_str}",
        "ATR", "ADX"
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

    # Applied HERE (before Trigger/Direction are derived) so the mode switch
    # actually takes effect, rather than being overwritten too late downstream.
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
# 5. TRADE MANAGEMENT: STOCK SIGNAL -> ATM OPTION TRANSLATION -> EXIT TRACKING
# ==============================================================================
def scan_institutional_tape(target_date_str):

    print(f"\n📡 Initiating Dual-Tier Execution & Exit Engine for {target_date_str}...")
    universe, options_by_underlying = get_fno_universe_and_options()
    if not universe:
        print(f"{COLOR_RED}[Error] No spot instruments mapped — cannot proceed.{COLOR_RESET}")
        return
    if not options_by_underlying:
        print(f"{COLOR_RED}[Error] No options chain data available — cannot translate stock signals into strikes.{COLOR_RESET}")
        return

    trading_days = get_past_trading_days(target_date_str, num_days=BACKTRACE_DAYS)
    if not trading_days: return

    target_dt = pd.to_datetime(target_date_str)
    current_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
    is_live_today = target_date_str == current_now.strftime("%Y-%m-%d")

    print(f"🚀 Multithreading Bulk Ingestion for {len(universe)} symbols (stock chart = signal engine)...")
    fetch_tasks = [(item, trading_days[0], target_date_str) for item in universe]
    historical_dfs = []

    def fetch_worker(task):
        item, start_date, end_date = task
        # Fyers' date-range history endpoint returns live-forming candles
        # automatically when the range includes "today" — no separate
        # intraday endpoint needed (unlike Upstox).
        df = fetch_fyers_candles(item["key"], start_date, end_date)
        if df is None or df.empty:
            return None
        df = df.drop_duplicates(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)
        df["Symbol"] = item["symbol"]
        return df

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

    if not historical_dfs:
        if _fyers_error_log_count > 0:
            print(f"{COLOR_RED}No historical stock data retrieved. See the [Fyers Diagnostic] lines above for the exact API error.{COLOR_RESET}")
        else:
            print(f"{COLOR_RED}No historical stock data retrieved, but no API errors were logged.{COLOR_RESET}")
            print(f"{COLOR_YELLOW}This usually means: the target date is a market holiday/weekend with no candles, "
                  f"or every request returned an empty candle set for some other silent reason.{COLOR_RESET}")
        return

    rolling_master_df = pd.concat(historical_dfs, ignore_index=True)
    print("⚙️ Computing 7-Pillar Scorecards & Velocity Matrices...")

    tape_exec = prepare_unified_execution_tape(rolling_master_df, MICRO_TIMEFRAME, MACRO_TIMEFRAMES, strategy_mode=GLOBAL_MACRO_STRATEGY_2D)

    all_anomalies = tape_exec[tape_exec["Direction"] != 0].copy()
    anomalies_by_time = all_anomalies.groupby("Datetime")

    closes_dict = tape_exec.set_index(["Datetime", "Symbol"])["Close"].to_dict()
    micro_price_renko = tape_exec.set_index(["Datetime", "Symbol"])[f"Renko_Count_{MICRO_TIMEFRAME}"].to_dict()
    micro_vol_renko = tape_exec.set_index(["Datetime", "Symbol"])[f"Vol_Renko_Count_{MICRO_TIMEFRAME}"].to_dict()
    micro_vel_bars = tape_exec.set_index(["Datetime", "Symbol"])[f"Bars_Since_Brick_{MICRO_TIMEFRAME}"].to_dict()

    macro_price_renkos = {tf: tape_exec.set_index(["Datetime", "Symbol"])[f"Renko_Count_{tf}"].to_dict() for tf in MACRO_TIMEFRAMES}
    macro_vol_renkos = {tf: tape_exec.set_index(["Datetime", "Symbol"])[f"Vol_Renko_Count_{tf}"].to_dict() for tf in MACRO_TIMEFRAMES}

    all_times = np.sort(tape_exec["Datetime"].unique())
    # 🌟 memory_bank: Symbol -> LIST of trade episodes (so multiple same-day
    # triggers on the same stock are all preserved, not overwritten).
    memory_bank = {}
    cutoff_time_obj = pd.to_datetime(ENTRY_CUTOFF_TIME).time()

    # 🌟 Options translation layer: cache fetched premium series per option key
    # so re-triggers landing on the same strike don't re-fetch from the API.
    option_series_cache = {}  # option_key -> (sorted_ts, price_map)
    option_meta_cache = {}    # option_key -> {"symbol":..., "strike":..., "type":...}

    def get_option_series(option_key):
        if option_key not in option_series_cache:
            option_series_cache[option_key] = fetch_option_day_series(
                option_key, target_date_str, is_live_today, current_now
            )
        return option_series_cache[option_key]

    skipped_no_chain = 0
    skipped_no_data = 0

    for t in all_times:
        t_dt = pd.to_datetime(t)

        # 1. Manage Active Trades (Velocity Stall Exit + Dual-Layered Trailing).
        # Exit TRIGGER logic still runs on the STOCK's own renko structure —
        # that's the signal engine and is left untouched. Only the PRICE
        # recorded (exit_price) is switched to the option premium below.
        for sym, episodes in memory_bank.items():
            if not episodes:
                continue
            st = episodes[-1]
            if st["state"] == "ACTIVE":
                stock_ltp = closes_dict.get((t_dt, sym))
                mi_p_count = micro_price_renko.get((t_dt, sym), 0)
                mi_v_count = micro_vol_renko.get((t_dt, sym), 0)
                mi_bars_stalled = micro_vel_bars.get((t_dt, sym), 0)

                if stock_ltp is not None:
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
                        sorted_ts, price_map = get_option_series(st["option_key"])
                        opt_exit_price = lookup_price_at_or_before(sorted_ts, price_map, t_dt)
                        if opt_exit_price is None:
                            opt_exit_price = st["origin"]  # no fresher option print available; hold last known
                        st["state"] = "EXITED"
                        st["exit_time"] = t_dt.strftime("%Y-%m-%d %H:%M")
                        st["exit_price"] = opt_exit_price
                        st["exit_stock_price"] = stock_ltp
                        st["exit_reason"] = exit_reason

        # 2. Process New Entrances -> Translate into ATM CE/PE and fetch premium.
        if t_dt in anomalies_by_time.groups and t_dt.time() <= cutoff_time_obj:
            for _, row in anomalies_by_time.get_group(t_dt).iterrows():
                sym = row["Symbol"]
                direction = row["Direction"]
                stock_price_at_entry = row["Close"]

                existing = memory_bank.get(sym, [])
                if existing and existing[-1]["state"] == "ACTIVE":
                    continue  # already in a live trade on this stock

                opt_type = "CE" if direction == 1 else "PE"
                selected = select_atm_option(sym, stock_price_at_entry, opt_type, options_by_underlying, target_date_str)
                if selected is None:
                    skipped_no_chain += 1
                    continue

                sorted_ts, price_map = get_option_series(selected["key"])
                origin_price = lookup_price_at_or_before(sorted_ts, price_map, t_dt)
                if origin_price is None or origin_price < MIN_OPT_PREMIUM_SANITY:
                    skipped_no_data += 1
                    continue

                triggered_m_tfs = []
                for tf in MACRO_TIMEFRAMES:
                    armed_col = f"Armed_Bull_{tf}" if direction == 1 else f"Armed_Bear_{tf}"
                    if row.get(armed_col, False):
                        triggered_m_tfs.append(tf)

                new_episode = {
                    "state": "ACTIVE",
                    "origin": origin_price,                      # option premium at entry
                    "entry_stock_price": stock_price_at_entry,   # stock price that triggered it (reference)
                    "date": t_dt.strftime("%Y-%m-%d"),
                    "time": t_dt.strftime("%H:%M"),
                    "dir": direction,
                    "option_symbol": selected["symbol"],
                    "option_key": selected["key"],
                    "strike": selected["strike"],
                    "opt_type": opt_type,
                    "exit_time": None,
                    "exit_price": None,
                    "exit_stock_price": None,
                    "exit_reason": None,
                    "triggering_macro_tfs": triggered_m_tfs,
                    "macro_scores": {tf: row.get(f"Score_Bull_{tf}" if direction == 1 else f"Score_Bear_{tf}", 0) for tf in MACRO_TIMEFRAMES},
                    "micro_score": row.get(f"Score_Bull_{MICRO_TIMEFRAME}" if direction == 1 else f"Score_Bear_{MICRO_TIMEFRAME}", 0)
                }
                memory_bank.setdefault(sym, []).append(new_episode)

        # 3. EOD Force Exit (15:15) — also priced off the option premium.
        if t_dt.hour == 15 and t_dt.minute >= 15:
            for sym, episodes in memory_bank.items():
                if episodes and episodes[-1]["state"] == "ACTIVE":
                    st = episodes[-1]
                    sorted_ts, price_map = get_option_series(st["option_key"])
                    opt_exit_price = lookup_price_at_or_before(sorted_ts, price_map, t_dt)
                    if opt_exit_price is None:
                        opt_exit_price = st["origin"]
                    st["state"] = "EXITED"
                    st["exit_time"] = t_dt.strftime("%Y-%m-%d %H:%M") + " (EOD)"
                    st["exit_price"] = opt_exit_price
                    st["exit_stock_price"] = closes_dict.get((t_dt, sym), st["entry_stock_price"])
                    st["exit_reason"] = "End of Day Market Close"

    today_master = tape_exec[tape_exec["Datetime"].dt.date == target_dt.date()]
    if today_master.empty:
        print(f"\n{COLOR_YELLOW}[Terminal Standby] Market data for {target_date_str} is empty.{COLOR_RESET}\n")
        return

    if skipped_no_chain:
        print(f"{COLOR_DIM}  ├─ Skipped {skipped_no_chain} signal(s): no matching option chain for that underlying/expiry.{COLOR_RESET}")
    if skipped_no_data:
        print(f"{COLOR_DIM}  ├─ Skipped {skipped_no_data} signal(s): selected option had no usable premium data.{COLOR_RESET}")

    # ==============================================================================
    # 6. TERMINAL OUTPUT — priced in OPTION PREMIUM, not stock price
    # ==============================================================================
    active_runners = []
    closed_trades = []
    for sym, episodes in memory_bank.items():
        for st in episodes:
            if st["state"] == "ACTIVE":
                active_runners.append({**st, "sym": sym})
            elif st["state"] == "EXITED" and st["date"] == target_date_str:
                closed_trades.append({**st, "sym": sym})
    closed_trades.sort(key=lambda x: (x["sym"], x["time"]))

    tf_display_str = " | ".join(MACRO_TIMEFRAMES)
    print(f"\n{COLOR_CYAN}================================================================================================{COLOR_RESET}")
    print(f"{COLOR_BOLD}7-PILLAR QUALIFYING-TF EXIT ENGINE [{MICRO_TIMEFRAME} Micro ⚡ Macro: {tf_display_str}] — OPTIONS MODE{COLOR_RESET}")
    print(f"{COLOR_CYAN}================================================================================================{COLOR_RESET}\n")

    if active_runners:
        print(f"{COLOR_BOLD}🟢 BASKET 1: ACTIVE RUNNERS (Riding the Trend){COLOR_RESET}")
        for st in active_runners:
            sorted_ts, price_map = get_option_series(st["option_key"])
            latest_opt_price = None
            if sorted_ts:
                latest_opt_price = price_map[sorted_ts[-1]]
            if latest_opt_price is None:
                latest_opt_price = st["origin"]

            pnl_pct = ((latest_opt_price - st["origin"]) / st["origin"]) * 100
            color = COLOR_GREEN if pnl_pct >= 0 else COLOR_RED
            d_str = "BULLISH (Long CE)" if st["dir"] == 1 else "BEARISH (Long PE)"

            print(f"  {color}⚡ {st['option_symbol']:<24} Open P&L: {pnl_pct:+.2f}% ({d_str}){COLOR_RESET}")
            print(f"      └─ 📈 Underlying / Strike        : {st['sym']} → {st['strike']} {st['opt_type']}")
            print(f"      └─ ⚓ Qualifying Macro TFs        : {', '.join(st['triggering_macro_tfs'])}")
            print(f"      └─ 🔫 Micro Execution [{MICRO_TIMEFRAME}] : Score >= {MICRO_MINIMUM_SCORE}/7 (Score={st['micro_score']})")
            print(f"      └─ ⚓ True Birth Anchor           : {st['date']} @ {st['time']} | Premium: ₹{st['origin']:.2f} (Stock @ ₹{st['entry_stock_price']:.2f})")
            print(f"      └─ 🎯 Latest Premium             : {target_date_str} @ EOD   | Premium: ₹{latest_opt_price:.2f}\n")

    if closed_trades:
        print(f"{COLOR_BOLD}🛑 BASKET 2: CLOSED TRADES (Renko Structure Broken / Stagnation){COLOR_RESET}")
        for st in closed_trades:
            pnl_pct = ((st["exit_price"] - st["origin"]) / st["origin"]) * 100
            color = COLOR_GREEN if pnl_pct >= 0 else COLOR_RED
            d_str = "BULLISH (Long CE)" if st["dir"] == 1 else "BEARISH (Long PE)"

            print(f"  {color}🛑 {st['option_symbol']:<24} Final P&L: {pnl_pct:+.2f}% ({d_str}){COLOR_RESET}")
            print(f"      └─ 📈 Underlying / Strike        : {st['sym']} → {st['strike']} {st['opt_type']}")
            print(f"      └─ ⚓ Qualifying Macro TFs        : {', '.join(st['triggering_macro_tfs'])}")
            print(f"      └─ 🔫 Micro Execution [{MICRO_TIMEFRAME}] : Score >= {MICRO_MINIMUM_SCORE}/7 (Score={st['micro_score']})")
            print(f"      └─ ⚓ True Birth Anchor           : {st['date']} @ {st['time']} | Premium: ₹{st['origin']:.2f} (Stock @ ₹{st['entry_stock_price']:.2f})")
            print(f"      └─ 🎯 Exit Time & Premium         : {st['exit_time']} | Premium: ₹{st['exit_price']:.2f} (Stock @ ₹{st['exit_stock_price']:.2f})")
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
        target_dt = datetime.utcnow() + timedelta(hours=5, minutes=30)
        if target_dt.weekday() == 5: target_dt -= timedelta(days=1)
        elif target_dt.weekday() == 6: target_dt -= timedelta(days=2)
        target_date_str = target_dt.strftime("%Y-%m-%d")
    else:
        target_date_str = datetime.strptime(raw_date_str, "%Y-%m-%d").strftime("%Y-%m-%d")

    if not validate_fyers_token():
        return

    scan_institutional_tape(target_date_str)

if __name__ == "__main__":
    run_production_sweep()
