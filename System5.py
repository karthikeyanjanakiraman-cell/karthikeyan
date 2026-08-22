"""system3.py - Asit Baran Pati Multi-Timeframe Trading System Implementation (Options Edition)

Production-Grade Universal N-Timeframe & Dual-Tier 45-Degree Renko Engine
- Target: F&O Stocks Only (Indices Excluded)
- Execution: Direct Options Premium Charting (Method A)
- DUAL-BROKER ARCHITECTURE: Seamlessly route between UPSTOX and FYERS
- TWO-STAGE INGESTION: Ultra-Fast Previous Day Volume & Premium Pre-Filtering
- 3-TIER FALLBACK FETCHER: Bypasses 'Birthdate' Empty Array Errors
- BULLETPROOF THREADING: Safe isolated workers to prevent traceback cascades
"""

import argparse
import concurrent.futures
import datetime
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
# GLOBAL CONFIGURATION: BROKER, OPTIONS & TIMEFRAMES
# ==============================================================================
ACTIVE_BROKER = "FYERS"

STRIKE_RANGE_OFFSET = 2
TARGET_EXPIRY = "CURRENT"
BACKTRACE_DAYS = 15

MIN_OPT_PREMIUM = 15.0
MIN_PREV_DAY_VOLUME = 250000

MAX_API_WORKERS = 40 if ACTIVE_BROKER == "UPSTOX" else 80

MICRO_TIMEFRAME = "1min"
MACRO_TIMEFRAMES = ["5min"]

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

# 🌟 BUY-ONLY MODE: user only buys options (never writes/shorts), so only
# rising-premium ("BULLISH") triggers are actionable — a buy only profits when
# the premium goes up. This applies per-contract: a bullish trigger on a PE
# still captures downside moves in the underlying (PE premium rises when the
# stock falls), so both directions are covered without ever needing to short.
GLOBAL_MACRO_STRATEGY_2D = "BULLISH"

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

MICRO_EXIT_PRICE_BRICKS = 5
MICRO_EXIT_VOL_BRICKS   = 5
MACRO_EXIT_PRICE_BRICKS = 1
MACRO_EXIT_VOL_BRICKS   = 1

RENKO_VELOCITY_MAX_BARS = 12
ENTRY_CUTOFF_TIME = "15:00"

EXCLUDED_INDICES = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX", "BANKEX", "NIFTY50", "NIFTYBANK"}

# 🌟 DEBUG / ERROR VISIBILITY (the root cause of "silent" zero-data failures was that
# every non-200 / non-"ok" response from the broker was swallowed with no diagnostics.
# This flag makes the first few real failures print their actual status code / body.)
FYERS_VERBOSE_ERRORS = True
_FYERS_ERROR_LOG_CAP = 5
_fyers_error_log_count = 0


def _log_fyers_error(context, status_code=None, body=None):
    """Prints the FIRST few real Fyers API errors so the true cause is visible
    instead of being hidden behind the generic 'token expired or rate-limited' guess."""
    global _fyers_error_log_count
    if not FYERS_VERBOSE_ERRORS or _fyers_error_log_count >= _FYERS_ERROR_LOG_CAP:
        return
    _fyers_error_log_count += 1
    snippet = str(body)[:300] if body is not None else ""
    print(f"{COLOR_YELLOW}  [Fyers Diagnostic #{_fyers_error_log_count}] {context}"
          f"{' | HTTP ' + str(status_code) if status_code else ''} {snippet}{COLOR_RESET}")


# ==============================================================================
# 1. DUAL-BROKER INGESTION ENGINE
# ==============================================================================
def validate_broker_auth():
    if ACTIVE_BROKER == "UPSTOX":
        if not os.environ.get("UPSTOX_ACCESS_TOKEN"):
            print(f"{COLOR_RED}Error: UPSTOX_ACCESS_TOKEN environment variable not found.{COLOR_RESET}")
            sys.exit(1)
    elif ACTIVE_BROKER == "FYERS":
        if not os.environ.get("FYERS_CLIENT_ID") or not os.environ.get("FYERS_ACCESS_TOKEN"):
            print(f"{COLOR_RED}Error: FYERS_CLIENT_ID or FYERS_ACCESS_TOKEN environment variables not found for Fyers.{COLOR_RESET}")
            sys.exit(1)

        # 🌟 REAL TOKEN VALIDATION (this is the actual fix for the reported bug).
        # The old code only checked that env vars were *set*, never that the token
        # actually *works*. That's why 214 symbols could all fail silently with the
        # same guessed message. We now call Fyers' lightweight /profile endpoint
        # once, up front, and fail fast with the real reason.
        try:
            headers = get_auth_headers()
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
                sys.exit(1)
            else:
                fy_name = body.get("data", {}).get("name", "Unknown")
                print(f"{COLOR_GREEN}✅ Fyers token validated OK (Account: {fy_name}){COLOR_RESET}")
        except requests.exceptions.RequestException as e:
            print(f"{COLOR_RED}❌ Could not reach Fyers to validate the token: {e}{COLOR_RESET}")
            sys.exit(1)

def get_auth_headers():
    if ACTIVE_BROKER == "UPSTOX":
        return {"Accept": "application/json", "Authorization": f"Bearer {os.environ.get('UPSTOX_ACCESS_TOKEN', '')}"}
    elif ACTIVE_BROKER == "FYERS":
        return {"Authorization": f"{os.environ.get('FYERS_CLIENT_ID', '')}:{os.environ.get('FYERS_ACCESS_TOKEN', '')}"}
    return {}

def fetch_json_gz(url):
    headers = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'}
    try:
        response = requests.get(url, headers=headers, timeout=45)
        if response.status_code == 200:
            return json.load(gzip.GzipFile(fileobj=io.BytesIO(response.content)))
    except Exception: pass
    return []

def get_universe_data():
    print(f"Fetching Master Instrument Matrix via {ACTIVE_BROKER}...")
    spot_inst, opt_inst = [], []

    if ACTIVE_BROKER == "UPSTOX":
        master_data = fetch_json_gz("https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz")
        if not master_data: return [], []

        fo_underlyings = {
            item.get("underlying_symbol") for item in master_data
            if item.get("instrument_type") in ("OPTSTK", "CE", "PE") and item.get("underlying_symbol") not in EXCLUDED_INDICES
        }
        fo_underlyings.discard(None)

        for item in master_data:
            if item.get("trading_symbol") in fo_underlyings and item.get("segment") == "NSE_EQ":
                spot_inst.append({"symbol": item["trading_symbol"], "key": item["instrument_key"], "underlying": item["trading_symbol"]})
            elif item.get("underlying_symbol") in fo_underlyings and item.get("instrument_type") in ("OPTSTK", "CE", "PE"):
                raw_strike = item.get("strike_price", item.get("strike"))
                try: strike_val = float(raw_strike)
                except: strike_val = None

                if strike_val is not None and item.get("expiry"):
                    opt_inst.append({
                        "symbol": item.get("trading_symbol", item.get("tradingsymbol", "UNKNOWN")),
                        "key": item["instrument_key"],
                        "underlying": item["underlying_symbol"],
                        "type": "CE" if "CE" in item.get("instrument_type", "") or "CE" in item.get("trading_symbol", "") else "PE",
                        "strike": strike_val,
                        "expiry": item["expiry"]
                    })

    elif ACTIVE_BROKER == "FYERS":
        try:
            print("  Downloading & Parsing FYERS F&O Data...")
            headers = {'User-Agent': 'Mozilla/5.0'}

            res_cm = requests.get("https://public.fyers.in/sym_details/NSE_CM.csv", headers=headers, timeout=15)
            spot_key_map = {}
            if res_cm.status_code == 200:
                for line in res_cm.text.strip().split('\n'):
                    cols = [c.strip() for c in line.split(',')]
                    for c in cols:
                        if c.startswith("NSE:") and c.endswith("-EQ"):
                            base = c.replace("NSE:", "").replace("-EQ", "")
                            spot_key_map[base] = c
                            break

            res_fo = requests.get("https://public.fyers.in/sym_details/NSE_FO.csv", headers=headers, timeout=15)
            valid_underlyings = set()

            for line in res_fo.text.strip().split('\n'):
                cols = [c.strip() for c in line.split(',')]

                opt_type = None
                type_idx = -1
                for i in range(len(cols)-1, -1, -1):
                    if cols[i] in ("CE", "PE"):
                        opt_type = cols[i]
                        type_idx = i
                        break

                if not opt_type or type_idx < 3:
                    continue

                try:
                    strike_val = float(cols[type_idx - 1])
                    # 🌟 FIX: base_symbol sits 3 columns left of CE/PE, not 2.
                    # The column at (type_idx - 2) is a raw numeric underlying instrument
                    # token (e.g. 26037 = NIFTY FIN SERVICE, 26009 = NIFTY BANK), NOT the
                    # symbol name. Reading it as the name caused EXCLUDED_INDICES (which
                    # matches on names like "BANKNIFTY") to silently miss index derivatives,
                    # which then produced bogus spot tickers like "NSE:26037-EQ" downstream.
                    base_symbol = cols[type_idx - 3].strip()

                    if base_symbol in EXCLUDED_INDICES:
                        continue

                    # Defensive guard: real NSE symbol names are never purely numeric.
                    # If parsing still lands on a numeric token for any row (format drift,
                    # malformed line, etc.), skip it instead of building an invalid ticker.
                    if base_symbol.isdigit() or not base_symbol:
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
                                expiry_date = dt.fromtimestamp(num).strftime("%Y-%m-%d")
                                break
                        except: pass

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
                        # 🌟 FIX: only trust a symbol that's a CONFIRMED equity ticker
                        # from the NSE_CM.csv map. The old fallback f"NSE:{base_symbol}-EQ"
                        # blindly guessed a ticker for anything not found there — which is
                        # exactly how non-equity index names (NIFTYNXT50, NIFTYFPI, etc.,
                        # which have F&O contracts but no underlying "-EQ" stock) leaked
                        # into the spot universe and got rejected by Fyers as invalid.
                        spot_ticker = spot_key_map.get(base_symbol)
                        if spot_ticker:
                            spot_inst.append({
                                "symbol": base_symbol,
                                "key": spot_ticker,
                                "underlying": base_symbol
                            })

                except Exception:
                    pass

        except Exception as e:
            print(f"{COLOR_RED}[Error] FYERS CSV fetch failed: {e}{COLOR_RESET}")

    print(f"  Mapped {len(spot_inst)} Spot Instruments & {len(opt_inst)} Options Contracts.")
    return spot_inst, opt_inst

def fetch_broker_data(key, tf_type, start_dt, end_dt, is_live=False):
    """Universal safe fetcher that routes requests cleanly without crashing"""
    headers = get_auth_headers()

    for attempt in range(3):
        try:
            if ACTIVE_BROKER == "UPSTOX":
                encoded_key = urllib.parse.quote(key)
                res_tf = "1minute" if tf_type == "1minute" else "day"

                if is_live and tf_type == "1minute":
                    url = f"https://api.upstox.com/v2/historical-candle/intraday/{encoded_key}/1minute"
                else:
                    url = f"https://api.upstox.com/v2/historical-candle/{encoded_key}/{res_tf}/{end_dt}/{start_dt}"

                res = requests.get(url, headers=headers, timeout=10)
                if res.status_code == 200:
                    body = res.json()
                    if not body: return None
                    data = body.get("data")
                    if not data: return None
                    candles = data.get("candles", [])

                    if not candles: return None
                    df = pd.DataFrame(candles, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume", "OI"])
                    df["Datetime"] = pd.to_datetime(df["Timestamp"]).dt.tz_localize(None).astype("datetime64[ns]")
                    return df

            elif ACTIVE_BROKER == "FYERS":
                time.sleep(0.2)
                res_tf = "1" if tf_type == "1minute" else "D"

                # 🌟 FIX: symbols like "NSE:GVT&D-EQ" contain a literal "&", which — left
                # unescaped — gets parsed by the HTTP layer as a query-string delimiter,
                # silently truncating/corrupting the request (Fyers then rejects it as an
                # "Invalid symbol"). We escape everything except ':' (Fyers' endpoint
                # rejects an encoded colon, per their API's documented quirk).
                encoded_symbol = urllib.parse.quote(key, safe=':')
                url = f"https://api-t1.fyers.in/data/history?symbol={encoded_symbol}&resolution={res_tf}&date_format=1&range_from={start_dt}&range_to={end_dt}"
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
                            # Valid response, genuinely no candles in this range (e.g. holiday/no trades).
                            return None
                        df = pd.DataFrame(candles, columns=["Epoch", "Open", "High", "Low", "Close", "Volume"])
                        df["Datetime"] = pd.to_datetime(df["Epoch"], unit='s', utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None).astype("datetime64[ns]")
                        return df

                    if status == "no_data":
                        # Legitimate "nothing to return" — not an error, don't retry, don't log.
                        return None

                    # status == "error" (e.g. code -16 "Could not authenticate the user",
                    # invalid symbol, bad date range, etc.) — THIS is what was previously hidden.
                    _log_fyers_error(
                        f"API error for {key} (code={data.get('code')}, msg={data.get('message')})",
                        res.status_code
                    )
                    if data.get("code") == -16:
                        # Auth errors won't fix themselves on retry — stop wasting attempts.
                        return None
                    # Otherwise fall through to the shared retry logic below.

                elif res.status_code in (429, 500, 502, 503):
                    time.sleep(random.uniform(1.0, 3.0) * (attempt + 1))
                    continue
                else:
                    _log_fyers_error(f"HTTP failure for {key}", res.status_code, res.text[:300])
                    return None

            if res.status_code == 429:
                time.sleep(random.uniform(1.0, 3.0) * (attempt + 1))
            else:
                break
        except requests.exceptions.RequestException as e:
            _log_fyers_error(f"Network exception on attempt {attempt + 1}: {e}")
            time.sleep(1)
        except Exception as e:
            _log_fyers_error(f"Unexpected exception on attempt {attempt + 1}: {e}")
            time.sleep(1)

    return None

def fetch_latest_spot_prices(spot_instruments):
    print(f"Fetching Spot Prices for {len(spot_instruments)} Underlyings ({ACTIVE_BROKER})...")
    spot_prices = {}

    def worker(inst):
        try:
            today_str = dt.utcnow().strftime("%Y-%m-%d")
            seven_days_ago = (dt.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")

            df = fetch_broker_data(inst["key"], "1minute", seven_days_ago, today_str, is_live=True)
            if df is not None and not df.empty:
                return inst["underlying"], df.iloc[0]["Close"] if ACTIVE_BROKER=="UPSTOX" else df.iloc[-1]["Close"]

            df_daily = fetch_broker_data(inst["key"], "day", seven_days_ago, today_str, is_live=False)
            if df_daily is not None and not df_daily.empty:
                return inst["underlying"], df_daily.iloc[0]["Close"] if ACTIVE_BROKER=="UPSTOX" else df_daily.iloc[-1]["Close"]
        except Exception:
            pass
        return inst["underlying"], None

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_API_WORKERS) as executor:
        futures = {executor.submit(worker, inst): inst for inst in spot_instruments}
        for future in concurrent.futures.as_completed(futures):
            try:
                sym, price = future.result()
                if price: spot_prices[sym] = price
            except Exception:
                pass

    if not spot_prices:
        if _fyers_error_log_count > 0:
            print(f"{COLOR_RED}  [Critical Error] Fyers returned NO spot prices. "
                  f"See the [Fyers Diagnostic] lines above for the exact API error.{COLOR_RESET}")
        else:
            print(f"{COLOR_RED}  [Critical Error] Fyers returned NO spot prices, but no API errors were logged. "
                  f"This usually means every request returned an empty candle set (e.g. market data "
                  f"permission not enabled on this app, or wrong symbol format).{COLOR_RESET}")

    return spot_prices

def build_options_matrix(spot_prices, options_instruments):
    print(f"Building Options Contract Matrix (Offset: +/-{STRIKE_RANGE_OFFSET}, Expiry: {TARGET_EXPIRY})...")
    grouped_options = {}
    for opt in options_instruments:
        grouped_options.setdefault(opt["underlying"], []).append(opt)

    target_contracts = []
    for symbol, spot_price in spot_prices.items():
        opts = grouped_options.get(symbol, [])
        if not opts: continue

        expiries = sorted(list(set(pd.to_datetime(o["expiry"]).date() for o in opts)))
        if not expiries: continue

        chosen_expiry = expiries[0] if TARGET_EXPIRY == "CURRENT" else (expiries[1] if len(expiries) > 1 else expiries[0])
        expiry_opts = [o for o in opts if pd.to_datetime(o["expiry"]).date() == chosen_expiry]

        unique_strikes = sorted(list(set(o["strike"] for o in expiry_opts)))
        if not unique_strikes: continue

        closest_strike = min(unique_strikes, key=lambda x: abs(x - spot_price))
        atm_idx = unique_strikes.index(closest_strike)

        start_idx = max(0, atm_idx - STRIKE_RANGE_OFFSET)
        end_idx = min(len(unique_strikes), atm_idx + STRIKE_RANGE_OFFSET + 1)
        selected_strikes = unique_strikes[start_idx:end_idx]

        final_opts = [o for o in expiry_opts if o["strike"] in selected_strikes]
        for opt in final_opts:
            target_contracts.append(opt)

    return target_contracts

def filter_liquid_options(target_contracts, target_date_str):
    print(f"\nSTAGE 1 INGESTION: Pre-Filtering {len(target_contracts)} contracts...")
    print(f"  Rules: Prev. Day Close >= Rs{MIN_OPT_PREMIUM} | Prev. Day Vol >= {MIN_PREV_DAY_VOLUME}")

    target_dt = dt.strptime(target_date_str, "%Y-%m-%d")
    prev_dt = target_dt - timedelta(days=1)
    while prev_dt.weekday() >= 5: prev_dt -= timedelta(days=1)
    prev_day = prev_dt.strftime("%Y-%m-%d")
    five_days_ago = (prev_dt - timedelta(days=7)).strftime("%Y-%m-%d")

    filtered_contracts = []

    def worker(contract):
        try:
            df = fetch_broker_data(contract["key"], "day", five_days_ago, prev_day)
            if df is not None and not df.empty:
                df = df.sort_values("Datetime")
                latest_candle = df.iloc[-1]
                if latest_candle["Close"] >= MIN_OPT_PREMIUM and latest_candle["Volume"] >= MIN_PREV_DAY_VOLUME:
                    return contract
        except Exception:
            pass
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_API_WORKERS) as executor:
        futures = {executor.submit(worker, c): c for c in target_contracts}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            sys.stdout.write(f"\r  Checking Liquidity... {completed}/{len(target_contracts)} processed")
            sys.stdout.flush()
            try:
                res = future.result()
                if res: filtered_contracts.append(res)
            except Exception:
                pass

    print(f"\n  Pre-Filter Complete: {len(filtered_contracts)} highly liquid contracts passed.")
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
    df_micro = apply_dual_tier_scorecard(df_micro, micro_tf, "MICRO")
    df_micro = df_micro.sort_values("Datetime").reset_index(drop=True)

    bull_gate_cols, bear_gate_cols = [], []
    for tf in macro_timeframes:
        print(f"   Evaluating Macro Context Gates + Price/Vol/Vel Renko for [{tf}]...")
        env_df = evaluate_single_timeframe_gates(rolling_master_df, tf)
        bull_col, bear_col = f"Armed_Bull_{tf}", f"Armed_Bear_{tf}"
        bull_gate_cols.append(bull_col)
        bear_gate_cols.append(bear_col)

        # 🌟 Defensive cast: pandas 2.x can produce mixed datetime64 precisions
        # ([s] vs [us] vs [ns]) depending on how a column was derived (raw epoch
        # conversion vs. arithmetic vs. groupby resampling). merge_asof requires
        # both join keys to share the exact same dtype, so we pin both explicitly
        # right before merging rather than relying on it matching by accident.
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

    # 🌟 FIX: this restriction MUST be applied here, before Trigger_Bull/Trigger_Bear/
    # Direction are derived below. The previous code applied it to tape_exec AFTER this
    # function returned — but Direction was already computed and frozen into the
    # dataframe by then, so overriding Master_Armed_Bear/Bull afterward had zero effect
    # and bearish trades kept firing even in "BULLISH" (buy-only) mode.
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
def scan_institutional_tape(target_date_str):
    print(f"\nInitiating Options Engine for {target_date_str} [{ACTIVE_BROKER}]...")

    spot_inst, options_inst = get_universe_data()
    if not spot_inst: return

    spot_prices = fetch_latest_spot_prices(spot_inst)
    if not spot_prices:
        return

    target_contracts = build_options_matrix(spot_prices, options_inst)

    if not target_contracts:
        print(f"{COLOR_RED}[Error] No options contracts mapped.{COLOR_RESET}")
        return

    print(f"Mapped {len(target_contracts)} total option contracts for analysis.")

    target_contracts = filter_liquid_options(target_contracts, target_date_str)

    if not target_contracts:
        print(f"{COLOR_YELLOW}All contracts failed the Liquidity (Vol >= {MIN_PREV_DAY_VOLUME}) or Premium (Price >= Rs{MIN_OPT_PREMIUM}) checks.{COLOR_RESET}")
        return

    trading_days = get_past_trading_days(target_date_str, num_days=BACKTRACE_DAYS)
    if not trading_days: return

    target_dt = pd.to_datetime(target_date_str)
    current_now = dt.utcnow() + timedelta(hours=5, minutes=30)
    is_live_today = target_date_str == current_now.strftime("%Y-%m-%d")

    print(f"\nSTAGE 2 INGESTION: Multithreading Bulk 1-Min Data for {len(target_contracts)} Contracts...")
    fetch_tasks = [(item, trading_days[0], target_date_str, is_live_today) for item in target_contracts]
    historical_dfs = []

    def fetch_worker(task):
        try:
            item, start_date, end_date, live = task
            dfs = []
            hist_end = end_date if not live else (current_now - timedelta(days=1)).strftime("%Y-%m-%d")

            df = fetch_broker_data(item["key"], "1minute", start_date, hist_end)

            if df is None or df.empty:
                fallback_start = get_past_trading_days(end_date, num_days=5)[0]
                df = fetch_broker_data(item["key"], "1minute", fallback_start, hist_end)

            if df is None or df.empty:
                extreme_start = get_past_trading_days(end_date, num_days=2)[0]
                df = fetch_broker_data(item["key"], "1minute", extreme_start, hist_end)

            if df is not None and not df.empty:
                dfs.append(df)

            if live:
                intra_df = fetch_broker_data(item["key"], "1minute", end_date, end_date, is_live=True)
                if intra_df is not None and not intra_df.empty:
                    dfs.append(intra_df)

            if not dfs:
                print(f"{COLOR_DIM}  [API Block] Broker returned ZERO data for {item['symbol']} (Likely illiquid or expired){COLOR_RESET}")
                return None

            final_df = pd.concat(dfs, ignore_index=True)
            final_df = final_df.drop_duplicates(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)
            final_df["Symbol"] = item["symbol"]
            return final_df
        except Exception:
            pass
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_API_WORKERS) as executor:
        futures = {executor.submit(fetch_worker, task): task for task in fetch_tasks}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            print(f"  Fetching 1-Min Data... {completed}/{len(fetch_tasks)} processed")
            try:
                res = future.result()
                if res is not None: historical_dfs.append(res)
            except Exception:
                pass
    print()

    if not historical_dfs:
        print(f"{COLOR_RED}No historical data retrieved.{COLOR_RESET}")
        return

    rolling_master_df = pd.concat(historical_dfs, ignore_index=True)
    print("Computing 7-Pillar Scorecards & Velocity Matrices on Premium Data...")

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
    # 🌟 FIX: memory_bank now maps Symbol -> LIST of trade episodes, not a single
    # episode. The old code overwrote memory_bank[sym] every time a new trigger
    # fired after a prior exit, so only the LAST birth-time of the day survived —
    # every earlier trigger/exit for that contract was silently lost.
    memory_bank = {}
    cutoff_time_obj = pd.to_datetime(ENTRY_CUTOFF_TIME).time()

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

                existing = memory_bank.get(sym, [])
                # Append a NEW episode whenever there's no prior episode, or the most
                # recent one has already exited — instead of overwriting it.
                if not existing or existing[-1]["state"] == "EXITED":
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
                        "macro_scores": {tf: row.get(f"Score_Bull_{tf}" if direction == 1 else f"Score_Bear_{tf}", 0) for tf in MACRO_TIMEFRAMES},
                        "micro_score": row.get(f"Score_Bull_{MICRO_TIMEFRAME}" if direction == 1 else f"Score_Bear_{MICRO_TIMEFRAME}", 0)
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

    today_master = tape_exec[tape_exec["Datetime"].dt.date == target_dt.date()]
    if today_master.empty:
        print(f"\n{COLOR_YELLOW}[Terminal Standby] Market data for {target_date_str} is empty.{COLOR_RESET}\n")
        return

    final_ltp_dict = today_master.groupby("Symbol")["Close"].last().to_dict()

    # ==============================================================================
    # 6. TERMINAL OUTPUT
    # ==============================================================================
    # 🌟 Flatten each symbol's list of episodes: every trigger of the day now
    # produces its own row in the output (previously only the LAST one survived).
    active_runners = {}
    closed_trades = []
    for sym, episodes in memory_bank.items():
        for st in episodes:
            if st["state"] == "ACTIVE":
                active_runners[sym] = st
            elif st["state"] == "EXITED" and st["date"] == target_date_str:
                closed_trades.append({**st, "sym": sym})
    # Sort chronologically so multiple triggers on the same contract print in order.
    closed_trades.sort(key=lambda x: (x["sym"], x["time"]))

    tf_display_str = " | ".join(MACRO_TIMEFRAMES)
    print(f"\n{COLOR_CYAN}================================================================================================{COLOR_RESET}")
    print(f"{COLOR_BOLD}7-PILLAR QUALIFYING-TF OPTIONS ENGINE [{MICRO_TIMEFRAME} Micro | Macro: {tf_display_str}]{COLOR_RESET}")
    print(f"{COLOR_CYAN}================================================================================================{COLOR_RESET}\n")

    if active_runners:
        print(f"{COLOR_BOLD}BASKET 1: ACTIVE RUNNERS (Riding the Trend){COLOR_RESET}")
        for sym, st in active_runners.items():
            ltp = final_ltp_dict.get(sym, st["origin"])
            pnl_pct = ((ltp - st["origin"]) / st["origin"]) * 100 if st["dir"] == 1 else ((st["origin"] - ltp) / st["origin"]) * 100
            color = COLOR_GREEN if pnl_pct >= 0 else COLOR_RED
            d_str = "BULLISH" if st["dir"] == 1 else "BEARISH"

            print(f"  {color}{sym:<20} Open P&L: {pnl_pct:+.2f}% ({d_str}){COLOR_RESET}")
            print(f"      Qualifying Macro TFs        : {', '.join(st['triggering_macro_tfs'])}")
            print(f"      Micro Execution [{MICRO_TIMEFRAME}] : Score >= {MICRO_MINIMUM_SCORE}/7 (Score={st['micro_score']})")
            print(f"      True Birth Anchor            : {st['date']} @ {st['time']} | Price: Rs{st['origin']:.2f}")
            print(f"      Latest LTP                  : {target_date_str} @ EOD   | Price: Rs{ltp:.2f}\n")

    if closed_trades:
        print(f"{COLOR_BOLD}BASKET 2: CLOSED TRADES (Renko Structure Broken / Stagnation){COLOR_RESET}")
        for st in closed_trades:
            pnl_pct = ((st["exit_price"] - st["origin"]) / st["origin"]) * 100 if st["dir"] == 1 else ((st["origin"] - st["exit_price"]) / st["origin"]) * 100
            color = COLOR_GREEN if pnl_pct >= 0 else COLOR_RED
            d_str = "BULLISH" if st["dir"] == 1 else "BEARISH"

            print(f"  {color}{st['sym']:<20} Final P&L: {pnl_pct:+.2f}% ({d_str}){COLOR_RESET}")
            print(f"      Qualifying Macro TFs        : {', '.join(st['triggering_macro_tfs'])}")
            print(f"      Micro Execution [{MICRO_TIMEFRAME}] : Score >= {MICRO_MINIMUM_SCORE}/7 (Score={st['micro_score']})")
            print(f"      True Birth Anchor            : {st['date']} @ {st['time']} | Price: Rs{st['origin']:.2f}")
            print(f"      Exit Time & Price            : {st['exit_time']} | Price: Rs{st['exit_price']:.2f}")
            print(f"      Reason                      : {st['exit_reason']}\n")

    if not active_runners and not closed_trades:
        print(f"{COLOR_DIM}[Terminal Silent] No trades triggered today.{COLOR_RESET}\n")

# ==============================================================================
# 7. RUN EXECUTOR
# ==============================================================================
def run_production_sweep():
    validate_broker_auth()

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

    scan_institutional_tape(target_date_str)

if __name__ == "__main__":
    run_production_sweep()
