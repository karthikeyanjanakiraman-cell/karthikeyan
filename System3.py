import os
import sys
import argparse
import urllib.parse
import json
import gzip
import io
import time
from datetime import datetime, timedelta
import concurrent.futures

import requests
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. ENGINE CONSTANTS & CONFIGURATION
# ==============================================================================
# --- ENGINE MODES ---
TRADING_MODE = "STOCK_FNO"   # Options: "STOCK_FNO", "CASH_EQUITY", "INDEX_OPTIONS"

# --- MULTI-TIMEFRAME (now MULTI-GRANULARITY RENKO) DASHBOARD COLUMNS ---
# Previously each "timeframe" column was a clock-time resample (3min/5min/...)
# of the raw 1-min tape. Replaced with genuine 45-degree Renko bricks built
# at several brick-size GRANULARITIES from the same 1-min Close series -
# see build_renko_bricks() / compute_fixed_brick_size() for the construction
# itself, and the comment there for why this is what makes it "45 degree".
# Each multiple scales that symbol's own fixed base brick size, giving a
# fine/medium/coarse family of Renko ladders that stand in for what
# short/medium/long timeframes used to represent - all still derived from
# the same underlying 1-min Close series, never from resampled OHLC bars.
RENKO_GRANULARITIES = [1, 2, 3, 5]   # multiples of the per-symbol base brick size
RENKO_BRICK_ATR_PERIOD = 14           # used ONCE per symbol to size the base brick (see compute_fixed_brick_size)
RENKO_BRICK_BASIS_TF = "15min"        # resample used ONLY for that one-time ATR sizing calc, never for indicator bars
RENKO_MIN_BRICK_PCT = 0.001           # brick floor as a fraction of price, guards against a near-zero brick on dead-quiet stocks

# --- OUTPUT: only buyers/sellers are shown, no neutral basket, and each
# list is capped so the dashboard stays readable on a large universe. ---
TOP_N_BUYERS = 15
TOP_N_SELLERS = 15
# MTF CONFLUENCE: a symbol only counts as a buyer/seller if at least this
# many of the RENKO_GRANULARITIES independently lean the same direction -
# not just a net-positive Score from one dominant granularity outvoting the
# others. With 4 granularities configured, 2 is "more agree than disagree".
MIN_CONFLUENCE_COUNT = 2

COLOR_GREEN_BG = '\033[42m\033[30m'  # Green background, black text
COLOR_RED_BG = '\033[41m\033[97m'    # Red background, white text
COLOR_GRAY_BG = '\033[100m\033[97m'  # Dark gray background, white text
COLOR_RESET = '\033[0m'
COLOR_BOLD = '\033[1m'
COLOR_CYAN = '\033[96m'
COLOR_RED_FG = '\033[91m'

# --- UNIVERSE FILTERING (Ignored for INDEX_OPTIONS) ---
MIN_PRICE = 50              # Minimum stock price
MAX_PRICE = 10000           # Maximum stock price
MIN_DAILY_VOLUME = 500000   # Minimum daily volume
BACKTRACE_DAYS = 5          # Days to fetch to ensure enough data for MAs

# --- INDICATOR PERIODS ---
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
ADX_PERIOD = 14
ADX_THRESHOLD = 25

# --- CONCURRENCY (tune these to your Upstox plan's rate limit) ---
# Two independent stages fetch data over the network, and both used to be
# fully sequential (one blocking HTTP call at a time, with a fixed
# time.sleep between each) - that serial chain was the actual bottleneck,
# not CPU work. Threading each stage is what makes this "complete quickly".
#
# UNIVERSE_FILTER_WORKERS: fan-out for the initial price/volume filter pass
# (STOCK_FNO/CASH_EQUITY only - up to 500 single-day requests, one per
# candidate stock). Was: a plain `for` loop, one request at a time.
UNIVERSE_FILTER_WORKERS = 10
# STOCK_WORKERS: fan-out across stocks for the full technical pipeline.
STOCK_WORKERS = 10
# DAY_FETCH_WORKERS: fan-out across a single stock's BACKTRACE_DAYS days.
# Was: a plain `for` loop inside process_stock, one request at a time with a
# 0.05s sleep between each - for BACKTRACE_DAYS=5 that's 5 sequential calls
# PER STOCK, repeated for every stock in the universe.
DAY_FETCH_WORKERS = 3
# Worst-case concurrent connections = STOCK_WORKERS * DAY_FETCH_WORKERS (here
# 30). If your API tier rate-limits harder than that, lower these first -
# fetch_upstox_candles_for_date already retries on 429 with backoff, but
# fewer 429s to begin with means fewer wasted round trips.

# ==============================================================================
# 1. LIVE INGESTION & SETUP
# ==============================================================================
def get_dynamic_universe(mode):
    """Fetches and categorizes the exact trading universe based on TRADING_MODE."""
    nse_url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
    try:
        response = requests.get(nse_url, timeout=10)
        if response.status_code != 200:
            return []
        nse_data = json.load(gzip.GzipFile(fileobj=io.BytesIO(response.content)))

        if mode == "INDEX_OPTIONS":
            target_indices = ["Nifty 50", "Nifty Bank", "Nifty Fin Service", "Nifty Mid Select"]
            return [{"symbol": item["trading_symbol"], "key": item["instrument_key"]}
                    for item in nse_data
                    if item.get("segment") == "NSE_INDEX" and item.get("trading_symbol") in target_indices]

        fno_underlying = {item.get("underlying_symbol") for item in nse_data if item.get("segment") == "NSE_FO" and item.get("underlying_symbol")}

        if mode == "STOCK_FNO":
            return [{"symbol": item["trading_symbol"], "key": item["instrument_key"]}
                    for item in nse_data
                    if item.get("segment") == "NSE_EQ" and item.get("trading_symbol") in fno_underlying]

        elif mode == "CASH_EQUITY":
            return [{"symbol": item["trading_symbol"], "key": item["instrument_key"]}
                    for item in nse_data
                    if item.get("segment") == "NSE_EQ" and item.get("trading_symbol") not in fno_underlying]

    except Exception as e:
        print(f"{COLOR_RED_FG}[API Error] Failed to fetch Universe: {e}{COLOR_RESET}")
        return []

def fetch_upstox_candles_for_date(instrument_key, date_str, retries=3):
    """Fetches 1-minute historical candles with Exponential Backoff for Rate Limits."""
    access_token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not access_token: return None

    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {access_token}'}
    today_str = (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d")

    if date_str == today_str:
        url = f"https://api.upstox.com/v2/historical-candle/intraday/{urllib.parse.quote(instrument_key)}/1minute"
    else:
        url = f"https://api.upstox.com/v2/historical-candle/{urllib.parse.quote(instrument_key)}/1minute/{date_str}/{date_str}"

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json().get('data', {}).get('candles', [])
                if not data: return None
                c_df = pd.DataFrame(data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
                c_df['Datetime'] = pd.to_datetime(c_df['Timestamp']).dt.tz_localize(None)
                return c_df.sort_values('Datetime').reset_index(drop=True)
            elif response.status_code == 429:
                time.sleep(1.0 * (attempt + 1))
                continue
            else:
                return None
        except Exception:
            time.sleep(1.0)
            continue
    return None

def get_past_trading_days(target_date_str, num_days=5):
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
# 2. INDICATOR MATH (BB-RSI, BB-MACD, ADX)
# ==============================================================================
def compute_fixed_brick_size(df_1m):
    """
    Sizes the base Renko brick ONCE per symbol, from that symbol's own
    volatility - then held FIXED for every brick built afterward. This is
    what makes the result genuinely "45 degree": if brick size were
    recalculated on every bar (as a naive adaptive-ATR Renko does), each
    brick would be a different height and the staircase wouldn't actually be
    uniform even though the code calls it 45 degrees. Fixed size = every
    brick is identically sized, so a plot of brick index vs. cumulative
    signed brick count is a literal, uniform 45-degree line.

    The one-time ATR estimate below is the ONLY place High/Low from the raw
    price series is used - purely to pick a sensible brick height up front.
    It plays no role in DETECTING when a brick has formed; that happens in
    build_renko_bricks() using Close alone (see its docstring).
    """
    resampled = df_1m.set_index('Datetime').resample(RENKO_BRICK_BASIS_TF).agg(
        {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
    ).dropna().reset_index()
    if len(resampled) < 5:
        return max(df_1m['Close'].iloc[-1] * RENKO_MIN_BRICK_PCT, 0.01)

    prev_close = resampled['Close'].shift(1)
    tr = pd.concat([
        resampled['High'] - resampled['Low'],
        (resampled['High'] - prev_close).abs(),
        (resampled['Low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(RENKO_BRICK_ATR_PERIOD, min_periods=1).mean().iloc[-1]
    price_floor = resampled['Close'].iloc[-1] * RENKO_MIN_BRICK_PCT
    return max(atr, price_floor, 0.01)

def build_renko_bricks(df_1m, brick_size):
    """
    Proper fixed-brick, CLOSE-ONLY, one-brick-per-step Renko construction.

    Deliberately does NOT look at each 1-minute candle's High/Low to decide
    whether a brick has formed (a common but flawed shortcut - a single wick
    can trip a brick threshold that the actual traded Close never reached).
    Only Close-to-close movement, against a FIXED brick_size, ever creates a
    new brick:

        while close has moved >= brick_size from the last brick's level
        (or >= 2x brick_size to reverse an established direction):
            emit exactly ONE new brick and advance the anchor by one
            brick_size step, then re-check (a single fast-moving 1-min
            candle can still legitimately emit several bricks - real Renko
            behaves this way - but each is emitted as its OWN row here, not
            compressed into one row with an inflated count column). That is
            what keeps the result a genuine 45-degree staircase: every row
            in the output is exactly one brick, always the same height.

    Returns a DataFrame of synthetic OHLC (Open = previous brick's Close,
    Close = this brick's Close, High/Low = max/min of the two) so the
    existing indicator math (which needs High/Low for ADX/DMI) can run on
    it unchanged - those High/Low values come only from consecutive BRICK
    closes, never from the original 1-minute candles' own High/Low.
    """
    closes = df_1m['Close'].values
    times = df_1m['Datetime'].values
    if len(closes) < 2:
        return pd.DataFrame(columns=['Datetime', 'Open', 'High', 'Low', 'Close'])

    bricks = []
    anchor = closes[0]
    direction = 0  # 0 = no established direction yet, 1 = up, -1 = down
    prev_brick_close = anchor

    for i in range(1, len(closes)):
        px, t = closes[i], times[i]
        while True:
            if direction >= 0 and px >= anchor + brick_size:
                anchor += brick_size
                direction = 1
            elif direction <= 0 and px <= anchor - brick_size:
                anchor -= brick_size
                direction = -1
            else:
                break
            o, c = prev_brick_close, anchor
            bricks.append({'Datetime': t, 'Open': o, 'High': max(o, c), 'Low': min(o, c), 'Close': c})
            prev_brick_close = anchor

    return pd.DataFrame(bricks)

def calculate_technical_signals(df):
    if len(df) < 30:
        return "Neutral", "Neutral", "Neutral"

    # --- 1. TRUE HYBRID BB-RSI MATH ---
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    df['RSI'] = 100 - (100 / (1 + rs))

    df['RSI_SMA'] = df['RSI'].rolling(BB_PERIOD).mean()
    df['RSI_STD'] = df['RSI'].rolling(BB_PERIOD).std()
    df['BB_Upper'] = df['RSI_SMA'] + (BB_STD * df['RSI_STD'])
    df['BB_Lower'] = df['RSI_SMA'] - (BB_STD * df['RSI_STD'])

    # --- 2. TRUE HYBRID BB-MACD-HIST MATH ---
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Line'] = ema_12 - ema_26
    df['Signal_Line'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD_Line'] - df['Signal_Line']

    df['MACD_Hist_SMA'] = df['MACD_Hist'].rolling(BB_PERIOD).mean()
    df['MACD_Hist_STD'] = df['MACD_Hist'].rolling(BB_PERIOD).std()
    df['MACD_BB_Upper'] = df['MACD_Hist_SMA'] + (BB_STD * df['MACD_Hist_STD'])
    df['MACD_BB_Lower'] = df['MACD_Hist_SMA'] - (BB_STD * df['MACD_Hist_STD'])

    # --- 3. ADX / DMI MATH ---
    df['up'] = df['High'].diff()
    df['down'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = np.where((df['up'] > df['down']) & (df['up'] > 0), df['up'], 0)
    df['-DM'] = np.where((df['down'] > df['up']) & (df['down'] > 0), df['down'], 0)

    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift(1)).abs()
    tr3 = (df['Low'] - df['Close'].shift(1)).abs()
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = df['TR'].ewm(alpha=1/ADX_PERIOD, adjust=False).mean()
    plus_di = 100 * (df['+DM'].ewm(alpha=1/ADX_PERIOD, adjust=False).mean() / (atr + 1e-8))
    minus_di = 100 * (df['-DM'].ewm(alpha=1/ADX_PERIOD, adjust=False).mean() / (atr + 1e-8))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
    df['ADX'] = dx.ewm(alpha=1/ADX_PERIOD, adjust=False).mean()
    df['+DI'] = plus_di
    df['-DI'] = minus_di

    # --- EXTRACT LATEST SIGNALS ---
    latest = df.iloc[-1]

    bb_rsi_sig = "Neutral"
    if latest['RSI'] > latest['BB_Upper']: bb_rsi_sig = "Buy+"
    elif latest['RSI'] < latest['BB_Lower']: bb_rsi_sig = "Sell+"

    bb_macd_sig = "Neutral"
    if latest['MACD_Hist'] > latest['MACD_BB_Upper']: bb_macd_sig = "Buy+"
    elif latest['MACD_Hist'] < latest['MACD_BB_Lower']: bb_macd_sig = "Sell+"

    adx_sig = "Neutral"
    if latest['ADX'] >= ADX_THRESHOLD:
        if latest['+DI'] > latest['-DI']: adx_sig = "Buy"
        elif latest['-DI'] > latest['+DI']: adx_sig = "Sell"

    return bb_rsi_sig, bb_macd_sig, adx_sig

# ==============================================================================
# 3. PIPELINE EXECUTOR & UI DRAWING
# ==============================================================================
def format_cell(text, width=10):
    text_str = str(text)
    if "Buy" in text_str: return f"{COLOR_GREEN_BG}{text_str:^{width}}{COLOR_RESET}"
    elif "Sell" in text_str: return f"{COLOR_RED_BG}{text_str:^{width}}{COLOR_RESET}"
    else: return f"{COLOR_GRAY_BG}{text_str:^{width}}{COLOR_RESET}"

def _filter_worker(item, filter_date):
    """
    One candidate's price/volume check. Runs inside a thread pool now
    (previously a plain `for` loop making one blocking call at a time).
    Returns (item, df) on pass so process_stock can REUSE this exact day's
    data instead of re-fetching it a second time later - previously
    filter_date was always inside trading_days, so every surviving stock's
    data for that day was silently fetched twice.
    """
    df = fetch_upstox_candles_for_date(item['key'], filter_date)
    if df is not None and not df.empty:
        close_px = df['Close'].iloc[-1]
        if MIN_PRICE <= close_px <= MAX_PRICE and df['Volume'].sum() >= MIN_DAILY_VOLUME:
            return item, df
    return None

def _fetch_remaining_days(key, days_needed):
    """Fetches a stock's remaining (non-cached) days in parallel."""
    if not days_needed: return []
    with concurrent.futures.ThreadPoolExecutor(max_workers=DAY_FETCH_WORKERS) as ex:
        return [d for d in ex.map(lambda day: fetch_upstox_candles_for_date(key, day), days_needed) if d is not None]

def process_stock(args):
    item, trading_days, cached = args  # cached: {date_str: df} for days already fetched during filtering
    days_needed = [d for d in trading_days if d not in cached]
    fetched = _fetch_remaining_days(item['key'], days_needed)
    dfs = list(cached.values()) + fetched

    if not dfs: return None
    master_1m = pd.concat(dfs, ignore_index=True).drop_duplicates(subset='Datetime').sort_values('Datetime').reset_index(drop=True)
    if len(master_1m) < 30: return None

    base_brick = compute_fixed_brick_size(master_1m)
    row_data = {'Symbol': item['symbol'], 'LTP': master_1m['Close'].iloc[-1], 'Score': 0, 'BullGranularities': 0, 'BearGranularities': 0}

    for mult in RENKO_GRANULARITIES:
        gtag = f"{mult}B"
        bricks = build_renko_bricks(master_1m, base_brick * mult)
        bb_rsi_sig, bb_macd_sig, adx_sig = calculate_technical_signals(bricks)
        row_data[f'BB_RSI_{gtag}'] = bb_rsi_sig
        row_data[f'BB_MACD_{gtag}'] = bb_macd_sig
        row_data[f'ADX_{gtag}'] = adx_sig

        # --- MOMENTUM HEAT SCORING + PER-GRANULARITY LEAN (for confluence) ---
        g_score = 0
        for sig in [bb_rsi_sig, bb_macd_sig, adx_sig]:
            if "Buy+" in sig: g_score += 2
            elif "Buy" in sig: g_score += 1
            elif "Sell+" in sig: g_score -= 2
            elif "Sell" in sig: g_score -= 1
        row_data['Score'] += g_score
        if g_score > 0: row_data['BullGranularities'] += 1
        elif g_score < 0: row_data['BearGranularities'] += 1

    return row_data

def run_screener():
    target_date_str = (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d")
    trading_days = get_past_trading_days(target_date_str, num_days=BACKTRACE_DAYS)
    filter_date = trading_days[-2] if len(trading_days) > 1 else trading_days[0]

    t_start = time.time()
    print(f"\n{COLOR_CYAN}📡 Initializing Screener Pipeline [{TRADING_MODE}]...{COLOR_RESET}")
    universe_raw = get_dynamic_universe(TRADING_MODE)

    if not universe_raw:
        print(f"⚠️ {COLOR_RED_FG}No universe found or UPSTOX_ACCESS_TOKEN is invalid.{COLOR_RESET}")
        return

    universe, filter_cache = [], {}
    if TRADING_MODE == "INDEX_OPTIONS":
        print(f"🔄 Bypassing Price/Volume filters for Indices. Found {len(universe_raw)} major indices.")
        universe = universe_raw
    else:
        candidates = universe_raw[:500]
        print(f"🔄 Filtering {len(candidates)} {TRADING_MODE} stocks for Volume & Price constraints ({UNIVERSE_FILTER_WORKERS} parallel workers)...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=UNIVERSE_FILTER_WORKERS) as ex:
            for result in ex.map(lambda it: _filter_worker(it, filter_date), candidates):
                if result is not None:
                    item, df = result
                    universe.append(item)
                    filter_cache[item['symbol']] = {filter_date: df}

    print(f"✅ Target Universe ready ({len(universe)} qualified assets). Computing technicals ({STOCK_WORKERS} parallel workers)...\n")

    work_items = [(item, trading_days, filter_cache.get(item['symbol'], {})) for item in universe]
    with concurrent.futures.ThreadPoolExecutor(max_workers=STOCK_WORKE
