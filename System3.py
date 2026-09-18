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
RENKO_GRANULARITIES = [1, 2, 3, 5]   # multiples of the per-symbol base brick size
RENKO_BRICK_ATR_PERIOD = 14           # used ONCE per symbol to size the base brick 
RENKO_BRICK_BASIS_TF = "15min"        # resample used ONLY for that one-time ATR sizing calc
RENKO_MIN_BRICK_PCT = 0.001           # brick floor as a fraction of price

# --- OUTPUT LIMITS & CONFLUENCE ---
TOP_N_BUYERS = 15
TOP_N_SELLERS = 15
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

# --- CONCURRENCY ---
UNIVERSE_FILTER_WORKERS = 10
STOCK_WORKERS = 10
DAY_FETCH_WORKERS = 3

# ==============================================================================
# 1. LIVE INGESTION & SETUP
# ==============================================================================
def get_dynamic_universe(mode):
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
    closes = df_1m['Close'].values
    times = df_1m['Datetime'].values
    if len(closes) < 2:
        return pd.DataFrame(columns=['Datetime', 'Open', 'High', 'Low', 'Close'])

    bricks = []
    anchor = closes[0]
    direction = 0  
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
    # FIX: Lowered required length to 5 to accommodate highly-compressed Renko granularities
    if len(df) < 5:
        return "Neutral", "Neutral", "Neutral"

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    df['RSI'] = 100 - (100 / (1 + rs))

    # FIX: Added min_periods=1 and fillna(0) to allow MAs to calculate on small datasets
    df['RSI_SMA'] = df['RSI'].rolling(BB_PERIOD, min_periods=1).mean()
    df['RSI_STD'] = df['RSI'].rolling(BB_PERIOD, min_periods=1).std().fillna(0)
    df['BB_Upper'] = df['RSI_SMA'] + (BB_STD * df['RSI_STD'])
    df['BB_Lower'] = df['RSI_SMA'] - (BB_STD * df['RSI_STD'])

    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Line'] = ema_12 - ema_26
    df['Signal_Line'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD_Line'] - df['Signal_Line']

    df['MACD_Hist_SMA'] = df['MACD_Hist'].rolling(BB_PERIOD, min_periods=1).mean()
    df['MACD_Hist_STD'] = df['MACD_Hist'].rolling(BB_PERIOD, min_periods=1).std().fillna(0)
    df['MACD_BB_Upper'] = df['MACD_Hist_SMA'] + (BB_STD * df['MACD_Hist_STD'])
    df['MACD_BB_Lower'] = df['MACD_Hist_SMA'] - (BB_STD * df['MACD_Hist_STD'])

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
    df = fetch_upstox_candles_for_date(item['key'], filter_date)
    if df is not None and not df.empty:
        close_px = df['Close'].iloc[-1]
        if MIN_PRICE <= close_px <= MAX_PRICE and df['Volume'].sum() >= MIN_DAILY_VOLUME:
            return item, df
    return None

def _fetch_remaining_days(key, days_needed):
    if not days_needed: return []
    with concurrent.futures.ThreadPoolExecutor(max_workers=DAY_FETCH_WORKERS) as ex:
        return [d for d in ex.map(lambda day: fetch_upstox_candles_for_date(key, day), days_needed) if d is not None]

def process_stock(args):
    item, trading_days, cached = args  
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
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=STOCK_WORKERS) as executor:
        results = list(executor.map(process_stock, work_items))
        
    dashboard_data = [r for r in results if r is not None]

    if not dashboard_data:
        print("No stocks passed the filtering criteria.")
        return

    # Split into Baskets based on Confluence
    bulls = [r for r in dashboard_data if r['BullGranularities'] >= MIN_CONFLUENCE_COUNT]
    bears = [r for r in dashboard_data if r['BearGranularities'] >= MIN_CONFLUENCE_COUNT]

    bulls.sort(key=lambda x: x['Score'], reverse=True)
    bears.sort(key=lambda x: x['Score'])

    bulls = bulls[:TOP_N_BUYERS]
    bears = bears[:TOP_N_SELLERS]

    print(f"{COLOR_BOLD}=== INSTITUTIONAL MULTI-GRANULARITY DASHBOARD [{TRADING_MODE}] ==={COLOR_RESET}\n")
    dash_len = 28 + len(RENKO_GRANULARITIES) * 39

    def print_basket(title, icon, data_list):
        if not data_list: return
        print(f"\n{COLOR_BOLD}{icon} {title}{COLOR_RESET}")
        header_str = f" {COLOR_CYAN}{'Script':<16} {'LTP':<8}"
        for mult in RENKO_GRANULARITIES:
            gtag = f"{mult}B"
            header_str += f" | {'BB-RSI '+gtag:^11} {'BB-MACD '+gtag:^12} {'ADX '+gtag:^9}"
        print(header_str + COLOR_RESET)
        print("-" * dash_len)

        for row in data_list:
            row_str = f" {row['Symbol']:<16} {row['LTP']:<8.2f}"
            for mult in RENKO_GRANULARITIES:
                gtag = f"{mult}B"
                bb_rsi_cell = format_cell(row[f'BB_RSI_{gtag}'], 11)
                bb_macd_cell = format_cell(row[f'BB_MACD_{gtag}'], 12)
                adx_cell = format_cell(row[f'ADX_{gtag}'], 9)
                row_str += f" | {bb_rsi_cell} {bb_macd_cell} {adx_cell}"
            print(row_str)

    print_basket(f"TOP BUYERS (Bullish Confluence >= {MIN_CONFLUENCE_COUNT})", "🔥", bulls)
    print_basket(f"TOP SELLERS (Bearish Confluence >= {MIN_CONFLUENCE_COUNT})", "🩸", bears)
    
    t_end = time.time()
    print(f"\n⏱️ Scan completed in {(t_end - t_start):.2f} seconds.\n")

if __name__ == "__main__":
    if not os.environ.get("UPSTOX_ACCESS_TOKEN"):
        print(f"❌ {COLOR_RED_FG}Error: UPSTOX_ACCESS_TOKEN environment variable not found.{COLOR_RESET}")
        sys.exit(1)
        
    run_screener()
        
