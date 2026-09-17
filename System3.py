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

# --- MULTI-TIMEFRAME DASHBOARD COLUMNS ---
TARGET_TIMEFRAMES = ["3min", "5min", "10min", "15min"]

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

def fetch_upstox_candles_for_date(instrument_key, date_str):
    access_token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not access_token: return None
    
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {access_token}'}
    today_str = (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d")
    
    if date_str == today_str:
        url = f"https://api.upstox.com/v2/historical-candle/intraday/{urllib.parse.quote(instrument_key)}/1minute"
    else:
        url = f"https://api.upstox.com/v2/historical-candle/{urllib.parse.quote(instrument_key)}/1minute/{date_str}/{date_str}"
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code != 200: return None
        data = response.json().get('data', {}).get('candles', [])
        if not data: return None
        c_df = pd.DataFrame(data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
        c_df['Datetime'] = pd.to_datetime(c_df['Timestamp']).dt.tz_localize(None) 
        return c_df.sort_values('Datetime').reset_index(drop=True)
    except:
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
# 2. COMBINED BB-RSI & ADX MATH
# ==============================================================================
def resample_tape(df_1m, tf_str):
    # FIX: Pandas 2.2+ no longer accepts 'T' for minutes. Pass 'min' directly.
    df = df_1m.set_index('Datetime')
    resampled = df.resample(tf_str).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    return resampled.reset_index()

def calculate_technical_signals(df):
    if len(df) < 30: 
        return "Neutral", "Neutral"

    # --- TRUE HYBRID BB-RSI MATH ---
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate Bollinger Bands OF the RSI line
    df['RSI_SMA'] = df['RSI'].rolling(BB_PERIOD).mean()
    df['RSI_STD'] = df['RSI'].rolling(BB_PERIOD).std()
    df['BB_Upper'] = df['RSI_SMA'] + (BB_STD * df['RSI_STD'])
    df['BB_Lower'] = df['RSI_SMA'] - (BB_STD * df['RSI_STD'])

    # --- ADX / DMI MATH ---
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
    
    # Combined BB-RSI Breakout Signal
    bb_rsi_sig = "Neutral"
    if latest['RSI'] > latest['BB_Upper']:
        bb_rsi_sig = "Buy+"
    elif latest['RSI'] < latest['BB_Lower']:
        bb_rsi_sig = "Sell+"
        
    # ADX Signal Generation
    adx_sig = "Neutral"
    if latest['ADX'] >= ADX_THRESHOLD:
        if latest['+DI'] > latest['-DI']:
            adx_sig = "Buy"
        elif latest['-DI'] > latest['+DI']:
            adx_sig = "Sell"

    return bb_rsi_sig, adx_sig

# ==============================================================================
# 3. PIPELINE EXECUTOR & UI DRAWING
# ==============================================================================
def format_cell(text, width=10):
    text_str = str(text)
    if "Buy" in text_str:
        return f"{COLOR_GREEN_BG}{text_str:^{width}}{COLOR_RESET}"
    elif "Sell" in text_str:
        return f"{COLOR_RED_BG}{text_str:^{width}}{COLOR_RESET}"
    else:
        return f"{COLOR_GRAY_BG}{text_str:^{width}}{COLOR_RESET}"

def run_screener(timeframes):
    target_date_str = (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d")
    trading_days = get_past_trading_days(target_date_str, num_days=BACKTRACE_DAYS)
    filter_date = trading_days[-2] if len(trading_days) > 1 else trading_days[0]
    
    print(f"\n{COLOR_CYAN}📡 Initializing Screener Pipeline [{TRADING_MODE}]...{COLOR_RESET}")
    universe_raw = get_dynamic_universe(TRADING_MODE)
    
    if not universe_raw:
        print(f"⚠️ {COLOR_RED_FG}No universe found or UPSTOX_ACCESS_TOKEN is invalid.{COLOR_RESET}")
        return

    universe = []
    if TRADING_MODE == "INDEX_OPTIONS":
        print(f"🔄 Bypassing Price/Volume filters for Indices. Found {len(universe_raw)} major indices.")
        universe = universe_raw
    else:
        print(f"🔄 Filtering {len(universe_raw)} {TRADING_MODE} stocks for Volume & Price constraints...")
        for item in universe_raw[:500]: # Capped to prevent aggressive Upstox rate limits
            df = fetch_upstox_candles_for_date(item['key'], filter_date)
            if df is not None and not df.empty:
                close_px = df['Close'].iloc[-1]
                if MIN_PRICE <= close_px <= MAX_PRICE and df['Volume'].sum() >= MIN_DAILY_VOLUME:
                    universe.append(item)
            time.sleep(0.01)

    print(f"✅ Target Universe ready. Computing technicals for {len(universe)} qualified assets...\n")

    dashboard_data = []
    def process_stock(item):
        dfs = []
        for day in trading_days:
            d = fetch_upstox_candles_for_date(item['key'], day)
            if d is not None: dfs.append(d)
            
        if not dfs: return None
        master_1m = pd.concat(dfs, ignore_index=True)
        
        row_data = {
            'Symbol': item['symbol'],
            'LTP': master_1m['Close'].iloc[-1]
        }
        
        for tf in timeframes:
            resampled_df = resample_tape(master_1m, tf)
            bb_sig, adx_sig = calculate_technical_signals(resampled_df)
            row_data[f'BB_RSI_{tf}'] = bb_sig
            row_data[f'ADX_{tf}'] = adx_sig
            
        return row_data

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(process_stock, universe))
        
    dashboard_data = [r for r in results if r is not None]

    if not dashboard_data:
        print("No stocks passed the filtering criteria.")
        return

    print(f"{COLOR_BOLD}=== INSTITUTIONAL MULTI-TIMEFRAME DASHBOARD [{TRADING_MODE}] ==={COLOR_RESET}\n")
    
    header_str = f" {COLOR_CYAN}{'Script':<16} {'LTP':<8}"
    for tf in timeframes:
        header_str += f" | {'BB-RSI '+tf:^11} {'ADX '+tf:^9}"
    print(header_str + COLOR_RESET)
    
    # Calculate exact dynamic divider length
    dash_len = 28 + len(timeframes) * 26
    print("-" * dash_len)

    # Sort Dashboard Alphabetically 
    dashboard_data.sort(key=lambda x: x['Symbol'])

    for row in dashboard_data:
        row_str = f" {row['Symbol']:<16} {row['LTP']:<8.2f}"
        for tf in timeframes:
            bb_rsi_cell = format_cell(row[f'BB_RSI_{tf}'], 11)
            adx_cell = format_cell(row[f'ADX_{tf}'], 9)
            row_str += f" | {bb_rsi_cell} {adx_cell}"
        print(row_str)
    print("\n")

if __name__ == "__main__":
    if not os.environ.get("UPSTOX_ACCESS_TOKEN"):
        print(f"❌ {COLOR_RED_FG}Error: UPSTOX_ACCESS_TOKEN environment variable not found.{COLOR_RESET}")
        sys.exit(1)
        
    run_screener(TARGET_TIMEFRAMES)
