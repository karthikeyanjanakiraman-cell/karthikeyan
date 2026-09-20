import os
import sys
import argparse
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
TRADING_MODE = "STOCK_FNO"   # Options: "STOCK_FNO", "CASH_EQUITY", "INDEX_OPTIONS"

# --- DECOUPLED HA-ATR ENGINE MULTIPLIERS ---
HA_ATR_MULTIPLIERS = [1, 2, 3, 5]     
ATR_BASIS_PERIOD = 14                 
ATR_BASIS_TF = "15min"                
MIN_ATR_PCT = 0.001                   

# --- OUTPUT LIMITS & CONFLUENCE ---
TOP_N_BUYERS = 15
TOP_N_SELLERS = 15
MIN_PERFECT_BLOCKS = 1 

COLOR_GREEN_BG = '\033[42m\033[30m'  
COLOR_RED_BG = '\033[41m\033[97m'    
COLOR_RESET = '\033[0m'
COLOR_BOLD = '\033[1m'
COLOR_CYAN = '\033[96m'
COLOR_RED_FG = '\033[91m'

# --- UNIVERSE FILTERING ---
MIN_PRICE = 50              
MAX_PRICE = 10000           
MIN_DAILY_VOLUME = 100000   
BACKTRACE_DAYS = 5          

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
# 1. FYERS API LIVE INGESTION & SETUP
# ==============================================================================
def get_dynamic_universe(mode):
    """Fetches and builds the instrument universe directly from FYERS CSV masters."""
    try:
        cm_url = "https://public.fyers.in/sym_details/NSE_CM.csv"
        cm_df = pd.read_csv(cm_url, header=None)
        
        if mode == "INDEX_OPTIONS":
            target_indices = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "NSE:FINNIFTY-INDEX", "NSE:MIDCPNIFTY-INDEX"]
            return [{"symbol": idx.split(":")[1].split("-")[0], "key": idx} for idx in target_indices]

        fo_url = "https://public.fyers.in/sym_details/NSE_FO.csv"
        fo_df = pd.read_csv(fo_url, header=None)
        
        # Column 12 in FYERS FO CSV contains the underlying short symbol (e.g., RELIANCE)
        fno_underlying = set(fo_df[12].dropna().unique())
        
        universe = []
        for _, row in cm_df.iterrows():
            sym_key = str(row[8])   # Column 8: FYERS format e.g., NSE:RELIANCE-EQ
            short_sym = str(row[12]) # Column 12: Short name e.g., RELIANCE
            
            if not sym_key.endswith("-EQ"):
                continue
                
            if mode == "STOCK_FNO" and short_sym in fno_underlying:
                universe.append({"symbol": short_sym, "key": sym_key})
            elif mode == "CASH_EQUITY" and short_sym not in fno_underlying:
                universe.append({"symbol": short_sym, "key": sym_key})
                
        return universe
    except Exception as e:
        print(f"{COLOR_RED_FG}[API Error] Failed to fetch Universe from FYERS: {e}{COLOR_RESET}")
        return []

def fetch_fyers_candles_for_date(instrument_key, date_str, retries=3):
    """Fetches 1-minute historical data seamlessly using FYERS API v3."""
    client_id = os.environ.get("CLIENT_ID")
    access_token = os.environ.get("ACCESS_TOKEN")
    if not client_id or not access_token: 
        return None

    headers = {'Authorization': f"{client_id}:{access_token}"}
    url = "https://api-t1.fyers.in/data/history"
    
    params = {
        "symbol": instrument_key,
        "resolution": "1",
        "date_format": "1",
        "range_from": date_str,
        "range_to": date_str
    }

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("s") == "ok" and "candles" in data and len(data["candles"]) > 0:
                    # FYERS output columns: Timestamp(epoch), Open, High, Low, Close, Volume
                    c_df = pd.DataFrame(data["candles"], columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                    
                    # Convert FYERS UTC epoch directly into local IST Pandas Datetime
                    c_df['Datetime'] = pd.to_datetime(c_df['Timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
                    
                    return c_df.sort_values('Datetime').reset_index(drop=True)
                else:
                    return None
            elif response.status_code == 429:
                time.sleep(1.0 * (attempt + 1))
            else:
                return None
        except Exception:
            time.sleep(1.0)
            
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
# 2. THE DECOUPLED HA-ATR ENGINE (WITH SESSION ISOLATION)
# ==============================================================================
def compute_base_atr(df_1m):
    resampled = df_1m.set_index('Datetime').resample(ATR_BASIS_TF).agg(
        {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
    ).dropna().reset_index()
    if len(resampled) < 5: return max(df_1m['Close'].iloc[-1] * MIN_ATR_PCT, 0.01)
    
    prev_close = resampled['Close'].shift(1)
    tr = pd.concat([
        resampled['High'] - resampled['Low'],
        (resampled['High'] - prev_close).abs(),
        (resampled['Low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(ATR_BASIS_PERIOD, min_periods=1).mean().iloc[-1]
    return max(atr, resampled['Close'].iloc[-1] * MIN_ATR_PCT, 0.01)

def build_isolated_range_bars(df_1m, target_range):
    df_1m['Date'] = df_1m['Datetime'].dt.date
    all_bars = []

    for date, group in df_1m.groupby('Date'):
        opens = group['Open'].values
        closes = group['Close'].values
        highs = group['High'].values
        lows = group['Low'].values
        times = group['Datetime'].values
        
        if len(closes) == 0: continue

        curr_O, curr_H, curr_L, curr_T = opens[0], highs[0], lows[0], times[0]
        curr_C = closes[0]
        raw_bars = []

        for i in range(len(closes)):
            curr_H = max(curr_H, highs[i])
            curr_L = min(curr_L, lows[i])
            curr_C = closes[i]

            if (curr_H - curr_L) >= target_range:
                raw_bars.append({'Datetime': curr_T, 'Open': curr_O, 'High': curr_H, 'Low': curr_L, 'Close': curr_C})
                curr_O = curr_C
                curr_H = curr_C
                curr_L = curr_C
                curr_T = times[i+1] if i < len(closes) - 1 else times[i]

        if curr_H > curr_L: 
            raw_bars.append({'Datetime': curr_T, 'Open': curr_O, 'High': curr_H, 'Low': curr_L, 'Close': curr_C})

        if not raw_bars: continue
        df_raw = pd.DataFrame(raw_bars)

        ha_closes = (df_raw['Open'] + df_raw['High'] + df_raw['Low'] + df_raw['Close']) / 4
        ha_opens = np.zeros(len(df_raw))
        ha_opens[0] = (df_raw['Open'].iloc[0] + df_raw['Close'].iloc[0]) / 2

        ha_closes_arr = ha_closes.to_numpy()
        for i in range(1, len(df_raw)):
            ha_opens[i] = (ha_opens[i-1] + ha_closes_arr[i-1]) / 2.0

        df_raw['HA_Trend'] = np.where(ha_closes >= ha_opens, 'Green', 'Red')
        all_bars.append(df_raw)

    if not all_bars: return pd.DataFrame()
    return pd.concat(all_bars, ignore_index=True)

def calculate_strict_signals(df):
    if len(df) < 5:
        return "", "", "", "NONE"

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-8))))

    df['RSI_SMA'] = df['RSI'].rolling(BB_PERIOD, min_periods=1).mean()
    df['RSI_STD'] = df['RSI'].rolling(BB_PERIOD, min_periods=1).std().fillna(0)
    df['BB_Upper'] = df['RSI_SMA'] + (BB_STD * df['RSI_STD'])
    df['BB_Lower'] = df['RSI_SMA'] - (BB_STD * df['RSI_STD'])

    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Hist'] = (ema_12 - ema_26) - (ema_12 - ema_26).ewm(span=9, adjust=False).mean()

    df['MACD_Hist_SMA'] = df['MACD_Hist'].rolling(BB_PERIOD, min_periods=1).mean()
    df['MACD_Hist_STD'] = df['MACD_Hist'].rolling(BB_PERIOD, min_periods=1).std().fillna(0)
    df['MACD_BB_Upper'] = df['MACD_Hist_SMA'] + (BB_STD * df['MACD_Hist_STD'])
    df['MACD_BB_Lower'] = df['MACD_Hist_SMA'] - (BB_STD * df['MACD_Hist_STD'])

    df['up'] = df['High'].diff()
    df['down'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = np.where((df['up'] > df['down']) & (df['up'] > 0), df['up'], 0)
    df['-DM'] = np.where((df['down'] > df['up']) & (df['down'] > 0), df['down'], 0)

    tr1, tr2, tr3 = df['High'] - df['Low'], (df['High'] - df['Close'].shift(1)).abs(), (df['Low'] - df['Close'].shift(1)).abs()
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = df['TR'].ewm(alpha=1/ADX_PERIOD, adjust=False).mean()
    df['+DI'] = 100 * (df['+DM'].ewm(alpha=1/ADX_PERIOD, adjust=False).mean() / (atr + 1e-8))
    df['-DI'] = 100 * (df['-DM'].ewm(alpha=1/ADX_PERIOD, adjust=False).mean() / (atr + 1e-8))
    df['ADX'] = (100 * (df['+DI'] - df['-DI']).abs() / (df['+DI'] + df['-DI'] + 1e-8)).ewm(alpha=1/ADX_PERIOD, adjust=False).mean()

    latest = df.iloc[-1]
    ha_trend = latest['HA_Trend']

    bb_rsi = "Neutral"
    if latest['RSI'] > latest['BB_Upper'] and latest['RSI_STD'] > 0: bb_rsi = "Buy+"
    elif latest['RSI'] > 55: bb_rsi = "Buy"
    elif latest['RSI'] < latest['BB_Lower'] and latest['RSI_STD'] > 0: bb_rsi = "Sell+"
    elif latest['RSI'] < 45: bb_rsi = "Sell"

    bb_macd = "Neutral"
    if latest['MACD_Hist'] > latest['MACD_BB_Upper'] and latest['MACD_Hist_STD'] > 0: bb_macd = "Buy+"
    elif latest['MACD_Hist'] > 0: bb_macd = "Buy"
    elif latest['MACD_Hist'] < latest['MACD_BB_Lower'] and latest['MACD_Hist_STD'] > 0: bb_macd = "Sell+"
    elif latest['MACD_Hist'] < 0: bb_macd = "Sell"

    adx_sig = "Neutral"
    if latest['ADX'] >= ADX_THRESHOLD:
        if latest['+DI'] > latest['-DI']: adx_sig = "Buy"
        elif latest['-DI'] > latest['+DI']: adx_sig = "Sell"

    is_bull_aligned = ("Buy" in bb_rsi) and ("Buy" in bb_macd) and ("Buy" in adx_sig) and (ha_trend == 'Green')
    is_bear_aligned = ("Sell" in bb_rsi) and ("Sell" in bb_macd) and ("Sell" in adx_sig) and (ha_trend == 'Red')

    if is_bull_aligned: return bb_rsi, bb_macd, adx_sig, "BULL"
    elif is_bear_aligned: return bb_rsi, bb_macd, adx_sig, "BEAR"
    else: return "", "", "", "NONE"

# ==============================================================================
# 3. PIPELINE EXECUTOR & UI DRAWING
# ==============================================================================
def format_cell(text, width=10):
    if not text:
        return " " * width
        
    text_str = str(text)
    spaces = width - len(text_str)
    left_pad = " " * (spaces // 2)
    right_pad = " " * (spaces - (spaces // 2))
    
    if "Buy" in text_str: 
        colored_text = f"{COLOR_GREEN_BG}{text_str}{COLOR_RESET}"
    elif "Sell" in text_str: 
        colored_text = f"{COLOR_RED_BG}{text_str}{COLOR_RESET}"
    else:
        colored_text = text_str
        
    return f"{left_pad}{colored_text}{right_pad}"

def _filter_worker(item, filter_date, latest_day):
    # Fyers doesn't need an 'is_latest_day' flag, the endpoint naturally covers it
    df = fetch_fyers_candles_for_date(item['key'], filter_date)
    if df is not None and not df.empty:
        if MIN_PRICE <= df['Close'].iloc[-1] <= MAX_PRICE and df['Volume'].sum() >= MIN_DAILY_VOLUME:
            return item, df
    return None

def process_stock(args):
    item, trading_days, cached = args  
    latest_day = trading_days[-1]
    days_needed = [d for d in trading_days if d not in cached]
    
    fetched = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=DAY_FETCH_WORKERS) as ex:
        futures = {
            ex.submit(fetch_fyers_candles_for_date, item['key'], day): day 
            for day in days_needed
        }
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res is not None and not res.empty:
                fetched.append(res)
    
    dfs = list(cached.values()) + fetched
    if not dfs: return None
    
    master_1m = pd.concat(dfs, ignore_index=True).drop_duplicates(subset='Datetime').sort_values('Datetime').reset_index(drop=True)
    
    # Data Integrity Guardrail
    if master_1m['Datetime'].dt.strftime("%Y-%m-%d").max() != latest_day:
        return None

    if len(master_1m) < 30: return None

    base_atr = compute_base_atr(master_1m)
    row_data = {'Symbol': item['symbol'], 'LTP': master_1m['Close'].iloc[-1], 'Score': 0, 'PerfectBullBlocks': 0, 'PerfectBearBlocks': 0}

    for mult in HA_ATR_MULTIPLIERS:
        gtag = f"{mult}X"
        ha_atr_df = build_isolated_range_bars(master_1m, base_atr * mult)
        
        bb_rsi, bb_macd, adx_sig, alignment = calculate_strict_signals(ha_atr_df)
        
        row_data[f'BB_RSI_{gtag}'] = bb_rsi
        row_data[f'BB_MACD_{gtag}'] = bb_macd
        row_data[f'ADX_{gtag}'] = adx_sig

        if alignment == "BULL":
            row_data['PerfectBullBlocks'] += 1
            row_data['Score'] += (bb_rsi.count('+') + bb_macd.count('+') + 1)
        elif alignment == "BEAR":
            row_data['PerfectBearBlocks'] += 1
            row_data['Score'] -= (bb_rsi.count('+') + bb_macd.count('+') + 1)

    return row_data

def run_screener():
    target_date_str = (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d")
    trading_days = get_past_trading_days(target_date_str, num_days=BACKTRACE_DAYS)
    latest_day = trading_days[-1]
    filter_date = trading_days[-2] if len(trading_days) > 1 else trading_days[0]

    t_start = time.time()
    print(f"\n{COLOR_CYAN}📡 Initializing Screener Pipeline [{TRADING_MODE}] via FYERS API...{COLOR_RESET}")
    universe_raw = get_dynamic_universe(TRADING_MODE)

    if not universe_raw: return

    universe, filter_cache = [], {}
    candidates = universe_raw  
    print(f"🔄 Filtering {len(candidates)} {TRADING_MODE} stocks for Volume & Price constraints...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=UNIVERSE_FILTER_WORKERS) as ex:
        for result in ex.map(lambda it: _filter_worker(it, filter_date, latest_day), candidates):
            if result is not None:
                item, df = result
                universe.append(item)
                filter_cache[item['symbol']] = {filter_date: df}

    print(f"✅ Target Universe ready ({len(universe)} qualified assets). Computing technicals...\n")

    work_items = [(item, trading_days, filter_cache.get(item['symbol'], {})) for item in universe]
    with concurrent.futures.ThreadPoolExecutor(max_workers=STOCK_WORKERS) as executor:
        results = list(executor.map(process_stock, work_items))
        
    dashboard_data = [r for r in results if r is not None]

    bulls = [r for r in dashboard_data if r['PerfectBullBlocks'] >= MIN_PERFECT_BLOCKS]
    bears = [r for r in dashboard_data if r['PerfectBearBlocks'] >= MIN_PERFECT_BLOCKS]

    bulls.sort(key=lambda x: (x['PerfectBullBlocks'], x['Score']), reverse=True)
    bears.sort(key=lambda x: (x['PerfectBearBlocks'], abs(x['Score'])), reverse=True)

    bulls = bulls[:TOP_N_BUYERS]
    bears = bears[:TOP_N_SELLERS]

    print(f"{COLOR_BOLD}=== INSTITUTIONAL DECOUPLED HA-ATR DASHBOARD [{TRADING_MODE}] ==={COLOR_RESET}\n")

    def print_basket(title, icon, data_list):
        if not data_list: return
        print(f"\n{COLOR_BOLD}{icon} {title}{COLOR_RESET}")
        
        header_str = f" {COLOR_CYAN}{'Script':<16} {'LTP':<8} |"
        for mult in HA_ATR_MULTIPLIERS:
            gtag = f"{mult}X"
            header_str += f"  {'BB-RSI '+gtag:^11} {'BB-MACD '+gtag:^12} {'ADX '+gtag:^9} |"
        
        dash_len = len(header_str) - 8 
        print(header_str + COLOR_RESET)
        print("-" * dash_len)

        for row in data_list:
            row_str = f" {row['Symbol']:<16} {row['LTP']:<8.2f} |"
            for mult in HA_ATR_MULTIPLIERS:
                gtag = f"{mult}X"
                bb_rsi_cell = format_cell(row[f'BB_RSI_{gtag}'], 11)
                bb_macd_cell = format_cell(row[f'BB_MACD_{gtag}'], 12)
                adx_cell = format_cell(row[f'ADX_{gtag}'], 9)
                row_str += f"  {bb_rsi_cell} {bb_macd_cell} {adx_cell} |"
            print(row_str)

    print_basket(f"TOP BUYERS (Pure Alignment >= {MIN_PERFECT_BLOCKS} Block)", "🔥", bulls)
    print_basket(f"TOP SELLERS (Pure Alignment >= {MIN_PERFECT_BLOCKS} Block)", "🩸", bears)
    
    print(f"\n⏱️ Scan completed in {(time.time() - t_start):.2f} seconds.\n")

if __name__ == "__main__":
    if not os.environ.get("CLIENT_ID") or not os.environ.get("ACCESS_TOKEN"):
        print(f"{COLOR_RED_FG}[!] Missing CLIENT_ID or ACCESS_TOKEN environment variables.{COLOR_RESET}")
        print(f"Please set them using: export CLIENT_ID='your_id' and export ACCESS_TOKEN='your_token'")
        sys.exit(1)
    run_screener()
