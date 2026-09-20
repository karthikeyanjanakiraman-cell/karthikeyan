import os
import sys
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
TRADING_MODE = "INDEX_OPTIONS"   # Options: "STOCK_FNO", "CASH_EQUITY", "INDEX_OPTIONS"

# --- OPTIONS CHAIN CONFIGURATION ---
EXPIRY_OFFSET = 0          # 0 = Current (Nearest) Expiry, 1 = Next Expiry
STRIKES_FROM_ATM = 5       # Generates ATM + 5 OTM + 5 ITM (Total 11 strikes per CE and PE)
OPT_MIN_PRICE = 30          # Filter out worthless deep OTM options below ₹5
OPT_MIN_VOLUME = 10000     # Minimum volume for options liquidity

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

# --- EQUITY UNIVERSE FILTERING (Only applies if not in OPTIONS mode) ---
MIN_PRICE = 50              
MAX_PRICE = 10000           
MIN_DAILY_VOLUME = 100000   
BACKTRACE_DAYS = 5          

# --- INDICATOR PERIODS ---
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 1           # Set to 1.5 to catch heavy institutional volume impulses
ADX_PERIOD = 14
ADX_THRESHOLD = 20

# --- CONCURRENCY ---
UNIVERSE_FILTER_WORKERS = 10
STOCK_WORKERS = 10
DAY_FETCH_WORKERS = 3

# ==============================================================================
# 1. UPSTOX API LIVE INGESTION & DYNAMIC OPTIONS STRIKE GENERATION
# ==============================================================================
def fetch_upstox_candles_for_date(instrument_key, date_str, is_latest_day=False, retries=3):
    access_token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not access_token: return None
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {access_token}'}
    encoded_key = urllib.parse.quote(instrument_key)

    urls_to_try = []
    if is_latest_day:
        urls_to_try.append(f"https://api.upstox.com/v2/historical-candle/intraday/{encoded_key}/1minute")
    urls_to_try.append(f"https://api.upstox.com/v2/historical-candle/{encoded_key}/1minute/{date_str}/{date_str}")

    for url in urls_to_try:
        for attempt in range(retries):
            try:
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json().get('data', {}).get('candles', [])
                    if data:
                        c_df = pd.DataFrame(data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
                        c_df['Datetime'] = pd.to_datetime(c_df['Timestamp']).dt.tz_localize(None)
                        if "intraday" in url:
                            c_df = c_df[c_df['Datetime'].dt.strftime("%Y-%m-%d") == date_str]
                            if c_df.empty: break  
                        return c_df.sort_values('Datetime').reset_index(drop=True)
                elif response.status_code == 429:
                    time.sleep(1.0 * (attempt + 1))
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

def get_dynamic_universe(mode, latest_day):
    print(f"🔄 Downloading Live Exchange Master JSONs (NSE & BSE)...")
    
    def fetch_gz_json(url):
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            raise Exception(f"HTTP {resp.status_code} for {url}")
        return json.load(gzip.GzipFile(fileobj=io.BytesIO(resp.content)))

    try:
        # Upstox bundles ALL segments (EQ, FO, CUR) inside the primary exchange file.
        nse_data = fetch_gz_json("https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz")
        bse_data = fetch_gz_json("https://assets.upstox.com/market-quote/instruments/exchange/BSE.json.gz")
    except Exception as e:
        print(f"{COLOR_RED_FG}[API Error] Failed to fetch Universe: {e}{COLOR_RESET}")
        return []

    # Merge NSE and BSE to search globally
    all_data = nse_data + bse_data

    if mode == "INDEX_OPTIONS":
        universe = []
        
        index_config = {
            "NIFTY": {"spot_key": "NSE_INDEX|Nifty 50", "step": 50, "match": ["NIFTY", "NIFTY 50"]},
            "BANKNIFTY": {"spot_key": "NSE_INDEX|Nifty Bank", "step": 100, "match": ["BANKNIFTY", "NIFTY BANK"]},
            "FINNIFTY": {"spot_key": "NSE_INDEX|Nifty Fin Service", "step": 50, "match": ["FINNIFTY", "NIFTY FIN SERVICE"]},
            "MIDCPNIFTY": {"spot_key": "NSE_INDEX|NIFTY MID SELECT", "step": 25, "match": ["MIDCPNIFTY", "NIFTY MID SELECT"]},
            "SENSEX": {"spot_key": "BSE_INDEX|SENSEX", "step": 100, "match": ["SENSEX", "BSE SENSEX"]}
        }

        print(f"🎯 Calculating ATM Strikes & Constructing Options Chain for {len(index_config)} Indices...\n")
        
        for idx_name, config in index_config.items():
            # 1. Get exact current spot price
            spot_df = fetch_upstox_candles_for_date(config["spot_key"], latest_day, is_latest_day=True)
            if spot_df is None or spot_df.empty:
                spot_df = fetch_upstox_candles_for_date(config["spot_key"], get_past_trading_days(latest_day, 2)[-2])
            
            if spot_df is None or spot_df.empty:
                print(f"   [!] {idx_name}: Could not fetch live spot price. Skipping.")
                continue
                
            spot_price = spot_df['Close'].iloc[-1]
            step = config["step"]
            atm_strike = int(round(spot_price / step) * step)
            
            # Generate range of strikes 
            target_strikes = [int(atm_strike + (i * step)) for i in range(-STRIKES_FROM_ATM, STRIKES_FROM_ATM + 1)]
            
            # Bulletproof Options Filter: Direct API Tag validation
            idx_opts = []
            valid_names = config["match"]
            
            for item in all_data:
                name = str(item.get("name", "")).upper()
                underlying = str(item.get("underlying_symbol", "")).upper()
                inst_type = str(item.get("instrument_type", "")).upper()
                
                # Verify it belongs to this index AND is mathematically flagged as an option
                if (name in valid_names or underlying in valid_names) and inst_type in ["CE", "PE"]:
                    strike_val = item.get("strike", item.get("strike_price"))
                    try:
                        if strike_val is not None:
                            strike = int(float(strike_val))
                            if strike > 0:
                                item['clean_strike'] = strike
                                item['clean_ts'] = str(item.get("tradingsymbol", item.get("trading_symbol", "")))
                                idx_opts.append(item)
                    except (ValueError, TypeError):
                        continue
            
            if not idx_opts: 
                print(f"   [!] {idx_name}: Could not find option chain in JSON. Skipping.")
                continue
                
            # Extract and Sort Expirations Chronologically (Handling Upstox Unix Timestamps)
            expiries_set = set(item.get("expiry") for item in idx_opts if item.get("expiry"))
            valid_expiries = []
            for e in expiries_set:
                try:
                    if isinstance(e, (int, float)) or (isinstance(e, str) and e.isdigit()):
                        dt = datetime.fromtimestamp(int(e) / 1000.0)
                        valid_expiries.append((dt, e))
                    else:
                        e_str = str(e).split('T')[0]
                        try: valid_expiries.append((datetime.strptime(e_str, "%Y-%m-%d"), e))
                        except: 
                            try: valid_expiries.append((datetime.strptime(e_str, "%d-%b-%Y"), e))
                            except: valid_expiries.append((datetime.max, e))
                except:
                    valid_expiries.append((datetime.max, e))
                    
            valid_expiries.sort(key=lambda x: x[0])
            unique_expiries = [x[1] for x in valid_expiries]

            if len(unique_expiries) <= EXPIRY_OFFSET: 
                print(f"   [!] {idx_name}: Required expiry offset not available. Found {len(unique_expiries)}. Skipping.")
                continue
            
            target_expiry = unique_expiries[EXPIRY_OFFSET]
            target_dt = [x[0] for x in valid_expiries if x[1] == target_expiry][0]
            expiry_str = target_dt.strftime("%d-%b-%Y") if target_dt != datetime.max else str(target_expiry)
            
            # Filter exact options based on calculated targets and chosen expiry
            matched_options = [opt for opt in idx_opts if opt.get("expiry") == target_expiry and opt.get("clean_strike") in target_strikes]
            
            print(f"   => {idx_name:<10} | Spot: {spot_price:<8.2f} | ATM: {atm_strike:<6} | Expiry: {expiry_str} | Grabbed {len(matched_options)} CE/PE Contracts")
            
            for opt in matched_options:
                universe.append({
                    "symbol": opt.get("clean_ts"), 
                    "key": opt.get("instrument_key")
                })
        
        print("\n")
        return universe

    # --- Standard Equity Execution below ---
    fno_underlying = {item.get("underlying_symbol") for item in nse_data if item.get("segment") == "NSE_FO" and item.get("underlying_symbol")}
    if mode == "STOCK_FNO":
        return [{"symbol": item.get("tradingsymbol", item.get("trading_symbol")), "key": item.get("instrument_key")} 
                for item in nse_data if item.get("segment") == "NSE_EQ" and item.get("tradingsymbol", item.get("trading_symbol")) in fno_underlying]
    elif mode == "CASH_EQUITY":
        return [{"symbol": item.get("tradingsymbol", item.get("trading_symbol")), "key": item.get("instrument_key")} 
                for item in nse_data if item.get("segment") == "NSE_EQ" and item.get("tradingsymbol", item.get("trading_symbol")) not in fno_underlying]

    return []

# ==============================================================================
# 2. THE DECOUPLED HA-ATR ENGINE (STRICT BOLLINGER BAND BREAKOUTS)
# ==============================================================================
def compute_base_atr(df_1m):
    resampled = df_1m.set_index('Datetime').resample(ATR_BASIS_TF).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna().reset_index()
    if len(resampled) < 5: return max(df_1m['Close'].iloc[-1] * MIN_ATR_PCT, 0.01)
    
    prev_close = resampled['Close'].shift(1)
    tr = pd.concat([resampled['High'] - resampled['Low'], (resampled['High'] - prev_close).abs(), (resampled['Low'] - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(ATR_BASIS_PERIOD, min_periods=1).mean().iloc[-1]
    return max(atr, resampled['Close'].iloc[-1] * MIN_ATR_PCT, 0.01)

def build_isolated_range_bars(df_1m, target_range):
    df_1m['Date'] = df_1m['Datetime'].dt.date
    all_bars = []
    for date, group in df_1m.groupby('Date'):
        opens, closes, highs, lows, times = group['Open'].values, group['Close'].values, group['High'].values, group['Low'].values, group['Datetime'].values
        if len(closes) == 0: continue

        curr_O, curr_H, curr_L, curr_T = opens[0], highs[0], lows[0], times[0]
        curr_C = closes[0]
        raw_bars = []

        for i in range(len(closes)):
            curr_H, curr_L, curr_C = max(curr_H, highs[i]), min(curr_L, lows[i]), closes[i]
            if (curr_H - curr_L) >= target_range:
                raw_bars.append({'Datetime': curr_T, 'Open': curr_O, 'High': curr_H, 'Low': curr_L, 'Close': curr_C})
                curr_O = curr_H = curr_L = curr_C
                curr_T = times[i+1] if i < len(closes) - 1 else times[i]

        if curr_H > curr_L: raw_bars.append({'Datetime': curr_T, 'Open': curr_O, 'High': curr_H, 'Low': curr_L, 'Close': curr_C})
        if not raw_bars: continue
        
        df_raw = pd.DataFrame(raw_bars)
        ha_closes = (df_raw['Open'] + df_raw['High'] + df_raw['Low'] + df_raw['Close']) / 4
        ha_opens = np.zeros(len(df_raw))
        ha_opens[0] = (df_raw['Open'].iloc[0] + df_raw['Close'].iloc[0]) / 2
        
        ha_closes_arr = ha_closes.to_numpy()
        for i in range(1, len(df_raw)): ha_opens[i] = (ha_opens[i-1] + ha_closes_arr[i-1]) / 2.0

        df_raw['HA_Trend'] = np.where(ha_closes >= ha_opens, 'Green', 'Red')
        all_bars.append(df_raw)

    return pd.concat(all_bars, ignore_index=True) if all_bars else pd.DataFrame()

def calculate_strict_signals(df):
    """Zero-lag momentum requiring strict Bollinger Band breakouts for RSI, MACD, +DI, and -DI."""
    if len(df) < 5: return "", "", "", "NONE"

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
    tr = pd.concat([df['High'] - df['Low'], (df['High'] - df['Close'].shift(1)).abs(), (df['Low'] - df['Close'].shift(1)).abs()], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/ADX_PERIOD, adjust=False).mean()
    df['+DI'] = 100 * (df['+DM'].ewm(alpha=1/ADX_PERIOD, adjust=False).mean() / (atr + 1e-8))
    df['-DI'] = 100 * (df['-DM'].ewm(alpha=1/ADX_PERIOD, adjust=False).mean() / (atr + 1e-8))
    df['ADX'] = (100 * (df['+DI'] - df['-DI']).abs() / (df['+DI'] + df['-DI'] + 1e-8)).ewm(alpha=1/ADX_PERIOD, adjust=False).mean()

    df['+DI_SMA'] = df['+DI'].rolling(BB_PERIOD, min_periods=1).mean()
    df['+DI_STD'] = df['+DI'].rolling(BB_PERIOD, min_periods=1).std().fillna(0)
    df['+DI_BB_Upper'] = df['+DI_SMA'] + (BB_STD * df['+DI_STD'])

    df['-DI_SMA'] = df['-DI'].rolling(BB_PERIOD, min_periods=1).mean()
    df['-DI_STD'] = df['-DI'].rolling(BB_PERIOD, min_periods=1).std().fillna(0)
    df['-DI_BB_Upper'] = df['-DI_SMA'] + (BB_STD * df['-DI_STD'])

    latest = df.iloc[-1]
    ha_trend = latest['HA_Trend']

    bb_rsi = "Neutral"
    if latest['RSI_STD'] > 0:
        if latest['RSI'] > latest['BB_Upper']: bb_rsi = "Buy"
        elif latest['RSI'] < latest['BB_Lower']: bb_rsi = "Sell"

    bb_macd = "Neutral"
    if latest['MACD_Hist_STD'] > 0:
        if latest['MACD_Hist'] > latest['MACD_BB_Upper']: bb_macd = "Buy"
        elif latest['MACD_Hist'] < latest['MACD_BB_Lower']: bb_macd = "Sell"

    adx_sig = "Neutral"
    if latest['+DI_STD'] > 0 and latest['+DI'] > latest['+DI_BB_Upper'] and latest['+DI'] > latest['-DI'] and latest['ADX'] >= ADX_THRESHOLD:
        adx_sig = "Buy"
    elif latest['-DI_STD'] > 0 and latest['-DI'] > latest['-DI_BB_Upper'] and latest['-DI'] > latest['+DI'] and latest['ADX'] >= ADX_THRESHOLD:
        adx_sig = "Sell"

    is_bull_aligned = (bb_rsi == "Buy") and (bb_macd == "Buy") and (adx_sig == "Buy") and (ha_trend == 'Green')
    is_bear_aligned = (bb_rsi == "Sell") and (bb_macd == "Sell") and (adx_sig == "Sell") and (ha_trend == 'Red')

    if is_bull_aligned: return bb_rsi, bb_macd, adx_sig, "BULL"
    elif is_bear_aligned: return bb_rsi, bb_macd, adx_sig, "BEAR"
    else: return "", "", "", "NONE"

# ==============================================================================
# 3. PIPELINE EXECUTOR & UI DRAWING
# ==============================================================================
def format_cell(text, width=10):
    if not text: return " " * width
    spaces = width - len(str(text))
    left_pad, right_pad = " " * (spaces // 2), " " * (spaces - (spaces // 2))
    colored_text = f"{COLOR_GREEN_BG}{text}{COLOR_RESET}" if text == "Buy" else f"{COLOR_RED_BG}{text}{COLOR_RESET}" if text == "Sell" else str(text)
    return f"{left_pad}{colored_text}{right_pad}"

def _filter_worker(item, filter_date, latest_day):
    df = fetch_upstox_candles_for_date(item['key'], filter_date, is_latest_day=(filter_date == latest_day))
    if df is not None and not df.empty:
        # Options specific filtering
        if TRADING_MODE == "INDEX_OPTIONS":
            if df['Close'].iloc[-1] >= OPT_MIN_PRICE and df['Volume'].sum() >= OPT_MIN_VOLUME:
                return item, df
        # Equity specific filtering
        elif MIN_PRICE <= df['Close'].iloc[-1] <= MAX_PRICE and df['Volume'].sum() >= MIN_DAILY_VOLUME:
            return item, df
    return None

def process_stock(args):
    item, trading_days, cached = args  
    latest_day = trading_days[-1]
    days_needed = [d for d in trading_days if d not in cached]
    
    fetched = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=DAY_FETCH_WORKERS) as ex:
        futures = {ex.submit(fetch_upstox_candles_for_date, item['key'], day, is_latest_day=(day == latest_day)): day for day in days_needed}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res is not None and not res.empty: fetched.append(res)
    
    dfs = list(cached.values()) + fetched
    if not dfs: return None
    
    master_1m = pd.concat(dfs, ignore_index=True).drop_duplicates(subset='Datetime').sort_values('Datetime').reset_index(drop=True)
    if master_1m['Datetime'].dt.strftime("%Y-%m-%d").max() != latest_day or len(master_1m) < 30: return None

    base_atr = compute_base_atr(master_1m)
    row_data = {'Symbol': item['symbol'], 'LTP': master_1m['Close'].iloc[-1], 'Score': 0, 'PerfectBullBlocks': 0, 'PerfectBearBlocks': 0}

    for mult in HA_ATR_MULTIPLIERS:
        gtag = f"{mult}X"
        ha_atr_df = build_isolated_range_bars(master_1m, base_atr * mult)
        bb_rsi, bb_macd, adx_sig, alignment = calculate_strict_signals(ha_atr_df)
        
        row_data[f'BB_RSI_{gtag}'], row_data[f'BB_MACD_{gtag}'], row_data[f'ADX_{gtag}'] = bb_rsi, bb_macd, adx_sig

        if alignment == "BULL":
            row_data['PerfectBullBlocks'] += 1
            row_data['Score'] += 1 
        elif alignment == "BEAR":
            row_data['PerfectBearBlocks'] += 1
            row_data['Score'] -= 1

    return row_data

def run_screener():
    target_date_str = (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d")
    trading_days = get_past_trading_days(target_date_str, num_days=BACKTRACE_DAYS)
    latest_day = trading_days[-1]
    filter_date = trading_days[-2] if len(trading_days) > 1 else trading_days[0]

    t_start = time.time()
    print(f"\n{COLOR_CYAN}📡 Initializing Screener Pipeline [{TRADING_MODE}] via UPSTOX API...{COLOR_RESET}")
    
    universe_raw = get_dynamic_universe(TRADING_MODE, latest_day)

    if not universe_raw: 
        print(f"{COLOR_RED_FG}[!] Failed to generate the Universe list. Exiting.{COLOR_RESET}")
        return

    universe, filter_cache = [], {}
    print(f"🔄 Filtering {len(universe_raw)} Options Strikes for Minimum Premium (₹{OPT_MIN_PRICE}) & Liquidity...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=UNIVERSE_FILTER_WORKERS) as ex:
        for result in ex.map(lambda it: _filter_worker(it, filter_date, latest_day), universe_raw):
            if result is not None:
                item, df = result
                universe.append(item)
                filter_cache[item['symbol']] = {filter_date: df}

    print(f"✅ Target Universe ready ({len(universe)} highly liquid Option Strikes). Computing technicals...\n")

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

    print(f"{COLOR_BOLD}=== STRICT INSTITUTIONAL VOLATILITY DASHBOARD [{TRADING_MODE}] ==={COLOR_RESET}\n")

    def print_basket(title, icon, data_list):
        if not data_list: return
        print(f"\n{COLOR_BOLD}{icon} {title}{COLOR_RESET}")
        
        header_str = f" {COLOR_CYAN}{'Options Strike':<22} {'LTP':<8} |"
        for mult in HA_ATR_MULTIPLIERS:
            gtag = f"{mult}X"
            header_str += f"  {'BB-RSI '+gtag:^11} {'BB-MACD '+gtag:^12} {'BB-DI '+gtag:^9} |"
        
        dash_len = len(header_str) - 8 
        print(header_str + COLOR_RESET)
        print("-" * dash_len)

        for row in data_list:
            row_str = f" {row['Symbol']:<22} {row['LTP']:<8.2f} |"
            for mult in HA_ATR_MULTIPLIERS:
                gtag = f"{mult}X"
                bb_rsi_cell = format_cell(row[f'BB_RSI_{gtag}'], 11)
                bb_macd_cell = format_cell(row[f'BB_MACD_{gtag}'], 12)
                adx_cell = format_cell(row[f'ADX_{gtag}'], 9)
                row_str += f"  {bb_rsi_cell} {bb_macd_cell} {adx_cell} |"
            print(row_str)

    print_basket(f"TOP PREMIUM BUYERS (Pure Alignment >= {MIN_PERFECT_BLOCKS} Block)", "🔥", bulls)
    print_basket(f"TOP PREMIUM SELLERS (Pure Alignment >= {MIN_PERFECT_BLOCKS} Block)", "🩸", bears)
    
    print(f"\n⏱️ Scan completed in {(time.time() - t_start):.2f} seconds.\n")

if __name__ == "__main__":
    if not os.environ.get("UPSTOX_ACCESS_TOKEN"):
        print(f"{COLOR_RED_FG}[!] Missing UPSTOX_ACCESS_TOKEN environment variable.{COLOR_RESET}")
        sys.exit(1)
    run_screener()
