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

# ==============================================================================
# 0. ENGINE CONSTANTS & TERMINAL COLORS
# ==============================================================================
COLOR_GREEN = '\033[92m'
COLOR_RED = '\033[91m'
COLOR_CYAN = '\033[96m'
COLOR_YELLOW = '\033[93m'
COLOR_MAGENTA = '\033[95m'
COLOR_DIM = '\033[2m'
COLOR_RESET = '\033[0m'
COLOR_BOLD = '\033[1m'

BACKTRACE_DAYS = 20      
MAX_BREACH_DAYS = 0      

# Global flag to stop spamming the console with the same API error
API_ERROR_LOGGED = False

# ==============================================================================
# ★ GLOBAL CONFIGURATION: TIMEFRAMES & INDICATORS ★
# ==============================================================================
# 1. Macro Environment (Permission)
TIMEFRAME = '15min'              
TIMEFRAME_MINS = 15              
ATR_PERIOD = 14
RSI_PERIOD = 14
BB_SMA_PERIOD = 20
BB_STD_DEV = 2.0               
ADX_PERIOD = 14
ADX_THRESHOLD = 20             

# 2. 1-Minute Micro Execution (Trigger)
RENKO_CONFIRM_BRICKS = 20       # The Asit Baran Pati 2-Brick Confirmation Rule
RENKO_MIN_BRICK = 0.05         
RENKO_DEFAULT_PCT = 0.005      

# 3. Risk Management
BREACH_PURGE_PCT = 0.015       

# ==============================================================================
# 1. LIVE INGESTION (F&O Universe & Parallel Bulk Fetching)
# ==============================================================================
def get_dynamic_fno_universe():
    nse_url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
    try:
        response = requests.get(nse_url, timeout=15)
        if response.status_code != 200: return []
        nse_data = json.load(gzip.GzipFile(fileobj=io.BytesIO(response.content)))
        fno_underlying = {item.get("underlying_symbol") for item in nse_data if item.get("segment") == "NSE_FO" and item.get("underlying_symbol")}
        return [{"symbol": item.get("trading_symbol"), "key": item.get("instrument_key")} for item in nse_data if item.get("segment") in ("NSE_EQ", "NSE_INDEX") and item.get("trading_symbol") in fno_underlying]
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
    except:
        return []

# ==============================================================================
# 2. ZERO-LAG DECOUPLED PRE-COMPUTATION ENGINE
# ==============================================================================
def apply_1m_renko_and_signals(df_1m):
    """
    Executes the pure time-independent Renko math on the 1-minute data feed.
    """
    renko_trends = np.zeros(len(df_1m))
    renko_counts = np.zeros(len(df_1m))
    
    for sym, indices in df_1m.groupby('Symbol').indices.items():
        sub_closes = df_1m['Close'].values[indices]
        sub_atrs = df_1m['ATR'].values[indices]
        
        if len(sub_closes) > 0:
            trends = np.zeros(len(sub_closes))
            counts = np.zeros(len(sub_closes))
            curr_trend = 0
            curr_count = 0
            curr_price = sub_closes[0]
            
            for i in range(1, len(sub_closes)):
                bs = max(sub_atrs[i], RENKO_MIN_BRICK)
                move = sub_closes[i] - curr_price
                
                if move >= bs:
                    bricks = int(move // bs)
                    if curr_trend == 1:
                        curr_count += bricks
                    else:
                        curr_trend = 1
                        curr_count = bricks # Reset counter on trend flip
                    curr_price += bricks * bs
                elif move <= -bs:
                    bricks = int(abs(move) // bs)
                    if curr_trend == -1:
                        curr_count -= bricks
                    else:
                        curr_trend = -1
                        curr_count = -bricks
                    curr_price -= bricks * bs
                    
                trends[i] = curr_trend
                counts[i] = curr_count
                
            renko_trends[indices] = trends
            renko_counts[indices] = counts
            
    df_1m['Renko_Trend'] = renko_trends
    df_1m['Renko_Brick_Count'] = renko_counts
    
    # -------------------------------------------------------------
    # THE ZERO-LAG EXECUTION TRIGGER
    # -------------------------------------------------------------
    # Trade only executes if Macro Environment grants Permission AND 1-Min Renko hits Brick Threshold
    df_1m['Trigger_Bull'] = df_1m['Armed_Bull'] & (df_1m['Renko_Brick_Count'] >= RENKO_CONFIRM_BRICKS)
    df_1m['Trigger_Bear'] = df_1m['Armed_Bear'] & (df_1m['Renko_Brick_Count'] <= -RENKO_CONFIRM_BRICKS)
    
    # Detect the EXACT 1-minute boundary where the trigger fires
    df_1m['Trigger_Bull_Prev'] = df_1m.groupby('Symbol')['Trigger_Bull'].shift(1).fillna(False)
    df_1m['Trigger_Bear_Prev'] = df_1m.groupby('Symbol')['Trigger_Bear'].shift(1).fillna(False)
    
    df_1m['New_Bull'] = df_1m['Trigger_Bull'] & ~df_1m['Trigger_Bull_Prev']
    df_1m['New_Bear'] = df_1m['Trigger_Bear'] & ~df_1m['Trigger_Bear_Prev']
    
    df_1m['Direction'] = np.where(df_1m['New_Bull'], 1, np.where(df_1m['New_Bear'], -1, 0))
    return df_1m


def prepare_technical_data(rolling_master_df):
    """
    Separates the Macro Environment (15m) from the Execution Engine (1m).
    """
    # 1. Macro Environment Build (15-Minute Aggregation)
    tech_15m = rolling_master_df.groupby(['Symbol', pd.Grouper(key='Datetime', freq=TIMEFRAME, closed='left', label='left')]).agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).reset_index()
    
    tech_15m = tech_15m.dropna(subset=['Close']).sort_values(['Symbol', 'Datetime'])
    
    # ATR & Technicals
    tech_15m['H-L'] = tech_15m['High'] - tech_15m['Low']
    tech_15m['H-PC'] = (tech_15m['High'] - tech_15m.groupby('Symbol')['Close'].shift(1)).abs()
    tech_15m['L-PC'] = (tech_15m['Low'] - tech_15m.groupby('Symbol')['Close'].shift(1)).abs()
    tech_15m['TR'] = tech_15m[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    atr_series = tech_15m.groupby('Symbol')['TR'].transform(lambda x: x.ewm(alpha=1/ATR_PERIOD, adjust=False).mean())
    tech_15m['ATR'] = atr_series

    delta = tech_15m.groupby('Symbol')['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.groupby(tech_15m['Symbol']).transform(lambda x: x.ewm(alpha=1/RSI_PERIOD, adjust=False).mean())
    avg_loss = loss.groupby(tech_15m['Symbol']).transform(lambda x: x.ewm(alpha=1/RSI_PERIOD, adjust=False).mean())
    tech_15m['RSI'] = 100 - (100 / (1 + (avg_gain / (avg_loss + 1e-8))))
    
    tech_15m['RSI_SMA'] = tech_15m.groupby('Symbol')['RSI'].transform(lambda x: x.rolling(BB_SMA_PERIOD).mean())
    tech_15m['RSI_STD'] = tech_15m.groupby('Symbol')['RSI'].transform(lambda x: x.rolling(BB_SMA_PERIOD).std())
    tech_15m['BB_Upper'] = tech_15m['RSI_SMA'] + (BB_STD_DEV * tech_15m['RSI_STD'])
    tech_15m['BB_Lower'] = tech_15m['RSI_SMA'] - (BB_STD_DEV * tech_15m['RSI_STD'])
    
    high_diff = tech_15m['High'] - tech_15m.groupby('Symbol')['High'].shift(1)
    low_diff = tech_15m.groupby('Symbol')['Low'].shift(1) - tech_15m['Low']
    tech_15m['+DM'] = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    tech_15m['-DM'] = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    tech_15m['+DI'] = 100 * (tech_15m.groupby('Symbol')['+DM'].transform(lambda x: x.ewm(alpha=1/ADX_PERIOD, adjust=False).mean()) / (atr_series + 1e-8))
    tech_15m['-DI'] = 100 * (tech_15m.groupby('Symbol')['-DM'].transform(lambda x: x.ewm(alpha=1/ADX_PERIOD, adjust=False).mean()) / (atr_series + 1e-8))
    tech_15m['DX'] = 100 * abs(tech_15m['+DI'] - tech_15m['-DI']) / (tech_15m['+DI'] + tech_15m['-DI'] + 1e-8)
    tech_15m['ADX'] = tech_15m.groupby('Symbol')['DX'].transform(lambda x: x.ewm(alpha=1/ADX_PERIOD, adjust=False).mean())
    tech_15m['ADX_prev'] = tech_15m.groupby('Symbol')['ADX'].shift(1)
    
    # 2. Gatekeeper Logic (Granting Permission)
    tech_15m['Armed_Bull'] = (tech_15m['RSI'] > tech_15m['BB_Upper']) & (tech_15m['ADX'] > ADX_THRESHOLD) & (tech_15m['ADX'] > tech_15m['ADX_prev']) & (tech_15m['+DI'] > tech_15m['-DI'])
    tech_15m['Armed_Bear'] = (tech_15m['RSI'] < tech_15m['BB_Lower']) & (tech_15m['ADX'] > ADX_THRESHOLD) & (tech_15m['ADX'] > tech_15m['ADX_prev']) & (tech_15m['-DI'] > tech_15m['+DI'])
    
    # Forward-shift evaluation time by 15 mins to absolutely prevent lookahead bias
    tech_15m['Eval_Time'] = tech_15m['Datetime'] + pd.Timedelta(minutes=TIMEFRAME_MINS)
    
    # Extract only the permission data
    env_df = tech_15m[['Symbol', 'Eval_Time', 'ATR', 'Armed_Bull', 'Armed_Bear', 'ADX']].copy()
    env_df = env_df.sort_values('Eval_Time').rename(columns={'Eval_Time': 'Datetime'})
    
    # 3. Micro Execution Build (1-Minute Merging)
    df_1m = rolling_master_df.sort_values('Datetime').copy()
    
    # Maps the latest confirmed 15-minute permission state onto every 1-minute tick
    df_1m_merged = pd.merge_asof(df_1m, env_df, on='Datetime', by='Symbol', direction='backward')
    df_1m_merged = df_1m_merged.sort_values(['Symbol', 'Datetime']).reset_index(drop=True)
    
    df_1m_merged['ATR'] = df_1m_merged['ATR'].fillna(df_1m_merged['Close'] * RENKO_DEFAULT_PCT)
    df_1m_merged['Armed_Bull'] = df_1m_merged['Armed_Bull'].fillna(False)
    df_1m_merged['Armed_Bear'] = df_1m_merged['Armed_Bear'].fillna(False)
    
    # 4. Generate the Exact 1-Minute Zero Lag Signals
    final_1m_tape = apply_1m_renko_and_signals(df_1m_merged)
    return final_1m_tape

# ==============================================================================
# 4. LIGHTNING STATE-BASED MEMORY ENGINE
# ==============================================================================
def scan_institutional_tape(target_date_str):
    global API_ERROR_LOGGED
    
    print(f"\n📡 Initiating Zero-Lag Decoupled Engine for {target_date_str}...")
    universe = get_dynamic_fno_universe()
    if not universe:
        print(f"⚠️ {COLOR_RED}No F&O universe found.{COLOR_RESET}")
        return

    trading_days = get_past_trading_days(target_date_str, num_days=BACKTRACE_DAYS)
    if not trading_days: return

    target_dt = pd.to_datetime(target_date_str)
    current_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
    is_live_today = (target_date_str == current_now.strftime("%Y-%m-%d"))

    print(f"🚀 Multithreading Bulk Fetch for {len(universe)} symbols (20 days at once)...")
    fetch_tasks = [(item, trading_days[0], target_date_str, is_live_today) for item in universe]
    historical_dfs = []

    def fetch_worker(task):
        global API_ERROR_LOGGED
        item, start_date, end_date, live = task
        key = urllib.parse.quote(item['key'])
        access_token = os.environ.get("UPSTOX_ACCESS_TOKEN")
        headers = {'Accept': 'application/json', 'Authorization': f'Bearer {access_token}'}
        
        dfs = []
        hist_end = end_date if not live else (current_now - timedelta(days=1)).strftime("%Y-%m-%d")
        
        for attempt in range(3):
            try:
                url = f"https://api.upstox.com/v2/historical-candle/{key}/1minute/{hist_end}/{start_date}"
                res = requests.get(url, headers=headers, timeout=15)
                
                if res.status_code == 429: 
                    time.sleep(1.5)
                    continue
                elif res.status_code == 200:
                    data = res.json().get('data', {}).get('candles')
                    if data: dfs.append(pd.DataFrame(data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI']))
                    break
                else:
                    if not API_ERROR_LOGGED:
                        print(f"\n\n{COLOR_RED}❌ [UPSTOX API REJECTION] HTTP {res.status_code}{COLOR_RESET}")
                        print(f"{COLOR_YELLOW}Response Message: {res.text}{COLOR_RESET}\n")
                        API_ERROR_LOGGED = True
                    break
            except Exception:
                time.sleep(1)
            
        if live:
            for attempt in range(3):
                try:
                    res = requests.get(f"https://api.upstox.com/v2/historical-candle/intraday/{key}/1minute", headers=headers, timeout=15)
                    if res.status_code == 429: time.sleep(1.5); continue
                    if res.status_code == 200 and res.json().get('data', {}).get('candles'):
                        dfs.append(pd.DataFrame(res.json()['data']['candles'], columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI']))
                    break
                except Exception:
                    time.sleep(1)

        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            df['Datetime'] = pd.to_datetime(df['Timestamp']).dt.tz_localize(None)
            df = df.drop_duplicates(subset=['Datetime']).sort_values('Datetime').reset_index(drop=True)
            df['Symbol'] = item['symbol']
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

    if not historical_dfs:
        print(f"⚠️ {COLOR_RED}Fatal Error: No data retrieved. Check the API Rejection message above.{COLOR_RESET}")
        return

    rolling_master_df = pd.concat(historical_dfs, ignore_index=True)
    
    print(f"⚙️ Computing Macro Environment & 1-Minute Micro Triggers...")
    tape_1m = prepare_technical_data(rolling_master_df)
    
    all_anomalies = tape_1m[tape_1m['Direction'] != 0].copy()
    anomalies_by_time = all_anomalies.groupby('Datetime')
    
    closes_dict = tape_1m.set_index(['Datetime', 'Symbol'])['Close'].to_dict()
    all_times = np.sort(tape_1m['Datetime'].unique())

    memory_bank = {} 
    historical_times = [t for t in all_times if pd.to_datetime(t).date() < target_dt.date()]

    # FAST HISTORICAL STATE BUILD (1-Minute Precision)
    for t in historical_times:
        t_dt = pd.to_datetime(t)
        
        # 09:15 Gap Checks
        if t_dt.hour == 9 and t_dt.minute == 15:
            day_str = t_dt.strftime('%Y-%m-%d')
            for sym, st in memory_bank.items():
                ltp = closes_dict.get((t_dt, sym))
                if ltp and st['state'] == 'ACTIVE':
                    if (st['dir'] == 1 and ltp < st['origin']) or (st['dir'] == -1 and ltp > st['origin']):
                        st['state'], st['breach_time'] = 'BREACHED', f"{day_str} 09:15 (GAP)"

        # 1-Minute Continuous Breach Detection
        for sym, st in memory_bank.items():
            ltp = closes_dict.get((t_dt, sym))
            if ltp:
                if st['state'] == 'ACTIVE' and ((st['dir'] == 1 and ltp < st['origin']) or (st['dir'] == -1 and ltp > st['origin'])):
                    st['state'], st['breach_time'] = 'BREACHED', t_dt.strftime('%Y-%m-%d %H:%M')
                elif st['state'] == 'BREACHED' and ((st['dir'] == 1 and ltp >= st['origin']) or (st['dir'] == -1 and ltp <= st['origin'])):
                    st['state'], st['breach_time'] = 'ACTIVE', None

        # Lock in exact 1-minute Triggers
        if t_dt in anomalies_by_time.groups:
            for _, row in anomalies_by_time.get_group(t_dt).iterrows():
                sym = row['Symbol']
                if sym not in memory_bank:
                    memory_bank[sym] = {'state': 'ACTIVE', 'origin': row['Close'], 'date': t_dt.strftime('%Y-%m-%d'), 'time': t_dt.strftime('%H:%M'), 'dir': row['Direction'], 'breach_time': None}
        
        # 15:15 EOD Purge check
        if t_dt.hour == 15 and t_dt.minute == 15:
            for sym in list(memory_bank.keys()):
                ltp = closes_dict.get((t_dt, sym))
                if ltp and memory_bank[sym]['state'] == 'BREACHED':
                    if (memory_bank[sym]['dir'] == 1 and ltp < memory_bank[sym]['origin'] * (1.0 - BREACH_PURGE_PCT)) or \
                       (memory_bank[sym]['dir'] == -1 and ltp > memory_bank[sym]['origin'] * (1.0 + BREACH_PURGE_PCT)):
                        del memory_bank[sym]

    # ----------------------------------------------------------------------
    # LIVE TARGET EVALUATION (1-Minute Precision)
    # ----------------------------------------------------------------------
    today_times = [t for t in all_times if pd.to_datetime(t).date() == target_dt.date()]
    if not today_times: 
        print(f"\n{COLOR_YELLOW}[Terminal Standby] Market data for {target_date_str} is empty or not available yet.{COLOR_RESET}\n")
        return

    today_master = tape_1m[tape_1m['Datetime'].dt.date == target_dt.date()]
    morning_opens = today_master[today_master['Datetime'].dt.time == pd.to_datetime('09:15').time()].set_index('Symbol')['Open'].to_dict()

    all_fresh_intrusions, all_reloads, all_reclaims = {}, {}, {}

    for t in today_times:
        t_dt = pd.to_datetime(t)
        
        # Real-time intraday breaches
        for sym, st in memory_bank.items():
            ltp = closes_dict.get((t_dt, sym))
            if ltp:
                if st['state'] == 'ACTIVE' and ((st['dir'] == 1 and ltp < st['origin']) or (st['dir'] == -1 and ltp > st['origin'])):
                    st['state'], st['breach_time'] = 'BREACHED', t_dt.strftime('%Y-%m-%d %H:%M')
                elif st['state'] == 'BREACHED' and ((st['dir'] == 1 and ltp >= st['origin']) or (st['dir'] == -1 and ltp <= st['origin'])):
                    st['state'], st['breach_time'] = 'ACTIVE', None

        # Real-time zero-lag triggers
        if t_dt in anomalies_by_time.groups:
            for _, row in anomalies_by_time.get_group(t_dt).iterrows():
                sym, price, direction = row['Symbol'], row['Close'], row['Direction']
                if sym not in memory_bank:
                    if sym not in all_fresh_intrusions:
                        row['Eval_Time_Str'] = t_dt.strftime('%H:%M')
                        all_fresh_intrusions[sym] = row
                        memory_bank[sym] = {'state': 'ACTIVE', 'origin': price, 'date': target_date_str, 'time': row['Eval_Time_Str'], 'dir': direction, 'breach_time': None}
                else:
                    st = memory_bank[sym]
                    row['Net_Drift'] = ((price - st['origin']) / st['origin'] * 100) if st['dir'] == 1 else ((st['origin'] - price) / st['origin'] * 100)
                    
                    if st['state'] == 'ACTIVE' and direction == st['dir']:
                        row['Eval_Time_Str'] = t_dt.strftime('%H:%M')
                        row['Macro_Price'], row['Macro_Date'], row['Micro_Price'] = st['origin'], st['date'], price
                        all_reloads[sym] = row
                    elif st['state'] == 'BREACHED' and direction == st['dir']:
                        st['state'], st['breach_time'] = 'ACTIVE', None
                        row['Eval_Time_Str'] = t_dt.strftime('%H:%M')
                        row['Origin'], row['First_Date'] = st['origin'], st['date']
                        all_reclaims[sym] = row

    final_ltp_dict = today_master.groupby('Symbol')['Close'].last().to_dict()
    breached = []

    for sym, st in memory_bank.items():
        if st['state'] == 'BREACHED' and sym in final_ltp_dict and sym not in all_reclaims:
            breached.append({'Symbol': sym, 'LTP': final_ltp_dict[sym], 'Origin': st['origin'], 'Dir': "BULLISH" if st['dir'] == 1 else "BEARISH", 'Time': st['breach_time'], 'First_Date': st['date'], 'Anchor_Time': st.get('time', '09:15')})

    # ----------------------------------------------------------------------
    # TERMINAL OUTPUT
    # ----------------------------------------------------------------------
    print(f"\n{COLOR_CYAN}================================================================================================{COLOR_RESET}")
    print(f"{COLOR_BOLD}FULL UNIVERSE TECHNICAL CONFLUENCE TAPE | DATE: {target_date_str}{COLOR_RESET}")
    print(f"{COLOR_CYAN}================================================================================================{COLOR_RESET}\n")

    if all_fresh_intrusions:
        print(f"{COLOR_BOLD}⚡ BASKET 1: FRESH INTRUSIONS (Phase 1 - Day-1 Births){COLOR_RESET}")
        for sym, row in all_fresh_intrusions.items():
            ltp = row['Close']
            pct_move = ((ltp - morning_opens.get(sym, ltp)) / morning_opens.get(sym, ltp)) * 100
            color, d_str = (COLOR_GREEN, "BULLISH") if row['Direction'] == 1 else (COLOR_RED, "BEARISH")
            print(f"  {color}🚨 {sym:<12} Day Move: {pct_move:+.2f}% ({d_str}){COLOR_RESET}")
            print(f"      └─ 🎯 Macro Permission Passed : 15m BB-RSI | ADX:{row.get('ADX', 0):.1f}")
            print(f"      └─ 🔫 1-Min Execution Trigger : Renko >= {RENKO_CONFIRM_BRICKS} Bricks")
            print(f"      └─ ⚓ Zero-Lag Anchor       : {target_date_str} @ {row['Eval_Time_Str']} | Price: ₹{ltp:.2f}")
            print(f"      └─ 🎯 Latest LTP          : {target_date_str} @ EOD   | Price: ₹{final_ltp_dict.get(sym, ltp):.2f}\n")

    if all_reloads:
        print(f"{COLOR_BOLD}🔄 BASKET 2: ALGORITHMIC RELOADS (Phase 2 - Institutional Continuations){COLOR_RESET}")
        for sym, row in all_reloads.items():
            ltp = row['Close']
            pct_move = ((ltp - morning_opens.get(sym, ltp)) / morning_opens.get(sym, ltp)) * 100
            color, d_str = (COLOR_GREEN, "BULLISH") if row['Direction'] == 1 else (COLOR_RED, "BEARISH")
            print(f"  {color}🔄 {sym:<12} Day Move: {pct_move:+.2f}% ({d_str}){COLOR_RESET}")
            print(f"      └─ 🎯 Macro Permission Passed : 15m BB-RSI | ADX:{row.get('ADX', 0):.1f}")
            print(f"      └─ 🔫 1-Min Execution Trigger : Renko >= {RENKO_CONFIRM_BRICKS} Bricks")
            print(f"      └─ ⚓ Macro Floor (Origin): {row['Macro_Date']} @ {memory_bank[sym].get('time', '09:15')} | Price: ₹{row['Macro_Price']:.2f}")
            print(f"      └─ ⚡ Micro Floor (Reload): {target_date_str} @ {row['Eval_Time_Str']} | Price: ₹{row['Micro_Price']:.2f}")
            print(f"      └─ 🎯 Latest LTP          : {target_date_str} @ EOD   | Price: ₹{final_ltp_dict.get(sym, ltp):.2f} (Trend Drift: {row['Net_Drift']:+.2f}%)\n")

    if breached:
        print(f"{COLOR_DIM}⚠️ BASKET 3: BREACHED PIVOTS (Phase 3 - Trapped Capital / Dead Trends){COLOR_RESET}")
        for b in breached:
            print(f"  {COLOR_YELLOW}⚠️ {b['Symbol']:<12} {b['Dir']} Anchor shattered!{COLOR_RESET}")
            print(f"      └─ ⚓ Anchor : {b['First_Date']} @ {b['Anchor_Time']} | LTP: ₹{b['Origin']:.2f}")
            print(f"      └─ 🎯 Latest : Breached At {b.get('Time', 'Pending')} | Current LTP: ₹{b['LTP']:.2f}\n")

    if all_reclaims:
        print(f"{COLOR_BOLD}🪤 BASKET 4: INSTITUTIONAL RECLAIMS (Phase 4 - Liquidity Traps){COLOR_RESET}")
        for sym, row in all_reclaims.items():
            ltp = row['Close']
            pct_move = ((ltp - morning_opens.get(sym, ltp)) / morning_opens.get(sym, ltp)) * 100
            d_str = "BULLISH" if row['Direction'] == 1 else "BEARISH"
            print(f"  {COLOR_MAGENTA}🔥 {sym:<12} Day Move: {pct_move:+.2f}% ({d_str}){COLOR_RESET}")
            print(f"      └─ 🎯 Macro Permission Passed : 15m BB-RSI | ADX:{row.get('ADX', 0):.1f}")
            print(f"      └─ 🔫 1-Min Execution Trigger : Renko >= {RENKO_CONFIRM_BRICKS} Bricks")
            print(f"      └─ ⚓ Anchor : {row['First_Date']} @ {memory_bank[sym].get('time', '09:15')} | LTP: ₹{row['Origin']:.2f}")
            print(f"      └─ 🎯 Latest : Reclaimed At {target_date_str} @ {row['Eval_Time_Str']} | LTP: ₹{ltp:.2f}\n")

    if not any([all_fresh_intrusions, all_reloads, all_reclaims, breached]):
        print(f"{COLOR_DIM}[Terminal Silent] No active institutional structure passing strict filters.{COLOR_RESET}\n")

# ==============================================================================
# 5. RUN EXECUTOR
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
    import warnings
    warnings.filterwarnings("ignore")
    run_production_sweep()
