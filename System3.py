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

BACKTRACE_DAYS = 20      # 1 F&O Monthly Derivative Cycle
MAX_BREACH_DAYS = 0      # Kill Switch: Days a stock can stay breached before memory purge

# ==============================================================================
# 1. LIVE INGESTION (F&O Universe with Error Shield)
# ==============================================================================
def get_dynamic_fno_universe():
    nse_url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
    try:
        response = requests.get(nse_url, timeout=5)
        if response.status_code != 200:
            return []
        nse_data = json.load(gzip.GzipFile(fileobj=io.BytesIO(response.content)))
        fno_underlying = {item.get("underlying_symbol") for item in nse_data if item.get("segment") == "NSE_FO" and item.get("underlying_symbol")}
        return [{"symbol": item.get("trading_symbol"), "key": item.get("instrument_key")} for item in nse_data if item.get("segment") in ("NSE_EQ", "NSE_INDEX") and item.get("trading_symbol") in fno_underlying]
    except Exception as e:
        print(f"{COLOR_RED}[API Error] Failed to fetch F&O universe: {e}{COLOR_RESET}")
        return []

def fetch_upstox_candles_for_date(instrument_key, date_str):
    access_token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not access_token:
        return None

    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {access_token}'}
    today_str = (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d")

    if date_str == today_str:
        url = f"https://api.upstox.com/v2/historical-candle/intraday/{urllib.parse.quote(instrument_key)}/1minute"
    else:
        url = f"https://api.upstox.com/v2/historical-candle/{urllib.parse.quote(instrument_key)}/1minute/{date_str}/{date_str}"

    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code != 200:
            return None
        data = response.json().get('data', {}).get('candles', [])
        if not data:
            return None
        c_df = pd.DataFrame(data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
        c_df['Datetime'] = pd.to_datetime(c_df['Timestamp']).dt.tz_localize(None) 
        c_df = c_df.sort_values('Datetime').reset_index(drop=True)
        return c_df
    except:
        return None

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
    except Exception as e:
        print(f"{COLOR_RED}[Date Error] {e}{COLOR_RESET}")
        return []

# ==============================================================================
# 2. TECHNICAL PRE-COMPUTATION ENGINE (BB-RSI, ADXBO, Renko)
# ==============================================================================
def prepare_technical_data(rolling_master_df):
    print(f"⚙️ Computing BB-RSI, ADXBO, and Renko Matrices on aggregated blocks...")
    df = rolling_master_df.copy()
    
    # 1. Group to 15m Blocks
    hist_15m = df.groupby(['Symbol', pd.Grouper(key='Datetime', freq='15min', closed='left', label='left')]).agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum', 'Turnover': 'sum', 'abs_move': 'sum'
    }).reset_index()
    
    hist_15m = hist_15m.dropna(subset=['Close']).sort_values(['Symbol', 'Datetime'])
    
    # 2. True Range & ATR
    hist_15m['H-L'] = hist_15m['High'] - hist_15m['Low']
    hist_15m['H-PC'] = (hist_15m['High'] - hist_15m.groupby('Symbol')['Close'].shift(1)).abs()
    hist_15m['L-PC'] = (hist_15m['Low'] - hist_15m.groupby('Symbol')['Close'].shift(1)).abs()
    hist_15m['TR'] = hist_15m[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    atr14 = hist_15m.groupby('Symbol')['TR'].transform(lambda x: x.ewm(alpha=1/14, adjust=False).mean())
    hist_15m['ATR'] = atr14
    
    # 3. RSI & BB-RSI
    delta = hist_15m.groupby('Symbol')['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.groupby(hist_15m['Symbol']).transform(lambda x: x.ewm(alpha=1/14, adjust=False).mean())
    avg_loss = loss.groupby(hist_15m['Symbol']).transform(lambda x: x.ewm(alpha=1/14, adjust=False).mean())
    rs = avg_gain / (avg_loss + 1e-8)
    hist_15m['RSI'] = 100 - (100 / (1 + rs))
    
    hist_15m['RSI_SMA'] = hist_15m.groupby('Symbol')['RSI'].transform(lambda x: x.rolling(20).mean())
    hist_15m['RSI_STD'] = hist_15m.groupby('Symbol')['RSI'].transform(lambda x: x.rolling(20).std())
    hist_15m['BB_Upper'] = hist_15m['RSI_SMA'] + (2 * hist_15m['RSI_STD'])
    hist_15m['BB_Lower'] = hist_15m['RSI_SMA'] - (2 * hist_15m['RSI_STD'])
    
    # 4. ADX, +DI, -DI (ADXBO Trigger)
    high_diff = hist_15m['High'] - hist_15m.groupby('Symbol')['High'].shift(1)
    low_diff = hist_15m.groupby('Symbol')['Low'].shift(1) - hist_15m['Low']
    
    hist_15m['+DM'] = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    hist_15m['-DM'] = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    hist_15m['+DI'] = 100 * (hist_15m.groupby('Symbol')['+DM'].transform(lambda x: x.ewm(alpha=1/14, adjust=False).mean()) / (atr14 + 1e-8))
    hist_15m['-DI'] = 100 * (hist_15m.groupby('Symbol')['-DM'].transform(lambda x: x.ewm(alpha=1/14, adjust=False).mean()) / (atr14 + 1e-8))
    
    dx = 100 * abs(hist_15m['+DI'] - hist_15m['-DI']) / (hist_15m['+DI'] + hist_15m['-DI'] + 1e-8)
    hist_15m['DX'] = dx
    hist_15m['ADX'] = hist_15m.groupby('Symbol')['DX'].transform(lambda x: x.ewm(alpha=1/14, adjust=False).mean())
    hist_15m['ADX_prev'] = hist_15m.groupby('Symbol')['ADX'].shift(1)
    
    # 5. ATR-Synthesized Renko Trend Filter
    renko_trends = np.ones(len(hist_15m))
    for sym, indices in hist_15m.groupby('Symbol').indices.items():
        sub_closes = hist_15m['Close'].values[indices]
        sub_atrs = hist_15m['ATR'].fillna(hist_15m['Close']*0.005).values[indices]
        if len(sub_closes) > 0:
            trends = np.ones(len(sub_closes))
            curr_trend = 1
            curr_price = sub_closes[0]
            for i in range(1, len(sub_closes)):
                bs = max(sub_atrs[i], 0.05)  
                move = sub_closes[i] - curr_price
                if move >= bs:
                    curr_trend = 1
                    curr_price += int(move // bs) * bs
                elif move <= -bs:
                    curr_trend = -1
                    curr_price -= int(abs(move) // bs) * bs
                trends[i] = curr_trend
            renko_trends[indices] = trends
            
    hist_15m['Renko_Trend'] = renko_trends
    return hist_15m

# ==============================================================================
# 3. PURE TECHNICAL CONFLUENCE GATEKEEPER
# ==============================================================================
def evaluate_technical_confluence(master_df, current_eval_time, hist_15m_tech=None, window_mins=15):
    try:
        if master_df is None or master_df.empty or 'Datetime' not in master_df.columns:
            return pd.DataFrame()

        recent_start = current_eval_time - pd.Timedelta(minutes=window_mins)
        rec_df = master_df[(master_df['Datetime'] > recent_start) & (master_df['Datetime'] <= current_eval_time)]

        if rec_df.empty:
            return pd.DataFrame()

        g_rec = rec_df.groupby('Symbol').agg({'Open': 'first', 'Close': 'last'}).reset_index()
        g_rec['Rec_Pct_Move'] = ((g_rec['Close'] - g_rec['Open']) / (g_rec['Open'] + 1e-8)) * 100

        if hist_15m_tech is not None and not hist_15m_tech.empty:
            tech_slice = hist_15m_tech[hist_15m_tech['Datetime'] <= current_eval_time].groupby('Symbol').last().reset_index()
            
            if not tech_slice.empty:
                merged = pd.merge(g_rec, tech_slice[['Symbol', 'RSI', 'BB_Upper', 'BB_Lower', 'ADX', 'ADX_prev', '+DI', '-DI', 'Renko_Trend']], on='Symbol', how='inner')
                
                bull_cond = (merged['RSI'] > merged['BB_Upper']) & \
                            (merged['ADX'] > 20) & \
                            (merged['ADX'] > merged['ADX_prev']) & \
                            (merged['+DI'] > merged['-DI']) & \
                            (merged['Renko_Trend'] == 1)
                            
                bear_cond = (merged['RSI'] < merged['BB_Lower']) & \
                            (merged['ADX'] > 20) & \
                            (merged['ADX'] > merged['ADX_prev']) & \
                            (merged['-DI'] > merged['+DI']) & \
                            (merged['Renko_Trend'] == -1)
                            
                merged = merged[bull_cond | bear_cond].copy()
                
                if merged.empty: 
                    return pd.DataFrame()

                merged['Direction'] = np.where(bull_cond, 1, -1)
                merged = merged.sort_values(by='Rec_Pct_Move', key=abs, ascending=False)
                return merged
                
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

# ==============================================================================
# 4. STATE-BASED MEMORY ENGINE
# ==============================================================================
def scan_institutional_tape(target_date_str):
    print(f"\n📡 Initiating State-Based Confluence Engine for {target_date_str}...")
    universe = get_dynamic_fno_universe()
    if not universe:
        print(f"⚠️ {COLOR_RED}No F&O universe found or API connection failed.{COLOR_RESET}")
        return

    trading_days = get_past_trading_days(target_date_str, num_days=BACKTRACE_DAYS)
    if not trading_days:
        print(f"⚠️ {COLOR_RED}Failed to generate trading days sequence.{COLOR_RESET}")
        return

    print(f"🔄 Backtracing structural memory across {len(trading_days)} trading days using Multithreading...")

    # MULTITHREADED API FETCHING
    fetch_tasks = [(item, day) for day in trading_days for item in universe]
    historical_dfs = []

    def fetch_worker(task):
        item, day = task
        df = fetch_upstox_candles_for_date(item['key'], day)
        if df is not None and not df.empty:
            df['Symbol'] = item['symbol']
            df['Turnover'] = df['Volume'] * df['Close']
            df['abs_move'] = (df['Close'] - df['Open']).abs()
            return df
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(fetch_worker, fetch_tasks))
        for res in results:
            if res is not None:
                historical_dfs.append(res)

    if not historical_dfs:
        print(f"⚠️ {COLOR_RED}Fatal Error: No valid market data fetched across the window.{COLOR_RESET}")
        return

    rolling_master_df = pd.concat(historical_dfs, ignore_index=True)
    
    # ✅ Initialize Technical Matrix Data
    hist_15m_tech = prepare_technical_data(rolling_master_df)

    current_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
    is_live_today = (target_date_str == current_now.strftime("%Y-%m-%d"))

    target_dt = pd.to_datetime(target_date_str)
    if not is_live_today or current_now.hour >= 16:
        eval_times = [
            target_dt + pd.Timedelta(hours=9, minutes=45),
            target_dt + pd.Timedelta(hours=10, minutes=30),
            target_dt + pd.Timedelta(hours=11, minutes=30),
            target_dt + pd.Timedelta(hours=12, minutes=30),
            target_dt + pd.Timedelta(hours=13, minutes=30),
            target_dt + pd.Timedelta(hours=14, minutes=30),
            target_dt + pd.Timedelta(hours=15, minutes=15)
        ]
    else:
        eval_times = [current_now.replace(second=0, microsecond=0) - timedelta(minutes=1)]

    memory_bank = {} 

    for day in trading_days:
        day_dt = pd.to_datetime(day)
        day_master = rolling_master_df[(rolling_master_df['Datetime'] >= day_dt) & (rolling_master_df['Datetime'] < day_dt + pd.Timedelta(days=1))]
        if day_master.empty:
            continue

        try:
            morning_open = day_master.groupby('Symbol').first().reset_index()
            m_dict = morning_open.set_index('Symbol')['Open'].to_dict()

            for sym, st in memory_bank.items():
                if sym in m_dict:
                    op = m_dict[sym]
                    if st['state'] == 'ACTIVE':
                        if (st['dir'] == 1 and op < st['origin']) or (st['dir'] == -1 and op > st['origin']):
                            st['state'] = 'BREACHED'
                            st['breach_time'] = f"{day} 09:15 (GAP)"
                            st['breach_days'] = 0  
        except:
            pass

        day_start = day_dt + pd.Timedelta(hours=9, minutes=15)
        day_end = day_dt + pd.Timedelta(hours=15, minutes=15) if day != target_date_str else eval_times[-1]

        try:
            time_steps = pd.date_range(start=day_start + pd.Timedelta(minutes=15), end=day_end, freq='15min')
            for t in time_steps:
                t_candles = day_master[day_master['Datetime'] == t].set_index('Symbol')['Close'].to_dict()
                for sym, st in memory_bank.items():
                    if sym in t_candles:
                        ltp = t_candles[sym]
                        if st['state'] == 'ACTIVE':
                            if (st['dir'] == 1 and ltp < st['origin']) or (st['dir'] == -1 and ltp > st['origin']):
                                st['state'] = 'BREACHED'
                                st['breach_time'] = t.strftime('%Y-%m-%d %H:%M')
                                st['breach_days'] = 0
                        elif st['state'] == 'BREACHED':
                            if (st['dir'] == 1 and ltp >= st['origin']) or (st['dir'] == -1 and ltp <= st['origin']):
                                st['state'] = 'ACTIVE'
                                st['breach_time'] = None
                                st['breach_days'] = 0

                anomalies = evaluate_technical_confluence(day_master, t, hist_15m_tech=hist_15m_tech, window_mins=15)
                if not anomalies.empty:
                    for _, row in anomalies.iterrows():
                        sym = row['Symbol']
                        price = row['Close']
                        direction = row['Direction']
                        if sym not in memory_bank:
                            memory_bank[sym] = {'state': 'ACTIVE', 'origin': price, 'date': day, 'time': t.strftime('%H:%M'), 'dir': direction, 'breach_time': None, 'breach_days': 0}
        except:
            pass

        try:
            daily_agg = day_master.groupby('Symbol').agg({'Close': 'last'}).reset_index()
            daily_dict = daily_agg.set_index('Symbol').to_dict('index')

            to_delete = []
            for sym, st in memory_bank.items():
                if sym not in daily_dict: continue
                d_close = daily_dict[sym]['Close']

                if st['state'] == 'BREACHED':
                    if st['dir'] == 1 and d_close < (st['origin'] * 0.985): 
                        to_delete.append(sym)
                        continue
                    elif st['dir'] == -1 and d_close > (st['origin'] * 1.015): 
                        to_delete.append(sym)
                        continue

                    st['breach_days'] += 1
                    if st['breach_days'] >= MAX_BREACH_DAYS:
                        to_delete.append(sym)

            for sym in to_delete: 
                del memory_bank[sym]
        except:
            pass

    # ----------------------------------------------------------------------
    # LIVE EVALUATION (Full-Day Sweep Loop)
    # ----------------------------------------------------------------------
    today_master = rolling_master_df[
        (rolling_master_df['Datetime'] >= target_dt) & 
        (rolling_master_df['Datetime'] <= target_dt + pd.Timedelta(days=1))
    ].copy()

    if today_master.empty: 
        print(f"\n{COLOR_YELLOW}[Terminal Standby] Market data for {target_date_str} is empty or not available yet.{COLOR_RESET}\n")
        return

    all_fresh_intrusions = {}
    all_reloads = {}
    all_reclaims = {}

    for eval_time_current in eval_times:
        current_slice = today_master[today_master['Datetime'] <= eval_time_current]
        if current_slice.empty: continue

        try:
            today_latest_ltp = current_slice.groupby('Symbol')['Close'].last().to_dict()
            for sym, st in memory_bank.items():
                if sym in today_latest_ltp:
                    ltp = today_latest_ltp[sym]
                    if st['state'] == 'ACTIVE':
                        if (st['dir'] == 1 and ltp < st['origin']) or (st['dir'] == -1 and ltp > st['origin']):
                            st['state'] = 'BREACHED'
                            st['breach_time'] = eval_time_current.strftime('%Y-%m-%d %H:%M')
                    elif st['state'] == 'BREACHED':
                        if (st['dir'] == 1 and ltp >= st['origin']) or (st['dir'] == -1 and ltp <= st['origin']):
                            st['state'] = 'ACTIVE'
                            st['breach_time'] = None

            curr_anomalies = evaluate_technical_confluence(current_slice, eval_time_current, hist_15m_tech=hist_15m_tech, window_mins=15)

            if not curr_anomalies.empty:
                for _, row in curr_anomalies.iterrows():
                    sym = row['Symbol']
                    price = row['Close']
                    direction = row['Direction']

                    if sym not in memory_bank:
                        if sym not in all_fresh_intrusions:
                            row['Eval_Time'] = eval_time_current.strftime('%H:%M')
                            launchpad_price = price
                            try:
                                launch_slice = rolling_master_df[
                                    (rolling_master_df['Symbol'] == sym) & 
                                    (rolling_master_df['Datetime'] < eval_time_current) & 
                                    (rolling_master_df['Datetime'] >= eval_time_current - pd.Timedelta(days=5))
                                ]
                                if not launch_slice.empty:
                                    if direction == 1: launchpad_price = launch_slice['Low'].min()
                                    else: launchpad_price = launch_slice['High'].max()
                            except: pass
                            row['Launchpad'] = launchpad_price
                            all_fresh_intrusions[sym] = row
                    else:
                        st = memory_bank[sym]
                        if st['dir'] == 1:
                            row['Net_Drift'] = ((price - st['origin']) / st['origin']) * 100
                        else:
                            row['Net_Drift'] = ((st['origin'] - price) / st['origin']) * 100

                        if st['state'] == 'ACTIVE' and row['Direction'] == st['dir']:
                            if (st['dir'] == 1 and price >= st['origin']) or (st['dir'] == -1 and price <= st['origin']):
                                row['Eval_Time'] = eval_time_current.strftime('%H:%M')
                                row['Macro_Price'] = st['origin']
                                row['Macro_Date'] = st['date']
                                row['Micro_Price'] = price
                                all_reloads[sym] = row
                            else:
                                st['state'] = 'BREACHED'
                                st['breach_time'] = eval_time_current.strftime('%Y-%m-%d %H:%M')

                        elif st['state'] == 'BREACHED' and row['Direction'] == st['dir']:
                            if (st['dir'] == 1 and price > st['origin']) or (st['dir'] == -1 and price < st['origin']):
                                st['state'] = 'ACTIVE' 
                                st['breach_time'] = None
                                row['Eval_Time'] = eval_time_current.strftime('%H:%M')
                                row['Origin'] = st['origin']
                                row['First_Date'] = st['date']
                                all_reclaims[sym] = row
        except:
            continue

    final_ltp_dict = today_master.groupby('Symbol')['Close'].last().to_dict()
    valid_fresh = {}
    
    for sym, row in all_fresh_intrusions.items():
        ltp = final_ltp_dict.get(sym, row['Close'])
        direction = row['Direction']
        birth_price = row['Close']

        if (direction == 1 and ltp < birth_price) or (direction == -1 and ltp > birth_price):
            memory_bank[sym] = {
                'state': 'BREACHED', 'origin': birth_price, 'date': target_date_str, 
                'time': row.get('Eval_Time', '15:15'), 'dir': direction, 
                'breach_time': f"{target_date_str} EOD Violation", 'breach_days': 0
            }
        else:
            valid_fresh[sym] = row

    breached = []
    for sym, st in memory_bank.items():
        if st['state'] == 'BREACHED' and sym in final_ltp_dict:
            if sym not in all_reclaims: 
                breached.append({
                    'Symbol': sym, 'LTP': final_ltp_dict[sym], 'Origin': st['origin'], 
                    'Dir': "BULLISH" if st['dir'] == 1 else "BEARISH",
                    'Time': st['breach_time'], 'First_Date': st['date'],
                    'Anchor_Time': st.get('time', '09:15')
                })

    # ----------------------------------------------------------------------
    # TERMINAL OUTPUT (4D Temporal UI Matrix with Tech Filtering)
    # ----------------------------------------------------------------------
    print(f"\n{COLOR_CYAN}================================================================================================{COLOR_RESET}")
    print(f"{COLOR_BOLD}FULL UNIVERSE TECHNICAL CONFLUENCE TAPE | DATE: {target_date_str}{COLOR_RESET}")
    print(f"{COLOR_CYAN}================================================================================================{COLOR_RESET}\n")

    if valid_fresh:
        print(f"{COLOR_BOLD}⚡ BASKET 1: FRESH INTRUSIONS (Phase 1 - Day-1 Births){COLOR_RESET}")
        for sym, row in valid_fresh.items():
            pct_move, ltp = row['Rec_Pct_Move'], row['Close']
            color = COLOR_GREEN if pct_move > 0 else COLOR_RED
            d_str = "BULLISH" if pct_move > 0 else "BEARISH"
            eval_t = row.get('Eval_Time', '15:15')
            launchpad = row.get('Launchpad', ltp)

            print(f"  {color}🚨 {sym:<12} Block Move: {pct_move:+.2f}% ({d_str}){COLOR_RESET}")
            print(f"      └─ 🎯 Tech Filters Passed : BB-RSI Breakout | ADX:{row.get('ADX', 0):.1f} | Renko: {d_str}")
            print(f"      └─ 🧱 Launchpad (Base)    : Price: ₹{launchpad:.2f}")
            print(f"      └─ ⚓ Breakout Anchor     : {target_date_str} @ {eval_t} | Price: ₹{ltp:.2f}")
            print(f"      └─ 🎯 Latest LTP          : {target_date_str} @ EOD    | Price: ₹{final_ltp_dict.get(sym, ltp):.2f}\n")

    if all_reloads:
        print(f"{COLOR_BOLD}🔄 BASKET 2: ALGORITHMIC RELOADS (Phase 2 - Institutional Continuations){COLOR_RESET}")
        for sym, row in all_reloads.items():
            pct_move, ltp = row['Rec_Pct_Move'], row['Close']
            true_drift = row['Net_Drift']
            color = COLOR_GREEN if pct_move > 0 else COLOR_RED
            d_str = "BULLISH" if pct_move > 0 else "BEARISH"
            eval_t = row.get('Eval_Time', '15:15')

            macro_date = row['Macro_Date']
            macro_time = memory_bank[sym].get('time', "09:15")
            macro_price = row['Macro_Price']
            micro_price = row['Micro_Price']

            print(f"  {color}🔄 {sym:<12} Block Move: {pct_move:+.2f}% ({d_str}){COLOR_RESET}")
            print(f"      └─ 🎯 Tech Filters Passed : BB-RSI Breakout | ADX:{row.get('ADX', 0):.1f} | Renko: {d_str}")
            print(f"      └─ ⚓ Macro Floor (Origin): {macro_date} @ {macro_time} | Price: ₹{macro_price:.2f}")
            print(f"      └─ ⚡ Micro Floor (Reload): {target_date_str} @ {eval_t} | Price: ₹{micro_price:.2f}")
            print(f"      └─ 🎯 Latest LTP          : {target_date_str} @ EOD    | Price: ₹{final_ltp_dict.get(sym, ltp):.2f} (Trend Drift: {true_drift:+.2f}%)\n")

    if breached:
        print(f"{COLOR_DIM}⚠️ BASKET 3: BREACHED PIVOTS (Phase 3 - Trapped Capital / Dead Trends){COLOR_RESET}")
        for b in breached:
            b_time = b['Time'] if b['Time'] else 'Pending Intraday Breakdown'
            print(f"  {COLOR_YELLOW}⚠️ {b['Symbol']:<12} {b['Dir']} Anchor shattered!{COLOR_RESET}")
            print(f"      └─ ⚓ Anchor : {b['First_Date']} @ {b['Anchor_Time']} | LTP: ₹{b['Origin']:.2f}")
            print(f"      └─ 🎯 Latest : Breached At {b_time} | Current LTP: ₹{b['LTP']:.2f}\n")

    if all_reclaims:
        print(f"{COLOR_BOLD}🪤 BASKET 4: INSTITUTIONAL RECLAIMS (Phase 4 - Liquidity Traps){COLOR_RESET}")
        for sym, row in all_reclaims.items():
            pct_move, ltp = row['Rec_Pct_Move'], row['Close']
            color = COLOR_MAGENTA
            d_str = "BULLISH" if pct_move > 0 else "BEARISH"
            anchor_time = memory_bank[sym].get('time', "09:15")
            eval_t = row.get('Eval_Time', '15:15')

            print(f"  {color}🔥 {sym:<12} Block Move: {pct_move:+.2f}% ({d_str}){COLOR_RESET}")
            print(f"      └─ 🎯 Tech Filters Passed : BB-RSI Breakout | ADX:{row.get('ADX', 0):.1f} | Renko: {d_str}")
            print(f"      └─ ⚓ Anchor : {row['First_Date']} @ {anchor_time} | LTP: ₹{row['Origin']:.2f}")
            print(f"      └─ 🎯 Latest : Reclaimed At {target_date_str} @ {eval_t} | LTP: ₹{ltp:.2f}\n")

    if not any([valid_fresh, all_reloads, all_reclaims, breached]):
        print(f"{COLOR_DIM}[Terminal Silent] No active institutional structure passing strict filters.{COLOR_RESET}\n")

# ==============================================================================
# 5. RUN EXECUTOR
# ==============================================================================
def run_production_sweep():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--date", type=str, default="")
    parser.add_argument("positional_date", nargs="?", default="")
    args, _ = parser.parse_known_args()

    raw_date_str = args.date or args.positional_date or os.environ.get("PARAM_BACKTEST_DATE", "").strip()
    is_backtest = bool(raw_date_str)

    if not is_backtest:
        target_dt = datetime.utcnow() + timedelta(hours=5, minutes=30)
        if target_dt.weekday() == 5: 
            print(f"{COLOR_YELLOW}[System Notice] Market closed (Saturday). Auto-rolling back to Friday's tape.{COLOR_RESET}")
            target_dt -= timedelta(days=1)
        elif target_dt.weekday() == 6: 
            print(f"{COLOR_YELLOW}[System Notice] Market closed (Sunday). Auto-rolling back to Friday's tape.{COLOR_RESET}")
            target_dt -= timedelta(days=2)
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
