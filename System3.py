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
# 2. TECHNICAL PRE-COMPUTATION ENGINE (BB-RSI, ADX, Renko)
# ==============================================================================
def prepare_technical_data(df):
    hist_15m = df.groupby(['Symbol', pd.Grouper(key='Datetime', freq='15min', closed='left', label='left')]).agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).reset_index()
    
    hist_15m = hist_15m.dropna(subset=['Close']).sort_values(['Symbol', 'Datetime'])
    
    hist_15m['H-L'] = hist_15m['High'] - hist_15m['Low']
    hist_15m['H-PC'] = (hist_15m['High'] - hist_15m.groupby('Symbol')['Close'].shift(1)).abs()
    hist_15m['L-PC'] = (hist_15m['Low'] - hist_15m.groupby('Symbol')['Close'].shift(1)).abs()
    hist_15m['TR'] = hist_15m[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    atr14 = hist_15m.groupby('Symbol')['TR'].transform(lambda x: x.ewm(alpha=1/14, adjust=False).mean())
    hist_15m['ATR'] = atr14

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
# 3. VECTORIZED ANOMALY DETECTION (FIXED SLICING BUG)
# ==============================================================================
def extract_all_anomalies(hist_15m):
    df = hist_15m.copy()
    df['Rec_Pct_Move'] = ((df['Close'] - df['Open']) / (df['Open'] + 1e-8)) * 100
    
    bull_cond = (df['RSI'] > df['BB_Upper']) & (df['ADX'] > 20) & (df['ADX'] > df['ADX_prev']) & (df['+DI'] > df['-DI']) & (df['Renko_Trend'] == 1)
    bear_cond = (df['RSI'] < df['BB_Lower']) & (df['ADX'] > 20) & (df['ADX'] > df['ADX_prev']) & (df['-DI'] > df['+DI']) & (df['Renko_Trend'] == -1)
    
    # Calculate Direction on the entire DataFrame before filtering
    df['Direction'] = np.where(bull_cond, 1, np.where(bear_cond, -1, 0))
    
    # Filter anomalies safely
    anomalies = df[df['Direction'] != 0].copy()
    
    if anomalies.empty:
        anomalies['Eval_Time'] = pd.Series(dtype='datetime64[ns]')
        return anomalies
    
    anomalies['Eval_Time'] = anomalies['Datetime'] + pd.Timedelta(minutes=15)
    return anomalies

# ==============================================================================
# 4. LIGHTNING STATE-BASED MEMORY ENGINE
# ==============================================================================
def scan_institutional_tape(target_date_str):
    print(f"\n📡 Initiating Vectorized Engine for {target_date_str}...")
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
        item, start_date, end_date, live = task
        key = urllib.parse.quote(item['key'])
        access_token = os.environ.get("UPSTOX_ACCESS_TOKEN")
        headers = {'Accept': 'application/json', 'Authorization': f'Bearer {access_token}'}
        
        dfs = []
        hist_end = end_date if not live else (current_now - timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Historical chunk
        for attempt in range(4):
            try:
                res = requests.get(f"https://api.upstox.com/v2/historical-candle/{key}/1minute/{hist_end}/{start_date}", headers=headers, timeout=15)
                if res.status_code == 429: time.sleep(1.5); continue
                if res.status_code == 200 and res.json().get('data', {}).get('candles'):
                    dfs.append(pd.DataFrame(res.json()['data']['candles'], columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI']))
                break
            except Exception:
                time.sleep(1)
            
        # Live intraday chunk
        if live:
            for attempt in range(4):
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

    # execution with progress tracker and reduced max_workers to avoid firewall DDOS triggers
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_worker, task): task for task in fetch_tasks}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            sys.stdout.write(f"\r📡 Fetching Data... {completed}/{len(fetch_tasks)} symbols processed")
            sys.stdout.flush()
            res = future.result()
            if res is not None:
                historical_dfs.append(res)
    print() # Clear line

    if not historical_dfs:
        print(f"⚠️ {COLOR_RED}Fatal Error: No data retrieved. Check API token or network limits.{COLOR_RESET}")
        return

    rolling_master_df = pd.concat(historical_dfs, ignore_index=True)
    
    print(f"⚙️ Computing Technical Matrices instantly...")
    hist_15m_tech = prepare_technical_data(rolling_master_df)
    
    all_anomalies = extract_all_anomalies(hist_15m_tech)
    anomalies_by_time = all_anomalies.groupby('Eval_Time')
    
    closes_15m = hist_15m_tech.copy()
    closes_15m['Eval_Time'] = closes_15m['Datetime'] + pd.Timedelta(minutes=15)
    closes_dict = closes_15m.set_index(['Eval_Time', 'Symbol'])['Close'].to_dict()

    memory_bank = {} 
    unique_days = sorted(rolling_master_df['Datetime'].dt.date.unique())
    historical_dates = [d for d in unique_days if d < target_dt.date()]

    for d in historical_dates:
        day_str = d.strftime('%Y-%m-%d')
        day_master = rolling_master_df[rolling_master_df['Datetime'].dt.date == d]
        
        morning_opens = day_master.groupby('Symbol')['Open'].first().to_dict()
        for sym, st in memory_bank.items():
            if sym in morning_opens and st['state'] == 'ACTIVE':
                if (st['dir'] == 1 and morning_opens[sym] < st['origin']) or (st['dir'] == -1 and morning_opens[sym] > st['origin']):
                    st['state'] = 'BREACHED'
                    st['breach_time'] = f"{day_str} 09:15 (GAP)"

        day_times = sorted([t for t in closes_15m['Eval_Time'].unique() if t.date() == d])
        
        for t in day_times:
            for sym, st in memory_bank.items():
                ltp = closes_dict.get((t, sym))
                if ltp:
                    if st['state'] == 'ACTIVE' and ((st['dir'] == 1 and ltp < st['origin']) or (st['dir'] == -1 and ltp > st['origin'])):
                        st['state'] = 'BREACHED'
                        st['breach_time'] = t.strftime('%Y-%m-%d %H:%M')
                    elif st['state'] == 'BREACHED' and ((st['dir'] == 1 and ltp >= st['origin']) or (st['dir'] == -1 and ltp <= st['origin'])):
                        st['state'] = 'ACTIVE'
                        st['breach_time'] = None

            if t in anomalies_by_time.groups:
                for _, row in anomalies_by_time.get_group(t).iterrows():
                    sym = row['Symbol']
                    if sym not in memory_bank:
                        memory_bank[sym] = {'state': 'ACTIVE', 'origin': row['Close'], 'date': day_str, 'time': t.strftime('%H:%M'), 'dir': row['Direction'], 'breach_time': None}
        
        daily_closes = day_master.groupby('Symbol')['Close'].last().to_dict()
        for sym in list(memory_bank.keys()):
            if sym in daily_closes:
                st = memory_bank[sym]
                if st['state'] == 'BREACHED':
                    if (st['dir'] == 1 and daily_closes[sym] < st['origin'] * 0.985) or (st['dir'] == -1 and daily_closes[sym] > st['origin'] * 1.015):
                        del memory_bank[sym]

    # ----------------------------------------------------------------------
    # LIVE TARGET EVALUATION 
    # ----------------------------------------------------------------------
    today_master = rolling_master_df[rolling_master_df['Datetime'].dt.date == target_dt.date()].copy()

    if today_master.empty: 
        print(f"\n{COLOR_YELLOW}[Terminal Standby] Market data for {target_date_str} is empty or not available yet.{COLOR_RESET}\n")
        return

    if not is_live_today or current_now.hour >= 16:
        eval_times = [target_dt + pd.Timedelta(hours=h, minutes=m) for h, m in [(9,45), (10,30), (11,30), (12,30), (13,30), (14,30), (15,15)]]
    else:
        eval_times = [current_now.replace(second=0, microsecond=0) - timedelta(minutes=1)]

    all_fresh_intrusions, all_reloads, all_reclaims = {}, {}, {}

    for eval_time_current in eval_times:
        current_slice = today_master[today_master['Datetime'] <= eval_time_current]
        if current_slice.empty: continue
        today_latest_ltp = current_slice.groupby('Symbol')['Close'].last().to_dict()
        
        for sym, st in memory_bank.items():
            ltp = today_latest_ltp.get(sym)
            if ltp:
                if st['state'] == 'ACTIVE' and ((st['dir'] == 1 and ltp < st['origin']) or (st['dir'] == -1 and ltp > st['origin'])):
                    st['state'], st['breach_time'] = 'BREACHED', eval_time_current.strftime('%Y-%m-%d %H:%M')
                elif st['state'] == 'BREACHED' and ((st['dir'] == 1 and ltp >= st['origin']) or (st['dir'] == -1 and ltp <= st['origin'])):
                    st['state'], st['breach_time'] = 'ACTIVE', None

        if not all_anomalies.empty:
            curr_anoms = all_anomalies[all_anomalies['Eval_Time'] == eval_time_current]
            for _, row in curr_anoms.iterrows():
                sym, price, direction = row['Symbol'], row['Close'], row['Direction']
                if sym not in memory_bank:
                    if sym not in all_fresh_intrusions:
                        row['Eval_Time_Str'] = eval_time_current.strftime('%H:%M')
                        all_fresh_intrusions[sym] = row
                else:
                    st = memory_bank[sym]
                    row['Net_Drift'] = ((price - st['origin']) / st['origin'] * 100) if st['dir'] == 1 else ((st['origin'] - price) / st['origin'] * 100)
                    
                    if st['state'] == 'ACTIVE' and direction == st['dir']:
                        row['Eval_Time_Str'] = eval_time_current.strftime('%H:%M')
                        row['Macro_Price'], row['Macro_Date'], row['Micro_Price'] = st['origin'], st['date'], price
                        all_reloads[sym] = row
                    elif st['state'] == 'BREACHED' and direction == st['dir']:
                        st['state'], st['breach_time'] = 'ACTIVE', None
                        row['Eval_Time_Str'] = eval_time_current.strftime('%H:%M')
                        row['Origin'], row['First_Date'] = st['origin'], st['date']
                        all_reclaims[sym] = row

    final_ltp_dict = today_master.groupby('Symbol')['Close'].last().to_dict()
    valid_fresh, breached = {}, []

    for sym, row in all_fresh_intrusions.items():
        ltp, direction, birth_price = final_ltp_dict.get(sym, row['Close']), row['Direction'], row['Close']
        if (direction == 1 and ltp < birth_price) or (direction == -1 and ltp > birth_price):
            memory_bank[sym] = {'state': 'BREACHED', 'origin': birth_price, 'date': target_date_str, 'time': row.get('Eval_Time_Str', '15:15'), 'dir': direction, 'breach_time': f"{target_date_str} EOD Violation"}
        else:
            valid_fresh[sym] = row

    for sym, st in memory_bank.items():
        if st['state'] == 'BREACHED' and sym in final_ltp_dict and sym not in all_reclaims:
            breached.append({'Symbol': sym, 'LTP': final_ltp_dict[sym], 'Origin': st['origin'], 'Dir': "BULLISH" if st['dir'] == 1 else "BEARISH", 'Time': st['breach_time'], 'First_Date': st['date'], 'Anchor_Time': st.get('time', '09:15')})

    # ----------------------------------------------------------------------
    # TERMINAL OUTPUT
    # ----------------------------------------------------------------------
    print(f"\n{COLOR_CYAN}================================================================================================{COLOR_RESET}")
    print(f"{COLOR_BOLD}FULL UNIVERSE TECHNICAL CONFLUENCE TAPE | DATE: {target_date_str}{COLOR_RESET}")
    print(f"{COLOR_CYAN}================================================================================================{COLOR_RESET}\n")

    if valid_fresh:
        print(f"{COLOR_BOLD}⚡ BASKET 1: FRESH INTRUSIONS (Phase 1 - Day-1 Births){COLOR_RESET}")
        for sym, row in valid_fresh.items():
            pct_move, ltp = row['Rec_Pct_Move'], row['Close']
            color, d_str = (COLOR_GREEN, "BULLISH") if pct_move > 0 else (COLOR_RED, "BEARISH")
            print(f"  {color}🚨 {sym:<12} Block Move: {pct_move:+.2f}% ({d_str}){COLOR_RESET}")
            print(f"      └─ 🎯 Tech Filters Passed : BB-RSI Breakout | ADX:{row.get('ADX', 0):.1f} | Renko: {d_str}")
            print(f"      └─ ⚓ Breakout Anchor     : {target_date_str} @ {row.get('Eval_Time_Str', '15:15')} | Price: ₹{ltp:.2f}")
            print(f"      └─ 🎯 Latest LTP          : {target_date_str} @ EOD    | Price: ₹{final_ltp_dict.get(sym, ltp):.2f}\n")

    if all_reloads:
        print(f"{COLOR_BOLD}🔄 BASKET 2: ALGORITHMIC RELOADS (Phase 2 - Institutional Continuations){COLOR_RESET}")
        for sym, row in all_reloads.items():
            pct_move, ltp = row['Rec_Pct_Move'], row['Close']
            color, d_str = (COLOR_GREEN, "BULLISH") if pct_move > 0 else (COLOR_RED, "BEARISH")
            print(f"  {color}🔄 {sym:<12} Block Move: {pct_move:+.2f}% ({d_str}){COLOR_RESET}")
            print(f"      └─ 🎯 Tech Filters Passed : BB-RSI Breakout | ADX:{row.get('ADX', 0):.1f} | Renko: {d_str}")
            print(f"      └─ ⚓ Macro Floor (Origin): {row['Macro_Date']} @ {memory_bank[sym].get('time', '09:15')} | Price: ₹{row['Macro_Price']:.2f}")
            print(f"      └─ ⚡ Micro Floor (Reload): {target_date_str} @ {row.get('Eval_Time_Str', '15:15')} | Price: ₹{row['Micro_Price']:.2f}")
            print(f"      └─ 🎯 Latest LTP          : {target_date_str} @ EOD    | Price: ₹{final_ltp_dict.get(sym, ltp):.2f} (Trend Drift: {row['Net_Drift']:+.2f}%)\n")

    if breached:
        print(f"{COLOR_DIM}⚠️ BASKET 3: BREACHED PIVOTS (Phase 3 - Trapped Capital / Dead Trends){COLOR_RESET}")
        for b in breached:
            print(f"  {COLOR_YELLOW}⚠️ {b['Symbol']:<12} {b['Dir']} Anchor shattered!{COLOR_RESET}")
            print(f"      └─ ⚓ Anchor : {b['First_Date']} @ {b['Anchor_Time']} | LTP: ₹{b['Origin']:.2f}")
            print(f"      └─ 🎯 Latest : Breached At {b.get('Time', 'Pending')} | Current LTP: ₹{b['LTP']:.2f}\n")

    if all_reclaims:
        print(f"{COLOR_BOLD}🪤 BASKET 4: INSTITUTIONAL RECLAIMS (Phase 4 - Liquidity Traps){COLOR_RESET}")
        for sym, row in all_reclaims.items():
            pct_move, ltp = row['Rec_Pct_Move'], row['Close']
            d_str = "BULLISH" if pct_move > 0 else "BEARISH"
            print(f"  {COLOR_MAGENTA}🔥 {sym:<12} Block Move: {pct_move:+.2f}% ({d_str}){COLOR_RESET}")
            print(f"      └─ 🎯 Tech Filters Passed : BB-RSI Breakout | ADX:{row.get('ADX', 0):.1f} | Renko: {d_str}")
            print(f"      └─ ⚓ Anchor : {row['First_Date']} @ {memory_bank[sym].get('time', '09:15')} | LTP: ₹{row['Origin']:.2f}")
            print(f"      └─ 🎯 Latest : Reclaimed At {target_date_str} @ {row.get('Eval_Time_Str', '15:15')} | LTP: ₹{ltp:.2f}\n")

    if not any([valid_fresh, all_reloads, all_reclaims, breached]):
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
