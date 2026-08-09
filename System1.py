import os
import argparse
import smtplib
from email.mime.text import MIMEText
import requests
import time
import io
from datetime import datetime, date, timedelta, timezone

import pandas as pd
import numpy as np
from fyers_apiv3 import fyersModel

# ==============================================================================
# 0. ENGINE CONSTANTS & GLOBAL CONFIGURATION DIALS
# ==============================================================================
COLOR_GREEN = '\033[92m'
COLOR_RED = '\033[91m'
COLOR_CYAN = '\033[96m'
COLOR_YELLOW = '\033[93m'
COLOR_MAGENTA = '\033[95m'
COLOR_DIM = '\033[2m'
COLOR_RESET = '\033[0m'
COLOR_BOLD = '\033[1m'

# 🎛️ GLOBAL TUNING DIALS
LOOKBACK_DAYS = 50            # Configurable: Set to any window (3 to 300+ days)
TOP_N_STRIKES = 5            # Max apex trades to display per basket/index
SCORE_THRESHOLD = 50        # Quad-Delta minimum score threshold
MIN_VECTOR_FLOOR = 3        # Minimum percentile contribution per variable
LIQUIDITY_MIN_PRICE = 95.0   # Global variable: Purges options trading below this price
MAX_BREACH_DAYS = 0          # Kill Switch: 0 for strict intraday scalping memory purge

GLOBAL_START_TIME = "09:30"  # Anchors trading day start, bypassing 09:15 auction noise
IST = timezone(timedelta(hours=5, minutes=30))

ACTIVE_INDICES = {
    "NIFTY": {"symbol": "NSE:NIFTY50-INDEX", "opt_prefix": "NSE:NIFTY"},
    "BANKNIFTY": {"symbol": "NSE:NIFTYBANK-INDEX", "opt_prefix": "NSE:BANKNIFTY"},
    "FINNIFTY": {"symbol": "NSE:NIFTY FIN SERVICE-INDEX", "opt_prefix": "NSE:FINNIFTY"},
    "SENSEX": {"symbol": "BSE:SENSEX-INDEX", "opt_prefix": "BSE:SENSEX"}
}

# ==============================================================================
# 1. AUTH & TARGET DATES (GITHUB UI & LIVE OVERRIDE)
# ==============================================================================
CLIENT_ID = os.getenv("FYERS_CLIENT_ID")
ACCESS_TOKEN = os.getenv("FYERS_ACCESS_TOKEN")
EMAIL_SENDER = os.getenv("EMAIL_SENDER")       
EMAIL_APP_PWD = os.getenv("EMAIL_APP_PWD")     
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")   

if not CLIENT_ID or not ACCESS_TOKEN:
    raise ValueError(f"{COLOR_RED}🚨 CRITICAL FAILURE: API credentials missing. Halting engine.{COLOR_RESET}")

def get_fyers_instance():
    return fyersModel.FyersModel(client_id=CLIENT_ID, is_async=False, token=ACCESS_TOKEN, log_path="")

def get_target_dates():
    param_date = os.getenv("PARAM_BACKTEST_DATE")
    if param_date and param_date.strip():
        end_date = datetime.strptime(param_date.strip(), "%Y-%m-%d").date()
        print(f"{COLOR_CYAN}⚙️ GITHUB UI OVERRIDE: Anchoring Target Date to {end_date}{COLOR_RESET}")
    else:
        end_date = datetime.now(IST).date()
        if end_date.weekday() == 5: end_date -= timedelta(days=1)
        elif end_date.weekday() == 6: end_date -= timedelta(days=2)
        
    start_date = end_date
    days_subtracted = 0
    while days_subtracted < LOOKBACK_DAYS:
        start_date -= timedelta(days=1)
        if start_date.weekday() < 5: 
            days_subtracted += 1
    return start_date, end_date

EPOCH_START, TARGET_END = get_target_dates()

# ==============================================================================
# 2. MASTER SYMBOL LOADER & LIQUIDITY SHIELD
# ==============================================================================
MASTER_SYMBOLS = {}
INDEX_EXPIRIES = {"NIFTY": set(), "BANKNIFTY": set(), "FINNIFTY": set(), "SENSEX": set()}

def load_symbol_master():
    print(f"{COLOR_DIM}📡 Downloading Exchange Symbol Master...{COLOR_RESET}")
    urls = ["https://public.fyers.in/sym_details/NSE_FO.csv", "https://public.fyers.in/sym_details/BSE_FO.csv"]
    for url in urls:
        try:
            res = requests.get(url, timeout=15)
            for line in res.text.split('\n'):
                parts = line.split(',')
                if len(parts) < 17: continue
                sym_ticker, opt_type, strike_str, expiry_val = parts[9], parts[16], parts[15], parts[8]
                if opt_type not in ["CE", "PE"]: continue
                
                idx = None
                for i_name, i_conf in ACTIVE_INDICES.items():
                    prefix = i_conf["opt_prefix"]
                    if sym_ticker.startswith(prefix) and len(sym_ticker) > len(prefix) and sym_ticker[len(prefix)].isdigit():
                        idx = i_name
                        break
                if not idx: continue
                
                try:
                    if expiry_val.isdigit():
                        expiry_date = datetime.fromtimestamp(int(expiry_val), tz=timezone.utc).astimezone(IST).date()
                    else:
                        expiry_date = datetime.strptime(expiry_val, "%Y-%m-%d").date()
                    INDEX_EXPIRIES[idx].add(expiry_date)
                    MASTER_SYMBOLS[(idx, expiry_date, int(float(strike_str)), opt_type)] = sym_ticker
                except: continue
        except: pass
    for k in INDEX_EXPIRIES: INDEX_EXPIRIES[k] = sorted(list(INDEX_EXPIRIES[k]))

def get_liquid_strikes(fyers, index_name, expiry_date):
    all_symbols = [sym for (idx, exp, strike, opt), sym in MASTER_SYMBOLS.items() if idx == index_name and exp == expiry_date]
    liquid_symbols = []
    for i in range(0, len(all_symbols), 50):
        batch = all_symbols[i:i+50]
        try:
            res = fyers.quotes({"symbols": ",".join(batch)})
            if 'd' in res:
                for data in res['d']:
                    sym, lp, vol = data['n'], data['v'].get('lp', 0), data['v'].get('volume', 0)
                    if lp >= LIQUIDITY_MIN_PRICE and vol > 0: liquid_symbols.append(sym)
        except: pass
    print(f"   🛡️ {index_name} Liquidity Filter (Threshold: ₹{LIQUIDITY_MIN_PRICE}): Tracking {len(liquid_symbols)} high-probability targets.")
    return liquid_symbols

def fetch_fyers_historical_df(fyers, symbol, index_name):
    res = fyers.history({
        "symbol": symbol, "resolution": "5", "date_format": "1",
        "range_from": EPOCH_START.strftime("%Y-%m-%d"), 
        "range_to": TARGET_END.strftime("%Y-%m-%d"), "cont_flag": "1"
    })
    candles = res.get('candles', [])
    if not candles: return None
    df = pd.DataFrame(candles, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Datetime'] = pd.to_datetime(df['Timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
    df['Symbol'] = symbol
    df['Index_Name'] = index_name
    df['Turnover'] = df['Volume'] * df['Close']
    df['abs_move'] = (df['Close'] - df['Open']).abs()
    return df

# ==============================================================================
# 3. INDEX-ISOLATED UNIFORM ROLLING BASELINE QUAD-DELTA ENGINE
# ==============================================================================
def calculate_velocity_leaderboard(master_df, current_eval_time, window_mins=15, baseline_mins=45):
    try:
        if master_df is None or master_df.empty or 'Datetime' not in master_df.columns: return pd.DataFrame()
        df_calc = master_df.copy()
        df_calc['candle_range'] = ((df_calc['High'] - df_calc['Low']) / (df_calc['Open'] + 1e-8)) * 100

        h, m = map(int, GLOBAL_START_TIME.split(':'))
        start_of_day = pd.to_datetime(current_eval_time.date()) + pd.Timedelta(hours=h, minutes=m)
        
        recent_start = current_eval_time - pd.Timedelta(minutes=window_mins)
        if recent_start <= start_of_day: return pd.DataFrame()
            
        baseline_start = max(start_of_day, recent_start - pd.Timedelta(minutes=baseline_mins))
        
        base_df = df_calc[(df_calc['Datetime'] >= baseline_start) & (df_calc['Datetime'] < recent_start)]
        rec_df = df_calc[(df_calc['Datetime'] >= recent_start) & (df_calc['Datetime'] <= current_eval_time)]
        if base_df.empty or rec_df.empty: return pd.DataFrame()
            
        g_base = base_df.groupby(['Symbol', 'Index_Name']).agg({'Turnover': 'sum', 'Open': 'first', 'Close': 'last', 'abs_move': 'sum', 'candle_range': 'mean'}).reset_index()
        g_base = g_base[g_base['Turnover'] > 0]
        if g_base.empty: return pd.DataFrame()
        
        g_base['Base_Pct_Move'] = ((g_base['Close'] - g_base['Open']) / (g_base['Open'] + 1e-8)) * 100
        g_base['Base_Efficiency'] = (g_base['Close'] - g_base['Open']).abs() / (g_base['abs_move'] + 1e-8)
        
        g_base['Base_Vol_Rank'] = g_base.groupby('Index_Name')['Turnover'].transform(lambda x: x.rank(pct=True) * 100)
        g_base['Base_P_Rank'] = g_base.groupby('Index_Name')['candle_range'].transform(lambda x: x.rank(pct=True) * 100)
        g_base['Base_Mom_Rank'] = g_base.groupby('Index_Name')['Base_Pct_Move'].transform(lambda x: x.abs().rank(pct=True) * 100)
        g_base['Base_Eff_Rank'] = g_base.groupby('Index_Name')['Base_Efficiency'].transform(lambda x: x.rank(pct=True) * 100)

        g_rec = rec_df.groupby(['Symbol', 'Index_Name']).agg({'Turnover': 'sum', 'Open': 'first', 'Close': 'last', 'abs_move': 'sum', 'candle_range': 'mean'}).reset_index()
        g_rec = g_rec[g_rec['Turnover'] > 0]
        if g_rec.empty: return pd.DataFrame()
        
        g_rec['Rec_Pct_Move'] = ((g_rec['Close'] - g_rec['Open']) / (g_rec['Open'] + 1e-8)) * 100
        g_rec['Rec_Efficiency'] = (g_rec['Close'] - g_rec['Open']).abs() / (g_rec['abs_move'] + 1e-8)
        
        g_rec['Rec_Vol_Rank'] = g_rec.groupby('Index_Name')['Turnover'].transform(lambda x: x.rank(pct=True) * 100)
        g_rec['Rec_P_Rank'] = g_rec.groupby('Index_Name')['candle_range'].transform(lambda x: x.rank(pct=True) * 100)
        g_rec['Rec_Mom_Rank'] = g_rec.groupby('Index_Name')['Rec_Pct_Move'].transform(lambda x: x.abs().rank(pct=True) * 100)
        g_rec['Rec_Eff_Rank'] = g_rec.groupby('Index_Name')['Rec_Efficiency'].transform(lambda x: x.rank(pct=True) * 100)

        merged = pd.merge(g_rec[['Symbol', 'Index_Name', 'Rec_Pct_Move', 'Close', 'Rec_Vol_Rank', 'Rec_P_Rank', 'Rec_Mom_Rank', 'Rec_Eff_Rank']], 
                          g_base[['Symbol', 'Index_Name', 'Base_Vol_Rank', 'Base_P_Rank', 'Base_Mom_Rank', 'Base_Eff_Rank']], on=['Symbol', 'Index_Name'], how='inner')
        if merged.empty: return pd.DataFrame()

        merged['Vol_Delta'] = merged['Rec_Vol_Rank'] - merged['Base_Vol_Rank']
        merged['P_Delta'] = merged['Rec_P_Rank'] - merged['Base_P_Rank']
        merged['Mom_Delta'] = merged['Rec_Mom_Rank'] - merged['Base_Mom_Rank']
        merged['Eff_Delta'] = merged['Rec_Eff_Rank'] - merged['Base_Eff_Rank']
        
        merged['Direction'] = np.where(merged['Rec_Pct_Move'] > 0, 1, -1)
        merged['V_Score'] = merged['Vol_Delta'] * merged['Direction']
        merged['P_Score'] = merged['P_Delta'] * merged['Direction']
        merged['M_Score'] = merged['Mom_Delta'] * merged['Direction']
        merged['E_Score'] = merged['Eff_Delta'] * merged['Direction']
        
        merged = merged[
            (merged['V_Score'].abs() >= MIN_VECTOR_FLOOR) &
            (merged['P_Score'].abs() >= MIN_VECTOR_FLOOR) &
            (merged['M_Score'].abs() >= MIN_VECTOR_FLOOR) &
            (merged['E_Score'].abs() >= MIN_VECTOR_FLOOR)
        ]
        
        merged['Valid_Tandem'] = (
            (np.sign(merged['V_Score']) == merged['Direction']) &
            (np.sign(merged['P_Score']) == merged['Direction']) &
            (np.sign(merged['M_Score']) == merged['Direction']) &
            (np.sign(merged['E_Score']) == merged['Direction'])
        )
        merged = merged[merged['Valid_Tandem']]

        merged['Total_Score'] = merged['V_Score'] + merged['P_Score'] + merged['M_Score'] + merged['E_Score']
        merged = merged[merged['Total_Score'].abs() >= SCORE_THRESHOLD]
        merged = merged.sort_values(by='Total_Score', key=abs, ascending=False)
        return merged
    except: return pd.DataFrame()

# ==============================================================================
# 4. STATE-BASED MEMORY ENGINE (Full Matrix Lifecycle)
# ==============================================================================
def execute_options_matrix():
    target_date_str = TARGET_END.strftime("%Y-%m-%d")
    print(f"\n{COLOR_CYAN}📡 Initiating State-Based Quad-Delta Engine for {target_date_str} (Lookback: {LOOKBACK_DAYS} Days)...{COLOR_RESET}")
    
    fyers = get_fyers_instance()
    load_symbol_master()
    
    historical_dfs = []
    for index_name in ACTIVE_INDICES.keys():
        valid_dates = [d for d in INDEX_EXPIRIES.get(index_name, []) if d >= TARGET_END]
        if not valid_dates: continue
        active_expiry = valid_dates[1] if TARGET_END == valid_dates[0] and len(valid_dates) > 1 else valid_dates[0]
        liquid_targets = get_liquid_strikes(fyers, index_name, active_expiry)
        
        for sym in liquid_targets:
            df = fetch_fyers_historical_df(fyers, sym, index_name)
            if df is not None and not df.empty: historical_dfs.append(df)
        
    if not historical_dfs:
        print(f"⚠️ {COLOR_RED}Fatal Error: No data returned from Fyers API across active strikes.{COLOR_RESET}")
        return

    rolling_master_df = pd.concat(historical_dfs, ignore_index=True)
    
    unique_dates = sorted(rolling_master_df['Datetime'].dt.date.unique())
    trading_days = [d.strftime("%Y-%m-%d") for d in unique_dates]
    
    current_now = datetime.now(IST).replace(tzinfo=None)
    is_live_today = (target_date_str == current_now.strftime("%Y-%m-%d"))
    
    target_dt = pd.to_datetime(target_date_str)
    if not is_live_today or current_now.hour >= 16:
        eval_times = [
            target_dt + pd.Timedelta(hours=9, minutes=45), target_dt + pd.Timedelta(hours=10, minutes=30),
            target_dt + pd.Timedelta(hours=11, minutes=30), target_dt + pd.Timedelta(hours=12, minutes=30),
            target_dt + pd.Timedelta(hours=13, minutes=30), target_dt + pd.Timedelta(hours=14, minutes=30),
            target_dt + pd.Timedelta(hours=15, minutes=15)
        ]
    else:
        eval_times = [current_now.replace(second=0, microsecond=0) - timedelta(minutes=1)]

    memory_bank = {} 
    
    for day in trading_days:
        day_dt = pd.to_datetime(day)
        day_master = rolling_master_df[(rolling_master_df['Datetime'] >= day_dt) & (rolling_master_df['Datetime'] < day_dt + pd.Timedelta(days=1))]
        if day_master.empty: continue

        h, m = map(int, GLOBAL_START_TIME.split(':'))
        day_start = day_dt + pd.Timedelta(hours=h, minutes=m)
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

                anomalies = calculate_velocity_leaderboard(day_master, t, window_mins=15)
                if not anomalies.empty:
                    for _, row in anomalies.iterrows():
                        sym, price, direction = row['Symbol'], row['Close'], row['Direction']
                        if sym not in memory_bank:
                            memory_bank[sym] = {
                                'state': 'ACTIVE', 'origin': price, 'date': day, 
                                'time': t.strftime('%H:%M'), 'datetime': t, 'dir': direction, 
                                'breach_time': None, 'breach_days': 0
                            }
        except: pass

    # ======================================================================
    # LIVE EVALUATION LOOP (TOP-N GUILLOTINE & MULTI-BASKET ROUTING)
    # ======================================================================
    today_master = rolling_master_df[(rolling_master_df['Datetime'] >= target_dt) & (rolling_master_df['Datetime'] <= target_dt + pd.Timedelta(days=1))].copy()

    if today_master.empty: 
        print(f"\n{COLOR_YELLOW}[Terminal Standby] Market data for {target_date_str} is empty or not available yet.{COLOR_RESET}\n")
        return

    all_fresh_intrusions, all_reloads, all_reclaims = {}, {}, {}

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

            curr_anomalies = calculate_velocity_leaderboard(current_slice, eval_time_current, window_mins=15)

            if not curr_anomalies.empty:
                for _, row in curr_anomalies.iterrows():
                    sym, price, direction = row['Symbol'], row['Close'], row['Direction']
                    
                    if sym not in memory_bank:
                        if sym not in all_fresh_intrusions:
                            row['Eval_Time'] = eval_time_current.strftime('%H:%M')
                            launchpad_price = price
                            try:
                                launch_slice = rolling_master_df[(rolling_master_df['Symbol'] == sym) & (rolling_master_df['Datetime'] < eval_time_current) & (rolling_master_df['Datetime'] >= eval_time_current - pd.Timedelta(days=5))]
                                if not launch_slice.empty:
                                    launchpad_price = launch_slice['Low'].min() if direction == 1 else launch_slice['High'].max()
                            except: pass
                            row['Launchpad'] = launchpad_price
                            all_fresh_intrusions[sym] = row
                            
                            # 🛡️ BRIDGE FIX: Register live Basket 1 entries into memory bank so they can transition to Basket 3 (Breached)
                            memory_bank[sym] = {
                                'state': 'ACTIVE', 
                                'origin': price, 
                                'date': target_date_str, 
                                'time': eval_time_current.strftime('%H:%M'), 
                                'datetime': eval_time_current, 
                                'dir': direction, 
                                'breach_time': None, 
                                'breach_days': 0
                            }
                    else:
                        st = memory_bank[sym]
                        row['Net_Drift'] = ((price - st['origin']) / st['origin']) * 100 if st['dir'] == 1 else ((st['origin'] - price) / st['origin']) * 100
                        
                        is_subsequent_bar = eval_time_current > st.get('datetime', target_dt)

                        if st['state'] == 'ACTIVE' and row['Direction'] == st['dir'] and is_subsequent_bar:
                            if (st['dir'] == 1 and price >= st['origin']) or (st['dir'] == -1 and price <= st['origin']):
                                row['Eval_Time'] = eval_time_current.strftime('%H:%M')
                                row['Macro_Price'], row['Macro_Date'], row['Micro_Price'] = st['origin'], st['date'], price
                                all_reloads[sym] = row
                        elif st['state'] == 'BREACHED' and row['Direction'] == st['dir']:
                            if (st['dir'] == 1 and price > st['origin']) or (st['dir'] == -1 and price < st['origin']):
                                st['state'] = 'ACTIVE' 
                                st['breach_time'] = None
                                row['Eval_Time'] = eval_time_current.strftime('%H:%M')
                                row['Origin'], row['First_Date'] = st['origin'], st['date']
                                all_reclaims[sym] = row
        except: continue

    final_ltp_dict = today_master.groupby('Symbol')['Close'].last().to_dict()
    valid_fresh = {}
    
    for sym, row in all_fresh_intrusions.items():
        ltp, direction, birth_price = final_ltp_dict.get(sym, row['Close']), row['Direction'], row['Close']
        if (direction == 1 and ltp < birth_price) or (direction == -1 and ltp > birth_price):
            memory_bank[sym] = {'state': 'BREACHED', 'origin': birth_price, 'date': target_date_str, 'time': row.get('Eval_Time', '15:15'), 'dir': direction, 'breach_time': f"{target_date_str} EOD Violation", 'breach_days': 0}
        else:
            valid_fresh[sym] = row

    breached = []
    for sym, st in memory_bank.items():
        if st['state'] == 'BREACHED' and sym in final_ltp_dict and sym not in all_reclaims: 
            sentiment_tag = "BULLISH" if st['dir'] == 1 else "BEARISH"
            breached.append({'Symbol': sym, 'LTP': final_ltp_dict[sym], 'Origin': st['origin'], 'Dir': sentiment_tag, 'Time': st['breach_time'], 'First_Date': st['date'], 'Anchor_Time': st.get('time', '09:15')})

    # ======================================================================
    # 5. TERMINAL & EMAIL DISPATCH UI
    # ======================================================================
    output_lines = []
    output_lines.append(f"\n{COLOR_CYAN}================================================================================================{COLOR_RESET}")
    output_lines.append(f"{COLOR_BOLD} 🦅 GLOBAL OPTIONS QUAD-DELTA SCOREBOARD | DATE: {target_date_str}{COLOR_RESET}")
    output_lines.append(f"{COLOR_CYAN}================================================================================================{COLOR_RESET}\n")

    if valid_fresh:
        output_lines.append(f"{COLOR_BOLD}⚡ BASKET 1: FRESH INTRUSIONS (Phase 1 - Day-1 Births){COLOR_RESET}")
        for sym, row in list(valid_fresh.items())[:TOP_N_STRIKES]:
            jump, ltp, launchpad = row['Total_Score'], row['Close'], row.get('Launchpad', row['Close'])
            color = COLOR_GREEN if jump > 0 else COLOR_RED
            sent_str = "BULLISH" if row['Direction'] == 1 else "BEARISH"
            
            output_lines.append(f"  {color}🚨 {sym:<22} {jump:+.0f} pts [V:{row['V_Score']:+.0f} P:{row['P_Score']:+.0f} M:{row['M_Score']:+.0f} E:{row['E_Score']:+.0f}] ({sent_str}){COLOR_RESET}")
            output_lines.append(f"      └─ 🧱 Launchpad (Kinetic Base) : Price: ₹{launchpad:.2f}")
            output_lines.append(f"      └─ ⚓ Breakout Anchor (Birth)  : {target_date_str} @ {row.get('Eval_Time', '15:15')} | Price: ₹{ltp:.2f}")
            output_lines.append(f"      └─ 🎯 Latest LTP               : {target_date_str} @ EOD   | Price: ₹{final_ltp_dict.get(sym, ltp):.2f}\n")

    if all_reloads:
        output_lines.append(f"{COLOR_BOLD}🔄 BASKET 2: ALGORITHMIC RELOADS (Phase 2 - Institutional Continuations){COLOR_RESET}")
        for sym, row in list(all_reloads.items())[:TOP_N_STRIKES]:
            jump, ltp, true_drift = row['Total_Score'], row['Close'], row['Net_Drift']
            color = COLOR_GREEN if jump > 0 else COLOR_RED
            sent_str = "BULLISH" if row['Direction'] == 1 else "BEARISH"
            macro_time = memory_bank[sym].get('time', "09:15")
            
            output_lines.append(f"  {color}🔄 {sym:<22} {jump:+.0f} pts [V:{row['V_Score']:+.0f} P:{row['P_Score']:+.0f} M:{row['M_Score']:+.0f} E:{row['E_Score']:+.0f}] ({sent_str}){COLOR_RESET}")
            output_lines.append(f"      └─ ⚓ Macro Floor (Origin) : {row['Macro_Date']} @ {macro_time} | Price: ₹{row['Macro_Price']:.2f}")
            output_lines.append(f"      └─ ⚡ Micro Floor (Reload) : {target_date_str} @ {row.get('Eval_Time', '15:15')} | Price: ₹{row['Micro_Price']:.2f}")
            output_lines.append(f"      └─ 🎯 Latest LTP           : {target_date_str} @ EOD   | Price: ₹{final_ltp_dict.get(sym, ltp):.2f} (Trend Drift: {true_drift:+.2f}%)\n")

    if breached:
        output_lines.append(f"{COLOR_DIM}⚠️ BASKET 3: BREACHED PIVOTS (Phase 3 - Trapped Capital / Dead Trends){COLOR_RESET}")
        for b in breached[:TOP_N_STRIKES]:
            b_time = b['Time'] if b['Time'] else 'Pending Intraday Breakdown'
            output_lines.append(f"  {COLOR_YELLOW}⚠️ {b['Symbol']:<22} {b['Dir']} Anchor shattered!{COLOR_RESET}")
            output_lines.append(f"      └─ ⚓ Anchor : {b['First_Date']} @ {b['Anchor_Time']} | LTP: ₹{b['Origin']:.2f}")
            output_lines.append(f"      └─ 🎯 Latest : Breached At {b_time} | Current LTP: ₹{b['LTP']:.2f}\n")

    if all_reclaims:
        output_lines.append(f"{COLOR_BOLD}🪤 BASKET 4: INSTITUTIONAL RECLAIMS (Phase 4 - Liquidity Traps){COLOR_RESET}")
        for sym, row in list(all_reclaims.items())[:TOP_N_STRIKES]:
            jump, ltp, anchor_time = row['Total_Score'], row['Close'], memory_bank[sym].get('time', "09:15")
            color, sent_str = COLOR_MAGENTA, "BULLISH" if row['Direction'] == 1 else "BEARISH"
            
            output_lines.append(f"  {color}🔥 {sym:<22} {jump:+.0f} pts [V:{row['V_Score']:+.0f} P:{row['P_Score']:+.0f} M:{row['M_Score']:+.0f} E:{row['E_Score']:+.0f}] ({sent_str}){COLOR_RESET}")
            output_lines.append(f"      └─ ⚓ Anchor : {row['First_Date']} @ {anchor_time} | LTP: ₹{row['Origin']:.2f}")
            output_lines.append(f"      └─ 🎯 Latest : Reclaimed At {target_date_str} @ {row.get('Eval_Time', '15:15')} | LTP: ₹{ltp:.2f}\n")

    if not any([valid_fresh, all_reloads, all_reclaims, breached]):
        output_lines.append(f"{COLOR_DIM}[Terminal Silent] No active institutional structure passing strict filters.{COLOR_RESET}\n")

    for line in output_lines: print(line)

    if not EMAIL_SENDER or not EMAIL_APP_PWD or not EMAIL_RECEIVER: return
    if not any([valid_fresh, all_reloads, all_reclaims, breached]): return

    import re
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    clean_body = "\n".join([ansi_escape.sub('', line) for line in output_lines])
    
    msg = MIMEText(clean_body)
    msg['Subject'] = f"⚡ GLOBAL OPTIONS ALERTS: {target_date_str}"
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_APP_PWD)
            smtp.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        print(f"{COLOR_GREEN}📧 Alert Successfully Dispatched to Inbox.{COLOR_RESET}")
    except Exception as e:
        print(f"{COLOR_RED}⚠️ Email failed to send: {e}{COLOR_RESET}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    execute_options_matrix()
    print("✅ System Core Shutting Down.")
