import os
import sys
import argparse
import urllib.parse
import json
import gzip
import io
import time
from datetime import datetime, timedelta

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

SCORE_THRESHOLD = 120    # Minimum absolute Tri-Delta score to trigger an anomaly
BACKTRACE_DAYS = 20      # 1 F&O Monthly Derivative Cycle

# ==============================================================================
# 1. LIVE INGESTION (F&O Universe)
# ==============================================================================
def get_dynamic_fno_universe():
    nse_url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
    try:
        response = requests.get(nse_url, timeout=5)
        if response.status_code != 200: return []
        nse_data = json.load(gzip.GzipFile(fileobj=io.BytesIO(response.content)))
        fno_underlying = {item.get("underlying_symbol") for item in nse_data if item.get("segment") == "NSE_FO" and item.get("underlying_symbol")}
        return [{"symbol": item.get("trading_symbol"), "key": item.get("instrument_key")} for item in nse_data if item.get("segment") in ("NSE_EQ", "NSE_INDEX") and item.get("trading_symbol") in fno_underlying]
    except:
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
        c_df = c_df.sort_values('Datetime').reset_index(drop=True)
        return c_df
    except:
        return None

def get_past_trading_days(target_date_str, num_days=20):
    target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
    trading_days = []
    current_dt = target_dt
    while len(trading_days) < num_days:
        if current_dt.weekday() < 5:  # Skip weekends (0=Mon, 4=Fri)
            trading_days.append(current_dt.strftime("%Y-%m-%d"))
        current_dt -= timedelta(days=1)
    trading_days.reverse()
    return trading_days

# ==============================================================================
# 2. TRI-DELTA VELOCITY ENGINE (Directional Percentile Math)
# ==============================================================================
def calculate_velocity_leaderboard(master_df, current_eval_time, window_mins=15):
    if master_df is None or master_df.empty or 'Datetime' not in master_df.columns:
        return pd.DataFrame()

    start_of_day = pd.to_datetime(current_eval_time.date()) + pd.Timedelta(hours=9, minutes=15)
    recent_start = current_eval_time - pd.Timedelta(minutes=window_mins)
    
    if recent_start <= start_of_day:
        return pd.DataFrame()
        
    cum_df = master_df[(master_df['Datetime'] >= start_of_day) & (master_df['Datetime'] < recent_start)]
    rec_df = master_df[(master_df['Datetime'] >= recent_start) & (master_df['Datetime'] <= current_eval_time)]
    
    if cum_df.empty or rec_df.empty:
        return pd.DataFrame()
        
    # --- BASELINE (Morning up to window start) ---
    g_cum = cum_df.groupby('Symbol').agg({'Turnover': 'sum', 'Open': 'first', 'Close': 'last', 'abs_move': 'sum'}).reset_index()
    g_cum = g_cum[g_cum['Turnover'] > 0]
    if g_cum.empty: return pd.DataFrame()
    
    g_cum['Cum_Pct_Move'] = ((g_cum['Close'] - g_cum['Open']) / (g_cum['Open'] + 1e-8)) * 100
    g_cum['Cum_Efficiency'] = (g_cum['Close'] - g_cum['Open']).abs() / (g_cum['abs_move'] + 1e-8)
    
    g_cum['Cum_Vol_Rank'] = g_cum['Turnover'].rank(pct=True) * 100
    g_cum['Cum_Mom_Rank'] = g_cum['Cum_Pct_Move'].abs().rank(pct=True) * 100
    g_cum['Cum_Eff_Rank'] = g_cum['Cum_Efficiency'].rank(pct=True) * 100

    # --- RECENT (Last window minutes) ---
    g_rec = rec_df.groupby('Symbol').agg({'Turnover': 'sum', 'Open': 'first', 'Close': 'last', 'abs_move': 'sum'}).reset_index()
    g_rec = g_rec[g_rec['Turnover'] > 0]
    if g_rec.empty: return pd.DataFrame()
    
    g_rec['Rec_Pct_Move'] = ((g_rec['Close'] - g_rec['Open']) / (g_rec['Open'] + 1e-8)) * 100
    g_rec['Rec_Efficiency'] = (g_rec['Close'] - g_rec['Open']).abs() / (g_rec['abs_move'] + 1e-8)
    
    g_rec['Rec_Vol_Rank'] = g_rec['Turnover'].rank(pct=True) * 100
    g_rec['Rec_Mom_Rank'] = g_rec['Rec_Pct_Move'].abs().rank(pct=True) * 100
    g_rec['Rec_Eff_Rank'] = g_rec['Rec_Efficiency'].rank(pct=True) * 100

    # --- THE DIRECTIONAL TRI-DELTA CALCULATION ---
    merged = pd.merge(g_rec[['Symbol', 'Rec_Pct_Move', 'Close', 'Rec_Vol_Rank', 'Rec_Mom_Rank', 'Rec_Eff_Rank']], 
                      g_cum[['Symbol', 'Cum_Vol_Rank', 'Cum_Mom_Rank', 'Cum_Eff_Rank']], on='Symbol', how='inner')
    
    if merged.empty: return pd.DataFrame()

    merged['Vol_Delta'] = merged['Rec_Vol_Rank'] - merged['Cum_Vol_Rank']
    merged['Mom_Delta'] = merged['Rec_Mom_Rank'] - merged['Cum_Mom_Rank']
    merged['Eff_Delta'] = merged['Rec_Eff_Rank'] - merged['Cum_Eff_Rank']
    
    # Vector Alignment: Force volume & efficiency to inherit price direction
    merged['Direction'] = np.where(merged['Rec_Pct_Move'] > 0, 1, -1)
    
    merged['V_Score'] = merged['Vol_Delta'] * merged['Direction']
    merged['M_Score'] = merged['Mom_Delta'] * merged['Direction']
    merged['E_Score'] = merged['Eff_Delta'] * merged['Direction']
    
    # Final Score
    merged['Total_Score'] = merged['V_Score'] + merged['M_Score'] + merged['E_Score']
    
    # Filter by Absolute Threshold (Uncapped Universe)
    merged = merged[merged['Total_Score'].abs() >= SCORE_THRESHOLD]
    merged = merged.sort_values(by='Total_Score', key=abs, ascending=False)
    
    return merged

# ==============================================================================
# 3. STATE-BASED MEMORY ENGINE & TAPE PRINTER
# ==============================================================================
def scan_institutional_tape(target_date_str):
    print(f"\n📡 Initiating State-Based Tri-Delta Engine for {target_date_str}...")
    universe = get_dynamic_fno_universe()
    if not universe:
        print(f"⚠️ {COLOR_RED}No F&O universe found. Please check Upstox API or Token.{COLOR_RESET}")
        return
        
    trading_days = get_past_trading_days(target_date_str, num_days=BACKTRACE_DAYS)
    print(f"🔄 Backtracing structural memory across {BACKTRACE_DAYS} trading days...")

    # Load history into RAM
    historical_dfs = []
    for day in trading_days:
        day_list = []
        for item in universe:
            df = fetch_upstox_candles_for_date(item['key'], day)
            if df is not None and not df.empty:
                df['Symbol'] = item['symbol']
                df['Turnover'] = df['Volume'] * df['Close']
                df['abs_move'] = (df['Close'] - df['Open']).abs()
                day_list.append(df)
            # time.sleep(0.01) # Optional rate-limiting for Upstox
        if day_list:
            historical_dfs.append(pd.concat(day_list, ignore_index=True))

    if not historical_dfs:
        print(f"⚠️ {COLOR_RED}Fatal Error: No valid market data fetched across the entire rolling window.{COLOR_RESET}")
        return

    rolling_master_df = pd.concat(historical_dfs, ignore_index=True)
    
    current_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
    is_live_today = (target_date_str == current_now.strftime("%Y-%m-%d"))
    
    if not is_live_today or current_now.hour >= 16:
        target_dt = pd.to_datetime(target_date_str)
        eval_time_current = target_dt + pd.Timedelta(hours=15, minutes=15) 
    else:
        eval_time_current = current_now.replace(second=0, microsecond=0) - timedelta(minutes=1)

    # ----------------------------------------------------------------------
    # REBUILDING THE STATE MACHINE (Historical Fast-Forward)
    # ----------------------------------------------------------------------
    memory_bank = {}  # { sym: {state, origin, date, time, dir} }
    
    for day in trading_days:
        day_dt = pd.to_datetime(day)
        day_master = rolling_master_df[(rolling_master_df['Datetime'] >= day_dt) & (rolling_master_df['Datetime'] < day_dt + pd.Timedelta(days=1))]
        if day_master.empty: continue

        day_start = day_dt + pd.Timedelta(hours=9, minutes=15)
        # Prevent engine from scanning future hours if evaluating today
        day_end = day_dt + pd.Timedelta(hours=15, minutes=15)
        if day == target_date_str:
            day_end = eval_time_current

        # Scan 15-min intervals to reconstruct state history
        time_steps = pd.date_range(start=day_start + pd.Timedelta(minutes=15), end=day_end, freq='15min')
                                   
        for t in time_steps:
            anomalies = calculate_velocity_leaderboard(day_master, t, window_mins=15)
            for _, row in anomalies.iterrows():
                sym = row['Symbol']
                price = row['Close']
                direction = row['Direction']
                
                if sym not in memory_bank:
                    memory_bank[sym] = {
                        'state': 'ACTIVE', 'origin': price, 'date': day, 'time': t.strftime('%H:%M'), 'dir': direction
                    }
                else:
                    st = memory_bank[sym]
                    if st['state'] == 'BREACHED' and st['dir'] == direction:
                        # Reclaim Logic
                        if (st['dir'] == 1 and price > st['origin']) or (st['dir'] == -1 and price < st['origin']):
                            st['state'] = 'ACTIVE'
                            
        # End-of-Day Purge Protocol
        daily_agg = day_master.groupby('Symbol').agg({'Low': 'min', 'High': 'max', 'Close': 'last'}).reset_index()
        daily_dict = daily_agg.set_index('Symbol').to_dict('index')
        
        to_delete = []
        for sym, st in memory_bank.items():
            if sym not in daily_dict: continue
            d_low, d_high, d_close = daily_dict[sym]['Low'], daily_dict[sym]['High'], daily_dict[sym]['Close']
            
            if st['dir'] == 1: # Bullish
                if st['state'] == 'ACTIVE' and d_low < st['origin']:
                    st['state'] = 'BREACHED'
                if st['state'] == 'BREACHED' and d_close < st['origin'] * 0.99: # 1% structural failure
                    to_delete.append(sym)
            else: # Bearish
                if st['state'] == 'ACTIVE' and d_high > st['origin']:
                    st['state'] = 'BREACHED'
                if st['state'] == 'BREACHED' and d_close > st['origin'] * 1.01:
                    to_delete.append(sym)
                    
        for sym in to_delete:
            del memory_bank[sym]

    # ----------------------------------------------------------------------
    # LIVE EVALUATION (Current Minute / Target Time)
    # ----------------------------------------------------------------------
    target_dt_obj = pd.to_datetime(target_date_str)
    today_master = rolling_master_df[
        (rolling_master_df['Datetime'] >= target_dt_obj) & 
        (rolling_master_df['Datetime'] < target_dt_obj + pd.Timedelta(days=1))
    ].copy()

    # The Empty Market Fix (Holiday/Data outage)
    if today_master.empty: 
        print(f"\n{COLOR_YELLOW}[Terminal Standby] Market data for {target_date_str} is entirely empty. (Market Closed / Holiday / No Liquidity).{COLOR_RESET}\n")
        return

    # Pre-check 09:15 Gaps vs Active Anchors
    today_latest_ltp = today_master.groupby('Symbol')['Close'].last().to_dict()
    for sym, st in memory_bank.items():
        if sym in today_latest_ltp:
            ltp = today_latest_ltp[sym]
            if st['dir'] == 1 and ltp < st['origin']: st['state'] = 'BREACHED'
            elif st['dir'] == -1 and ltp > st['origin']: st['state'] = 'BREACHED'

    curr_anomalies = calculate_velocity_leaderboard(today_master, eval_time_current, window_mins=15)

    fresh_intrusions, reloads, reclaims, breached = [], [], [], []

    if not curr_anomalies.empty:
        for _, row in curr_anomalies.iterrows():
            sym = row['Symbol']
            if sym not in memory_bank:
                fresh_intrusions.append(row)
            else:
                st = memory_bank[sym]
                row['Origin'] = st['origin']
                row['First_Date'] = st['date']
                row['Net_Drift'] = ((row['Close'] - st['origin']) / st['origin']) * 100
                
                if st['state'] == 'ACTIVE' and row['Direction'] == st['dir']:
                    reloads.append(row)
                elif st['state'] == 'BREACHED' and row['Direction'] == st['dir']:
                    if (st['dir'] == 1 and row['Close'] > st['origin']) or (st['dir'] == -1 and row['Close'] < st['origin']):
                        st['state'] = 'ACTIVE' # Reclaimed today
                        reclaims.append(row)

    # Gather any remaining breached pivots for observation
    for sym, st in memory_bank.items():
        if st['state'] == 'BREACHED' and sym in today_latest_ltp:
            if not any(sym == r['Symbol'] for r in reclaims): 
                breached.append({'Symbol': sym, 'LTP': today_latest_ltp[sym], 'Origin': st['origin'], 'Dir': "BULLISH" if st['dir']==1 else "BEARISH"})

    # ----------------------------------------------------------------------
    # TERMINAL OUTPUT
    # ----------------------------------------------------------------------
    print(f"\n{COLOR_CYAN}================================================================================================{COLOR_RESET}")
    print(f"{COLOR_BOLD}FULL UNIVERSE TRI-DELTA TAPE | TIME: {eval_time_current.strftime('%H:%M')} IST | DATE: {target_date_str}{COLOR_RESET}")
    print(f"{COLOR_CYAN}================================================================================================{COLOR_RESET}\n")

    if fresh_intrusions:
        print(f"{COLOR_BOLD}⚡ FRESH INTRUSIONS (Phase 1 - Day-1 Births){COLOR_RESET}")
        for row in fresh_intrusions:
            sym, jump, ltp = row['Symbol'], row['Total_Score'], row['Close']
            color = COLOR_GREEN if jump > 0 else COLOR_RED
            d_str = "BULLISH" if jump > 0 else "BEARISH"
            print(f"  {color}🚨 {sym:<12} {jump:+.0f} pts [V:{row['V_Score']:+.0f} M:{row['M_Score']:+.0f} E:{row['E_Score']:+.0f}] ({d_str:<7}) | LTP: ₹{ltp:<8.2f}{COLOR_RESET}")
        print("")

    if reloads:
        print(f"{COLOR_BOLD}🔄 ALGORITHMIC RELOADS (Phase 2 - Second Waves){COLOR_RESET}")
        for row in reloads:
            sym, jump, ltp, drift = row['Symbol'], row['Total_Score'], row['Close'], row['Net_Drift']
            color = COLOR_GREEN if jump > 0 else COLOR_RED
            d_str = "BULLISH" if jump > 0 else "BEARISH"
            print(f"  {color}🔄 {sym:<12} {jump:+.0f} pts [V:{row['V_Score']:+.0f} M:{row['M_Score']:+.0f} E:{row['E_Score']:+.0f}] ({d_str:<7}) | LTP: ₹{ltp:<8.2f} --> [Anchor: {row['First_Date']} @ ₹{row['Origin']:.2f} | Drift: {drift:+.2f}%]{COLOR_RESET}")
        print("")

    if reclaims:
        print(f"{COLOR_BOLD}🪤 INSTITUTIONAL RECLAIMS (Phase 4 - Liquidity Traps / Squeezes){COLOR_RESET}")
        for row in reclaims:
            sym, jump, ltp = row['Symbol'], row['Total_Score'], row['Close']
            color = COLOR_MAGENTA
            d_str = "BULLISH" if jump > 0 else "BEARISH"
            print(f"  {color}🔥 {sym:<12} {jump:+.0f} pts [V:{row['V_Score']:+.0f} M:{row['M_Score']:+.0f} E:{row['E_Score']:+.0f}] ({d_str:<7}) | RECLAIMED ANCHOR: ₹{row['Origin']:.2f}{COLOR_RESET}")
        print("")

    if breached:
        print(f"{COLOR_DIM}⚠️ BREACHED PIVOTS (Phase 3 - Under Observation / Awaiting Resolution){COLOR_RESET}")
        for b in breached:
            print(f"  {COLOR_YELLOW}⚠️ {b['Symbol']:<12} Anchor: ₹{b['Origin']:.2f} ({b['Dir']}) | Currently trading at: ₹{b['LTP']:.2f} (Awaiting Reclaim or Daily Purge){COLOR_RESET}")
        print("")

    if not any([fresh_intrusions, reloads, reclaims, breached]):
        print(f"{COLOR_DIM}[Terminal Silent] No active institutional structure passing strict filters at this minute.{COLOR_RESET}\n")


# ==============================================================================
# 4. RUN EXECUTOR (With Weekend Rollover Defense)
# ==============================================================================
def run_production_sweep():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--date", type=str, default="")
    parser.add_argument("positional_date", nargs="?", default="")
    args, _ = parser.parse_known_args()

    raw_date_str = args.date or args.positional_date or os.environ.get("PARAM_BACKTEST_DATE", "").strip()
    is_backtest = bool(raw_date_str)

    if not is_backtest:
        # Live run: Get current date and handle weekends automatically
        target_dt = datetime.utcnow() + timedelta(hours=5, minutes=30)
        
        # Weekend Fallback Protocol
        if target_dt.weekday() == 5: # Saturday
            print(f"{COLOR_YELLOW}[System Notice] Market closed (Saturday). Auto-rolling back to Friday's closing tape.{COLOR_RESET}")
            target_dt -= timedelta(days=1)
        elif target_dt.weekday() == 6: # Sunday
            print(f"{COLOR_YELLOW}[System Notice] Market closed (Sunday). Auto-rolling back to Friday's closing tape.{COLOR_RESET}")
            target_dt -= timedelta(days=2)
            
        target_date_str = target_dt.strftime("%Y-%m-%d")
    else:
        # Backtest run: Assume user explicitly wants this date
        target_date_str = datetime.strptime(raw_date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
    
    if not os.environ.get("UPSTOX_ACCESS_TOKEN"):
        print(f"❌ {COLOR_RED}Error: UPSTOX_ACCESS_TOKEN not found in environment variables.{COLOR_RESET}")
        return

    scan_institutional_tape(target_date_str)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    run_production_sweep()

