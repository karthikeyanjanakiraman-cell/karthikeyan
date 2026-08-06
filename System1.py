import os
import argparse
import urllib.parse
import json
import gzip
import io
from datetime import datetime, timedelta

import requests
import pandas as pd

# ==============================================================================
# 1. LIVE INGESTION & +/- 5 STRIKES OPTIONS PACK
# ==============================================================================
def fetch_upstox_intraday_candles(instrument_key, target_date_str):
    access_token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not access_token:
        return None
    
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {access_token}'}
    # FORCE IST TIME
    today_str = (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d")
    
    if target_date_str == today_str:
        url = f"https://api.upstox.com/v2/historical-candle/intraday/{urllib.parse.quote(instrument_key)}/1minute"
    else:
        url = f"https://api.upstox.com/v2/historical-candle/{urllib.parse.quote(instrument_key)}/1minute/{target_date_str}/{target_date_str}"
    
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
    except Exception:
        return None

def get_options_universe(target_date_str):
    print(f"\n📡 Building Options Universe (+/- 5 Strikes) for {target_date_str}...")
    
    master_data = []
    for url in ["https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz", 
                "https://assets.upstox.com/market-quote/instruments/exchange/BSE.json.gz"]:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = json.load(gzip.GzipFile(fileobj=io.BytesIO(resp.content)))
                master_data.extend(data)
        except Exception as e:
            print(f"⚠️ Error fetching master JSON: {e}")
            
    df_inst = pd.DataFrame(master_data)
    if df_inst.empty:
        print("⚠️ Failed to load broker instrument list.")
        return []
        
    df_opt = df_inst[df_inst['instrument_type'] == 'OPTIDX'].copy()
    
    # Parse expiries 
    df_opt['expiry_dt'] = pd.to_datetime(df_opt['expiry'], unit='ms', errors='coerce')
    mask = df_opt['expiry_dt'].isna()
    if mask.any():
        df_opt.loc[mask, 'expiry_dt'] = pd.to_datetime(df_opt.loc[mask, 'expiry'], errors='coerce')
        
    indices_config = {
        "NIFTY": {"key": "NSE_INDEX|Nifty 50", "step": 50},
        "BANKNIFTY": {"key": "NSE_INDEX|Nifty Bank", "step": 100},
        "SENSEX": {"key": "BSE_INDEX|SENSEX", "step": 100}
    }
    
    target_dt = pd.to_datetime(target_date_str)
    final_universe = []
    
    for idx_name, info in indices_config.items():
        # Step 1: Get Spot Price to calculate ATM
        spot_df = fetch_upstox_intraday_candles(info["key"], target_date_str)
        if spot_df is None or spot_df.empty:
            print(f"⚠️ Warning: Could not fetch spot price for {idx_name}. Skipping index.")
            continue
            
        latest_spot = spot_df['Close'].iloc[-1]
        step = info["step"]
        
        # Step 2: Lock onto the ATM
        atm_strike = round(latest_spot / step) * step
        
        # Step 3: Build the 11-Strike Array (+5, ATM, -5)
        target_strikes = [atm_strike + (i * step) for i in range(-5, 6)]
        
        # Step 4: Filter master list for this index
        idx_opts = df_opt[df_opt['name'] == idx_name].copy()
        if idx_opts.empty: continue
        
        # Find closest upcoming expiry
        valid_expiries = idx_opts[idx_opts['expiry_dt'] >= target_dt]['expiry_dt'].unique()
        if len(valid_expiries) == 0:
            continue
        closest_expiry = min(valid_expiries)
        
        # Filter strictly by closest expiry and the 11 target strikes
        filtered_opts = idx_opts[(idx_opts['expiry_dt'] == closest_expiry) & (idx_opts['strike'].isin(target_strikes))]
        
        print(f"🔍 {idx_name} -> Spot: {latest_spot:.2f} | ATM: {atm_strike} | Found {len(filtered_opts)} CE/PE contracts.")
        
        for _, row in filtered_opts.iterrows():
            final_universe.append({
                "symbol": row['trading_symbol'],
                "key": row['instrument_key']
            })
            
    return final_universe

# ==============================================================================
# 2. CONFLUENCE LEADERS SCANNER (Options Only: Cumulative × Discrete)
# ==============================================================================
def scan_discrete_hourly_turnover(target_date_str):
    universe = get_options_universe(target_date_str)
    if not universe:
        print("⚠️ No valid Option universe could be built.")
        return
        
    print(f"\n⏳ Initializing Confluence Leaders Scanner for {len(universe)} Option Contracts...")
    
    target_dt = pd.to_datetime(target_date_str)
    start_of_day = target_dt + pd.Timedelta(hours=9, minutes=15)
    
    # STRICT IST CLOCK ENFORCEMENT
    current_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
    is_live_today = (target_date_str == current_now.strftime("%Y-%m-%d"))

    windows = [
        (target_dt + pd.Timedelta(hours=9, minutes=15), target_dt + pd.Timedelta(hours=10, minutes=15), "09:15 - 10:15"),
        (target_dt + pd.Timedelta(hours=10, minutes=15), target_dt + pd.Timedelta(hours=11, minutes=15), "10:15 - 11:15"),
        (target_dt + pd.Timedelta(hours=11, minutes=15), target_dt + pd.Timedelta(hours=12, minutes=15), "11:15 - 12:15"),
        (target_dt + pd.Timedelta(hours=12, minutes=15), target_dt + pd.Timedelta(hours=13, minutes=15), "12:15 - 13:15"),
        (target_dt + pd.Timedelta(hours=13, minutes=15), target_dt + pd.Timedelta(hours=14, minutes=15), "13:15 - 14:15"),
        (target_dt + pd.Timedelta(hours=14, minutes=15), target_dt + pd.Timedelta(hours=15, minutes=15), "14:15 - 15:15"),
        (target_dt + pd.Timedelta(hours=15, minutes=15), target_dt + pd.Timedelta(hours=15, minutes=30), "15:15 - 15:30")
    ]
    
    print(f"📡 Downloading intraday data for {len(universe)} option contracts...")
    
    master_intraday_list = []
    for item in universe:
        df = fetch_upstox_intraday_candles(item['key'], target_date_str)
        if df is not None and not df.empty:
            df['Symbol'] = item['symbol']
            df['Turnover'] = df['Volume'] * df['Close']
            df['abs_move'] = (df['Close'] - df['Open']).abs()
            master_intraday_list.append(df)
            
    if not master_intraday_list:
        print("⚠️ Warning: No valid intraday options data found.")
        return
        
    master_df = pd.concat(master_intraday_list, ignore_index=True)
    
    print("\n" + "="*155)
    print(f"🔥 TOP 5 INSTITUTIONAL OPTION STRIKES (Best in Both: Cumulative × Discrete) | DATE: {target_date_str}")
    print("="*155)
    
    for start_time, end_time, base_label in windows:
        if is_live_today and current_now < start_time:
            break
            
        is_active_live = False
        label = base_label
        
        if is_live_today and start_time <= current_now < end_time:
            is_active_live = True
            end_time = current_now
            label = f"{start_time.strftime('%H:%M')} - {current_now.strftime('%H:%M')} (LIVE ONGOING)"
            
        # 1. ISOLATED DISCRETE WINDOW DATA
        df_discrete = master_df[(master_df['Datetime'] >= start_time) & (master_df['Datetime'] < end_time)]
        # 2. CUMULATIVE SESSION DATA (From 09:15 up to current window end)
        df_cumulative = master_df[(master_df['Datetime'] >= start_of_day) & (master_df['Datetime'] < end_time)]
        
        if df_discrete.empty or df_cumulative.empty: 
            if is_active_live: break
            continue
            
        # --- COMPUTE DISCRETE SCORES ---
        grouped_disc = df_discrete.groupby('Symbol').agg({
            'Turnover': 'sum', 'Volume': 'sum', 'Open': 'first', 'Close': 'last', 'abs_move': 'sum'
        }).reset_index()
        # Options must have volume to rank
        grouped_disc = grouped_disc[grouped_disc['Turnover'] > 0]
        if grouped_disc.empty: 
            if is_active_live: break
            continue
            
        grouped_disc['Turnover_PR'] = grouped_disc['Turnover'].rank(pct=True) * 100
        grouped_disc['Pct_Move'] = ((grouped_disc['Close'] - grouped_disc['Open']) / grouped_disc['Open']) * 100
        grouped_disc['Momentum_PR'] = grouped_disc['Pct_Move'].abs().rank(pct=True) * 100
        grouped_disc['Net_Disp'] = (grouped_disc['Close'] - grouped_disc['Open']).abs()
        grouped_disc['Efficiency'] = grouped_disc['Net_Disp'] / (grouped_disc['abs_move'] + 1e-8)
        grouped_disc['Hurst_PR'] = grouped_disc['Efficiency'].rank(pct=True) * 100
        grouped_disc['Discrete_Power'] = (grouped_disc['Turnover_PR'] * grouped_disc['Momentum_PR'] * grouped_disc['Hurst_PR']) / 100.0

        # --- COMPUTE CUMULATIVE SCORES ---
        grouped_cum = df_cumulative.groupby('Symbol').agg({
            'Turnover': 'sum', 'Volume': 'sum', 'Open': 'first', 'Close': 'last', 'abs_move': 'sum'
        }).reset_index()
        grouped_cum = grouped_cum[grouped_cum['Turnover'] > 0]
        
        grouped_cum['Cum_Turnover_PR'] = grouped_cum['Turnover'].rank(pct=True) * 100
        grouped_cum['Cum_Pct_Move'] = ((grouped_cum['Close'] - grouped_cum['Open']) / grouped_cum['Open']) * 100
        grouped_cum['Cum_Momentum_PR'] = grouped_cum['Cum_Pct_Move'].abs().rank(pct=True) * 100
        grouped_cum['Cum_Net_Disp'] = (grouped_cum['Close'] - grouped_cum['Open']).abs()
        grouped_cum['Cum_Efficiency'] = grouped_cum['Cum_Net_Disp'] / (grouped_cum['abs_move'] + 1e-8)
        grouped_cum['Cum_Hurst_PR'] = grouped_cum['Cum_Efficiency'].rank(pct=True) * 100
        grouped_cum['Cumulative_Power'] = (grouped_cum['Cum_Turnover_PR'] * grouped_cum['Cum_Momentum_PR'] * grouped_cum['Cum_Hurst_PR']) / 100.0

        # --- MERGE AND CALCULATE COMBINED POWER ---
        merged = pd.merge(grouped_disc[['Symbol', 'Discrete_Power', 'Turnover_PR', 'Momentum_PR', 'Hurst_PR', 'Pct_Move', 'Close']], 
                          grouped_cum[['Symbol', 'Cumulative_Power']], 
                          on='Symbol', how='inner')
        
        # Strict Agreement: Multiply them to naturally filter for strikes high in BOTH timeframes
        merged['Combined_Power'] = (merged['Discrete_Power'] * merged['Cumulative_Power']) / 10000.0
        
        # Select Top 5 by highest Combined Power overall for the hour
        top5 = merged.nlargest(5, 'Combined_Power')
        
        print(f"\n⏰ DISCRETE WINDOW: {label} IST")
        print(f"{'Rank':<5} {'Symbol':<22} {'Combined Pwr':<12} | {'Disc Pwr':<10} | {'Cum Pwr':<9} | {'Turnover':<11} | {'Momentum':<11} | {'Hurst':<9} | {'% Move':<8} {'LTP (₹)':<10}")
        print("-" * 155)
        
        for rank, (_, row) in enumerate(top5.iterrows(), 1):
            move_sign = "+" if row['Pct_Move'] > 0 else ""
            print(f"{rank:<5} {row['Symbol']:<22} {row['Combined_Power']:<12.1f} | {row['Discrete_Power']:>8.1f}   | {row['Cumulative_Power']:>7.1f}   | {row['Turnover_PR']:>7.2f} PR | {row['Momentum_PR']:>7.2f} PR | {row['Hurst_PR']:>5.2f} PR | {move_sign}{row['Pct_Move']:<6.2f}%   ₹{row['Close']:<10.2f}")

        if is_active_live:
            break

# ==============================================================================
# 3. MAIN CONTROLLER
# ==============================================================================
def run_production_sweep():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--date", type=str, default="")
    parser.add_argument("positional_date", nargs="?", default="")
    args, _ = parser.parse_known_args()

    raw_date_str = args.date or args.positional_date or os.environ.get("PARAM_BACKTEST_DATE", "").strip()
    is_backtest = bool(raw_date_str)

    if not is_backtest:
        target_date_str = (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d")
    else:
        try:
            target_date_str = datetime.strptime(raw_date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            print(f"❌ Critical Error: Date '{raw_date_str}' is invalid. Use YYYY-MM-DD.")
            return

    print(f"⚙️ METRICS DATE ACTIVE: {target_date_str} | MODE: {'BACKTEST' if is_backtest else 'LIVE'}")
    
    scan_discrete_hourly_turnover(target_date_str)

if __name__ == "__main__":
    run_production_sweep()

