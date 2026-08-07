import os
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
def fetch_live_intraday_candles(instrument_key):
    """Fetches purely the live intraday candles for today."""
    access_token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not access_token:
        print("⚠️ CRITICAL: UPSTOX_ACCESS_TOKEN not found in environment!")
        return None
    
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {access_token}'}
    url = f"https://api.upstox.com/v2/historical-candle/intraday/{urllib.parse.quote(instrument_key)}/1minute"
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code != 200:
            return None
            
        data = response.json().get('data', {}).get('candles', [])
        if not data:
            return None
            
        c_df = pd.DataFrame(data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
        c_df['Datetime'] = pd.to_datetime(c_df['Timestamp']).apply(lambda x: x.replace(tzinfo=None))
        c_df = c_df.sort_values('Datetime').reset_index(drop=True)
        return c_df
    except Exception:
        return None

def get_live_options_universe():
    print(f"📡 Downloading Live Exchange Master Files from Broker...")
    
    master_data = []
    for url in ["https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz", 
                "https://assets.upstox.com/market-quote/instruments/exchange/BSE.json.gz"]:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                master_data.extend(json.load(gzip.GzipFile(fileobj=io.BytesIO(resp.content))))
        except Exception as e:
            print(f"⚠️ Error fetching master JSON: {e}")
            
    df_inst = pd.DataFrame(master_data)
    if df_inst.empty:
        print("⚠️ Failed to load broker instrument list.")
        return []
        
    # Dynamically handle Upstox changing column names silently
    strike_col = 'strike_price' if 'strike_price' in df_inst.columns else 'strike'
    ts_col = 'trading_symbol' if 'trading_symbol' in df_inst.columns else 'tradingsymbol'
    
    df_inst[strike_col] = pd.to_numeric(df_inst[strike_col], errors='coerce')
    if 'expiry' in df_inst.columns:
        df_inst['expiry_dt'] = pd.to_datetime(df_inst['expiry'], unit='ms', errors='coerce')
    
    indices_config = {
        "NIFTY": {"key": "NSE_INDEX|Nifty 50", "step": 50},
        "BANKNIFTY": {"key": "NSE_INDEX|Nifty Bank", "step": 100},
        "SENSEX": {"key": "BSE_INDEX|SENSEX", "step": 100}
    }
    
    final_universe = []
    
    for idx_name, info in indices_config.items():
        # Step 1: Get LIVE Spot Price
        spot_df = fetch_live_intraday_candles(info["key"])
        if spot_df is None or spot_df.empty:
            print(f"⚠️ Warning: Could not fetch LIVE Spot price for {idx_name}. Market may not be open yet.")
            continue
            
        latest_spot = spot_df['Close'].iloc[-1]
        step = info["step"]
        
        # Step 2: ATM Calculation
        atm_strike = round(latest_spot / step) * step
        target_strikes = [atm_strike + (i * step) for i in range(-5, 6)]
        
        # Step 3: Parse Expiries & Build Universe
        # We strictly rely on the ticker string prefix and math (strike > 0) to avoid broker tags
        match_condition = df_inst[ts_col].astype(str).str.upper().str.startswith(idx_name)
        idx_opts = df_inst[match_condition & (df_inst[strike_col] > 0)].copy()
        
        if idx_opts.empty:
            continue
            
        # The broker automatically deletes expired contracts, so the absolute minimum date is the current active weekly
        valid_expiries = idx_opts['expiry_dt'].dropna().unique()
        if not len(valid_expiries): 
            continue
            
        closest_expiry = min(valid_expiries)
        
        filtered_opts = idx_opts[(idx_opts['expiry_dt'] == closest_expiry) & (idx_opts[strike_col].isin(target_strikes))]
        print(f"✅ {idx_name:<10} | Spot: {latest_spot:<8.2f} | ATM Locked: {atm_strike:<6} | Mapped {len(filtered_opts)} LIVE Contracts.")
        
        for _, row in filtered_opts.iterrows():
            final_universe.append({"symbol": row[ts_col], "key": row['instrument_key']})
            
    return final_universe

# ==============================================================================
# 2. CONFLUENCE LEADERS SCANNER (Cumulative × Discrete)
# ==============================================================================
def run_live_scanner():
    current_now_ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
    
    print("\n" + "="*80)
    print(f"🚀 INITIATING LIVE MARKET SCANNER | TIME: {current_now_ist.strftime('%I:%M %p IST')}")
    print("="*80)

    if current_now_ist.hour < 9 or (current_now_ist.hour == 9 and current_now_ist.minute < 15):
        print("\n⏳ MARKET IS CLOSED/PRE-OPEN. Please run this script after 09:15 AM IST.")
        return

    universe = get_live_options_universe()
    if not universe:
        print("\n⚠️ SYSTEM HALT: No options mapped. Exiting scanner.")
        return
        
    print(f"\n⚡ Downloading live volume flow for {len(universe)} contracts...")
    
    master_intraday_list = []
    for item in universe:
        df = fetch_live_intraday_candles(item['key'])
        if df is not None and not df.empty:
            df['Symbol'] = item['symbol']
            df['Turnover'] = df['Volume'] * df['Close']
            df['abs_move'] = (df['Close'] - df['Open']).abs()
            master_intraday_list.append(df)
            
    if not master_intraday_list:
        print("⚠️ Waiting on broker volume... Try again in 1 minute.")
        return
        
    master_df = pd.concat(master_intraday_list, ignore_index=True)
    real_market_date = master_df['Datetime'].max().normalize()
    start_of_day = real_market_date + pd.Timedelta(hours=9, minutes=15)
    
    windows = [
        (real_market_date + pd.Timedelta(hours=9, minutes=15), real_market_date + pd.Timedelta(hours=10, minutes=15), "09:15 - 10:15"),
        (real_market_date + pd.Timedelta(hours=10, minutes=15), real_market_date + pd.Timedelta(hours=11, minutes=15), "10:15 - 11:15"),
        (real_market_date + pd.Timedelta(hours=11, minutes=15), real_market_date + pd.Timedelta(hours=12, minutes=15), "11:15 - 12:15"),
        (real_market_date + pd.Timedelta(hours=12, minutes=15), real_market_date + pd.Timedelta(hours=13, minutes=15), "12:15 - 13:15"),
        (real_market_date + pd.Timedelta(hours=13, minutes=15), real_market_date + pd.Timedelta(hours=14, minutes=15), "13:15 - 14:15"),
        (real_market_date + pd.Timedelta(hours=14, minutes=15), real_market_date + pd.Timedelta(hours=15, minutes=15), "14:15 - 15:15"),
        (real_market_date + pd.Timedelta(hours=15, minutes=15), real_market_date + pd.Timedelta(hours=15, minutes=30), "15:15 - 15:30")
    ]
    
    print("\n" + "="*155)
    print(f"🔥 TOP 5 INSTITUTIONAL OPTION STRIKES (Cumulative × Discrete) | DATE: {real_market_date.strftime('%Y-%m-%d')}")
    print("="*155)
    
    for start_time, end_time, base_label in windows:
        if current_now_ist < start_time:
            break
            
        is_active_live = False
        label = base_label
        
        if start_time <= current_now_ist < end_time:
            is_active_live = True
            end_time = current_now_ist
            label = f"{start_time.strftime('%H:%M')} - {current_now_ist.strftime('%H:%M')} (LIVE ONGOING)"
            
        df_discrete = master_df[(master_df['Datetime'] >= start_time) & (master_df['Datetime'] < end_time)]
        df_cumulative = master_df[(master_df['Datetime'] >= start_of_day) & (master_df['Datetime'] < end_time)]
        
        if df_discrete.empty or df_cumulative.empty: 
            if is_active_live: break
            continue
            
        grouped_disc = df_discrete.groupby('Symbol').agg({
            'Turnover': 'sum', 'Volume': 'sum', 'Open': 'first', 'Close': 'last', 'abs_move': 'sum'
        }).reset_index()
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

        grouped_cum = df_cumulative.groupby('Symbol').agg({
            'Turnover': 'sum', 'Volume': 'sum', 'Open': 'first', 'Close': 'last', 'abs_move': 'sum'
        }).reset_index()
        grouped_cum = grouped_cum[grouped_cum['Turnover'] > 0]
        
        if grouped_cum.empty: continue
            
        grouped_cum['Cum_Turnover_PR'] = grouped_cum['Turnover'].rank(pct=True) * 100
        grouped_cum['Cum_Pct_Move'] = ((grouped_cum['Close'] - grouped_cum['Open']) / grouped_cum['Open']) * 100
        grouped_cum['Cum_Momentum_PR'] = grouped_cum['Cum_Pct_Move'].abs().rank(pct=True) * 100
        grouped_cum['Cum_Net_Disp'] = (grouped_cum['Close'] - grouped_cum['Open']).abs()
        grouped_cum['Cum_Efficiency'] = grouped_cum['Cum_Net_Disp'] / (grouped_cum['abs_move'] + 1e-8)
        grouped_cum['Cum_Hurst_PR'] = grouped_cum['Cum_Efficiency'].rank(pct=True) * 100
        grouped_cum['Cumulative_Power'] = (grouped_cum['Cum_Turnover_PR'] * grouped_cum['Cum_Momentum_PR'] * grouped_cum['Cum_Hurst_PR']) / 100.0

        merged = pd.merge(grouped_disc[['Symbol', 'Discrete_Power', 'Turnover_PR', 'Momentum_PR', 'Hurst_PR', 'Pct_Move', 'Close']], 
                          grouped_cum[['Symbol', 'Cumulative_Power']], 
                          on='Symbol', how='inner')
        
        if merged.empty: continue
            
        merged['Combined_Power'] = (merged['Discrete_Power'] * merged['Cumulative_Power']) / 10000.0
        top5 = merged.nlargest(5, 'Combined_Power')
        
        print(f"\n⏰ {label} IST")
        print(f"{'Rank':<5} {'Symbol':<26} {'Score':<10} | {'Disc Pwr':<9} | {'Cum Pwr':<8} | {'Turnover':<11} | {'Momentum':<11} | {'Hurst':<9} | {'% Move':<8} {'LTP (₹)':<10}")
        print("-" * 150)
        
        for rank, (_, row) in enumerate(top5.iterrows(), 1):
            move_sign = "+" if row['Pct_Move'] > 0 else ""
            print(f"{rank:<5} {row['Symbol']:<26} {row['Combined_Power']:<10.1f} | {row['Discrete_Power']:>7.1f}   | {row['Cumulative_Power']:>6.1f}   | {row['Turnover_PR']:>7.2f} PR | {row['Momentum_PR']:>7.2f} PR | {row['Hurst_PR']:>5.2f} PR | {move_sign}{row['Pct_Move']:<6.2f}%   ₹{row['Close']:<10.2f}")

        if is_active_live:
            break

if __name__ == "__main__":
    run_live_scanner()
