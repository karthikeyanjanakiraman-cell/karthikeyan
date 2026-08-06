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
        print("⚠️ CRITICAL: UPSTOX_ACCESS_TOKEN not found in environment!")
        return None
    
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {access_token}'}
    
    current_now_ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
    today_str = current_now_ist.strftime("%Y-%m-%d")
    
    # If the target date is strictly today AND the market is open, use intraday live endpoint
    if target_date_str == today_str and current_now_ist.hour >= 9:
        url = f"https://api.upstox.com/v2/historical-candle/intraday/{urllib.parse.quote(instrument_key)}/1minute"
    else:
        # It's a historical date (or yesterday). Pull a 3-day range to bypass the Upstox boundary bug.
        target_dt = pd.to_datetime(target_date_str)
        from_date_str = (target_dt - timedelta(days=3)).strftime("%Y-%m-%d")
        url = f"https://api.upstox.com/v2/historical-candle/{urllib.parse.quote(instrument_key)}/1minute/{target_date_str}/{from_date_str}"
        
    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code != 200:
            print(f"⚠️ API Error {response.status_code} for {instrument_key}")
            return None
            
        data = response.json().get('data', {}).get('candles', [])
        if not data:
            return None
            
        c_df = pd.DataFrame(data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
        c_df['Datetime'] = pd.to_datetime(c_df['Timestamp']).dt.tz_localize(None) 
        c_df = c_df.sort_values('Datetime').reset_index(drop=True)
        
        # Isolate exactly the date requested so previous days don't bleed into the scanner
        c_df = c_df[c_df['Datetime'].dt.strftime('%Y-%m-%d') == target_date_str].reset_index(drop=True)
        
        return c_df
    except Exception as e:
        print(f"⚠️ Exception fetching data for {instrument_key}: {e}")
        return None

def get_options_universe(target_date_str):
    print(f"\n📡 Building Options Universe (+/- 5 Strikes) from Broker API...")
    
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
        
    # Safely detect the exact column name Upstox is using for strike prices
    strike_col = 'strike_price' if 'strike_price' in df_inst.columns else 'strike' if 'strike' in df_inst.columns else None
    if strike_col:
        df_inst[strike_col] = pd.to_numeric(df_inst[strike_col], errors='coerce')
    else:
        print("⚠️ Critical API Change: Strike price column missing from broker JSON.")
        return []
        
    # Safely detect trading symbol column
    ts_col = 'trading_symbol' if 'trading_symbol' in df_inst.columns else 'tradingsymbol' if 'tradingsymbol' in df_inst.columns else None
    if not ts_col:
        print("⚠️ Critical API Change: Trading symbol column missing from broker JSON.")
        return []
    
    # Parse expiries safely
    if 'expiry' in df_inst.columns:
        df_inst['expiry_dt'] = pd.to_datetime(df_inst['expiry'], unit='ms', errors='coerce')
        mask = df_inst['expiry_dt'].isna()
        if mask.any():
            df_inst.loc[mask, 'expiry_dt'] = pd.to_datetime(df_inst.loc[mask, 'expiry'], errors='coerce')
    else:
        print("⚠️ Critical API Change: 'expiry' column missing from broker JSON.")
        return []

    indices_config = {
        "NIFTY": {"key": "NSE_INDEX|Nifty 50", "step": 50},
        "BANKNIFTY": {"key": "NSE_INDEX|Nifty Bank", "step": 100},
        "SENSEX": {"key": "BSE_INDEX|SENSEX", "step": 100}
    }
    
    final_universe = []
    
    for idx_name, info in indices_config.items():
        # Step 1: Spot Price Check
        spot_df = fetch_upstox_intraday_candles(info["key"], target_date_str)
        if spot_df is None or spot_df.empty:
            print(f"⚠️ Warning: Could not fetch Spot price for {idx_name} on {target_date_str}. Skipping.")
            continue
            
        latest_spot = spot_df['Close'].iloc[-1]
        step = info["step"]
        
        # Step 2: ATM Calculation
        atm_strike = round(latest_spot / step) * step
        target_strikes = [atm_strike + (i * step) for i in range(-5, 6)]
        
        # Step 3: Match the Symbol securely
        match_condition = df_inst[ts_col].astype(str).str.upper().str.startswith(idx_name)
        idx_opts = df_inst[match_condition].copy()
        
        # Mathematically guarantees we only pull options (equities/futures have NaN or 0 strikes)
        idx_opts = idx_opts[idx_opts[strike_col] > 0]
        
        if idx_opts.empty:
            print(f"⚠️ Warning: Found 0 option contracts starting with '{idx_name}'. Skipping.")
            continue
        
        # Step 4: Expiry Matching
        valid_expiries = idx_opts['expiry_dt'].dropna().unique()
        if not len(valid_expiries):
            print(f"⚠️ Warning: Found {idx_name} options, but 0 valid expiries parsed. Skipping.")
            continue
            
        closest_expiry = min(valid_expiries)
        
        # Step 5: Strike Matching
        filtered_opts = idx_opts[(idx_opts['expiry_dt'] == closest_expiry) & (idx_opts[strike_col].isin(target_strikes))]
        
        if filtered_opts.empty:
            print(f"⚠️ Warning: Found {idx_name} options, but 0 matched our 11 strikes for {closest_expiry.strftime('%Y-%m-%d')}.")
            continue
            
        print(f"🔍 {idx_name} -> Spot: {latest_spot:.2f} | ATM: {atm_strike} | Found {len(filtered_opts)} CE/PE contracts for {closest_expiry.strftime('%Y-%m-%d')}.")
        
        for _, row in filtered_opts.iterrows():
            final_universe.append({
                "symbol": row[ts_col],
                "key": row['instrument_key']
            })
            
    return final_universe

# ==============================================================================
# 2. CONFLUENCE LEADERS SCANNER (Options Only: Cumulative × Discrete)
# ==============================================================================
def scan_discrete_hourly_turnover(target_date_str):
    universe = get_options_universe(target_date_str)
    if not universe:
        print("\n⚠️ SYSTEM HALT: No option contracts survived the filter. Exiting scanner.")
        return
        
    print(f"\n⏳ Downloading intraday data for {len(universe)} option contracts...")
    
    master_intraday_list = []
    for item in universe:
        df = fetch_upstox_intraday_candles(item['key'], target_date_str)
        if df is not None and not df.empty:
            df['Symbol'] = item['symbol']
            df['Turnover'] = df['Volume'] * df['Close']
            df['abs_move'] = (df['Close'] - df['Open']).abs()
            master_intraday_list.append(df)
            
    if not master_intraday_list:
        print("⚠️ Warning: Pulled the contracts, but Upstox returned 0 intraday volume for them.")
        return
        
    master_df = pd.concat(master_intraday_list, ignore_index=True)

    real_market_date = master_df['Datetime'].max().normalize()
    start_of_day = real_market_date + pd.Timedelta(hours=9, minutes=15)
    
    current_now_ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
    is_live_today = (real_market_date.date() == current_now_ist.date())

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
    print(f"🔥 TOP 5 INSTITUTIONAL OPTION STRIKES (Best in Both: Cumulative × Discrete) | MARKET DATE: {real_market_date.strftime('%Y-%m-%d')}")
    print("="*155)
    
    for start_time, end_time, base_label in windows:
        if is_live_today and current_now_ist < start_time:
            break
            
        is_active_live = False
        label = base_label
        
        if is_live_today and start_time <= current_now_ist < end_time:
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
        
        merged['Combined_Power'] = (merged['Discrete_Power'] * merged['Cumulative_Power']) / 10000.0
        top5 = merged.nlargest(5, 'Combined_Power')
        
        print(f"\n⏰ DISCRETE WINDOW: {label} IST")
        print(f"{'Rank':<5} {'Symbol':<26} {'Combined Pwr':<12} | {'Disc Pwr':<10} | {'Cum Pwr':<9} | {'Turnover':<11} | {'Momentum':<11} | {'Hurst':<9} | {'% Move':<8} {'LTP (₹)':<10}")
        print("-" * 155)
        
        for rank, (_, row) in enumerate(top5.iterrows(), 1):
            move_sign = "+" if row['Pct_Move'] > 0 else ""
            print(f"{rank:<5} {row['Symbol']:<26} {row['Combined_Power']:<12.1f} | {row['Discrete_Power']:>8.1f}   | {row['Cumulative_Power']:>7.1f}   | {row['Turnover_PR']:>7.2f} PR | {row['Momentum_PR']:>7.2f} PR | {row['Hurst_PR']:>5.2f} PR | {move_sign}{row['Pct_Move']:<6.2f}%   ₹{row['Close']:<10.2f}")

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

    current_now_ist = datetime.utcnow() + timedelta(hours=5, minutes=30)

    if not is_backtest:
        # MIDNIGHT BUG FIX: If it is before 9:15 AM IST, default to the previous trading day.
        if current_now_ist.hour < 9 or (current_now_ist.hour == 9 and current_now_ist.minute < 15):
            target_dt = current_now_ist - timedelta(days=1)
            while target_dt.weekday() > 4:  # Skip Sat (5) and Sun (6)
                target_dt -= timedelta(days=1)
            target_date_str = target_dt.strftime("%Y-%m-%d")
        else:
            target_date_str = current_now_ist.strftime("%Y-%m-%d")
    else:
        try:
            target_date_str = datetime.strptime(raw_date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            print(f"❌ Critical Error: Date '{raw_date_str}' is invalid. Use YYYY-MM-DD.")
            return

    print(f"⚙️ METRICS TARGET DATE: {target_date_str} | MODE: {'BACKTEST' if is_backtest else 'LIVE'}")
    
    scan_discrete_hourly_turnover(target_date_str)

if __name__ == "__main__":
    run_production_sweep()
