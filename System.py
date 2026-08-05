import os
import requests
import pandas as pd
import numpy as np
import urllib.parse
import json
import gzip
import io
import time
from datetime import datetime, timedelta

# ==============================================================================
# 1. HURST EXPONENT (Directional Persistence / One-Way Volatility Filter)
# ==============================================================================
def calculate_hurst(price_series):
    try:
        prices = np.array(price_series, dtype=np.float32)
        if len(prices) < 15:
            return 0.5
        lags = range(2, min(len(prices) // 2, 20))
        tau = [np.std(prices[lag:] - prices[:-lag]) for lag in lags]
        lags_arr = np.array(list(lags))
        tau_arr = np.array(tau)
        valid = tau_arr > 0
        if np.sum(valid) < 2:
            return 0.5
        poly = np.polyfit(np.log(lags_arr[valid]), np.log(tau_arr[valid]), 1)
        return float(poly[0] * 2.0)
    except:
        return 0.5

# ==============================================================================
# 2. HISTORICAL 5-YEAR SCANNER: Long-Only Pristine Breakouts
# ==============================================================================
def scan_historical_pristine_breakouts(csv_filename="historical_fno.csv", min_pct=5.0, max_drawdown=0.5, max_gap=0.2):
    """
    Scans the 5-year historical daily database for rare, trap-free LONG breakouts:
    - Minimum 2-day net positive move >= +5.0%
    - Zero overnight gaps (Open matches prior Close within max_gap %)
    - Zero mean reversion / drawdown during the move (Lows do not breach baseline)
    """
    if not os.path.exists(csv_filename):
        print(f"❌ Error: '{csv_filename}' not found. Run the downloader script first.")
        return

    print(f"\n⏳ Scanning historical F&O database for zero-gap, zero-drawdown LONG breakouts (>= +{min_pct}% in 2 days)...")
    df = pd.read_csv(csv_filename)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)

    pristine_breakouts = []

    for symbol, group in df.groupby('Symbol'):
        group = group.reset_index(drop=True)
        if len(group) < 3:
            continue

        dates = group['Date'].values
        opens = group['Open'].values
        highs = group['High'].values
        lows = group['Low'].values
        closes = group['Close'].values

        for i in range(len(group) - 2):
            t0_close = closes[i]
            t1_open, t1_close, t1_low = opens[i+1], closes[i+1], lows[i+1]
            t2_open, t2_close, t2_low, t2_high = opens[i+2], closes[i+2], lows[i+2], highs[i+2]

            # 1. Check for Zero Gaps (Open must match prior close within max_gap %)
            gap_t1 = abs((t1_open - t0_close) / t0_close) * 100
            gap_t2 = abs((t2_open - t1_close) / t1_close) * 100
            if gap_t1 > max_gap or gap_t2 > max_gap:
                continue

            # 2. Calculate 2-Day Net Move (Strictly Positive for Long Breakouts)
            net_move = ((t2_close - t0_close) / t0_close) * 100
            if net_move < min_pct:
                continue

            # 3. Check for Zero Drawdown / Mean Reversion (Lows cannot drop below baseline)
            drawdown_t1 = ((t1_low - t0_close) / t0_close) * 100
            drawdown_t2 = ((t2_low - t1_close) / t1_close) * 100
            if drawdown_t1 < -max_drawdown or drawdown_t2 < -max_drawdown:
                continue

            pristine_breakouts.append({
                'Symbol': symbol,
                'Breakout_Date': pd.to_datetime(dates[i+2]).strftime('%Y-%m-%d'),
                'Net_Move_%': round(net_move, 2),
                'Base_Price': round(t0_close, 2),
                'Target_Price': round(t2_close, 2)
            })

    result_df = pd.DataFrame(pristine_breakouts)
    if not result_df.empty:
        print(f"\n🎉 Found {len(result_df)} pristine, gap-free, zero-drawdown LONG breakout instances across history!")
        print(result_df.head(15).to_string(index=False))
        result_df.to_csv("pristine_long_breakouts_catalog.csv", index=False)
    else:
        print("⚠️ No long instances matched the strict criteria.")

# ==============================================================================
# 3. LIVE CUMULATIVE PRE-BREAKOUT COMPRESSION SCANNER (Intraday)
# ==============================================================================
def get_dynamic_fno_universe():
    nse_url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
    response = requests.get(nse_url)
    if response.status_code != 200:
        return []
    try:
        nse_data = json.load(gzip.GzipFile(fileobj=io.BytesIO(response.content)))
        fno_underlying = {item.get("underlying_symbol") for item in nse_data if item.get("segment") == "NSE_FO" and item.get("underlying_symbol")}
        return [{"symbol": item.get("trading_symbol"), "key": item.get("instrument_key")} for item in nse_data if item.get("segment") in ("NSE_EQ", "NSE_INDEX") and item.get("trading_symbol") in fno_underlying]
    except:
        return []

def fetch_upstox_intraday_candles(instrument_key, target_date_str):
    access_token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not access_token:
        return None
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {access_token}'}
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    if target_date_str == today_str:
        url = f"https://api.upstox.com/v2/historical-candle/intraday/{urllib.parse.quote(instrument_key)}/1minute"
    else:
        target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
        next_date_str = (target_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        url = f"https://api.upstox.com/v2/historical-candle/{urllib.parse.quote(instrument_key)}/1minute/{next_date_str}/{target_date_str}"
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        # Diagnostic print for debugging API token/limit issues
        print(f"⚠️ API Error [{response.status_code}] for {instrument_key}: {response.text}")
        return None
        
    data = response.json().get('data', {}).get('candles', [])
    if not data:
        return None
    c_df = pd.DataFrame(data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
    c_df['Datetime'] = pd.to_datetime(c_df['Timestamp']).dt.tz_localize(None)
    return c_df.sort_values('Datetime').reset_index(drop=True)

def scan_live_cumulative_compression(target_date_str):
    access_token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not access_token:
        print("\n⚠️ Skipping Live Intraday Scan: 'UPSTOX_ACCESS_TOKEN' environment variable is missing or not set.")
        print("💡 Tip: Set your token in your terminal using: export UPSTOX_ACCESS_TOKEN='your_token_here'")
        return

    print(f"\n🔍 Scanning session for cumulative pre-breakout compression on {target_date_str}...")
    universe = get_dynamic_fno_universe()
    if not universe:
        print("⚠️ No F&O universe found.")
        return

    records = []
    target_dt = pd.to_datetime(target_date_str)
    session_open = target_dt + pd.Timedelta(hours=9, minutes=15)
    current_checkpoint = datetime.now() if target_date_str == datetime.now().strftime("%Y-%m-%d") else target_dt + pd.Timedelta(hours=15, minutes=30)

    for item in universe:
        df = fetch_upstox_intraday_candles(item['key'], target_date_str)
        if df is None or df.empty:
            continue

        sub_df = df[(df['Datetime'] >= session_open) & (df['Datetime'] <= current_checkpoint)]
        if len(sub_df) < 15:
            continue

        sub_df['Turnover'] = sub_df['Volume'] * sub_df['Close']
        cum_turnover = sub_df['Turnover'].sum()
        open_val = sub_df['Open'].iloc[0]
        close_val = sub_df['Close'].iloc[-1]
        pct_move = ((close_val - open_val) / open_val) * 100
        hurst = calculate_hurst(sub_df['Close'].values)

        records.append({
            'Symbol': item['symbol'],
            'Cum_Turnover': cum_turnover,
            'Pct_Move': pct_move,
            'Abs_Move': abs(pct_move),
            'Hurst': hurst,
            'LTP': close_val
        })
        time.sleep(0.1)

    if not records:
        print("❌ No intraday records collected (Market might be closed, token invalid, or date format rejected).")
        return

    master = pd.DataFrame(records)
    master['Turnover_PR'] = master['Cum_Turnover'].rank(pct=True) * 100
    master['Momentum_PR'] = master['Abs_Move'].rank(pct=True) * 100
    master['Hurst_PR'] = master['Hurst'].rank(pct=True) * 100
    master['Power_Score'] = master['Turnover_PR'] * master['Momentum_PR'] * (master['Hurst_PR'] / 100.0)

    top_picks = master.sort_values(by='Power_Score', ascending=False).head(5)
    print("\n" + "="*100)
    print(f"🔥 TOP CUMULATIVE PRE-BREAKOUT COMPRESSION PICKS | DATE: {target_date_str}")
    print("="*100)
    print(f"{'Symbol':<15} {'Power Score':<12} | {'Turnover PR':<12} | {'Momentum PR':<12} | {'Hurst PR':<10} | {'% Move':<8} {'LTP':<10}")
    print("-" * 100)
    for _, row in top_picks.iterrows():
        move_str = f"+{row['Pct_Move']:.2f}%" if row['Pct_Move'] > 0 else f"{row['Pct_Move']:.2f}%"
        print(f"{row['Symbol']:<15} {row['Power_Score']:<12.1f} | {row['Turnover_PR']:>8.2f} PR | {row['Momentum_PR']:>8.2f} PR | {row['Hurst_PR']:>6.2f} PR | {move_str:<8} ₹{row['LTP']:<10.2f}")

if __name__ == "__main__":
    if os.path.exists("historical_fno.csv"):
        scan_historical_pristine_breakouts()
    
    target_date = datetime.now().strftime("%Y-%m-%d")
    scan_live_cumulative_compression(target_date)
