import os
import gzip
import json
import io
import urllib.parse
from datetime import datetime, timedelta
import requests
import pandas as pd
import argparse
import time

INDICES_TO_FETCH = {
    "NIFTY 50": "NIFTY_50",
    "NIFTY BANK": "NIFTY_BANK",
    "NIFTY IT": "NIFTY_IT",
    "NIFTY AUTO": "NIFTY_AUTO",
    "NIFTY METAL": "NIFTY_METAL",
    "NIFTY FMCG": "NIFTY_FMCG",
    "NIFTY ENERGY": "NIFTY_ENERGY"
}

def fetch_chunked_historical_data(instrument_key, access_token, from_date_str, to_date_str, chunk_size=90):
    """Loops backward in safe chunks between the specified from_date and to_date."""
    all_candles = []
    
    start_dt = datetime.strptime(from_date_str, "%Y-%m-%d")
    end_dt = datetime.strptime(to_date_str, "%Y-%m-%d")
    
    if start_dt > end_dt:
        print("❌ Error: 'from_date' cannot be later than 'to_date'.")
        return []
        
    current_end = end_dt
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {access_token}'}
    encoded_key = urllib.parse.quote(instrument_key)
    
    while current_end >= start_dt:
        current_start = max(current_end - timedelta(days=chunk_size), start_dt)
        
        to_str = current_end.strftime("%Y-%m-%d")
        from_str = current_start.strftime("%Y-%m-%d")
        
        url = f"https://api.upstox.com/v2/historical-candle/{encoded_key}/day/{to_str}/{from_str}"
        
        try:
            res = requests.get(url, headers=headers)
            if res.status_code == 200:
                candles = res.json().get('data', {}).get('candles', [])
                if candles:
                    all_candles.extend(candles)
            else:
                print(f"⚠️ API Chunk Error ({from_str} to {to_str}): HTTP {res.status_code}")
        except Exception as e:
            print(f"❌ Connection error during chunk fetch: {e}")
            
        # Shift window backward
        current_end = current_start - timedelta(days=1)
        time.sleep(0.15) # Rate limit safety
        
    return all_candles

def generate_historical_indices_csv(from_date_str, to_date_str, filename="output/historical_indices.csv"):
    print(f"🌐 Downloading Upstox NSE Master Contract (Range: {from_date_str} to {to_date_str})...")
    nse_url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
    response = requests.get(nse_url)
    
    if response.status_code != 200:
        print("❌ Failed to download Upstox Master Contract.")
        return
        
    try:
        nse_data = json.load(gzip.GzipFile(fileobj=io.BytesIO(response.content)))
        index_keys = {}
        for item in nse_data:
            if item.get("segment") == "NSE_INDEX":
                ts = item.get("trading_symbol")
                if ts in INDICES_TO_FETCH:
                    index_keys[INDICES_TO_FETCH[ts]] = item.get("instrument_key")
        print(f"✅ Found keys for: {list(index_keys.keys())}")
    except Exception as e:
        print(f"❌ Error parsing Master Contract: {e}")
        return
        
    access_token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not access_token:
        print("❌ UPSTOX_ACCESS_TOKEN missing!")
        return
        
    all_data = []
    
    for symbol_name, instrument_key in index_keys.items():
        print(f"Fetching chunked history for {symbol_name}...")
        candles = fetch_chunked_historical_data(instrument_key, access_token, from_date_str, to_date_str, chunk_size=90)
        
        if candles:
            df = pd.DataFrame(candles, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
            # Safely handle Upstox ISO-8601 date format strings 
            df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.strftime('%Y-%m-%d')
            df['Symbol'] = symbol_name
            
            # Filter strictly within the requested bounds, drop duplicates, sort
            df = df[(df['Date'] >= from_date_str) & (df['Date'] <= to_date_str)]
            df = df.drop_duplicates(subset=['Date']).sort_values('Date').reset_index(drop=True)
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol']]
            
            all_data.append(df)
            print(f"✅ {symbol_name}: Compiled {len(df)} candles.")
        else:
            print(f"⚠️ No data retrieved for {symbol_name}")
            
    if all_data:
        # Ensure the output directory exists before saving
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv(filename, index=False)
        print(f"🎉 Success! '{filename}' generated safely between {from_date_str} and {to_date_str}.")
    else:
        print("❌ No index data was compiled.")

if __name__ == "__main__":
    # Use argparse to properly map the flags passed by your GitHub Actions YAML
    parser = argparse.ArgumentParser(description="Generate Historical Indices Data")
    parser.add_argument("--start-date", type=str, default="2024-01-01", help="Start Date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=datetime.now().strftime("%Y-%m-%d"), help="End Date (YYYY-MM-DD)")
    
    # Accept the other parameters sent by your YAML so the script doesn't crash on unrecognized arguments
    parser.add_argument("--interval", type=str, help="Interval parameter (ignored by this specific script)")
    parser.add_argument("--lookback", type=str, help="Lookback window (ignored by this specific script)")
    parser.add_argument("--universe", type=str, help="Asset universe (ignored by this specific script)")
    
    args = parser.parse_args()

    f_date = args.start_date.strip()
    t_date = args.end_date.strip()
    
    print(f"ℹ️ Running query from {f_date} to {t_date}")
        
    generate_historical_indices_csv(f_date, t_date, filename="output/historical_indices.csv")
    
