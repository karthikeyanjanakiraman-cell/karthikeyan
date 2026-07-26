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
            
        current_end = current_start - timedelta(days=1)
        time.sleep(0.15) # Rate limit safety
        
    return all_candles

def generate_historical_csv(from_date_str, to_date_str, filename="output/historical_fno.csv"):
    # 1. Load the target symbols from the CSV
    target_symbols = set()
    try:
        stock_df = pd.read_csv("fno_stock_list.csv")
        # Auto-detect the symbol column (looks for 'Symbol', 'Ticker', 'Name', or defaults to the first column)
        col_name = None
        for col in stock_df.columns:
            if col.strip().upper() in ["SYMBOL", "TICKER", "STOCK", "NAME"]:
                col_name = col
                break
        if not col_name:
            col_name = stock_df.columns[0]
            
        target_symbols = set(stock_df[col_name].astype(str).str.upper().str.strip())
        print(f"✅ Loaded {len(target_symbols)} targets from fno_stock_list.csv (Column: {col_name})")
    except Exception as e:
        print(f"❌ Could not read fno_stock_list.csv: {e}")
        return

    # 2. Download Master Contract
    print(f"🌐 Downloading Upstox NSE Master Contract (Range: {from_date_str} to {to_date_str})...")
    nse_url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
    response = requests.get(nse_url)
    
    if response.status_code != 200:
        print("❌ Failed to download Upstox Master Contract.")
        return
        
    # 3. Map target symbols to Upstox Instrument Keys
    try:
        nse_data = json.load(gzip.GzipFile(fileobj=io.BytesIO(response.content)))
        index_keys = {}
        
        for item in nse_data:
            # Check both stocks (NSE_EQ) and indices (NSE_INDEX)
            if item.get("segment") in ["NSE_EQ", "NSE_INDEX"]:
                ts = str(item.get("trading_symbol", "")).upper()
                nm = str(item.get("name", "")).upper()
                
                # Upstox usually adds -EQ to stock symbols (e.g., RELIANCE-EQ). Strip it for matching.
                base_ts = ts.replace("-EQ", "")
                
                if base_ts in target_symbols:
                    index_keys[base_ts] = item.get("instrument_key")
                elif nm in target_symbols:
                    index_keys[nm] = item.get("instrument_key")
                    
        print(f"✅ Successfully mapped {len(index_keys)} out of {len(target_symbols)} targets to Upstox Instrument Keys.")
    except Exception as e:
        print(f"❌ Error parsing Master Contract: {e}")
        return
        
    # 4. Fetch the data
    access_token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not access_token:
        print("❌ UPSTOX_ACCESS_TOKEN missing! (Check GitHub Repository Settings > Secrets)")
        return
        
    all_data = []
    
    for symbol_name, instrument_key in index_keys.items():
        print(f"Fetching chunked history for {symbol_name}...")
        candles = fetch_chunked_historical_data(instrument_key, access_token, from_date_str, to_date_str, chunk_size=90)
        
        if candles:
            df = pd.DataFrame(candles, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
            df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.strftime('%Y-%m-%d')
            df['Symbol'] = symbol_name
            
            df = df[(df['Date'] >= from_date_str) & (df['Date'] <= to_date_str)]
            df = df.drop_duplicates(subset=['Date']).sort_values('Date').reset_index(drop=True)
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol']]
            
            all_data.append(df)
        else:
            print(f"⚠️ No data retrieved for {symbol_name}")
            
    # 5. Compile and save
    if all_data:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv(filename, index=False)
        print(f"🎉 Success! '{filename}' generated safely with {len(all_data)} distinct assets.")
    else:
        print("❌ No data was compiled.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Historical F&O Data")
    parser.add_argument("--start-date", type=str, default="2024-01-01", help="Start Date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=datetime.now().strftime("%Y-%m-%d"), help="End Date (YYYY-MM-DD)")
    parser.add_argument("--interval", type=str, help="Interval parameter")
    parser.add_argument("--lookback", type=str, help="Lookback window")
    parser.add_argument("--universe", type=str, help="Asset universe")
    
    args = parser.parse_args()

    f_date = args.start_date.strip()
    t_date = args.end_date.strip()
    
    print(f"ℹ️ Running query from {f_date} to {t_date}")
    # Changed output filename to match your ML pipeline structure
    generate_historical_csv(f_date, t_date, filename="output/historical_fno.csv")
    
