import os
import gzip
import json
import io
import urllib.parse
from datetime import datetime, timedelta
import requests
import pandas as pd

# 1. Map the Upstox Index Names to CSV Symbols
INDICES_TO_FETCH = {
    "NIFTY 50": "NIFTY_50",
    "NIFTY BANK": "NIFTY_BANK",
    "NIFTY IT": "NIFTY_IT",
    "NIFTY AUTO": "NIFTY_AUTO",
    "NIFTY METAL": "NIFTY_METAL",
    "NIFTY FMCG": "NIFTY_FMCG",
    "NIFTY ENERGY": "NIFTY_ENERGY"
}

def generate_historical_indices_csv(filename="historical_indices.csv", days_back=365):
    print("🌐 Downloading Upstox NSE Master Contract to find index keys...")
    nse_url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
    response = requests.get(nse_url)
    
    if response.status_code != 200:
        print("❌ Failed to download Upstox Master Contract.")
        return
        
    try:
        nse_data = json.load(gzip.GzipFile(fileobj=io.BytesIO(response.content)))
        
        # Extract instrument keys for indices
        index_keys = {}
        for item in nse_data:
            if item.get("segment") == "NSE_INDEX":
                ts = item.get("trading_symbol")
                if ts in INDICES_TO_FETCH:
                    index_keys[INDICES_TO_FETCH[ts]] = item.get("instrument_key")
                    
        print(f"✅ Found keys for indices: {list(index_keys.keys())}")
        
    except Exception as e:
        print(f"❌ Error parsing Master Contract: {e}")
        return
        
    access_token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not access_token:
        print("❌ UPSTOX_ACCESS_TOKEN environment variable missing!")
        return
        
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {access_token}'}
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    all_data = []
    
    for symbol_name, instrument_key in index_keys.items():
        print(f"Fetching historical data for {symbol_name}...")
        encoded_key = urllib.parse.quote(instrument_key)
        url = f"https://api.upstox.com/v2/historical-candle/{encoded_key}/day/{to_date}/{from_date}"
        
        try:
            res = requests.get(url, headers=headers)
            if res.status_code == 200:
                candles = res.json().get('data', {}).get('candles', [])
                if candles:
                    # Upstox format: [timestamp, open, high, low, close, volume, oi]
                    df = pd.DataFrame(candles, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
                    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
                    df['Symbol'] = symbol_name
                    
                    # Sort oldest to newest
                    df = df.sort_values('Date').reset_index(drop=True)
                    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol']]
                    
                    all_data.append(df)
                    print(f"✅ {symbol_name}: Loaded {len(df)} candles.")
                else:
                    print(f"⚠️ No candles returned for {symbol_name}")
            else:
                print(f"⚠️ API Error for {symbol_name}: HTTP {res.status_code} - {res.text}")
                
        except Exception as e:
            print(f"❌ Failed processing {symbol_name}: {e}")
            
        # Rate limit protection
        import time
        time.sleep(0.2)
        
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv(filename, index=False)
        print(f"🎉 Success! '{filename}' generated with {len(final_df)} rows from Upstox.")
    else:
        print("❌ No index data was compiled.")

if __name__ == "__main__":
    generate_historical_indices_csv(filename="historical_indices.csv", days_back=3650)
