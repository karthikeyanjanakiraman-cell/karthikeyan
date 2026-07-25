import os
import requests
import pandas as pd
import json
import gzip
import io
import urllib.parse
import time  # <-- Added for speed braking
from datetime import datetime, timedelta

def get_dynamic_fno_universe():
    print("🌐 Downloading Live Upstox NSE Master Contract...")
    nse_url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
    
    response = requests.get(nse_url)
    if response.status_code != 200:
        print(f"❌ Failed to download NSE Master. HTTP {response.status_code}")
        return []

    try:
        nse_data = json.load(gzip.GzipFile(fileobj=io.BytesIO(response.content)))
        
        fno_underlying_symbols = set()
        for item in nse_data:
            if item.get("segment") == "NSE_FO" and item.get("underlying_symbol"):
                fno_underlying_symbols.add(item.get("underlying_symbol"))
                
        fno_universe = []
        for item in nse_data:
            if item.get("segment") in ("NSE_EQ", "NSE_INDEX") and item.get("trading_symbol") in fno_underlying_symbols:
                fno_universe.append({
                    "symbol": item.get("trading_symbol"),
                    "key": item.get("instrument_key")
                })
                
        print(f"✅ Successfully mapped {len(fno_universe)} F&O Instrument Keys.")
        return fno_universe
        
    except Exception as e:
        print(f"❌ JSON Parsing Error: {str(e)}")
        return []

def download_fno_history():
    access_token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    from_date_str = os.environ.get("PARAM_FROM_DATE", "2024-01-01")
    to_date_str = os.environ.get("PARAM_TO_DATE", datetime.now().strftime("%Y-%m-%d"))
    
    if not access_token:
        print("❌ Error: UPSTOX_ACCESS_TOKEN missing.")
        return

    fno_universe = get_dynamic_fno_universe()
    if not fno_universe:
        return
    
    all_rows = []
    print(f"📥 Fetching Daily Candles from {from_date_str} to {to_date_str} (Using 100-Day Chunks)...")

    final_start = datetime.strptime(from_date_str, "%Y-%m-%d")
    final_end = datetime.strptime(to_date_str, "%Y-%m-%d")

    for asset in fno_universe:
        symbol = asset["symbol"]
        encoded_key = urllib.parse.quote(asset["key"])
        
        current_end = final_end
        asset_candles = []
        
        while current_end > final_start:
            # SHIFT TO 100-DAY CHUNKS to bypass Upstox Data Limits
            current_start = max(current_end - timedelta(days=100), final_start)
            
            str_to = current_end.strftime("%Y-%m-%d")
            str_from = current_start.strftime("%Y-%m-%d")
            
            url = f"https://api.upstox.com/v2/historical-candle/{encoded_key}/day/{str_to}/{str_from}"
            headers = {'Accept': 'application/json', 'Authorization': f'Bearer {access_token}'}
            
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json().get('data', {}).get('candles', [])
                if data:
                    asset_candles.extend(data)
            else:
                print(f"⚠️ API Error for {symbol} ({str_from} to {str_to}): HTTP {response.status_code} - {response.text}")
                break 
                
            current_end = current_start - timedelta(days=1)
            
            # CRITICAL SPEED BRAKE: Prevents the 10 requests/second rate limit crash
            time.sleep(0.3) 
            
        if asset_candles:
            for c in asset_candles:
                all_rows.append({
                    'Date': c[0].split('T')[0],
                    'Symbol': symbol,
                    'Open': float(c[1]),
                    'High': float(c[2]),
                    'Low': float(c[3]),
                    'Close': float(c[4]),
                    'Volume': float(c[5])
                })
            print(f"✅ Extracted {len(asset_candles)} candles for {symbol}")

    if all_rows:
        df = pd.DataFrame(all_rows)
        df = df.drop_duplicates(subset=['Symbol', 'Date'])
        df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)
        df.to_csv("historical_fno.csv", index=False)
        print(f"🎉 Success! Saved {len(df)} rows of data to 'historical_fno.csv'.")
    else:
        print("❌ No data collected. Check the API error messages above.")

if __name__ == "__main__":
    download_fno_history()
