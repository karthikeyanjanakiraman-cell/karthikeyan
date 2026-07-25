import os
import requests
import pandas as pd
import json
import gzip
import io
import urllib.parse  # <-- Added to translate the pipe character
from datetime import datetime

def get_dynamic_fno_universe():
    print("🌐 Downloading Live Upstox NSE Master Contract (this may take a few seconds)...")
    nse_url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
    
    response = requests.get(nse_url)
    if response.status_code != 200:
        print(f"❌ Failed to download NSE Master. HTTP {response.status_code}")
        return []

    print("📦 Decompressing and mapping the exchange file...")
    try:
        nse_data = json.load(gzip.GzipFile(fileobj=io.BytesIO(response.content)))
        
        fno_underlying_symbols = set()
        for item in nse_data:
            if item.get("segment") == "NSE_FO" and item.get("underlying_symbol"):
                fno_underlying_symbols.add(item.get("underlying_symbol"))
                
        print(f"✅ Discovered {len(fno_underlying_symbols)} underlying F&O assets.")
        
        fno_universe = []
        for item in nse_data:
            segment = item.get("segment")
            symbol = item.get("trading_symbol")
            
            if segment in ("NSE_EQ", "NSE_INDEX") and symbol in fno_underlying_symbols:
                fno_universe.append({
                    "symbol": symbol,
                    "key": item.get("instrument_key")
                })
                
        print(f"✅ Successfully mapped {len(fno_universe)} Instrument Keys for download.")
        return fno_universe
        
    except Exception as e:
        print(f"❌ JSON Parsing Error: {str(e)}")
        return []

def download_fno_history():
    access_token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    from_date = os.environ.get("PARAM_FROM_DATE", "2024-01-01")
    to_date = os.environ.get("PARAM_TO_DATE", datetime.now().strftime("%Y-%m-%d"))
    
    if not access_token:
        print("❌ Error: UPSTOX_ACCESS_TOKEN missing in GitHub Secrets.")
        return

    fno_universe = get_dynamic_fno_universe()
    if not fno_universe:
        print("⚠️ Universe empty. Aborting download.")
        return
    
    all_rows = []
    print(f"📥 Fetching Daily Candles from {from_date} to {to_date}...")

    for asset in fno_universe:
        symbol = asset["symbol"]
        key = asset["key"]
        
        # URL-ENCODE THE KEY to fix the HTTP 400 error (translates the '|' to '%7C')
        encoded_key = urllib.parse.quote(key)
        
        url = f"https://api.upstox.com/v2/historical-candle/{encoded_key}/day/{to_date}/{from_date}"
        headers = {'Accept': 'application/json', 'Authorization': f'Bearer {access_token}'}
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"⚠️ Failed to fetch {symbol}: HTTP {response.status_code}")
            continue
            
        candles = response.json().get('data', {}).get('candles', [])
        for c in candles:
            all_rows.append({
                'Date': c[0].split('T')[0],
                'Symbol': symbol,
                'Open': float(c[1]),
                'High': float(c[2]),
                'Low': float(c[3]),
                'Close': float(c[4]),
                'Volume': float(c[5])
            })
        print(f"✅ Extracted {len(candles)} candles for {symbol}")

    if all_rows:
        df = pd.DataFrame(all_rows)
        df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)
        df.to_csv("historical_fno.csv", index=False)
        print(f"🎉 Success! Saved {len(df)} rows of data to 'historical_fno.csv'.")
    else:
        print("❌ No data collected. Please verify your UPSTOX_ACCESS_TOKEN and dates.")

if __name__ == "__main__":
    download_fno_history()
