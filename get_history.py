import os
import requests
import pandas as pd
import json
import gzip
import io

def get_dynamic_fno_universe():
    print("🌐 Step 1: Downloading Live NSE_FO Master to identify F&O stocks...")
    fo_url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE_FO.json.gz"
    fo_response = requests.get(fo_url)
    fo_data = json.load(gzip.GzipFile(fileobj=io.BytesIO(fo_response.content)))
    
    # Extract unique underlying stock symbols that have active F&O contracts
    fno_symbols = set()
    for item in fo_data:
        if item.get("underlying_symbol"):
            fno_symbols.add(item.get("underlying_symbol"))
            
    print(f"✅ Found {len(fno_symbols)} active F&O underlying symbols.")
    
    print("🌐 Step 2: Downloading Live NSE_EQ Master to get exact Instrument Keys...")
    eq_url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
    eq_response = requests.get(eq_url)
    eq_data = json.load(gzip.GzipFile(fileobj=io.BytesIO(eq_response.content)))
    
    # Match the F&O symbols to their Equity keys
    fno_universe = []
    for item in eq_data:
        if item.get("segment") == "NSE_EQ" and item.get("trading_symbol") in fno_symbols:
            fno_universe.append({
                "symbol": item.get("trading_symbol"),
                "key": item.get("instrument_key")
            })
            
    print(f"✅ Successfully mapped {len(fno_universe)} F&O Instrument Keys.")
    return fno_universe

def download_fno_history():
    access_token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    from_date = os.environ.get("PARAM_FROM_DATE")
    to_date = os.environ.get("PARAM_TO_DATE")
    
    if not access_token:
        print("❌ Error: UPSTOX_ACCESS_TOKEN missing.")
        return

    # Fetch the live list directly from the exchange files
    fno_universe = get_dynamic_fno_universe()
    
    all_rows = []
    print(f"📥 Fetching Daily Candles from {from_date} to {to_date}...")

    for asset in fno_universe:
        symbol = asset["symbol"]
        key = asset["key"]
        
        url = f"https://api.upstox.com/v2/historical-candle/{key}/day/{to_date}/{from_date}"
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
        # Sort chronologically so the Neural Network reads it in order
        df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)
        df.to_csv("historical_fno.csv", index=False)
        print("🎉 Success! Saved all data to 'historical_fno.csv'.")
    else:
        print("❌ No data collected.")

if __name__ == "__main__":
    download_fno_history()

