import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from fyers_apiv3 import fyersModel

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger('VolScanner')

def get_fyers():
    return fyersModel.FyersModel(
        client_id=os.getenv('CLIENT_ID'), 
        token=os.getenv('ACCESS_TOKEN')
    )

def load_all_fno_symbols(root_dir='sectors'):
    symbols = set()
    if os.path.exists(root_dir):
        for f in os.listdir(root_dir):
            if f.endswith('.csv'):
                df = pd.read_csv(os.path.join(root_dir, f))
                symbols.update(df.iloc[:,0].dropna().astype(str).str.strip().unique())
    return sorted(list(symbols))

def fetch_chain(fyers, symbol):
    s = symbol.upper().replace('NSE:', '').replace('-EQ', '')
    try:
        # Fyers requires specific expiry format
        res = fyers.optionchain(data={
            'symbol': f"NSE:{s}-EQ", 
            'strikecount': 50, 
            'expiry': os.getenv('ACTIVE_EXPIRY', '2026-04-30')
        })
        if res.get('s') == 'ok':
            chain = res.get('data', {}).get('optionsChain') or res.get('data', {}).get('optionschain')
            if chain: return pd.DataFrame(chain)
    except: pass
    return pd.DataFrame()

def process_iteration():
    fyers = get_fyers()
    symbols = load_all_fno_symbols()

    all_data = []
    for sym in symbols:
        df = fetch_chain(fyers, sym)
        if not df.empty:
            df['Underlying'] = sym
            # Relative Volume Logic (Simplified)
            df['vol'] = pd.to_numeric(df.get('volume', 0), errors='coerce')
            # Assuming you have a way to pull yesterday's data; for now, we surge by vol
            df['Surge'] = df['vol'] 
            all_data.append(df)

    if all_data:
        full_df = pd.concat(all_data)
        # Sort by Primary Volume
        full_df = full_df.sort_values('vol', ascending=False)

        # Rank Long (CE/Price UP) vs Short (PE/Price DOWN)
        # Using simple heuristic: CE vol vs PE vol
        longs = full_df[full_df['optionType'] == 'CE'].head(15)
        shorts = full_df[full_df['optionType'] == 'PE'].head(15)

        longs.to_csv('top_15_longs.csv', index=False)
        shorts.to_csv('top_15_shorts.csv', index=False)
        logger.info("Iteration complete. Top 15 candidates saved.")

if __name__ == '__main__':
    process_iteration()
