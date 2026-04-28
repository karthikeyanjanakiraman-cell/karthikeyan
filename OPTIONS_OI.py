
import os
import logging
import pandas as pd
import numpy as np
from fyers_apiv3 import fyersModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger('Scanner')

def get_fyers():
    return fyersModel.FyersModel(
        client_id=os.getenv('CLIENT_ID'), 
        token=os.getenv('ACCESS_TOKEN')
    )

def load_fno_symbols(root_dir='sectors'):
    symbols = set()
    if os.path.exists(root_dir):
        for f in os.listdir(root_dir):
            if f.endswith('.csv'):
                df = pd.read_csv(os.path.join(root_dir, f))
                symbols.update(df.iloc[:,0].dropna().astype(str).str.strip().unique())
    return sorted(list(symbols))

def run_standalone_scanner():
    fyers = get_fyers()
    expiry = os.getenv('ACTIVE_EXPIRY', '2026-04-30')
    symbols = load_fno_symbols()
    current_data = []

    for sym in symbols:
        try:
            s = sym.upper().replace('NSE:', '').replace('-EQ', '')
            res = fyers.optionchain(data={'symbol': f"NSE:{s}-EQ", 'strikecount': 50, 'expiry': expiry})
            if res.get('s') == 'ok':
                data = res.get('data', {}).get('optionsChain') or res.get('data', {}).get('optionschain')
                if data:
                    df = pd.DataFrame(data)
                    df['Underlying'] = sym
                    current_data.append(df)
        except Exception:
            continue

    if current_data:
        full_df = pd.concat(current_data)
        full_df['volume'] = pd.to_numeric(full_df.get('volume', 0), errors='coerce')
        full_df = full_df.sort_values('volume', ascending=False)

        longs = full_df[full_df['optionType'] == 'CE'].head(15)
        shorts = full_df[full_df['optionType'] == 'PE'].head(15)

        longs.to_csv('top_15_longs.csv', index=False)
        shorts.to_csv('top_15_shorts.csv', index=False)
        logger.info("Top 15 candidates calculated and saved.")

if __name__ == '__main__':
    run_standalone_scanner()
