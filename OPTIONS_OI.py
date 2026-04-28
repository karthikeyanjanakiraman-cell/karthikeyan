import os
import logging
import pandas as pd
from fyers_apiv3 import fyersModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def run():
    client_id = os.getenv('CLIENT_ID')
    token = os.getenv('ACCESS_TOKEN')
    if not client_id or not token:
        print("Credentials missing")
        return

    fyers = fyersModel.FyersModel(client_id=client_id, token=token)
    expiry = os.getenv('ACTIVE_EXPIRY', '2026-04-30')
    symbols = []

    if os.path.isdir('sectors'):
        for f in os.listdir('sectors'):
            if f.endswith('.csv'):
                df = pd.read_csv(os.path.join('sectors', f))
                symbols.extend(df.iloc[:, 0].dropna().astype(str).tolist())

    all_chains = []
    for sym in list(set(symbols)):
        try:
            s = sym.strip().upper().replace('NSE:', '').replace('-EQ', '')
            res = fyers.optionchain(data={'symbol': f"NSE:{s}-EQ", 'strikecount': 50, 'expiry': expiry})
            if res.get('s') == 'ok':
                data = res.get('data', {}).get('optionsChain') or res.get('data', {}).get('optionschain')
                if data:
                    df = pd.DataFrame(data)
                    df['Underlying'] = s
                    all_chains.append(df)
        except Exception: continue

    if all_chains:
        df = pd.concat(all_chains)
        df['volume'] = pd.to_numeric(df.get('volume', 0), errors='coerce')
        df[df['optionType'] == 'CE'].sort_values('volume', ascending=False).head(15).to_csv('top_15_longs.csv', index=False)
        df[df['optionType'] == 'PE'].sort_values('volume', ascending=False).head(15).to_csv('top_15_shorts.csv', index=False)
        print("Done.")

if __name__ == '__main__':
    run()
