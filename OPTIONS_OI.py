import os
import logging
import pandas as pd
from fyers_apiv3 import fyersModel

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def run():
    client_id = os.getenv('CLIENT_ID')
    token = os.getenv('ACCESS_TOKEN')
    expiry = os.getenv('ACTIVE_EXPIRY', '2026-04-28')

    if not client_id or not token:
        logging.error("Credentials missing.")
        return

    fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=token)

    symbols = []
    if os.path.isdir('sectors'):
        for f in os.listdir('sectors'):
            if f.endswith('.csv'):
                try:
                    df = pd.read_csv(os.path.join('sectors', f))
                    symbols.extend(df.iloc[:, 0].dropna().astype(str).tolist())
                except: continue

    unique_symbols = list(set(symbols))
    logging.info(f"Scanning {len(unique_symbols)} symbols for expiry {expiry}")

    all_chains = []
    for sym in unique_symbols:
        s = sym.strip().upper().replace('NSE:', '').replace('-EQ', '')

        # Try two common variants for V3 API symbols
        for symbol_variant in [f"NSE:{s}-EQ", f"NSE:{s}"]:
            try:
                payload = {"symbol": symbol_variant, "strikecount": 50, "expiry": expiry}
                res = fyers.optionchain(data=payload)

                if isinstance(res, dict) and res.get('s') == 'ok':
                    data = res.get('data', {}).get('optionsChain') or res.get('data', {}).get('optionschain')
                    if data:
                        temp_df = pd.DataFrame(data)
                        temp_df['Underlying'] = s
                        all_chains.append(temp_df)
                        logging.info(f"Found data for {symbol_variant}")
                        break # Successfully fetched, move to next symbol
            except Exception as e:
                continue

    if all_chains:
        full_df = pd.concat(all_chains, ignore_index=True)
        full_df['volume'] = pd.to_numeric(full_df.get('volume', 0), errors='coerce').fillna(0)

        longs = full_df[full_df['optionType'] == 'CE'].sort_values('volume', ascending=False).head(15)
        shorts = full_df[full_df['optionType'] == 'PE'].sort_values('volume', ascending=False).head(15)

        longs.to_csv('top_15_longs.csv', index=False)
        shorts.to_csv('top_15_shorts.csv', index=False)
        logging.info("Analysis complete: top_15_longs.csv and top_15_shorts.csv generated.")
    else:
        logging.error("No option data retrieved. Please verify ACTIVE_EXPIRY is a Thursday date in YYYY-MM-DD format.")

if __name__ == '__main__':
    run()
