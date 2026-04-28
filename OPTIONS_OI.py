
import os
import logging
import pandas as pd
from fyers_apiv3 import fyersModel

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def run():
    client_id = os.getenv('CLIENT_ID')
    token = os.getenv('ACCESS_TOKEN')
    expiry = os.getenv('ACTIVE_EXPIRY', '2026-04-30')

    if not client_id or not token:
        logging.error("Credentials (CLIENT_ID/ACCESS_TOKEN) are missing.")
        return

    # Initialize Fyers Model
    fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=token, log_path="")

    symbols = []
    if os.path.isdir('sectors'):
        for f in os.listdir('sectors'):
            if f.endswith('.csv'):
                try:
                    df = pd.read_csv(os.path.join('sectors', f))
                    symbols.extend(df.iloc[:, 0].dropna().astype(str).tolist())
                except Exception as e:
                    logging.debug(f"Error reading {f}: {e}")

    unique_symbols = list(set(symbols))
    logging.info(f"Loaded {len(unique_symbols)} unique symbols from sectors/")

    all_chains = []
    for sym in unique_symbols:
        s = sym.strip().upper().replace('NSE:', '').replace('-EQ', '')
        payload = {
            "symbol": f"NSE:{s}-EQ",
            "strikecount": 50,
            "expiry": expiry
        }

        try:
            # MUST use data=payload to avoid "Invalid method" 400 error in Fyers v3
            res = fyers.optionchain(data=payload)

            if isinstance(res, dict) and res.get('s') == 'ok':
                data = res.get('data', {}).get('optionsChain')
                if data:
                    temp_df = pd.DataFrame(data)
                    temp_df['Underlying'] = s
                    all_chains.append(temp_df)
            else:
                logging.debug(f"API Error for {s}: {res}")
        except Exception as e:
            logging.debug(f"Exception for {s}: {e}")
            continue

    if all_chains:
        full_df = pd.concat(all_chains, ignore_index=True)
        # Ensure volume is numeric for sorting
        full_df['volume'] = pd.to_numeric(full_df.get('volume', 0), errors='coerce').fillna(0)

        # Rank by volume desc
        longs = full_df[full_df['optionType'] == 'CE'].sort_values('volume', ascending=False).head(15)
        shorts = full_df[full_df['optionType'] == 'PE'].sort_values('volume', ascending=False).head(15)

        longs.to_csv('top_15_longs.csv', index=False)
        shorts.to_csv('top_15_shorts.csv', index=False)
        logging.info("Successfully generated top_15_longs.csv and top_15_shorts.csv.")
    else:
        logging.error("No option chain data retrieved. Check ACTIVE_EXPIRY format (YYYY-MM-DD) and API token.")

if __name__ == '__main__':
    run()
