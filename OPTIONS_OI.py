
import os
import sys
import logging
import sqlite3
import smtplib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

try:
    from fyers_apiv3 import fyersModel
except Exception:
    fyersModel = None

# --- Logger Setup ---
logger = logging.getLogger('OptionsScanner')
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(ch)

# --- Config ---
ACTIVE_EXPIRY = os.getenv('ACTIVE_EXPIRY', '2026-04-30')
ATM_STRIKE_RANGE = int(os.getenv('ATM_STRIKE_RANGE', '3'))

def format_equity_symbol(symbol: str) -> str:
    s = str(symbol).strip().upper()
    return s if s.startswith('NSE:') else f'NSE:{s}-EQ'

# --- FIXED API CALL ---
def fetch_option_chain_equity(fyers, symbol, expiry):
    # Fyers API V3 Option Chain payload requirements:
    # 1. symbol: Underlying equity symbol (e.g., NSE:RELIANCE-EQ)
    # 2. strikecount: integer (e.g., 50)
    # 3. expiry: date string (YYYY-MM-DD)

    # Try the two most common formats to resolve "valid inputs" errors
    possible_syms = [format_equity_symbol(symbol), symbol.strip().upper()]

    for sym in possible_syms:
        try:
            payload = {
                'symbol': sym,
                'strikecount': 50,
                'expiry': expiry
            }
            res = fyers.optionchain(data=payload)

            # Check for API success
            if isinstance(res, dict) and res.get('s') == 'ok':
                data = res.get('data', {})
                chain = data.get('optionsChain', []) or data.get('optionschain', []) or []
                if chain:
                    logger.info(f'[OK] Chain found for {sym}')
                    df = pd.DataFrame(chain)
                    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
                    return df

            # Log failure details for debugging
            msg = res.get("message", "No message") if isinstance(res, dict) else "Non-dict response"
            logger.info(f'[DEBUG] Empty/Failed chain for {sym}: {msg}')

        except Exception as e:
            logger.warning(f'[WARN] API call failed for {sym}: {e}')

    return pd.DataFrame()

def main():
    # Placeholder for the rest of your logic...
    print("Pipeline ready with corrected API parameters.")

if __name__ == '__main__':
    main()
